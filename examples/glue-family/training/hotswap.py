import time
from transformers import TrainerCallback
import torch.distributed as dist
import torch
import logging
from training.helper import *
from model.custom_module import LinearColWise
from model.linear_elementwise import LinearElementwise
from accelerate.utils import extract_model_from_parallel

logger = logging.getLogger(__name__)

def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class HotSwapCallback(TrainerCallback):
    def __init__(self, swap_iters=100, elementwise_scheme= "all"):
        self.model = None  # will be set on first call
        self.swap_iters = swap_iters
        self.elementwise_scheme = elementwise_scheme


    def _unwrap_optimizer(self, opt):
        # Walk through any wrappers until we hit the real torch optimizer
        while hasattr(opt, "optimizer"):   # AcceleratedOptimizer has .optimizer
            opt = opt.optimizer
        return opt
    
    def ddp_bucket_bytes(self, ddp_model):
        r = ddp_model.reducer
        return sum(b.buffer().numel() * b.buffer().element_size() for b in r._buckets)
    
    def bytes_on_cuda_state(self, optim):
        total = 0
        for st in optim.state.values():
            for v in st.values():
                if torch.is_tensor(v) and v.is_cuda:
                    total += v.numel() * v.element_size()
        return total


    #TODO: this is before gradients are zeroed out. make sure this is not an issue
    def on_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step % self.swap_iters != 0 or state.global_step <= 0:
            return
        wrapped = kwargs.get("model")
        if wrapped is None:
            logger.warning("HotSwapCallback: model not found in kwargs")
            return  # rare, but be defensive
        self.model = extract_model_from_parallel(wrapped)
                
        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            logger.warning("no optimizer :(")
            return
        optimizer = self._unwrap_optimizer(optimizer)
        #print(f"settings {optimizer.foreach} {optimizer.capturable} {optimizer.fused}")
        print("CUDA bytes in optimizer state:", self.bytes_on_cuda_state(optimizer))
        optimizer.zero_grad(set_to_none=True)

        logger.info(f"[hotswap on_optimizer_step] global_step={state.global_step}")
        
        with torch.no_grad():
            layers: nn.Module = get_decoder_layers(self.model)
            for layer in layers:
                for name, mod in layer.named_modules():
                    if isinstance(mod, LinearColWise):
                        #logger.info(f"found {name}")
                        new_hot_idx = make_hot_idx_n(out_features=mod.out_features, n=mod.hot_idx.numel(), device=mod.W_hot.device)
                        assert new_hot_idx.shape == mod.hot_idx.shape
                        #new_hot_idx = mod.hot_idx.clone()
                        mod.switch_hot(new_hot_idx=new_hot_idx, keep_state=True, optimizer=optimizer)
                        mod._assert_optimizer_has(optimizer)
                    if isinstance(mod, LinearElementwise):
                        assert False, "hotswap for LinearElementwise is not fully tested yet"
                        if self.elementwise_scheme == "neuron":
                            new_hot_idx = make_hot_idx_n(out_features=mod.out_features, n=mod.bias_idx.numel(), device=mod.vals.device)
                            w_idx, b_idx = build_elementwise_indices_from_hotidx(
                                out_features=mod.out_features,
                                in_features=mod.in_features,
                                hot_idx=new_hot_idx,
                                device=mod.vals.device,
                            )
                            mod.hotswap(
                                new_weight_indices=w_idx,
                                new_bias_indices=b_idx,
                                keep_state=True,
                                optimizer=optimizer,
                            )
                        elif self.elementwise_scheme == "all":
                            frac = float(mod.bias_idx.numel()) / mod.out_features
                            w_idx, b_idx = build_elementwise_indices_from_random(
                                out_features=mod.out_features,
                                in_features=mod.in_features,
                                frac=frac,
                                device=mod.vals.device,
                            )
                        elif self.elementwise_scheme == "input":
                            new_hot_idx = make_hot_idx_n(out_features=mod.in_features, n=mod.store_n, device=mod.vals.device)
                            w_idx, b_idx = build_elementwise_indices_from_hotidx_input_features(
                                out_features=mod.out_features,
                                in_features=mod.in_features,
                                hot_idx=new_hot_idx,
                                device=mod.vals.device,
                            )
                            mod.hotswap(
                                new_weight_indices=w_idx,
                                new_bias_indices=b_idx,
                                keep_state=True,
                                optimizer=optimizer,
                            )
                        else:
                            raise ValueError(f"Unsupported elementwise_scheme: {self.elementwise_scheme}")

        # verify_shapes_across_ranks(self.model)
        # verify_weights_across_ranks(self.model)
            # for g in optimizer.param_groups:
            #     g['params'] = list(g['params'])  # reassign to break potential cached views
