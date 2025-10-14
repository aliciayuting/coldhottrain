import time
from transformers import TrainerCallback
import torch.distributed as dist
import torch
import logging
from helper import *
from module import EmbeddingColWise, LinearColWise
from accelerate.utils import extract_model_from_parallel

logger = logging.getLogger(__name__)

def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class HotSwapCallback(TrainerCallback):
    # def on_train_begin(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[on_train_begin] max_steps={state.max_steps} epochs={args.num_train_epochs}")

    # def on_epoch_begin(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[on_epoch_begin] epoch_float={state.epoch}")

    # def on_train_batch_end(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[on_train_batch_end] global_step={state.global_step}")

    # def on_step_end(self, args, state, control, **kwargs):
    #     if _is_main():
    #         if state.global_step % 5 == 0:
    #             print(f"[probe on_step_end] global_step={state.global_step}")
    #             log_memory_stats()
    #     # Fires after optimizer step; here global_step has just incremented
        
            
    # def on_substep_end(self, args, state, control, **kwargs):
    #     # Fires after optimizer step; here global_step has just incremented
    #     if _is_main():
    #         print(f"[probe on_substep_end] global_step={state.global_step}")
    
    
    
    # def on_pre_optimizer_step(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[probe on_pre_optimizer_step] global_step={state.global_step}")


    def _unwrap_optimizer(self, opt):
        # Walk through any wrappers until we hit the real torch optimizer
        while hasattr(opt, "optimizer"):   # AcceleratedOptimizer has .optimizer
            opt = opt.optimizer
        return opt
    
    def ddp_bucket_bytes(self, ddp_model):
        r = ddp_model.reducer
        return sum(b.buffer().numel() * b.buffer().element_size() for b in r._buckets)

    #TODO: this is before gradients are zeroed out. make sure this is not an issue
    def on_optimizer_step(self, args, state, control, **kwargs):
        #if state.global_step % 20 != 0 or state.global_step <= 0:
        if state.global_step % 20 != 0:
            return
        wrapped = kwargs.get("model")
        wrapped2 = kwargs.get("model_wrapped")
        if wrapped is None:
            logger.warning("HotSwapCallback: model not found in kwargs")
            return  # rare, but be defensive
        self.model = extract_model_from_parallel(wrapped)
                
        optimizer = kwargs.get("optimizer", None)
        if optimizer is None:
            logger.warning("no optimizer :(")
            return
        optimizer = self._unwrap_optimizer(optimizer)

        optimizer.zero_grad(set_to_none=True)

        logger.info(f"[hotswap on_optimizer_step] global_step={state.global_step}")
        logger.info("bucket bytes BEFORE:", self.ddp_bucket_bytes(wrapped2))

        with torch.no_grad():
            layers: nn.Module = get_decoder_layers(self.model)
            for layer in layers:
                for name, mod in layer.named_modules():
                    if isinstance(mod, LinearColWise):
                        #logger.info(f"found {name}")
                        mod.switch_hot(new_hot_idx=make_hot_idx_n(out_features=mod.out_features, n=mod.hot_idx.numel(), device=mod.W_hot.device), keep_state=True, optimizer=optimizer)
        logger.info("bucket bytes AFTER:", self.ddp_bucket_bytes(wrapped2))