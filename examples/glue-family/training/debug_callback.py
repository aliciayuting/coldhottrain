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

class DebugCallback(TrainerCallback):
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
        wrapped = kwargs.get("model")
        if wrapped is None:
            logger.warning("DebugCallback: model not found in kwargs")
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

        for param, state in optimizer.state.items():
            print(f"--- param: {id(param)} shape: {param.shape} ---")
            for skey, sval in state.items():
                if torch.is_tensor(sval):
                    print(f"\tstate key: {skey} shape: {sval.shape} device:{sval.device} dtype: {sval.dtype}")
                else:
                    print(f"\tstate key: {skey} value: {sval} type: {type(sval)}")
