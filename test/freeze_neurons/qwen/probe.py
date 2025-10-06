from transformers import TrainerCallback
import torch.distributed as dist
import torch
import logging

logger = logging.getLogger(__name__)

def log_memory_stats():
        """Log current GPU memory statistics"""
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Max Allocated: {max_allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        logging.info(f"torch.cuda.memory_summary():\n{torch.cuda.memory_summary()}")

def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class Probe(TrainerCallback):
    # def on_train_begin(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[on_train_begin] max_steps={state.max_steps} epochs={args.num_train_epochs}")

    # def on_epoch_begin(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[on_epoch_begin] epoch_float={state.epoch}")

    # def on_train_batch_end(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[on_train_batch_end] global_step={state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        if _is_main():
            if state.global_step % 5 == 0:
                print(f"[probe on_step_end] global_step={state.global_step}")
                log_memory_stats()
        # Fires after optimizer step; here global_step has just incremented
        
            
    # def on_substep_end(self, args, state, control, **kwargs):
    #     # Fires after optimizer step; here global_step has just incremented
    #     if _is_main():
    #         print(f"[probe on_substep_end] global_step={state.global_step}")
    
    
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[probe on_pre_optimizer_step] global_step={state.global_step}")