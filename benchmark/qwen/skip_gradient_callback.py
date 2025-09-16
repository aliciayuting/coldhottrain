from transformers import TrainerCallback, PreTrainedModel
import torch.distributed as dist

def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class SkipGradientCallback(TrainerCallback):
    def __init__(self,
                 model: PreTrainedModel,
                 epoch_start_track=1,
                 epoch_compute_masks=2,
                 zero_bottom_k_percent=0.1,
                 zero_mode="neurons",
                 ):
            self.model = model
            self.epoch_start_track = epoch_start_track
            self.epoch_compute_masks = epoch_compute_masks
            self.zero_bottom_k_percent = zero_bottom_k_percent
            self.zero_mode = zero_mode
            self._has_computed_masks = False

            self.neuron_masks = {}

           
    def on_train_begin(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[on_train_begin] max_steps={state.max_steps} epochs={args.num_train_epochs}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[on_epoch_begin] epoch_float={state.epoch}")
        if state.epoch == self.epoch_compute_masks:
            self._compute_masks()

    def on_train_batch_end(self, args, state, control, **kwargs):
        if _is_main():
            print(f"[on_train_batch_end] global_step={state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        # Fires after optimizer step; here global_step has just incremented
        if _is_main():
            print(f"[on_step_end] global_step={state.global_step}")
            
    def on_substep_end(self, args, state, control, **kwargs):
        # Fires after optimizer step; here global_step has just incremented
        if _is_main():
            print(f"[on_substep_end] global_step={state.global_step}")
