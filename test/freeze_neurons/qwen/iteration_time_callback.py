import time
import torch
from transformers import TrainerCallback

class IterationTimeCallback(TrainerCallback):
    def __init__(self, iter_times=None):
        self._t0 = None
        self.iter_times = iter_times if iter_times is not None else []

    def on_step_begin(self, args, state, control, **kwargs):
        # Start a stopwatch for the *optimizer step* (not micro-steps).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # (Optional) make sure all ranks finished this step before we stop the clock
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self._t0 is not None and state.is_world_process_zero:
            dt = time.time() - self._t0
            self.iter_times.append(dt)
            # print (or write to a file) only on rank 0
            print(f"[step {state.global_step}] iteration_time_after_agg_sec = {dt:.6f}")
            # If you prefer structured logs that integrate with Trainer’s logger:
            # kwargs.get("logs", {}) won't help here; instead do:
            # from transformers.trainer_utils import speed_metrics
            # metrics = {"iteration_time_sec": dt}
            # control.should_log = True  # ensure on_log gets called soon
            # return control  # (printing is usually simplest)
        return control