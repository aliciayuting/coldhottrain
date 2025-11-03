from typing import Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

def _bytes_to_mb(x: int) -> float:
    return x / (1024 ** 2)

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size() if t is not None else 0

def model_param_nbytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

def model_buffer_nbytes(model: torch.nn.Module) -> int:
    return sum(b.numel() * b.element_size() for b in model.buffers())

def model_grad_nbytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += tensor_nbytes(p.grad)
    return total

def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    # Sum all tensors stored in optimizer.state for each param
    total = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for v in state.values():
                if torch.is_tensor(v):
                    total += tensor_nbytes(v)
    return total

class VramBreakdownCallback(TrainerCallback):
    """
    Logs breakdown to TensorBoard:
      - params, buffers, grads, optimizer_state
      - activation_estimate (pre/post forward alloc delta)
      - cuda_now_allocated / cuda_max_allocated
    Works with gradient accumulation: logs on the last microstep of each optimizer step.
    """
    def __init__(self, log_dir: str = None):
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()
        self._pre_fwd_alloc = None
        self._post_fwd_alloc = None
        self._handles = []
        self._global_step = 0

    # ---- register hooks once we see the model ----
    def _register_forward_hooks(self, model: torch.nn.Module):
        if not torch.cuda.is_available():
            return
        def _pre(_m, _inp):
            torch.cuda.synchronize()
            self._pre_fwd_alloc = torch.cuda.memory_allocated()
        def _post(_m, _inp, _out):
            torch.cuda.synchronize()
            self._post_fwd_alloc = torch.cuda.memory_allocated()
        # Use top module hooks (one pair is enough)
        self._handles = [
            model.register_forward_pre_hook(_pre),
            model.register_forward_hook(_post),
        ]

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            self._register_forward_hooks(model)

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called every training step, *after* loss.backward() and optimizer.step() if it occurs on this step.
        We’ll log on `state.global_step` changes (HF manages accumulation internally).
        """
        self._global_step = state.global_step
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)

        if model is None:
            return
        

        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"No gradient for {name}")
            else:
                print(f"Gradient for {name}: {param.grad.norm()}")

        # Compute sizes
        params_bytes = model_param_nbytes(model)
        buffers_bytes = model_buffer_nbytes(model)
        grads_bytes = model_grad_nbytes(model)

        opt_bytes = optimizer_state_nbytes(optimizer) if optimizer is not None else 0

        # Activation estimate from forward hook
        activation_est = 0
        if self._pre_fwd_alloc is not None and self._post_fwd_alloc is not None:
            diff = self._post_fwd_alloc - self._pre_fwd_alloc
            activation_est = max(diff, 0)

        # CUDA allocator telemetry
        cuda_now = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        cuda_max = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        cuda_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        cuda_reserved_max = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0


        sum = params_bytes + buffers_bytes + grads_bytes + opt_bytes + activation_est


        step = self._global_step
        self.writer.add_scalar("mem/params_MB", _bytes_to_mb(params_bytes), step)
        self.writer.add_scalar("mem/buffers_MB", _bytes_to_mb(buffers_bytes), step)
        self.writer.add_scalar("mem/grads_MB", _bytes_to_mb(grads_bytes), step)
        self.writer.add_scalar("mem/optimizer_state_MB", _bytes_to_mb(opt_bytes), step)
        self.writer.add_scalar("mem/activation_estimate_MB", _bytes_to_mb(activation_est), step)
        self.writer.add_scalar("mem/sum_MB", _bytes_to_mb(sum), step)
        self.writer.add_scalar("mem/cuda_now_MB", _bytes_to_mb(cuda_now), step)
        self.writer.add_scalar("mem/cuda_peak_MB", _bytes_to_mb(cuda_max), step)

        # Compact console line (useful in logs)
        print(
            f"[Pre optimizer {step}] params={_bytes_to_mb(params_bytes):.1f}MB "
            f"buf={_bytes_to_mb(buffers_bytes):.1f}MB "
            f"grads={_bytes_to_mb(grads_bytes):.1f}MB "
            f"opt={_bytes_to_mb(opt_bytes):.1f}MB "
            f"act~={_bytes_to_mb(activation_est):.1f}MB "
            f"sum={_bytes_to_mb(sum):.1f}MB "
            f"now={_bytes_to_mb(cuda_now):.1f}MB "
            f"peak={_bytes_to_mb(cuda_max):.1f}MB"
            f" resv={_bytes_to_mb(cuda_reserved):.1f}MB"
            f" resv_peak={_bytes_to_mb(cuda_reserved_max):.1f}MB"
        )


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called every training step, *after* loss.backward() and optimizer.step() if it occurs on this step.
        We’ll log on `state.global_step` changes (HF manages accumulation internally).
        """
        self._global_step = state.global_step
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)

        if model is None:
            return

        # Compute sizes
        params_bytes = model_param_nbytes(model)
        buffers_bytes = model_buffer_nbytes(model)
        grads_bytes = model_grad_nbytes(model)

        opt_bytes = optimizer_state_nbytes(optimizer) if optimizer is not None else 0

        # Activation estimate from forward hook
        activation_est = 0
        if self._pre_fwd_alloc is not None and self._post_fwd_alloc is not None:
            diff = self._post_fwd_alloc - self._pre_fwd_alloc
            activation_est = max(diff, 0)

        # CUDA allocator telemetry
        cuda_now = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        cuda_max = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        cuda_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        cuda_reserved_max = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0


        sum = params_bytes + buffers_bytes + grads_bytes + opt_bytes + activation_est


        step = self._global_step
        self.writer.add_scalar("mem/params_MB", _bytes_to_mb(params_bytes), step)
        self.writer.add_scalar("mem/buffers_MB", _bytes_to_mb(buffers_bytes), step)
        self.writer.add_scalar("mem/grads_MB", _bytes_to_mb(grads_bytes), step)
        self.writer.add_scalar("mem/optimizer_state_MB", _bytes_to_mb(opt_bytes), step)
        self.writer.add_scalar("mem/activation_estimate_MB", _bytes_to_mb(activation_est), step)
        self.writer.add_scalar("mem/sum_MB", _bytes_to_mb(sum), step)
        self.writer.add_scalar("mem/cuda_now_MB", _bytes_to_mb(cuda_now), step)
        self.writer.add_scalar("mem/cuda_peak_MB", _bytes_to_mb(cuda_max), step)

        # Compact console line (useful in logs)
        print(
            f"[Step end {step}] params={_bytes_to_mb(params_bytes):.1f}MB "
            f"buf={_bytes_to_mb(buffers_bytes):.1f}MB "
            f"grads={_bytes_to_mb(grads_bytes):.1f}MB "
            f"opt={_bytes_to_mb(opt_bytes):.1f}MB "
            f"act~={_bytes_to_mb(activation_est):.1f}MB "
            f"sum={_bytes_to_mb(sum):.1f}MB "
            f"now={_bytes_to_mb(cuda_now):.1f}MB "
            f"peak={_bytes_to_mb(cuda_max):.1f}MB"
            f" resv={_bytes_to_mb(cuda_reserved):.1f}MB"
            f" resv_peak={_bytes_to_mb(cuda_reserved_max):.1f}MB"
        )
    

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.writer.flush()
        self.writer.close()
