# vram_breakdown_callback.py
import torch
from transformers import TrainerCallback

def _tensor_nbytes(t):
    return 0 if t is None else t.numel() * t.element_size()


def _unique_params(module):
    seen = set()
    for p in module.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            yield p

def _model_param_bytes(model):
    return sum(_tensor_nbytes(p) for p in _unique_params(model))

# def _model_buffer_bytes(model):
#     return sum(_tensor_nbytes(b) for b in model.buffers())

def _model_buffer_bytes(model):
    seen = set()
    total = 0
    for b in model.buffers():
        bid = id(b)
        if bid not in seen:
            seen.add(bid)
            total += _tensor_nbytes(b)
    return total

def _grad_bytes(model):
    return sum(_tensor_nbytes(p.grad) for p in model.parameters() if getattr(p, "grad", None) is not None)

def _optimizer_state_bytes(optimizer):
    if optimizer is None:
        return 0
    total = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += _tensor_nbytes(v)
    return total

def _fmt_mb(b):
    return round(b / (1024 ** 2), 2)


class VramBreakdownCallback(TrainerCallback):
    """
    Logs VRAM breakdown with correct gradient timing:
      - params, buffers, grads (captured at substep end), optimizer state
      - activations estimate: forward pre→post alloc delta
      - now/peak allocated

    Works with gradient accumulation: logs on the *last* micro-step of each optimizer step,
    before grads are zeroed.
    """

    def __init__(self):
        self._pre_fwd_alloc = None
        self._post_fwd_alloc = None
        self._handles = []
        self._microstep_idx = 0  # within accumulation window

    # ---- hook helpers ----
    def _register_forward_hooks(self, model):
        if not torch.cuda.is_available():
            return

        def _pre(_mod, _inp):
            torch.cuda.synchronize()
            self._pre_fwd_alloc = torch.cuda.memory_allocated()

        def _post(_mod, _inp, _out):
            torch.cuda.synchronize()
            self._post_fwd_alloc = torch.cuda.memory_allocated()

        self._handles = [
            model.register_forward_pre_hook(_pre, with_kwargs=False),
            model.register_forward_hook(_post, with_kwargs=False),
        ]

    def _remove_hooks(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []

    # ---- TrainerCallback API ----
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            self._register_forward_hooks(model)

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_hooks()

    def on_train_batch_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            self._pre_fwd_alloc = None
            self._post_fwd_alloc = None
            #torch.cuda.reset_peak_memory_stats()
        self._microstep_idx = 0

    def on_step_end(self, args, state, control, **kwargs):
        """
        Fallback for when gradient_accumulation_steps=1
        Called after optimizer.step()
        """
        # Only use this if no gradient accumulation
        if args.gradient_accumulation_steps == 1:
            self._log_memory_stats(args, state, control, **kwargs)
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        """For gradient accumulation scenarios"""
        self._microstep_idx += 1
        
        # Only log on last micro-step
        if self._microstep_idx % args.gradient_accumulation_steps != 0:
            return
            
        self._log_memory_stats(args, state, control, **kwargs)

    def _log_memory_stats(self, args, state, control, **kwargs):
        """Extracted logging logic"""
        next_global_step = state.global_step + 1
        if args.logging_strategy == "steps" and (next_global_step % args.logging_steps != 0):
            return

        if not torch.cuda.is_available():
            return

        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        trainer = kwargs.get("trainer", None)

        torch.cuda.synchronize()
        now_alloc = torch.cuda.memory_allocated()
        peak_alloc = torch.cuda.max_memory_allocated()
        reserved_alloc = torch.cuda.memory_reserved()

        p_bytes = _model_param_bytes(model) if model is not None else 0
        b_bytes = _model_buffer_bytes(model) if model is not None else 0
        g_bytes = _grad_bytes(model) if model is not None else 0
        o_bytes = _optimizer_state_bytes(optimizer)

        act_bytes_est = 0
        if self._pre_fwd_alloc is not None and self._post_fwd_alloc is not None:
            act_bytes_est = max(0, self._post_fwd_alloc - self._pre_fwd_alloc)

        log_record = {
            "mem/params_mb": _fmt_mb(p_bytes),
            "mem/buffers_mb": _fmt_mb(b_bytes),
            "mem/grads_mb": _fmt_mb(g_bytes),
            "mem/optimizer_mb": _fmt_mb(o_bytes),
            "mem/activations_mb~": _fmt_mb(act_bytes_est),
            "mem/now_allocated_mb": _fmt_mb(now_alloc),
            "mem/peak_allocated_mb": _fmt_mb(peak_alloc),
            "mem/reserved_mb": _fmt_mb(reserved_alloc),
        }

        if trainer is not None:
            trainer.log(log_record)
        else:
            state.log_history.append(log_record)

        print(f"[mem @ next_step {next_global_step}] "
              f"params={log_record['mem/params_mb']}MB, "
              f"buffers={log_record['mem/buffers_mb']}MB, "
              f"grads={log_record['mem/grads_mb']}MB, "
              f"opt={log_record['mem/optimizer_mb']}MB, "
              f"acts~={log_record['mem/activations_mb~']}MB, "
              f"now={log_record['mem/now_allocated_mb']}MB, "
              f"peak={log_record['mem/peak_allocated_mb']}MB,"
              f"reserved={log_record['mem/reserved_mb']}MB")