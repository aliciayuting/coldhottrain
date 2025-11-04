import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from collections import defaultdict

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
    Logs VRAM breakdown to TensorBoard:
      - per-module params_bytes
      - per-module activation_estimate_bytes (delta alloc around each module's forward)
      - global cuda_now_allocated / cuda_max_allocated

    Works with gradient accumulation: you can log only on the last microstep.
    """
    def __init__(self, log_dir: str = None):
        # self.writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()
        self._handles = []
        self._entire_model_handles = []
        self._pre_fwd_alloc = None
        self._post_fwd_alloc = None
        self._global_step = 0

        # Per-module bookkeeping
        self._mod_pre_alloc = {}
        self._mod_act_bytes = defaultdict(int)
        self._mod_param_bytes = {}

        # In case you want to know when hooks are registered
        self._hooks_registered = False

    # Helper: register hooks on leaf modules
    def _register_forward_hooks(self, model: torch.nn.Module):
        if self._hooks_registered:
            return
        if not torch.cuda.is_available():
            return
        

        def _pre(_m, _inp):
            torch.cuda.synchronize()
            self._pre_fwd_alloc = torch.cuda.memory_allocated()
        def _post(_m, _inp, _out):
            torch.cuda.synchronize()
            self._post_fwd_alloc = torch.cuda.memory_allocated()

        self._entire_model_handles = [
            model.register_forward_pre_hook(_pre),
            model.register_forward_hook(_post),
        ]

        for name, module in model.named_modules():
            # Skip top-level container modules; only hook leaves
            if any(module.children()):
                continue

            # Compute and store param bytes for this module
            param_bytes = 0
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    param_bytes += p.numel() * p.element_size()
            self._mod_param_bytes[name] = param_bytes

            def _pre(m, inp, name=name):
                torch.cuda.synchronize()
                self._mod_pre_alloc[name] = torch.cuda.memory_allocated()

            def _post(m, inp, out, name=name):
                torch.cuda.synchronize()
                pre = self._mod_pre_alloc.get(name, None)
                if pre is None:
                    return
                post = torch.cuda.memory_allocated()
                # delta = max(0, post - pre)
                delta = post - pre
                # accumulate across the batch / microsteps
                self._mod_act_bytes[name] += delta

            self._handles.append(module.register_forward_pre_hook(_pre))
            self._handles.append(module.register_forward_hook(_post))

        self._hooks_registered = True

    # ---- HF Trainer integration ----

    def on_train_begin(self, args, state, control, **kwargs):
        # We get the model from kwargs["model"]
        model = kwargs.get("model", None)
        if model is not None:
            self._register_forward_hooks(model)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """
        Called at the end of a *optimizer* step.
        If you need per-microstep logging, use on_substep_end instead.
        """
        if not torch.cuda.is_available():
            return
        
        # if self._global_step < 4:
        #     return

        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
 
        # Global stats
        torch.cuda.synchronize()
        now = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        step = int(state.global_step)

        # self.writer.add_scalar("vram/global_now_bytes", now, step)
        # self.writer.add_scalar("vram/global_peak_bytes", peak, step)

        # # Per-module params + activation estimates
        # for name, pbytes in self._mod_param_bytes.items():
        #     self.writer.add_scalar(f"vram/params_bytes/{name}", pbytes, step)


        s = 0
        for name, abytes in self._mod_act_bytes.items():
            # abytes is the *accumulated* per-step activation estimate
            # self.writer.add_scalar(f"vram/activations_bytes/{name}", abytes, step)
            print(f"Step {step} - Module {name}: Activation: {_bytes_to_mb(abytes)} MB")
            s += abytes


        # Reset per-step activation accumulators
        self._mod_act_bytes = defaultdict(int)


        cuda_max = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0


        activation_est = 0
        if self._pre_fwd_alloc is not None and self._post_fwd_alloc is not None:
            diff = self._post_fwd_alloc - self._pre_fwd_alloc
            activation_est = max(diff, 0)

        params_bytes = model_param_nbytes(model)
        buffers_bytes = model_buffer_nbytes(model)
        grads_bytes = model_grad_nbytes(model)
        opt_bytes = optimizer_state_nbytes(optimizer) if optimizer is not None else 0


        print(f"Step {step} - Entire model: Activation estimate: {_bytes_to_mb(activation_est):.1f} MB sum of per-module: {_bytes_to_mb(s):.1f} MB peak: {_bytes_to_mb(cuda_max):.1f} MB params: {_bytes_to_mb(params_bytes):.1f} MB buffers: {_bytes_to_mb(buffers_bytes):.1f} MB grads: {_bytes_to_mb(grads_bytes):.1f} MB optim: {_bytes_to_mb(opt_bytes):.1f} MB")

        self._global_step = step

    def on_train_end(self, args, state, control, **kwargs):
        for h in self._handles:
            h.remove()
        for h in self._entire_model_handles:
            h.remove()
        # self.writer.close()
