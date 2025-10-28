import os, csv, time
import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback


def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class PerModuleGradDumper(TrainerCallback):
    """
    At specified global steps, dumps gradients for selected submodules to .npy files
    and appends rows to index.csv with metadata including epoch indices.
    """

    def __init__(self,
                 out_dir="grad_dump",
                 model=None,
                 capture_steps=100,
                 include_bias=False,
                 also_embeddings=False,
                 mkdir_mode=0o755):
        self.out_dir = out_dir
        self.capture_steps = capture_steps
        self.include_bias = bool(include_bias)
        self.also_embeddings = bool(also_embeddings)
        self.mkdir_mode = mkdir_mode

        os.makedirs(self.out_dir, exist_ok=True)
        self.index_path = os.path.join(self.out_dir, "index.csv")
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "global_step","epoch_idx","epoch_float","step_in_epoch",
                    "layer","submodule","param","shape","dtype","numel","sum_g2","file"
                ])

        self._rank = 0
        self._world = 1

        # Track epoch & local steps inside the epoch
        self._epoch_idx = -1           # integer epoch counter we control
        self._step_in_epoch = 0
        
        self._model = model
        

    # def _is_main(self):
    #     return (self._rank == 0)

    # def _setup_ddp(self):
    #     if dist.is_available() and dist.is_initialized():
    #         self._rank = dist.get_rank()
    #         self._world = dist.get_world_size()

    # -------- Trainer hooks --------
    # def on_train_begin(self, args, state, control, model=None, **kwargs):
    #     if _is_main():
    #         print(f"[on_train_begin] max_steps={state.max_steps} epochs={args.num_train_epochs}")
            # self._model = kwargs.get("model", None)
            # print(f"!!! {self._model is not None}")

    
    # def on_epoch_begin(self, args, state, control, **kwargs):
    #     if _is_main():
            # print(f"[on_epoch_begin] epoch_float={state.epoch}")
            # New epoch started; bump integer epoch counter and reset local step
            # self._epoch_idx += 1
            # self._step_in_epoch = 0

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # if _is_main():
        #     print(f"[grad dump on_step_end] global_step={state.global_step}")

        # Increment local step-in-epoch first (so first batch is 1)
        self._step_in_epoch += 1

        step = state.global_step
        # if step not in self.capture_steps:
        #     return
        if step % self.capture_steps != 0:
            return
        if not _is_main():
            return

        print(f"[PerModuleGradDumper] Capturing gradients at step {step}")

        # epoch_float from TrainerState (may be None early on, so guard)
        epoch_float = float(state.epoch) if state.epoch is not None else float(self._epoch_idx)
        
        
        model = self._model

        # Resolve layers list (Qwen/LLaMA style)
        # layers = None
        # for cand in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
        #     try:
        #         layers = eval(f"model.{cand}")
        #         break
        #     except Exception:
        #         pass
            
        layers = None
        for cand in ["model.model.layers", "model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
            try:
                layers = eval(f"m.{cand}", {"m": self._model})
                break
            except Exception:
                pass
        
        if layers is None:
            print("[PerModuleGradDumper] Could not locate transformer layers; skipping.")
            return
        # else:
        #     print(f"layers: {layers}")

        # Prepare step dir
        step_dir = os.path.join(self.out_dir, f"step{step:06d}")
        os.makedirs(step_dir, exist_ok=True, mode=self.mkdir_mode)

        rows = []  # rows to append to CSV at the end

        def dump_param(layer_id: int, submod_name: str, pname: str, tensor: torch.Tensor):
            if tensor is None or tensor.grad is None:
                print(f"[PerModuleGradDumper WARNING] {submod_name}.{pname} (no grad)")
                return
            g = tensor.grad.detach().to(torch.float32)  # keep exact shape in .npy
            shape = tuple(g.shape)
            numel = int(g.numel())
            sum_g2 = float((g.flatten()**2).sum().item())
            dtype = "float32"

            fn = f"L{layer_id:02d}_{submod_name}_{pname.replace('.', '_')}.npy"
            fpath = os.path.join(step_dir, fn)
            np.save(fpath, g.cpu().numpy(), allow_pickle=False)

            rows.append([
                step,                      # global_step
                self._epoch_idx,           # epoch_idx (int, starts at 0)
                f"{epoch_float:.6f}",      # epoch_float (from TrainerState)
                self._step_in_epoch,       # step_in_epoch (local within epoch)
                layer_id, submod_name, pname,
                str(shape), dtype, numel, f"{sum_g2:.6e}",
                os.path.relpath(fpath, start=self.out_dir)
            ])

        # Iterate layers and collect standard projections
        for lid, blk in enumerate(layers):
            attn = getattr(blk, "self_attn", getattr(blk, "attention", None))
            mlp  = getattr(blk, "mlp", getattr(blk, "feed_forward", None))

            # Attention projections
            if attn is not None:
                for nm in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "qkv_proj"]:
                    sub = getattr(attn, nm, None)
                    if sub is None:
                        continue
                    if hasattr(sub, "weight"):
                        dump_param(lid, "self_attn", f"{nm}.weight", sub.weight)
                    if self.include_bias and hasattr(sub, "bias") and sub.bias is not None:
                        dump_param(lid, "self_attn", f"{nm}.bias", sub.bias)
            else:
                print(f"attn is None")

            # MLP projections (SwiGLU)
            if mlp is not None:
                for nm in ["gate_proj", "up_proj", "down_proj"]:
                    sub = getattr(mlp, nm, None)
                    if sub is None:
                        continue
                    if hasattr(sub, "weight"):
                        dump_param(lid, "mlp", f"{nm}.weight", sub.weight)
                    if self.include_bias and hasattr(sub, "bias") and sub.bias is not None:
                        dump_param(lid, "mlp", f"{nm}.bias", sub.bias)

        # Optionally embeddings / lm_head
        if self.also_embeddings:
            for name in ["model.embed_tokens", "transformer.wte", "wte", "tok_embeddings"]:
                emb = _safe_getattr_chain(self._model, name)
                if emb is not None and hasattr(emb, "weight"):
                    dump_param(-1, "embeddings", "weight", emb.weight)
                    break
            for name in ["lm_head"]:
                lm = _safe_getattr_chain(self._model, name)
                if lm is not None:
                    if hasattr(lm, "weight"):
                        dump_param(-1, "lm_head", "weight", lm.weight)
                    elif isinstance(lm, torch.Tensor) and lm.grad is not None:
                        class _Wrap: pass
                        w = _Wrap(); w.grad = lm.grad
                        dump_param(-1, "lm_head", "weight", w)

        if rows:
            with open(self.index_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerows(rows)

        print(f"[PerModuleGradDumper] step={step} epoch_idx={self._epoch_idx} "
              f"step_in_epoch={self._step_in_epoch} dumped {len(rows)} tensors -> {step_dir}")

    
    # def on_substep_end(self, args, state, control, **kwargs):
    #     # Fires after optimizer step; here global_step has just incremented
    #     if _is_main():
    #         print(f"[probe on_substep_end] global_step={state.global_step}")
            
    # def on_pre_optimizer_step(self, args, state, control, **kwargs):
    #     if _is_main():
    #         print(f"[probe on_pre_optimizer_step] global_step={state.global_step}")

def _safe_getattr_chain(root, dotted):
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur
