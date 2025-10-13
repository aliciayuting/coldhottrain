from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification

# import torch.nn
import torch
import torch.nn as nn
import torch.distributed as dist
import hashlib
from module import EmbeddingColWise, LinearColWise


def get_decoder_layers(m: nn.Module):
    """
    Return the list-like container of decoder blocks for common HF decoder-only models.
    Works for Qwen2ForCausalLM (m.model.layers).
    """
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
        return m.transformer.layers
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError("Could not locate decoder layers (tried model.layers / transformer.layers / layers).")

# ---- your previously defined helpers (from my last message) ----
def make_hot_idx(out_features: int, frac: float | None = None, idx: torch.Tensor | None = None, device=None):
    if idx is not None:
        hot = torch.as_tensor(idx, dtype=torch.long, device=device)
        assert hot.ndim == 1 and hot.numel() > 0
        assert hot.min().item() >= 0 and hot.max().item() < out_features
        return hot
    assert frac is not None and 0.0 <= frac <= 1.0
    k = max(0, min(out_features, int(round(frac * out_features))))
    #TODO: remove this later
    if k == out_features:
        k = out_features - 1
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    perm = torch.randperm(out_features, device=device)
    return perm[:k].sort().values

import torch

def make_hot_idx_n(
    out_features: int,
    n: int | None = None,
    device=None,
) -> torch.Tensor:

    assert n is not None and isinstance(n, int)
    k = max(0, min(out_features, int(n)))

    # TODO: remove this later (kept to match original semantics)
    if k == out_features:
        k = out_features - 1

    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    perm = torch.randperm(out_features, device=device)
    return perm[:k].sort().values

def replace_linear_with_colwise(mod: nn.Module, hot_idx: torch.Tensor, mode: str = "1linear_efficient") -> LinearColWise:
    assert isinstance(mod, nn.Linear)
    hot_idx = hot_idx.to(mod.weight.device)
    wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx, mode=mode)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped

def replace_embedding_with_colwise(mod: nn.Module, hot_idx: torch.Tensor) -> EmbeddingColWise:
    assert isinstance(mod, nn.Embedding)
    hot_idx = hot_idx.to(mod.weight.device)
    wrapped = EmbeddingColWise.from_embedding(mod, hot_idx=hot_idx)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped

# ---- apply to layer 23 of Qwen2.5 -------------------------------------------
policy_by_name = {
    "self_attn.q_proj": 0.30,
    "self_attn.k_proj": 0.30,
    "self_attn.v_proj": 0.30,
    "self_attn.o_proj": 0.30,
    "mlp.up_proj":      0.30,
    "mlp.down_proj":    0.30,
    # "mlp.gate_proj":  0.30,  # include if you want to split the gate too
}


def model_shape_signature(model):
    """Return an ordered list of (name, shape, dtype, is_buffer) for params + buffers."""
    sig = []
    # Parameters
    for n, p in model.named_parameters():
        sig.append((f"PARAM::{n}", tuple(p.shape), str(p.dtype), False))
    # Buffers (e.g., running stats in norm layers)
    for n, b in model.named_buffers():
        sig.append((f"BUFFER::{n}", tuple(b.shape), str(b.dtype), True))
    # Keep a deterministic order
    sig.sort(key=lambda x: x[0])
    return sig

def verify_shapes_across_ranks(model):
    """Gather all ranks' shape signatures and assert they are identical."""
    my_sig = model_shape_signature(model)
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world == 1:
        print("[verify] Single process – shapes OK by definition.")
        return

    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, my_sig)

    # Compare against rank 0's signature
    ref = gathered[0]
    for r, sig in enumerate(gathered):
        if sig != ref:
            # Find and print the diff for easier debugging
            ref_set = set(ref)
            sig_set = set(sig)
            missing = ref_set - sig_set
            extra   = sig_set - ref_set
            raise RuntimeError(
                f"[verify] Rank {r} model signature differs!\n"
                f"  Missing on rank {r} (present on rank 0): {list(missing)[:10]}\n"
                f"  Extra on rank {r} (absent on rank 0): {list(extra)[:10]}"
            )
    if dist.get_rank() == 0:
        print("[verify] All ranks have identical parameter/buffer dimensions and dtypes.")

def tensor_sha256(t: torch.Tensor) -> str:
    # Move to CPU, make contiguous, then view as bytes without changing values.
    tb = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(tb).hexdigest()

def model_weights_checksum(model):
    """Return ordered list of (name, sha256) for params + buffers."""
    checks = []
    for n, p in model.named_parameters():
        checks.append((f"PARAM::{n}", tensor_sha256(p)))
    for n, b in model.named_buffers():
        checks.append((f"BUFFER::{n}", tensor_sha256(b)))
    checks.sort(key=lambda x: x[0])
    return checks

def verify_weights_across_ranks(model):
    """Assert params/buffers are bit-identical across ranks."""
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world == 1:
        print("[verify] Single process – weights OK by definition.")
        return

    my_chk = model_weights_checksum(model)
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, my_chk)

    ref = gathered[0]
    for r, chk in enumerate(gathered):
        if chk != ref:
            # Find the first mismatch to help you locate it
            for (n0, h0), (n1, h1) in zip(ref, chk):
                if (n0 != n1) or (h0 != h1):
                    raise RuntimeError(
                        f"[verify] Weight mismatch at rank {r}: {n0} vs {n1}; "
                        f"hash {h0[:8]} != {h1[:8]}"
                    )
            raise RuntimeError(f"[verify] Weight mismatch at rank {r}")
    if dist.get_rank() == 0:
        print("[verify] All ranks have bit-identical weights/buffers.")