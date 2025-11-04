from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification

# import torch.nn
import torch
import torch.nn as nn
import torch.distributed as dist
import hashlib
from module import EmbeddingColWise, LinearColWise
from linear_elementwise import LinearElementwise

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
    all_out = torch.arange(mod.out_features, device=hot_idx.device)
    cold_idx = all_out[~torch.isin(all_out, hot_idx)]

    #print("Non-trainable output indices:", cold_idx)

    wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx, mode=mode)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped

def get_non_trainable_indices(in_features, out_features: int, w, b) -> torch.Tensor:
    total_out, total_in = out_features, in_features
    all_w = torch.arange(total_out * total_in, device=w.device)
    hot_w = w[:, 0] * total_in + w[:, 1]
    cold_w = all_w[~torch.isin(all_w, hot_w)]
    cold_w_pairs = torch.stack((cold_w // total_in, cold_w % total_in), dim=1)

    all_b = torch.arange(total_out, device=b.device)
    cold_b = all_b[~torch.isin(all_b, b)]

    print("Non-trainable weight indices:", cold_w_pairs)
    print("Non-trainable bias indices:", cold_b)

def build_elementwise_indices_from_hotidx(out_features: int, in_features: int, hot_idx: torch.Tensor, device=None):
    device = hot_idx.device

    # If no rows are hot, return empty indices in the right shape/dtype
    if hot_idx.numel() == 0:
        raise ValueError("hot_rows must contain at least one index")
        w_idx = torch.empty(0, 2, dtype=torch.long, device=device)
        b_idx = torch.empty(0, dtype=torch.long, device=device)
    else:
        # Build all (row, col) pairs for those rows
        cols = torch.arange(in_features, device=device, dtype=torch.long)
        # meshgrid gives two [#hot, in_features] grids; stack -> [..., 2] then flatten
        R, C = torch.meshgrid(hot_idx, cols, indexing="ij")  # PyTorch >=1.10
        w_idx = torch.stack((R.reshape(-1), C.reshape(-1)), dim=1).contiguous()

        # Bias indices are just the hot rows themselves
        b_idx = hot_idx.contiguous()
    return w_idx, b_idx

def build_elementwise_indices_from_hotidx_input_features(out_features: int, in_features: int, hot_idx: torch.Tensor, device=None):
    device = hot_idx.device

    if hot_idx.numel() == 0:
        raise ValueError("hot_cols must contain at least one index")

    rows = torch.arange(out_features, device=device, dtype=torch.long)
    R, C = torch.meshgrid(rows, hot_idx, indexing="ij")
    w_idx = torch.stack((R.reshape(-1), C.reshape(-1)), dim=1).contiguous()

    b_idx = None
    return w_idx, b_idx


def build_elementwise_indices_from_random(out_features: int, in_features: int, frac: float, device=None):
    num_w = out_features * in_features
    k_w = int(round(frac * num_w))
    if frac > 0.0 and k_w == 0:
        k_w = 1  # ensure at least one when frac > 0

    if k_w > 0:
        flat = torch.randperm(num_w, device="cpu")[:k_w]      # unique linear indices
        w_row = (flat // in_features).long()
        w_col = (flat %  in_features).long()
        train_weight_indices = torch.stack([w_row, w_col], dim=1)  # [k_w, 2]
    else:
        train_weight_indices = torch.empty(0, 2, dtype=torch.long)

    # ----- pick trainable BIAS indices -----
    num_b = out_features
    k_b = int(round(frac * num_b))
    if frac > 0.0 and k_b == 0:
        k_b = 1

    train_bias_indices: Optional[torch.Tensor]
    if k_b > 0:
        train_bias_indices = torch.randperm(num_b, device="cpu")[:k_b].long()  # [k_b]
    else:
        train_bias_indices = None  # no trainable bias entries
    return train_weight_indices.to(device=device), train_bias_indices.to(device=device) if train_bias_indices is not None else train_bias_indices

def replace_linear_with_elementwise_hotidx(mod: nn.Linear, hot_rows: torch.Tensor) -> "LinearElementwise":
    
    #print(f"Replacing {mod._get_name()} with LinearElementwise: {w_idx.size(0)} trainable weights, {b_idx.size(0)} trainable biases")
    #print(get_non_trainable_indices(mod.in_features, mod.out_features, w_idx, b_idx))
    w_idx, b_idx = build_elementwise_indices_from_hotidx(
        out_features=mod.out_features,
        in_features=mod.in_features,
        hot_idx=hot_rows,
        device=mod.weight.device,
    )
    le = LinearElementwise.from_linear(
        mod,
        train_weight_indices=w_idx,
        train_bias_indices=b_idx,
        )
    return le.to(device=mod.weight.device, dtype=mod.weight.dtype)

def replace_linear_with_elementwise_hotidx_input_features(mod: nn.Linear, hot_cols: torch.Tensor) -> "LinearElementwise":
    
    n = hot_cols.numel()
    w_idx, b_idx = build_elementwise_indices_from_hotidx_input_features(
        out_features=mod.out_features,
        in_features=mod.in_features,
        hot_idx=hot_cols,
        device=mod.weight.device,
    )
    print(f"Replacing {mod._get_name()} with LinearElementwise: {w_idx.size(0)} trainable weights")
    print(get_non_trainable_indices(mod.in_features, mod.out_features, w_idx, b_idx))
    le = LinearElementwise.from_linear(
        mod,
        train_weight_indices=w_idx,
        train_bias_indices=b_idx,
        store_n = n
    )
    return le.to(device=mod.weight.device, dtype=mod.weight.dtype)

def replace_linear_with_elementwise_random(mod: nn.Module, percent_hot: float) -> "LinearElementwise":
    """
    Wrap an nn.Linear in a LinearElementwise with a random subset of weights/biases trainable.

    Args:
        mod: nn.Linear to convert.
        percent_hot: fraction in [0, 1] of weights and biases to mark trainable.

    Returns:
        LinearElementwise initialized from `mod` via LinearElementwise.from_linear(...)
    """
    if not isinstance(mod, nn.Linear):
        raise TypeError("mod must be an nn.Linear")
    if not (0.0 <= percent_hot <= 1.0):
        raise ValueError("percent_hot must be in [0, 1]")

    out_features, in_features = mod.out_features, mod.in_features

    # ----- pick trainable WEIGHT indices -----
    train_weight_indices, train_bias_indices = build_elementwise_indices_from_random(
        out_features=out_features,
        in_features=in_features,
        frac=percent_hot,
        device=mod.weight.device,
    )
    # ----- build via the convenience constructor -----
    return LinearElementwise.from_linear(
        base=mod,
        train_weight_indices=train_weight_indices,
        train_bias_indices=train_bias_indices,
    )

def replace_embedding_with_colwise(mod: nn.Module, hot_idx: torch.Tensor) -> EmbeddingColWise:
    assert isinstance(mod, nn.Embedding)
    hot_idx = hot_idx.to(mod.weight.device)
    print(f"Replacing {mod._get_name()} with LinearColWise: {hot_idx.size(0)} trainable weights")
    print(hot_idx)
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

@torch.no_grad()
def model_weights_checksum(model):
    """Return ordered list of (name, sha256) for params + buffers."""
    checks = []
    for n, p in model.named_parameters():
        checks.append((f"PARAM::{n}", tensor_sha256(p)))
    for n, b in model.named_buffers():
        checks.append((f"BUFFER::{n}", tensor_sha256(b)))
    checks.sort(key=lambda x: x[0])
    return checks

@torch.no_grad()
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
