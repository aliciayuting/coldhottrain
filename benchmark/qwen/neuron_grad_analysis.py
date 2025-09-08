#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_neuron_drift_pre_post.py
Build neuron-wise pre→post drift heatmaps across training steps.

What it does
------------
- Scans weight_dump/ for pairs: step{S}_pre and step{S}_post (S from your callback).
- For each (layer, operator) you choose, loads the weight tensors from both checkpoints,
  computes ΔW = post - pre, then reduces to a neuron-level vector:
    * agg='col'  : per-column   L2^2 of ΔW     (default; 'neuron' ~ column)
    * agg='row'  : per-row      L2^2 of ΔW
    * agg='head' : per-head     L2^2 of ΔW (contiguous column slices of size head_dim)
- Emits one long CSV plus a pivoted table per (layer, operator),
  and plots a heatmap: rows = neuron index, columns = global_step S, values = L2 drift.

Assumptions
-----------
- Qwen/LLaMA-like state_dict keys, e.g.:
  model.model.layers.{L}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
  model.model.layers.{L}.mlp.{gate_proj,up_proj,down_proj}.weight
- Checkpoints saved by your callback: weight_dump/stepXXXXXX_pre, stepXXXXXX_post
- Supports safetensors or pytorch_model.bin

Edit the CONFIG section and run.
"""

# =======================
# CONFIG (edit these)
# =======================
WEIGHT_ROOT = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/weight_dump"
OUT_DIR     = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/neuron_drifts"

# Which layers / operators to include
LAYERS = [0, 1, 2, 3]  # e.g., range(0, 32) for 7B
OPS = [
    # attention
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    # MLP
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
]

# Neuron aggregation mode: 'col' (default), 'row', or 'head'
AGG = "col"

# If AGG == 'head', you may provide (optional) head meta to avoid inference
HEADS_HINT = None  # e.g., {"n_heads": 8, "head_dim": 64}

# Tick density for heatmaps
MAX_X_TICKS = 12
MAX_Y_TICKS = 30

# =======================
# Imports
# =======================
import os, re, glob, math, gc, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

try:
    from safetensors.torch import load_file as safetensors_load
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False


# =======================
# Helpers
# =======================
def _gather_steps(weight_root: str):
    """Return sorted list of steps S where both dirs exist: step{S:06d}_pre and _post."""
    pre_dirs  = glob.glob(os.path.join(weight_root, "step*_pre"))
    post_dirs = set(os.path.basename(p).replace("_post", "") for p in glob.glob(os.path.join(weight_root, "step*_post")))
    steps = []
    for p in pre_dirs:
        base = os.path.basename(p)
        if not base.endswith("_pre"): 
            continue
        stem = base[:-4]  # drop _pre
        if stem in post_dirs:
            # extract integer step
            m = re.match(r"step(\d{6})", stem)
            if m:
                steps.append(int(m.group(1)))
    steps = sorted(set(steps))
    return steps

def _gather_weight_files(ckpt_dir: str):
    safes = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if safes:
        return safes
    pt = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(pt):
        return [pt]
    raise FileNotFoundError(f"No weight files in {ckpt_dir}")

def _load_state_dict_any(ckpt_dir: str) -> dict:
    files = _gather_weight_files(ckpt_dir)
    sd = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".safetensors":
            if not _HAVE_SAFETENSORS:
                raise RuntimeError("safetensors not installed; pip install safetensors")
            part = safetensors_load(f, device="cpu")
            for k, v in part.items():
                sd[k] = v.detach().to(torch.float32)
            del part
        elif ext in [".bin", ".pt", ".pth"]:
            obj = torch.load(f, map_location="cpu")
            if not isinstance(obj, dict):
                raise RuntimeError(f"Unexpected object in {f}: {type(obj)}")
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    sd[k] = v.detach().to(torch.float32)
        else:
            warnings.warn(f"Skipping {f}")
    return sd

def _key_for(layer: int, sub: str, proj: str):
    # Qwen/LLaMA path variants — adjust if needed
    return f"model.model.layers.{layer}.{sub}.{proj}.weight"

def _reduce_neuron(delta: torch.Tensor, agg: str, heads_hint=None) -> np.ndarray:
    """
    delta: weight_post - weight_pre (tensor)
    Returns 1D numpy array of neuron-level L2^2 deltas.
    """
    if delta.ndim == 1:
        # bias: treat each element as a 'neuron'
        return (delta.pow(2)).to(torch.float32).cpu().numpy()

    if agg == "col":
        # columns = input features => sum across rows (dim=0 in [out, in]?)
        # PyTorch Linear: weight [out_features, in_features]
        # per-column => sum over out_features (dim=0)
        return (delta.pow(2).sum(dim=0)).to(torch.float32).cpu().numpy()

    if agg == "row":
        # per-row => sum over in_features (dim=1)
        return (delta.pow(2).sum(dim=1)).to(torch.float32).cpu().numpy()

    if agg == "head":
        # Slice contiguous column chunks of size head_dim; sum each chunk
        if heads_hint is None:
            # try to infer n_heads/head_dim from columns
            cols = delta.shape[1]
            pick = None
            for h in range(1, 256):
                if cols % h == 0:
                    d = cols // h
                    if 16 <= d <= 256:
                        pick = (h, d); break
            if pick is None:
                raise RuntimeError(f"Cannot infer (n_heads, head_dim) from shape {tuple(delta.shape)}")
            n_heads, head_dim = pick
        else:
            n_heads = heads_hint["n_heads"]; head_dim = heads_hint["head_dim"]
            assert delta.shape[1] == n_heads * head_dim, f"columns {delta.shape[1]} != n_heads*head_dim"
        parts = []
        W = delta.to(torch.float32)
        for h in range(n_heads):
            sl = W[:, h*head_dim:(h+1)*head_dim]
            parts.append(sl.pow(2).sum().item())
        return np.array(parts, dtype=np.float64)

    raise ValueError(f"Unknown agg={agg}")

def autoscale_ticks(ax, values, axis="x", max_ticks=12):
    n = len(values)
    if n <= max_ticks:
        ticks = np.arange(n)
    else:
        ticks = np.unique(np.linspace(0, n - 1, max_ticks, dtype=int))
    labels = [str(values[i]) for i in ticks]
    if axis == "x":
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=0)
    else:
        ax.set_yticks(ticks); ax.set_yticklabels(labels)

# =======================
# Build long table
# =======================
def build_neuron_drift_long(weight_root: str, layers, ops, agg="col", heads_hint=None) -> pd.DataFrame:
    steps = _gather_steps(weight_root)
    if not steps:
        raise SystemExit(f"No step*_pre/post pairs found in {weight_root}")
    rows = []
    for S in steps:
        pre_dir  = os.path.join(weight_root, f"step{S:06d}_pre")
        post_dir = os.path.join(weight_root, f"step{S:06d}_post")
        sd_pre   = _load_state_dict_any(pre_dir)
        sd_post  = _load_state_dict_any(post_dir)

        for layer in layers:
            for sub, proj in ops:
                key = _key_for(layer, sub, proj)
                w_pre = sd_pre.get(key, None)
                w_post= sd_post.get(key, None)
                if w_pre is None or w_post is None:
                    # silently skip missing modules (some variants omit biases, fused qkv, etc.)
                    continue
                if w_pre.shape != w_post.shape:
                    warnings.warn(f"[skip] shape mismatch at step {S}: {key} {tuple(w_pre.shape)} vs {tuple(w_post.shape)}")
                    continue
                delta = (w_post - w_pre)
                vec = _reduce_neuron(delta, agg=agg, heads_hint=heads_hint)  # 1D array
                for idx, val in enumerate(vec):
                    rows.append({
                        "global_step": S,
                        "layer": layer,
                        "operator": f"{sub}.{proj}",
                        "index": idx,
                        "l2_abs_change": float(val),
                        "shape": str(tuple(w_pre.shape)),
                        "agg": agg
                    })
        # free per-step dicts
        del sd_pre, sd_post; gc.collect()

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No data collected (check LAYERS/OPS and key patterns).")
    return df.sort_values(["layer", "operator", "index", "global_step"]).reset_index(drop=True)

def build_pivot(df_op: pd.DataFrame):
    """
    Given rows for one operator (fixed layer & operator),
    return (P, steps) — pivot index=index(neuron), columns=global_step, values=l2_abs_change
    """
    df_op = df_op.copy()
    df_op["index"] = df_op["index"].astype(int)
    df_op["global_step"] = df_op["global_step"].astype(int)
    P = (df_op.pivot(index="index", columns="global_step", values="l2_abs_change")
                .sort_index(axis=0)
                .sort_index(axis=1))
    steps = P.columns.to_numpy()
    return P, steps

def plot_heatmap(P: pd.DataFrame, steps: np.ndarray, title: str, out_png: str):
    M = P.to_numpy(dtype=np.float32)
    M_masked = np.ma.masked_invalid(M)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    im = ax.imshow(M_masked, origin="lower", aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("L2 drift (neuron-wise) pre→post")

    ax.set_xlabel("Global step (pre→post)")
    ax.set_ylabel("Neuron index")

    autoscale_ticks(ax, steps, axis="x", max_ticks=MAX_X_TICKS)
    autoscale_ticks(ax, np.arange(M.shape[0]), axis="y", max_ticks=MAX_Y_TICKS)

    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    plt.close(fig)

# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[scan] reading step*_pre/_post pairs in: {WEIGHT_ROOT}")
    df = build_neuron_drift_long(WEIGHT_ROOT, LAYERS, OPS, agg=AGG, heads_hint=HEADS_HINT)

    # Save the long-form table
    long_csv = os.path.join(OUT_DIR, f"neuron_drift_long_agg-{AGG}.csv")
    df.to_csv(long_csv, index=False)
    print(f"[saved] long table → {long_csv}  (rows={len(df):,})")

    # For each (layer, operator), build a pivot + heatmap
    for layer in sorted(df["layer"].unique()):
        for op in sorted(df[df["layer"] == layer]["operator"].unique()):
            df_op = df[(df["layer"] == layer) & (df["operator"] == op)]
            if df_op.empty:
                continue
            P, steps = build_pivot(df_op)
            table_path = os.path.join(OUT_DIR, f"table_layer{layer}_{op.replace('.','_')}_agg-{AGG}.csv")
            P.to_csv(table_path, float_format="%.6g")
            print(f"[saved] pivot → {table_path}  (rows={P.shape[0]}, cols={P.shape[1]})")

            png_path = os.path.join(OUT_DIR, f"heatmap_layer{layer}_{op.replace('.','_')}_agg-{AGG}.png")
            title = f"Layer {layer} · {op} · agg={AGG}"
            plot_heatmap(P, steps, title, png_path)

if __name__ == "__main__":
    main()