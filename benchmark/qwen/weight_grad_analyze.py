#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot grad energy curve vs weight-delta energy curve for a given training step S.

- Gradients: reads grad_dump/index.csv entries where global_step == S, loads tensors,
  and builds Lorenz-style cumulative curve of g^2 (L2^2 energy vs proportion of elements).

- Weights: loads HF checkpoints from weight_dump/step{S:06d}_pre and _post, computes
  ΔW = W_post - W_pre (elementwise), and builds the same curve for (ΔW)^2.

Saves:
  - grad_curve_stepXXXXXX.png
  - weight_delta_curve_stepXXXXXX.png
  - grad_vs_weight_delta_stepXXXXXX.png
"""

# ========================
# Config (edit these)
# ========================
GRAD_BASE_DIR   = "/pscratch/sd/l/lsx/yyt_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
WEIGHT_ROOT     = "/pscratch/sd/l/lsx/yyt_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/weight_dump"
GLOBAL_STEP     = 200        # plots grads @ this step; compares stepXXXXXX_pre vs stepXXXXXX_post
OUT_DIR         = "/pscratch/sd/l/lsx/yyt_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/plots"

SAMPLE_FRAC     = 1.0        # <1.0 to uniformly subsample to save RAM (e.g., 0.25)
TOP_P           = 0.01       # annotate top 1% capture on each curve

# Optional parameter-name filters for weight delta (regex). Keep None to include all weights.
INCLUDE_PATTERNS = None      # e.g., [r"self_attn\\..*\\.weight", r"mlp\\..*\\.weight"]
EXCLUDE_PATTERNS = None      # e.g., [r"lm_head", r"embed"]
INCLUDE_BIAS     = False     # set True to include *.bias in ΔW

# ========================
# Script
# ========================
import os, re, gc, math, warnings, glob
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Optional safetensors support
try:
    from safetensors.torch import load_file as safetensors_load
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False

# ---- Helpers ----

def _load_any_tensor(path: str) -> torch.Tensor:
    """
    Load a tensor from path (supports torch .pt/.pth/.bin and numpy .npy/.npz).
    Returns CPU float32 tensor.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".pt", ".pth", ".bin"]:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                t = obj
            elif isinstance(obj, dict) and "tensor" in obj:
                t = obj["tensor"]
            else:
                if isinstance(obj, dict):
                    t = None
                    for v in obj.values():
                        if isinstance(v, torch.Tensor):
                            t = v; break
                    if t is None:
                        raise ValueError(f"No tensor found in {path}")
                else:
                    raise ValueError(f"Unsupported torch object in {path}: {type(obj)}")
            return t.detach().to(dtype=torch.float32, device="cpu")
        elif ext == ".npy":
            arr = np.load(path, allow_pickle=False)
            return torch.from_numpy(np.array(arr, dtype=np.float32))
        elif ext == ".npz":
            npz = np.load(path, allow_pickle=False)
            key = list(npz.keys())[0]
            return torch.from_numpy(np.array(npz[key], dtype=np.float32))
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {path}: {e}")

def _tensor_to_sampled_sq_1d(t: torch.Tensor, sample_frac: float) -> np.ndarray:
    """
    Flatten -> optional uniform subsample -> return squared magnitudes as numpy (x^2).
    """
    t = t.reshape(-1)
    if sample_frac < 1.0:
        n = t.numel()
        k = max(1, int(math.ceil(n * sample_frac)))
        idx = torch.randperm(n)[:k]
        t = t[idx]
    sq = (t**2).to(torch.float32).cpu().numpy()
    del t
    return sq

def _make_curve(values_sq: np.ndarray):
    """
    Given squared magnitudes, build Lorenz-style cumulative curve.
    Returns x (proportion), y (cumulative energy), N (count).
    """
    order = np.argsort(values_sq)[::-1]
    s = values_sq[order]
    N = s.size
    x = (np.arange(1, N + 1, dtype=np.float64)) / float(N)
    cum = np.cumsum(s, dtype=np.float64)
    y = cum / cum[-1]
    return x, y, N

def _plot_curve(x, y, label, save_path, top_p=0.01, figsize=(5.0, 4.0), dpi=200):
    k = max(1, int(math.ceil(top_p * y.size)))
    y_top = y[k - 1]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, y, linewidth=2)
    ax.axvline(top_p, linestyle="--", linewidth=1.5)
    ax.axhline(y_top, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Proportion of Elements", fontsize=12)
    ax.set_ylabel("Cumulative L2 Norm\u00b2", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.legend([label], loc="lower right", frameon=True)
    ax.text(top_p + 0.002, min(0.98, y_top + 0.02),
            f"Top {int(top_p*100)}%\n{y_top*100:.1f}% of L2\u00b2",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return float(y_top)

# ---- Gradients from index.csv ----

def load_grad_sq_from_index(grad_base_dir: str, global_step: int, sample_frac: float = 1.0,
                            index_csv: Optional[str] = None) -> np.ndarray:
    if index_csv is None:
        index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(f"Missing index.csv: {index_csv}")

    df = pd.read_csv(index_csv)
    if "global_step" not in df.columns or "file" not in df.columns:
        raise ValueError("index.csv must contain columns 'global_step' and 'file'.")

    rows = df[df["global_step"] == global_step]
    if rows.empty:
        raise ValueError(f"No entries for global_step={global_step} in {index_csv}")

    file_paths = [os.path.join(grad_base_dir, f) for f in rows["file"].tolist()]

    chunks = []
    for p in file_paths:
        if not os.path.isfile(p):
            warnings.warn(f"[grad] Missing file: {p}")
            continue
        t = _load_any_tensor(p)
        g2 = _tensor_to_sampled_sq_1d(t, sample_frac)
        chunks.append(g2)

    if not chunks:
        raise RuntimeError("No gradients loaded (all files missing or empty).")

    return np.concatenate(chunks, axis=0)

# ---- Weights: pre vs post -> delta ----

def _param_name_passes(name: str,
                       include_patterns: Optional[List[str]],
                       exclude_patterns: Optional[List[str]],
                       include_bias: bool) -> bool:
    if not include_bias:
        if name.endswith(".bias") or name.split(".")[-1] == "bias":
            return False
    def _match_any(patterns, s):
        return any(re.search(p, s) for p in patterns) if patterns else False
    if include_patterns and not _match_any(include_patterns, name):
        return False
    if exclude_patterns and _match_any(exclude_patterns, name):
        return False
    return True

def _gather_weight_files(ckpt_dir: str) -> List[str]:
    safes = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if len(safes) > 0:
        return safes
    pt_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(pt_bin):
        return [pt_bin]
    raise FileNotFoundError(f"No weight files found in {ckpt_dir} (safetensors/bin)")

def _load_state_dict_any(ckpt_dir: str) -> dict:
    files = _gather_weight_files(ckpt_dir)
    state = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".safetensors":
            if not _HAVE_SAFETENSORS:
                raise RuntimeError("safetensors required; pip install safetensors")
            sd = safetensors_load(f, device="cpu")
            for k, v in sd.items():
                state[k] = v.detach().to(torch.float32)
            del sd
        elif ext in [".bin", ".pt", ".pth"]:
            obj = torch.load(f, map_location="cpu")
            if not isinstance(obj, dict):
                raise RuntimeError(f"Unexpected object in {f}: {type(obj)}")
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.detach().to(torch.float32)
        else:
            warnings.warn(f"Skipping unsupported weight file: {f}")
    return state

def load_delta_sq_from_pre_post(weight_root: str, step: int, sample_frac: float = 1.0,
                                include_patterns: Optional[List[str]] = None,
                                exclude_patterns: Optional[List[str]] = None,
                                include_bias: bool = False) -> np.ndarray:
    pre_dir  = os.path.join(weight_root, f"step{step:06d}_pre")
    post_dir = os.path.join(weight_root, f"step{step:06d}_post")
    if not os.path.isdir(pre_dir):
        raise FileNotFoundError(f"Missing pre checkpoint dir: {pre_dir}")
    if not os.path.isdir(post_dir):
        raise FileNotFoundError(f"Missing post checkpoint dir: {post_dir}")

    sd_pre  = _load_state_dict_any(pre_dir)
    sd_post = _load_state_dict_any(post_dir)

    chunks = []
    missing = 0
    for name, w_pre in sd_pre.items():
        if not _param_name_passes(name, include_patterns, exclude_patterns, include_bias):
            continue
        w_post = sd_post.get(name, None)
        if w_post is None:
            missing += 1
            continue
        if w_post.shape != w_pre.shape:
            warnings.warn(f"Shape mismatch for {name}: pre {tuple(w_pre.shape)} vs post {tuple(w_post.shape)}")
            continue
        d = (w_post - w_pre).reshape(-1)
        if sample_frac < 1.0:
            n = d.numel()
            k = max(1, int(math.ceil(n * sample_frac)))
            idx = torch.randperm(n)[:k]
            d = d[idx]
        d2 = (d**2).to(torch.float32).cpu().numpy()
        chunks.append(d2)

    if missing > 0:
        warnings.warn(f"{missing} parameters present in pre but missing in post; skipped")

    if not chunks:
        raise RuntimeError("No delta weights computed (filters too strict or empty checkpoints).")
    return np.concatenate(chunks, axis=0)

# ---- Run ----

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Gradients
    g2_all = load_grad_sq_from_index(GRAD_BASE_DIR, GLOBAL_STEP, SAMPLE_FRAC)
    xg, yg, Ng = _make_curve(g2_all)
    grad_png = os.path.join(OUT_DIR, f"grad_curve_step{GLOBAL_STEP:06d}.png")
    g_cap = _plot_curve(xg, yg, f"Gradients @ step {GLOBAL_STEP}", grad_png, top_p=TOP_P)

    # Weight deltas (post - pre)
    d2_all = load_delta_sq_from_pre_post(
        WEIGHT_ROOT, GLOBAL_STEP, SAMPLE_FRAC,
        include_patterns=INCLUDE_PATTERNS,
        exclude_patterns=EXCLUDE_PATTERNS,
        include_bias=INCLUDE_BIAS
    )
    xd, yd, Nd = _make_curve(d2_all)
    delta_png = os.path.join(OUT_DIR, f"weight_delta_curve_step{GLOBAL_STEP:06d}.png")
    d_cap = _plot_curve(xd, yd, f"ΔW @ step {GLOBAL_STEP} (post − pre)", delta_png, top_p=TOP_P)

    # Overlay
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=200)
    ax.plot(xg, yg, linewidth=2, label=f"Grad @ {GLOBAL_STEP}")
    ax.plot(xd, yd, linewidth=2, linestyle="--", label=f"ΔW @ {GLOBAL_STEP}")
    ax.set_xlabel("Proportion of Elements", fontsize=12)
    ax.set_ylabel("Cumulative L2 Norm\u00b2", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True)
    overlay_png = os.path.join(OUT_DIR, f"grad_vs_weight_delta_step{GLOBAL_STEP:06d}.png")
    plt.tight_layout()
    plt.savefig(overlay_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[GRAD]    N={Ng}  top{int(TOP_P*100)}% -> {g_cap*100:.2f}%   saved {grad_png}")
    print(f"[ΔW^2]   N={Nd}  top{int(TOP_P*100)}% -> {d_cap*100:.2f}%   saved {delta_png}")
    print(f"[OVERLAY] saved {overlay_png}")

if __name__ == "__main__":
    main()