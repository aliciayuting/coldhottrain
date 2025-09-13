#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel-level grad vs ΔW curves for a given training step.

Definitions:
- MLP channel = one intermediate channel: row i of up_proj.weight (incoming) or column i of down_proj.weight (outgoing).
- Attention channel = one column in Weight matrix

This script:
  1) Loads gradients at GLOBAL_STEP from grad_dump/index.csv, aggregates to channel-level energy.
  2) Loads weights from weight_dump/stepXXXXXX_pre and _post, computes ΔW, aggregates to channel-level energy.
  3) Plots Lorenz-style curves (grad vs ΔW) for MLP channels and for attention heads (and an optional overall).

No edits to training/callback code required.
"""

# ========================
# Config (edit these)
# ========================
GRAD_BASE_DIR   = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
WEIGHT_ROOT     = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/weight_dump"
GLOBAL_STEP     = 200
OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/test_plots_channel"
# OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/exact_channel"


SAMPLE_FRAC     = 1.0      # element subsample before reduction (for very large tensors)
TOP_P           = 0.01     # annotate top-1%
INCLUDE_BIAS    = True     # match your callback (you set include_bias=True)

# If you want to include/exclude modules beyond the dumper’s default, tweak these
INCLUDE_EMBEDDINGS = False   # your dumper sets also_embeddings=False; keep False to match
INCLUDE_LM_HEAD    = False

# ========================
# Script
# ========================
import os, re, gc, math, warnings, glob, csv
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    from safetensors.torch import load_file as safetensors_load
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False

os.makedirs(OUT_DIR, exist_ok=True)


# ---------- basic loaders ----------
def _load_any_tensor(path: str) -> torch.Tensor:
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

def _tensor_sq_1d(t: torch.Tensor, sample_frac: float) -> np.ndarray:
    # elementwise squared values as 1D (use only for element-wise paths)
    x = t.reshape(-1)
    if sample_frac < 1.0:
        n = x.numel()
        k = max(1, int(math.ceil(n * sample_frac)))
        idx = torch.randperm(n)[:k]
        x = x[idx]
    return (x**2).to(torch.float32).cpu().numpy()

# ---------- plotting ----------
def _make_curve(values: np.ndarray):
    s = np.sort(values)[::-1]
    n = s.size
    x = (np.arange(1, n+1, dtype=np.float64))/float(n)
    y = np.cumsum(s, dtype=np.float64); y /= y[-1]
    return x, y, n

def _plot_curve(x, y, label, save_path, top_p=0.01, figsize=(5,4), dpi=200):
    k = max(1, int(math.ceil(top_p * y.size)))
    yk = y[k-1]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, y, linewidth=2)
    ax.axvline(top_p, linestyle="--", linewidth=1.5)
    ax.axhline(yk, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Proportion of channels", fontsize=12)
    ax.set_ylabel("Cumulative energy (L2²)", fontsize=12)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.grid(False)
    ax.legend([label], loc="lower right", frameon=True)
    ax.text(top_p + 0.002, min(0.98, yk + 0.02),
            f"Top {int(top_p*100)}%\n{yk*100:.1f}% of L2²",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return float(yk)

# ---------- grad: aggregate to channels from index.csv ----------
def load_grad_channel_energy(grad_base_dir: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mlp_channel_grad: concatenated per-channel grad energy across layers
      attn_head_grad:  concatenated per-head grad energy across layers
    """
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    df = pd.read_csv(index_csv)

    rows = df[df["global_step"] == step]
    if rows.empty:
        raise ValueError(f"No entries for global_step={step} in index.csv")

    mlp_energy_per_layer: Dict[int, Dict[str, np.ndarray]] = {}
    

    def _ensure_mlp(layer_id, key, size):
        d = mlp_energy_per_layer.setdefault(layer_id, {})
        if key not in d:
            d[key] = np.zeros(size, dtype=np.float64)
    for _, r in rows.iterrows():
        layer_id = int(r["layer"])
        sub = r["submodule"]
        param = r["param"]
        f = os.path.join(grad_base_dir, r["file"])
        if not os.path.isfile(f):
            warnings.warn(f"[grad] missing {f}"); continue
        G = _load_any_tensor(f)  # [out,in]
        if G.ndim != 2:
            # ignore 1D (bias etc) unless INCLUDE_BIAS
            if INCLUDE_BIAS and G.ndim == 1 and param.endswith(".bias"):
                # bias: treat as its own 'channel contribution' for matching index
                pass
            else:
                continue

        if sub == "mlp":
            if param == "up_proj.weight":
                # per-channel incoming energy = row-wise sum
                e = (G.to(torch.float32).pow(2).sum(dim=1)).cpu().numpy()  # [d_hidden]
                _ensure_mlp(layer_id, "incoming", e.size)
                mlp_energy_per_layer[layer_id]["incoming"] += e
            elif param == "down_proj.weight":
                # per-channel outgoing energy = column-wise sum
                e = (G.to(torch.float32).pow(2).sum(dim=0)).cpu().numpy()  # [d_hidden]
                _ensure_mlp(layer_id, "outgoing", e.size)
                mlp_energy_per_layer[layer_id]["outgoing"] += e
            # (optionally include gate_proj if you want; by default we focus on up/down as channel definition)
        
    # Combine incoming+outgoing for each MLP channel
    mlp_all = []
    for lid, d in mlp_energy_per_layer.items():
        inc = d.get("incoming", None)
        out = d.get("outgoing", None)
        if inc is None and out is None:
            continue
        if inc is None:  mlp = out
        elif out is None: mlp = inc
        else:            mlp = inc + out
        mlp_all.append(inc)
        mlp_all.append(out)

    mlp_channel_grad = np.concatenate(mlp_all, axis=0) if len(mlp_all) else np.array([], dtype=np.float64)
    # write top mlp channels to CSV
    top_rows = []
    if mlp_channel_grad.size:
        top_idx = np.argsort(mlp_channel_grad)[::-1][:200]
        for rank, idx in enumerate(top_idx, 1):
            top_rows.append({
                "kind": "MLP",
                "index": int(idx),
                "value": float(mlp_channel_grad[idx]),
                "rank": rank
            })
    if top_rows:
        df = pd.DataFrame(top_rows)
        out_csv = os.path.join(OUT_DIR, f"top200_grad_step{step:06d}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[DEBUG] Wrote top-200 grad values to {out_csv}")
    print(f"[DEBUG] grad MLP channels: {sum(m.size for m in mlp_all) if mlp_all else 0} ")
    return mlp_channel_grad, np.array([], dtype=np.float64)  # no attn in this version

# ---------- weights pre/post: aggregate ΔW to channels ----------
def _gather_weight_files(ckpt_dir: str) -> List[str]:
    safes = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if len(safes) > 0:
        return safes
    pt_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(pt_bin):
        return [pt_bin]
    raise FileNotFoundError(f"No weight files found in {ckpt_dir}")

def _load_state_dict_any(ckpt_dir: str) -> Dict[str, torch.Tensor]:
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
        else:
            obj = torch.load(f, map_location="cpu")
            if not isinstance(obj, dict):
                raise RuntimeError(f"Unexpected object in {f}: {type(obj)}")
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.detach().to(torch.float32)
    return state

def _infer_heads_from_Wq(Wq: torch.Tensor) -> Tuple[int,int]:
    # For q_proj: weight shape is [out = n_heads * d_head, in = d_model]
    rows, cols = int(Wq.shape[0]), int(Wq.shape[1])
    candidates = [h for h in range(1, 256) if rows % h == 0]
    for h in candidates:
        d_h = rows // h
        if 8 <= d_h <= 256:
            return h, d_h
    # fallback common dims
    if rows % 64 == 0: return rows // 64, 64
    if rows % 128 == 0: return rows // 128, 128
    raise RuntimeError(f"Cannot infer (n_heads, d_head) from q_proj shape {tuple(Wq.shape)}")

def load_delta_channel_energy(weight_root: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
    pre_dir  = os.path.join(weight_root, f"step{step:06d}_pre")
    post_dir = os.path.join(weight_root, f"step{step:06d}_post")
    if not os.path.isdir(pre_dir):  raise FileNotFoundError(pre_dir)
    if not os.path.isdir(post_dir): raise FileNotFoundError(post_dir)

    sd_pre  = _load_state_dict_any(pre_dir)
    sd_post = _load_state_dict_any(post_dir)

    # collect per-layer MLP and attention keys
    # Typical keys: model.model.layers.{L}.mlp.up_proj.weight etc.
    layer_regex = re.compile(r"\.layers\.(\d+)\.")
    mlp_channel_chunks = []
    attn_head_chunks = []

    # Group by layer
    keys_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, w in sd_pre.items():
        m = layer_regex.search(k)
        if not m: continue
        lid = int(m.group(1))
        d = keys_by_layer.setdefault(lid, {})
        d[k] = w

    for lid, d in keys_by_layer.items():
        # MLP up/down
        up_k   = [k for k in d if k.endswith(".mlp.up_proj.weight")]
        down_k = [k for k in d if k.endswith(".mlp.down_proj.weight")]
        if up_k and up_k[0] in sd_post and down_k and down_k[0] in sd_post:
            Wup_pre   = sd_pre[up_k[0]];   Wup_post   = sd_post[up_k[0]]
            Wdown_pre = sd_pre[down_k[0]]; Wdown_post = sd_post[down_k[0]]
            if Wup_pre.shape == Wup_post.shape and Wdown_pre.shape == Wdown_post.shape:
                Dup   = (Wup_post - Wup_pre)     # [d_hidden, d_model]  (row = channel incoming)
                Ddown = (Wdown_post - Wdown_pre) # [d_model, d_hidden]  (col = channel outgoing)
                per_up_channel = Dup.pow(2).sum(dim=1).cpu().numpy()
                per_down_channel = Ddown.pow(2).sum(dim=0).cpu().numpy()
                mlp_channel_chunks.append(per_up_channel)
                mlp_channel_chunks.append(per_down_channel)
                print("per_up_channel dim", per_up_channel.shape, "per_down_channel dim", per_down_channel.shape)

    mlp_channel_delta = np.concatenate(mlp_channel_chunks, axis=0) if mlp_channel_chunks else np.array([], dtype=np.float64)
    print(f"[DEBUG] ΔW MLP channels: {sum(x.size for x in mlp_channel_chunks)} ")
    top_rows = []
    if mlp_channel_delta.size:
        top_idx = np.argsort(mlp_channel_delta)[::-1][:200]
        for rank, idx in enumerate(top_idx, 1):
            top_rows.append({
                "kind": "MLP",
                "index": int(idx),
                "value": float(mlp_channel_delta[idx]),
                "rank": rank
            })
    if top_rows:
        df = pd.DataFrame(top_rows)
        out_csv = os.path.join(OUT_DIR, f"top200_deltas_step{step:06d}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[DEBUG] Wrote top-200 delta values to {out_csv}")
    print(f"[DEBUG] ΔW MLP channels: {mlp_channel_delta.size} ")
    return mlp_channel_delta, np.array([], dtype=np.float64)  # no attn in this version


# ---------- main ----------
def main():
    # 1) Grad channel energies
    mlp_g, attn_g = load_grad_channel_energy(GRAD_BASE_DIR, GLOBAL_STEP)
    print(f"[DEBUG] grad: MLP channels={mlp_g.size}, ATTN heads={attn_g.size}, totals L2²: "
          f"MLP={mlp_g.sum():.3e}, ATTN={attn_g.sum():.3e}")

    # 2) ΔW channel energies
    mlp_d, attn_d = load_delta_channel_energy(WEIGHT_ROOT, GLOBAL_STEP)
    print(f"[DEBUG] ΔW:   MLP channels={mlp_d.size}, ATTN heads={attn_d.size}, totals L2²: "
          f"MLP={mlp_d.sum():.3e}, ATTN={attn_d.sum():.3e}")

    # 3) Plot MLP channels
    if mlp_g.size and mlp_d.size:
        xg, yg, _ = _make_curve(mlp_g)
        xd, yd, _ = _make_curve(mlp_d)
        fig, ax = plt.subplots(figsize=(5,4), dpi=200)
        ax.plot(xg, yg, label="Grad (MLP channels)", linewidth=2)
        ax.plot(xd, yd, label="ΔW (MLP channels)", linewidth=2, linestyle="--")
        ax.set_xlabel("Proportion of channels"); ax.set_ylabel("Cumulative L2²")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right"); ax.grid(False)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"channels_mlp_grad_vs_delta_step{GLOBAL_STEP:06d}.png")
        plt.savefig(path, bbox_inches="tight"); plt.close(fig)
        print(f"[PLOT] MLP channels overlay -> {path}")

        # standalone curves
        gcap = _plot_curve(xg, yg, f"Grad (MLP channels) @ step {GLOBAL_STEP}",
                           os.path.join(OUT_DIR, f"channels_mlp_grad_step{GLOBAL_STEP:06d}.png"), top_p=TOP_P)
        dcap = _plot_curve(xd, yd, f"ΔW (MLP channels) @ step {GLOBAL_STEP}",
                           os.path.join(OUT_DIR, f"channels_mlp_delta_step{GLOBAL_STEP:06d}.png"), top_p=TOP_P)
        print(f"[MLP] top {int(TOP_P*100)}% capture: grad={gcap*100:.2f}%  ΔW={dcap*100:.2f}%")

    # # 5) Optional: overall channels (concatenate MLP + heads)
    # if (mlp_g.size or attn_g.size) and (mlp_d.size or attn_d.size):
    #     g_all = np.concatenate([x for x in [mlp_g, attn_g] if x.size], axis=0)
    #     d_all = np.concatenate([x for x in [mlp_d, attn_d] if x.size], axis=0)
    #     xg, yg, _ = _make_curve(g_all)
    #     xd, yd, _ = _make_curve(d_all)
    #     fig, ax = plt.subplots(figsize=(5,4), dpi=200)
    #     ax.plot(xg, yg, label="Grad (all channels)", linewidth=2)
    #     ax.plot(xd, yd, label="ΔW (all channels)", linewidth=2, linestyle="--")
    #     ax.set_xlabel("Proportion of channels"); ax.set_ylabel("Cumulative L2²")
    #     ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right"); ax.grid(False)
    #     plt.tight_layout()
    #     path = os.path.join(OUT_DIR, f"channels_all_grad_vs_delta_step{GLOBAL_STEP:06d}.png")
    #     plt.savefig(path, bbox_inches="tight"); plt.close(fig)
    #     print(f"[PLOT] OVERALL channels overlay -> {path}")
        

if __name__ == "__main__":
    main()