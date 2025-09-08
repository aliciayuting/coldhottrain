#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuron-level grad vs ΔW curves for a given training step.

Definitions:
- MLP neuron = one intermediate channel: row i of up_proj.weight (incoming) + column i of down_proj.weight (outgoing).
- Attention neuron = one head: contiguous column slice of size head_dim in q/k/v/o projection matrices.

This script:
  1) Loads gradients at GLOBAL_STEP from grad_dump/index.csv, aggregates to neuron-level energy.
  2) Loads weights from weight_dump/stepXXXXXX_pre and _post, computes ΔW, aggregates to neuron-level energy.
  3) Plots Lorenz-style curves (grad vs ΔW) for MLP neurons and for attention heads (and an optional overall).

No edits to training/callback code required.
"""

# ========================
# Config (edit these)
# ========================
GRAD_BASE_DIR   = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
WEIGHT_ROOT     = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/weight_dump"
GLOBAL_STEP     = 200
OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/plots_neuron"

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
    ax.set_xlabel("Proportion of neurons", fontsize=12)
    ax.set_ylabel("Cumulative energy (L2²)", fontsize=12)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.grid(False)
    ax.legend([label], loc="lower right", frameon=True)
    ax.text(top_p + 0.002, min(0.98, yk + 0.02),
            f"Top {int(top_p*100)}%\n{yk*100:.1f}% of L2²",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return float(yk)

# ---------- grad: aggregate to neurons from index.csv ----------
def load_grad_neuron_energy(grad_base_dir: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mlp_neuron_grad: concatenated per-neuron grad energy across layers
      attn_head_grad:  concatenated per-head grad energy across layers
    """
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    df = pd.read_csv(index_csv)

    rows = df[df["global_step"] == step]
    if rows.empty:
        raise ValueError(f"No entries for global_step={step} in index.csv")

    # Collect by module name
    # Files are stored as e.g. L07_self_attn_q_proj_weight.npy
    # We'll load tensors and reduce to:
    #  - MLP: per-row for up_proj.weight + per-col for down_proj.weight, then sum per neuron index
    #  - ATTENTION: per-head by slicing columns into head_dim chunks
    mlp_energy_per_layer: Dict[int, Dict[str, np.ndarray]] = {}
    attn_energy_per_layer: Dict[int, Dict[str, np.ndarray]] = {}

    def _ensure_mlp(layer_id, key, size):
        d = mlp_energy_per_layer.setdefault(layer_id, {})
        if key not in d:
            d[key] = np.zeros(size, dtype=np.float64)

    def _ensure_attn(layer_id, key, size):
        d = attn_energy_per_layer.setdefault(layer_id, {})
        if key not in d:
            d[key] = np.zeros(size, dtype=np.float64)

    # First pass: discover head_dim for attention per layer using q_proj shapes
    head_meta: Dict[int, Tuple[int,int]] = {}  # layer_id -> (n_heads, head_dim)
    for _, r in rows.iterrows():
        sub = r["submodule"]; param = r["param"]; layer_id = int(r["layer"])
        if sub != "self_attn" or param != "q_proj.weight":
            continue
        f = os.path.join(grad_base_dir, r["file"])
        if not os.path.isfile(f):
            continue
        Wqg = _load_any_tensor(f)  # grad shape [out,in] for Linear
        # Expect shape like [d_model, n_heads*d_head]
        dm, cols = int(Wqg.shape[0]), int(Wqg.shape[1])
        # Infer n_heads, head_dim from divisors
        candidates = [h for h in range(1, 256) if cols % h == 0]
        pick = None
        for h in candidates:
            d_h = cols // h
            if 8 <= d_h <= 256:  # reasonable head dims
                pick = (h, d_h); break
        if pick is None:
            # fallback: guess a typical d_head like 64
            d_h = 64 if cols % 64 == 0 else 128 if cols % 128 == 0 else None
            if d_h is None:
                warnings.warn(f"[L{layer_id}] could not infer head_dim from q_proj.grad shape {tuple(Wqg.shape)}")
                continue
            pick = (cols // d_h, d_h)
        head_meta[layer_id] = pick

    # Second pass: load grads and accumulate
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
                # bias: treat as its own 'neuron contribution' for matching index
                pass
            else:
                continue

        if sub == "mlp":
            if param == "up_proj.weight":
                # per-neuron incoming energy = row-wise sum
                e = (G.to(torch.float32).pow(2).sum(dim=1)).cpu().numpy()  # [d_hidden]
                _ensure_mlp(layer_id, "incoming", e.size)
                mlp_energy_per_layer[layer_id]["incoming"] += e
            elif param == "down_proj.weight":
                # per-neuron outgoing energy = column-wise sum
                e = (G.to(torch.float32).pow(2).sum(dim=0)).cpu().numpy()  # [d_hidden]
                _ensure_mlp(layer_id, "outgoing", e.size)
                mlp_energy_per_layer[layer_id]["outgoing"] += e
            # (optionally include gate_proj if you want; by default we focus on up/down as neuron definition)
        elif sub == "self_attn":
            # per-head: slice columns by head_dim
            meta = head_meta.get(layer_id, None)
            if meta is None:
                continue
            n_heads, d_head = meta
            if param in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]:
                cols = G.shape[1]
                if cols % (n_heads*d_head) != 0 and param != "o_proj.weight":
                    warnings.warn(f"[L{layer_id}] {param} col mismatch; skip")
                    continue
                # For o_proj we still slice columns into head_dim chunks (common layout)
                Gh = G.to(torch.float32)
                per_head = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    sl = Gh[:, h*d_head:(h+1)*d_head]  # [d_model, d_head]
                    per_head[h] += float((sl*sl).sum().item())
                _ensure_attn(layer_id, "heads", n_heads)
                attn_energy_per_layer[layer_id]["heads"] += per_head

    # Combine incoming+outgoing for each MLP neuron
    mlp_all = []
    for lid, d in mlp_energy_per_layer.items():
        inc = d.get("incoming", None)
        out = d.get("outgoing", None)
        if inc is None and out is None:
            continue
        if inc is None:  mlp = out
        elif out is None: mlp = inc
        else:            mlp = inc + out
        mlp_all.append(mlp)
    attn_all = [v["heads"] for v in attn_energy_per_layer.values() if "heads" in v]

    mlp_neuron_grad = np.concatenate(mlp_all, axis=0) if len(mlp_all) else np.array([], dtype=np.float64)
    attn_head_grad  = np.concatenate(attn_all, axis=0) if len(attn_all) else np.array([], dtype=np.float64)
    return mlp_neuron_grad, attn_head_grad

# ---------- weights pre/post: aggregate ΔW to neurons ----------
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
    dm, cols = int(Wq.shape[0]), int(Wq.shape[1])
    candidates = [h for h in range(1, 256) if cols % h == 0]
    for h in candidates:
        d_h = cols // h
        if 8 <= d_h <= 256:
            return h, d_h
    # fallback common dims
    if cols % 64 == 0: return cols//64, 64
    if cols % 128 == 0: return cols//128, 128
    raise RuntimeError(f"Cannot infer (n_heads, head_dim) from q_proj shape {tuple(Wq.shape)}")

def load_delta_neuron_energy(weight_root: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
    pre_dir  = os.path.join(weight_root, f"step{step:06d}_pre")
    post_dir = os.path.join(weight_root, f"step{step:06d}_post")
    if not os.path.isdir(pre_dir):  raise FileNotFoundError(pre_dir)
    if not os.path.isdir(post_dir): raise FileNotFoundError(post_dir)

    sd_pre  = _load_state_dict_any(pre_dir)
    sd_post = _load_state_dict_any(post_dir)

    # collect per-layer MLP and attention keys
    # Typical keys: model.model.layers.{L}.mlp.up_proj.weight etc.
    layer_regex = re.compile(r"\.layers\.(\d+)\.")
    mlp_neuron_chunks = []
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
                Dup   = (Wup_post - Wup_pre)     # [d_hidden, d_model]  (row = neuron incoming)
                Ddown = (Wdown_post - Wdown_pre) # [d_model, d_hidden]  (col = neuron outgoing)
                per_neuron = (Dup.pow(2).sum(dim=1) + Ddown.pow(2).sum(dim=0)).cpu().numpy()
                mlp_neuron_chunks.append(per_neuron)

        # Attention q/k/v/o
        qk = [k for k in d if k.endswith(".self_attn.q_proj.weight")]
        ok = [k for k in d if k.endswith(".self_attn.o_proj.weight") or k.endswith(".self_attn.out_proj.weight")]
        if qk and ok and qk[0] in sd_post and ok[0] in sd_post:
            Wq_pre = sd_pre[qk[0]]; Wq_post = sd_post[qk[0]]
            Wo_pre = sd_pre[ok[0]]; Wo_post = sd_post[ok[0]]
            if Wq_pre.shape == Wq_post.shape and Wo_pre.shape == Wo_post.shape:
                n_heads, d_head = _infer_heads_from_Wq(Wq_pre)
                # helper to accumulate head energies
                def head_delta_cols(Wpre, Wpost):
                    Wd = (Wpost - Wpre).to(torch.float32)     # [d_model, n_heads*d_head]
                    per = np.zeros(n_heads, dtype=np.float64)
                    for h in range(n_heads):
                        sl = Wd[:, h*d_head:(h+1)*d_head]
                        per[h] += float((sl*sl).sum().item())
                    return per
                per_head = head_delta_cols(Wq_pre, Wq_post)

                # add K/V if exist
                kk = [k for k in d if k.endswith(".self_attn.k_proj.weight")]
                vk = [k for k in d if k.endswith(".self_attn.v_proj.weight")]
                if kk and kk[0] in sd_post:
                    per_head += head_delta_cols(sd_pre[kk[0]], sd_post[kk[0]])
                if vk and vk[0] in sd_post:
                    per_head += head_delta_cols(sd_pre[vk[0]], sd_post[vk[0]])

                # add O
                per_head += head_delta_cols(Wo_pre, Wo_post)
                attn_head_chunks.append(per_head)

    mlp_neuron_delta = np.concatenate(mlp_neuron_chunks, axis=0) if mlp_neuron_chunks else np.array([], dtype=np.float64)
    attn_head_delta  = np.concatenate(attn_head_chunks, axis=0) if attn_head_chunks else np.array([], dtype=np.float64)
    return mlp_neuron_delta, attn_head_delta

# ---------- main ----------
def main():
    # 1) Grad neuron energies
    mlp_g, attn_g = load_grad_neuron_energy(GRAD_BASE_DIR, GLOBAL_STEP)
    print(f"[DEBUG] grad: MLP neurons={mlp_g.size}, ATTN heads={attn_g.size}, totals L2²: "
          f"MLP={mlp_g.sum():.3e}, ATTN={attn_g.sum():.3e}")

    # 2) ΔW neuron energies
    mlp_d, attn_d = load_delta_neuron_energy(WEIGHT_ROOT, GLOBAL_STEP)
    print(f"[DEBUG] ΔW:   MLP neurons={mlp_d.size}, ATTN heads={attn_d.size}, totals L2²: "
          f"MLP={mlp_d.sum():.3e}, ATTN={attn_d.sum():.3e}")

    # 3) Plot MLP neurons
    if mlp_g.size and mlp_d.size:
        xg, yg, _ = _make_curve(mlp_g)
        xd, yd, _ = _make_curve(mlp_d)
        fig, ax = plt.subplots(figsize=(5,4), dpi=200)
        ax.plot(xg, yg, label="Grad (MLP neurons)", linewidth=2)
        ax.plot(xd, yd, label="ΔW (MLP neurons)", linewidth=2, linestyle="--")
        ax.set_xlabel("Proportion of neurons"); ax.set_ylabel("Cumulative L2²")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right"); ax.grid(False)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"neurons_mlp_grad_vs_delta_step{GLOBAL_STEP:06d}.png")
        plt.savefig(path, bbox_inches="tight"); plt.close(fig)
        print(f"[PLOT] MLP neurons overlay -> {path}")

        # standalone curves
        gcap = _plot_curve(xg, yg, f"Grad (MLP neurons) @ step {GLOBAL_STEP}",
                           os.path.join(OUT_DIR, f"neurons_mlp_grad_step{GLOBAL_STEP:06d}.png"), top_p=TOP_P)
        dcap = _plot_curve(xd, yd, f"ΔW (MLP neurons) @ step {GLOBAL_STEP}",
                           os.path.join(OUT_DIR, f"neurons_mlp_delta_step{GLOBAL_STEP:06d}.png"), top_p=TOP_P)
        print(f"[MLP] top {int(TOP_P*100)}% capture: grad={gcap*100:.2f}%  ΔW={dcap*100:.2f}%")

    # 4) Plot ATTENTION heads
    if attn_g.size and attn_d.size:
        xg, yg, _ = _make_curve(attn_g)
        xd, yd, _ = _make_curve(attn_d)
        fig, ax = plt.subplots(figsize=(5,4), dpi=200)
        ax.plot(xg, yg, label="Grad (attention heads)", linewidth=2)
        ax.plot(xd, yd, label="ΔW (attention heads)", linewidth=2, linestyle="--")
        ax.set_xlabel("Proportion of heads"); ax.set_ylabel("Cumulative L2²")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right"); ax.grid(False)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"neurons_attn_grad_vs_delta_step{GLOBAL_STEP:06d}.png")
        plt.savefig(path, bbox_inches="tight"); plt.close(fig)
        print(f"[PLOT] ATTENTION heads overlay -> {path}")

        gcap = _plot_curve(xg, yg, f"Grad (attention heads) @ step {GLOBAL_STEP}",
                           os.path.join(OUT_DIR, f"neurons_attn_grad_step{GLOBAL_STEP:06d}.png"), top_p=TOP_P)
        dcap = _plot_curve(xd, yd, f"ΔW (attention heads) @ step {GLOBAL_STEP}",
                           os.path.join(OUT_DIR, f"neurons_attn_delta_step{GLOBAL_STEP:06d}.png"), top_p=TOP_P)
        print(f"[ATTN] top {int(TOP_P*100)}% capture: grad={gcap*100:.2f}%  ΔW={dcap*100:.2f}%")

    # 5) Optional: overall neurons (concatenate MLP + heads)
    if (mlp_g.size or attn_g.size) and (mlp_d.size or attn_d.size):
        g_all = np.concatenate([x for x in [mlp_g, attn_g] if x.size], axis=0)
        d_all = np.concatenate([x for x in [mlp_d, attn_d] if x.size], axis=0)
        xg, yg, _ = _make_curve(g_all)
        xd, yd, _ = _make_curve(d_all)
        fig, ax = plt.subplots(figsize=(5,4), dpi=200)
        ax.plot(xg, yg, label="Grad (all neurons)", linewidth=2)
        ax.plot(xd, yd, label="ΔW (all neurons)", linewidth=2, linestyle="--")
        ax.set_xlabel("Proportion of neurons"); ax.set_ylabel("Cumulative L2²")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right"); ax.grid(False)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"neurons_all_grad_vs_delta_step{GLOBAL_STEP:06d}.png")
        plt.savefig(path, bbox_inches="tight"); plt.close(fig)
        print(f"[PLOT] OVERALL neurons overlay -> {path}")

if __name__ == "__main__":
    main()