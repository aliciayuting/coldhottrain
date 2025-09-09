#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuron-level grad vs ΔW curves for a given training step.

Definitions:
- MLP neuron = one intermediate channel: row i of up_proj.weight (incoming) + column i of down_proj.weight (outgoing).
- Attention neuron = one head (GQA-aware): q/k/v use contiguous ROW slices; o uses contiguous COLUMN slices. If n_kv_heads < n_heads, K/V per-head energies are broadcast to the Q/O head groups (group_size = n_heads // n_kv_heads).

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
# OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/exact_neuron"


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


def _infer_gqa_meta(q_rows: int, kv_rows: int | None) -> tuple[int, int, int]:
    """Infer (n_heads, n_kv, d_head) from q_rows and optional kv_rows.
    Prefers small/common head_dims typical for LLMs.
    """
    preferred = [32, 40, 48, 64, 80, 96, 104, 112, 128, 160, 192, 224, 256]
    for d in preferred:
        if q_rows % d == 0 and (kv_rows is None or kv_rows % d == 0):
            n_heads = q_rows // d
            n_kv = (kv_rows // d) if kv_rows is not None else n_heads
            return n_heads, n_kv, d
    # fallback: try any divisor 16..256
    for d in range(16, 257):
        if q_rows % d == 0 and (kv_rows is None or kv_rows % d == 0):
            n_heads = q_rows // d
            n_kv = (kv_rows // d) if kv_rows is not None else n_heads
            return n_heads, n_kv, d
    # last resort: treat as MHA
    d = max(1, min(128, q_rows))
    n_heads = max(1, q_rows // d)
    n_kv = n_heads if kv_rows is None else max(1, kv_rows // d)
    return n_heads, n_kv, d

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

    # First pass (GQA-aware): collect q_rows and kv_rows per layer, then infer (n_heads, n_kv, d_head)
    head_meta: Dict[int, Tuple[int,int,int]] = {}
    q_rows_layer: Dict[int, int] = {}
    kv_rows_layer: Dict[int, int] = {}
    for _, r in rows.iterrows():
        sub = r["submodule"]; param = r["param"]; layer_id = int(r["layer"])
        if sub != "self_attn":
            continue
        f = os.path.join(grad_base_dir, r["file"])
        if not os.path.isfile(f):
            continue
        if param == "q_proj.weight":
            G = _load_any_tensor(f)
            q_rows_layer[layer_id] = int(G.shape[0])
            print(f"[DEBUG][head-meta] L{layer_id} q_proj.grad shape={G.shape} -> rows={int(G.shape[0])} cols={int(G.shape[1])}")
        elif param in ("k_proj.weight", "v_proj.weight"):
            G = _load_any_tensor(f)
            kv_rows_layer[layer_id] = max(kv_rows_layer.get(layer_id, 0), int(G.shape[0]))

    for lid, q_rows in q_rows_layer.items():
        kv_rows = kv_rows_layer.get(lid, None)
        n_heads, n_kv, d_head = _infer_gqa_meta(q_rows, kv_rows)
        head_meta[lid] = (n_heads, n_kv, d_head)
        if kv_rows is None:
            print(f"[DEBUG][GQA] L{lid} q_rows={q_rows} -> n_heads={n_heads} d_head={d_head} (MHA assumption; no kv_rows)")
        else:
            print(f"[DEBUG][GQA] L{lid} q_rows={q_rows} kv_rows={kv_rows} -> n_heads={n_heads} n_kv={n_kv} d_head={d_head}")

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
            meta = head_meta.get(layer_id, None)
            if meta is None:
                continue
            n_heads, n_kv, d_head = meta
            Gh = G.to(torch.float32)

            if param == "q_proj.weight":
                rows = Gh.shape[0]
                if rows != n_heads * d_head:
                    warnings.warn(f"[L{layer_id}] q_proj row mismatch; expected {n_heads*d_head}, got {rows}")
                    continue
                per_head_q = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    sl = Gh[h*d_head:(h+1)*d_head, :]
                    per_head_q[h] = float((sl * sl).sum().item())
                _ensure_attn(layer_id, "heads", n_heads)
                attn_energy_per_layer[layer_id]["heads"] += per_head_q
                print(f"[DEBUG][GRAD][L{layer_id}] q ROW slices: {tuple(Gh.shape)} -> n_heads={n_heads} d_head={d_head}")

            elif param in ["k_proj.weight", "v_proj.weight"]:
                rows = Gh.shape[0]
                if rows != n_kv * d_head:
                    warnings.warn(f"[L{layer_id}] {param} row mismatch; expected {n_kv*d_head}, got {rows}")
                    continue
                per_kv = np.zeros(n_kv, dtype=np.float64)
                for h in range(n_kv):
                    sl = Gh[h*d_head:(h+1)*d_head, :]
                    per_kv[h] = float((sl * sl).sum().item())
                group_size = max(1, n_heads // n_kv)
                per_kv_expanded = np.repeat(per_kv, group_size)[:n_heads]
                _ensure_attn(layer_id, "heads", n_heads)
                attn_energy_per_layer[layer_id]["heads"] += per_kv_expanded
                print(f"[DEBUG][GRAD][L{layer_id}] {param} ROW slices: {tuple(Gh.shape)} -> n_kv={n_kv} d_head={d_head} broadcast x{group_size}")

            elif param in ["o_proj.weight", "out_proj.weight"]:
                cols = Gh.shape[1]
                if cols != n_heads * d_head:
                    warnings.warn(f"[L{layer_id}] o_proj col mismatch; expected {n_heads*d_head}, got {cols}")
                    continue
                per_head_o = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    sl = Gh[:, h*d_head:(h+1)*d_head]
                    per_head_o[h] = float((sl * sl).sum().item())
                _ensure_attn(layer_id, "heads", n_heads)
                attn_energy_per_layer[layer_id]["heads"] += per_head_o
                print(f"[DEBUG][GRAD][L{layer_id}] o COL slices: {tuple(Gh.shape)} -> n_heads={n_heads} d_head={d_head}")
            

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
    print(f"[DEBUG] grad MLP neurons: {sum(m.size for m in mlp_all) if mlp_all else 0} "
          f"heads: {sum(v['heads'].size for v in attn_energy_per_layer.values() if 'heads' in v)}")
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

        # Attention q/k/v/o (GQA-aware)
        qk = [k for k in d if k.endswith(".self_attn.q_proj.weight")]
        ok = [k for k in d if k.endswith(".self_attn.o_proj.weight") or k.endswith(".self_attn.out_proj.weight")]
        kk = [k for k in d if k.endswith(".self_attn.k_proj.weight")]
        vk = [k for k in d if k.endswith(".self_attn.v_proj.weight")]
        if qk and ok and qk[0] in sd_post and ok[0] in sd_post:
            Wq_pre = sd_pre[qk[0]]; Wq_post = sd_post[qk[0]]
            Wo_pre = sd_pre[ok[0]]; Wo_post = sd_post[ok[0]]
            kv_rows = None
            if kk and kk[0] in sd_post:
                kv_rows = int(sd_pre[kk[0]].shape[0])
            elif vk and vk[0] in sd_post:
                kv_rows = int(sd_pre[vk[0]].shape[0])
            n_heads, n_kv, d_head = _infer_gqa_meta(int(Wq_pre.shape[0]), kv_rows)
            print(f"[DEBUG][DELTA][L{lid}] q={tuple(Wq_pre.shape)} o={tuple(Wo_pre.shape)} n_heads={n_heads} n_kv={n_kv} d_head={d_head}")

            def head_delta_rows(Wpre, Wpost, expected_heads):
                Wd = (Wpost - Wpre).to(torch.float32)
                rows = Wd.shape[0]
                assert rows == expected_heads * d_head, f"rows {rows} != {expected_heads}*{d_head}"
                per = np.zeros(expected_heads, dtype=np.float64)
                for h in range(expected_heads):
                    sl = Wd[h*d_head:(h+1)*d_head, :]
                    per[h] = float((sl*sl).sum().item())
                return per

            def head_delta_cols(Wpre, Wpost, expected_heads):
                Wd = (Wpost - Wpre).to(torch.float32)
                cols = Wd.shape[1]
                assert cols == expected_heads * d_head, f"cols {cols} != {expected_heads}*{d_head}"
                per = np.zeros(expected_heads, dtype=np.float64)
                for h in range(expected_heads):
                    sl = Wd[:, h*d_head:(h+1)*d_head]
                    per[h] = float((sl*sl).sum().item())
                return per

            per_head = head_delta_rows(Wq_pre, Wq_post, n_heads)  # Q
            if kk and kk[0] in sd_post:
                per_k = head_delta_rows(sd_pre[kk[0]], sd_post[kk[0]], n_kv)
                per_head += np.repeat(per_k, max(1, n_heads//n_kv))[:n_heads]
            if vk and vk[0] in sd_post:
                per_v = head_delta_rows(sd_pre[vk[0]], sd_post[vk[0]], n_kv)
                per_head += np.repeat(per_v, max(1, n_heads//n_kv))[:n_heads]
            per_head += head_delta_cols(Wo_pre, Wo_post, n_heads)  # O

            attn_head_chunks.append(per_head)
            print(f"[DEBUG][DELTA][L{lid}] per-head ΔW sum={per_head.sum():.3e} min={per_head.min():.3e} max={per_head.max():.3e}")

    mlp_neuron_delta = np.concatenate(mlp_neuron_chunks, axis=0) if mlp_neuron_chunks else np.array([], dtype=np.float64)
    attn_head_delta  = np.concatenate(attn_head_chunks, axis=0) if attn_head_chunks else np.array([], dtype=np.float64)
    print(f"[DEBUG] ΔW MLP neurons: {sum(x.size for x in mlp_neuron_chunks)} "
          f"heads: {sum(x.size for x in attn_head_chunks)}")
    return mlp_neuron_delta, attn_head_delta


# ================= ADDED =========================

import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings

TOP_K = 200  # how many units to list in CSV/plots
CSV_OUT = os.path.join(OUT_DIR, f"topK_neurons_step{GLOBAL_STEP:06d}.csv")
SCATTER_PNG = os.path.join(OUT_DIR, f"topK_scatter_step{GLOBAL_STEP:06d}.png")
BARS_PNG = os.path.join(OUT_DIR, f"top50_bars_step{GLOBAL_STEP:06d}.png")

def _concat_and_rank(records, value_key):
    """Return dataframe with ranks and cumulative fractions for chosen value_key ('grad' or 'delta_total')."""
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    vals = df[value_key].to_numpy(dtype=np.float64)
    order = np.argsort(vals)[::-1]
    df = df.iloc[order].reset_index(drop=True)
    total = float(vals[order].sum())
    df["rank"] = np.arange(1, len(df)+1)
    df["cum"] = np.cumsum(df[value_key].to_numpy(dtype=np.float64))
    df["cum_frac"] = df["cum"] / (total if total > 0 else 1.0)
    df["value_frac"] = df[value_key] / (total if total > 0 else 1.0)
    return df

def collect_annotated_neurons(step: int):
    """
    Recomputes neuron/head energies with annotations:
      kind: 'MLP' or 'ATTN'
      layer: int
      unit: neuron_id (MLP) or head_id (ATTN)
      component: 'mlp_up','mlp_down','q','k','v','o','total'
      grad: value (component or total)
      delta_*: matching ΔW components and totals
    """
    # ---- Grad with annotations ----
    df_idx = pd.read_csv(os.path.join(GRAD_BASE_DIR, "index.csv"))
    rows = df_idx[df_idx["global_step"] == step]
    if rows.empty:
        raise ValueError(f"No entries for global_step={step} in index.csv")

    # First pass: infer (n_heads, n_kv, d_head) per layer (GQA-aware)
    q_rows_layer, kv_rows_layer = {}, {}
    for _, r in rows.iterrows():
        if r["submodule"] != "self_attn": continue
        t = _load_any_tensor(os.path.join(GRAD_BASE_DIR, r["file"]))
        if r["param"] == "q_proj.weight":
            q_rows_layer[int(r["layer"])] = int(t.shape[0])
        elif r["param"] in ("k_proj.weight","v_proj.weight"):
            kv_rows_layer[int(r["layer"])] = max(kv_rows_layer.get(int(r["layer"]),0), int(t.shape[0]))
    head_meta = {}
    for lid, qrows in q_rows_layer.items():
        kvrows = kv_rows_layer.get(lid, None)
        head_meta[lid] = _infer_gqa_meta(qrows, kvrows)

    # Annotated container
    recs = []

    # Second pass: accumulate grad per component
    # For MLP we need to sum row-wise (up) and col-wise (down) into matching neuron ids
    mlp_grad_inc = defaultdict(lambda: None)  # lid -> np.array[d_hidden]
    mlp_grad_out = defaultdict(lambda: None)
    attn_grad_q = defaultdict(lambda: None)   # lid -> np.array[n_heads]
    attn_grad_k = defaultdict(lambda: None)   # after broadcast to n_heads
    attn_grad_v = defaultdict(lambda: None)
    attn_grad_o = defaultdict(lambda: None)

    for _, r in rows.iterrows():
        lid = int(r["layer"]); sub = r["submodule"]; param = r["param"]
        t = _load_any_tensor(os.path.join(GRAD_BASE_DIR, r["file"])).to(torch.float32)

        if sub == "mlp" and param == "up_proj.weight" and t.ndim == 2:
            e = (t.pow(2).sum(dim=1)).cpu().numpy()  # [d_hidden]
            mlp_grad_inc[lid] = e if mlp_grad_inc[lid] is None else (mlp_grad_inc[lid] + e)
        elif sub == "mlp" and param == "down_proj.weight" and t.ndim == 2:
            e = (t.pow(2).sum(dim=0)).cpu().numpy()  # [d_hidden]
            mlp_grad_out[lid] = e if mlp_grad_out[lid] is None else (mlp_grad_out[lid] + e)
        elif sub == "self_attn" and t.ndim == 2 and lid in head_meta:
            n_heads, n_kv, d_head = head_meta[lid]
            if param == "q_proj.weight":
                v = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    v[h] = float((t[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                attn_grad_q[lid] = v if attn_grad_q[lid] is None else (attn_grad_q[lid] + v)
            elif param in ("k_proj.weight","v_proj.weight"):
                nk = n_kv
                if t.shape[0] != nk*d_head: continue
                tmp = np.zeros(nk, dtype=np.float64)
                for h in range(nk):
                    tmp[h] = float((t[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                group = max(1, n_heads // nk)
                tmp = np.repeat(tmp, group)[:n_heads]
                if param.startswith("k_"):
                    attn_grad_k[lid] = tmp if attn_grad_k[lid] is None else (attn_grad_k[lid] + tmp)
                else:
                    attn_grad_v[lid] = tmp if attn_grad_v[lid] is None else (attn_grad_v[lid] + tmp)
            elif param in ("o_proj.weight","out_proj.weight"):
                v = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    v[h] = float((t[:, h*d_head:(h+1)*d_head].pow(2)).sum().item())
                attn_grad_o[lid] = v if attn_grad_o[lid] is None else (attn_grad_o[lid] + v)

    # Emit annotated grad records
    for lid in sorted(set(list(mlp_grad_inc.keys()) + list(mlp_grad_out.keys()))):
        inc = mlp_grad_inc[lid]; out = mlp_grad_out[lid]
        if inc is None and out is None: continue
        if inc is None: inc = np.zeros_like(out)
        if out is None: out = np.zeros_like(inc)
        total = inc + out
        for i in range(total.size):
            recs.append(dict(kind="MLP", layer=lid, unit=i,
                             component="mlp_up",  grad=float(inc[i]), delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="MLP", layer=lid, unit=i,
                             component="mlp_down",grad=float(out[i]), delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="MLP", layer=lid, unit=i,
                             component="total",   grad=float(total[i]), delta_q=0, delta_k=0, delta_v=0, delta_o=0))

    for lid, (n_heads, n_kv, d_head) in sorted(head_meta.items()):
        q = attn_grad_q.get(lid); k = attn_grad_k.get(lid); v = attn_grad_v.get(lid); o = attn_grad_o.get(lid)
        if q is None and k is None and v is None and o is None: continue
        q = q if q is not None else np.zeros(n_heads)
        k = k if k is not None else np.zeros(n_heads)
        v = v if v is not None else np.zeros(n_heads)
        o = o if o is not None else np.zeros(n_heads)
        tot = q + k + v + o
        for h in range(n_heads):
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="q",     grad=float(q[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="k",     grad=float(k[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="v",     grad=float(v[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="o",     grad=float(o[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="total", grad=float(tot[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))

    # ---- ΔW with annotations (components) ----
    pre_dir  = os.path.join(WEIGHT_ROOT, f"step{step:06d}_pre")
    post_dir = os.path.join(WEIGHT_ROOT, f"step{step:06d}_post")
    sd_pre  = _load_state_dict_any(pre_dir)
    sd_post = _load_state_dict_any(post_dir)

    # group by layer
    import re
    layer_regex = re.compile(r"\.layers\.(\d+)\.")
    by_layer = defaultdict(dict)
    for k in sd_pre.keys():
        m = layer_regex.search(k)
        if not m: continue
        by_layer[int(m.group(1))][k] = True

    for lid, _ in sorted(by_layer.items()):
        # MLP ΔW
        up = f"model.model.layers.{lid}.mlp.up_proj.weight"
        down = f"model.model.layers.{lid}.mlp.down_proj.weight"
        if up in sd_pre and up in sd_post and down in sd_pre and down in sd_post:
            Dup   = (sd_post[up]   - sd_pre[up]).to(torch.float32)   # [d_hidden, d_model]
            Ddown = (sd_post[down] - sd_pre[down]).to(torch.float32) # [d_model, d_hidden]
            inc = Dup.pow(2).sum(dim=1).cpu().numpy()
            out = Ddown.pow(2).sum(dim=0).cpu().numpy()
            tot = inc + out
            for i in range(tot.size):
                recs.append(dict(kind="MLP", layer=lid, unit=i, component="mlp_up",
                                grad=0.0,
                                delta_q=0.0, delta_k=0.0, delta_v=0.0, delta_o=0.0,
                                delta_total=float(inc[i])))
                recs.append(dict(kind="MLP", layer=lid, unit=i, component="mlp_down",
                                grad=0.0,
                                delta_q=0.0, delta_k=0.0, delta_v=0.0, delta_o=0.0,
                                delta_total=float(out[i])))
                recs.append(dict(kind="MLP", layer=lid, unit=i, component="total",
                                grad=0.0,
                                delta_q=0.0, delta_k=0.0, delta_v=0.0, delta_o=0.0,
                                delta_total=float(tot[i])))

        # ATTN ΔW (GQA-aware)
        qk = f"model.model.layers.{lid}.self_attn.q_proj.weight"
        ok = f"model.model.layers.{lid}.self_attn.o_proj.weight"
        kk = f"model.model.layers.{lid}.self_attn.k_proj.weight"
        vk = f"model.model.layers.{lid}.self_attn.v_proj.weight"
        if qk in sd_pre and qk in sd_post and ok in sd_pre and ok in sd_post:
            Wq_pre, Wq_post = sd_pre[qk], sd_post[qk]
            n_heads, n_kv, d_head = _infer_gqa_meta(int(Wq_pre.shape[0]),
                                                    int(sd_pre[kk].shape[0]) if kk in sd_pre else (int(sd_pre[vk].shape[0]) if vk in sd_pre else None))

            # q rows
            per_q = np.zeros(n_heads, dtype=np.float64)
            Wd = (Wq_post - Wq_pre).to(torch.float32)
            for h in range(n_heads):
                per_q[h] = float((Wd[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())

            # k rows (broadcast)
            per_k = np.zeros(n_heads, dtype=np.float64)
            if kk in sd_pre and kk in sd_post:
                Wd = (sd_post[kk] - sd_pre[kk]).to(torch.float32)
                nk = n_kv
                tmp = np.zeros(nk, dtype=np.float64)
                for h in range(nk):
                    tmp[h] = float((Wd[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                per_k = np.repeat(tmp, max(1, n_heads//nk))[:n_heads]

            # v rows (broadcast)
            per_v = np.zeros(n_heads, dtype=np.float64)
            if vk in sd_pre and vk in sd_post:
                Wd = (sd_post[vk] - sd_pre[vk]).to(torch.float32)
                nk = n_kv
                tmp = np.zeros(nk, dtype=np.float64)
                for h in range(nk):
                    tmp[h] = float((Wd[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                per_v = np.repeat(tmp, max(1, n_heads//nk))[:n_heads]

            # o cols
            per_o = np.zeros(n_heads, dtype=np.float64)
            Wd = (sd_post[ok] - sd_pre[ok]).to(torch.float32)
            for h in range(n_heads):
                per_o[h] = float((Wd[:, h*d_head:(h+1)*d_head].pow(2)).sum().item())

            tot = per_q + per_k + per_v + per_o
            for h in range(n_heads):
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="q",
                                grad=0.0,
                                delta_q=float(per_q[h]), delta_k=0.0, delta_v=0.0, delta_o=0.0,
                                delta_total=float(per_q[h])))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="k",
                                grad=0.0,
                                delta_q=0.0, delta_k=float(per_k[h]), delta_v=0.0, delta_o=0.0,
                                delta_total=float(per_k[h])))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="v",
                                grad=0.0,
                                delta_q=0.0, delta_k=0.0, delta_v=float(per_v[h]), delta_o=0.0,
                                delta_total=float(per_v[h])))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="o",
                                grad=0.0,
                                delta_q=0.0, delta_k=0.0, delta_v=0.0, delta_o=float(per_o[h]),
                                delta_total=float(per_o[h])))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="total",
                                grad=0.0,
                                delta_q=0.0, delta_k=0.0, delta_v=0.0, delta_o=0.0,
                                delta_total=float(tot[h])))

    # # Final tidy: add delta_total column
    # for r in recs:
    #     # For MLP we packed ΔW totals in delta_q; for ATTN we split q/k/v/o
    #     if r["kind"] == "MLP":
    #         r["delta_total"] = r["delta_q"] if r["component"] in ("mlp_up","mlp_down","total") else 0.0
    #     else:
    #         if r["component"] == "total":
    #             # for convenience recompute from components later during ranking
    #             r["delta_total"] = r["delta_q"]  # will overwrite below after we collect per-head rows
    #         else:
    #             r["delta_total"] = r["delta_q"]

    # For attention, ensure 'total' rows truly hold q+k+v+o
    df_recs = pd.DataFrame.from_records(recs)
    # if not df_recs.empty:
    #     # Build a per-(ATTN,layer,unit) sum for components
    #     mask_attn = (df_recs["kind"] == "ATTN")
    #     comp_sum = (df_recs[mask_attn & df_recs["component"].isin(["q","k","v","o"])]
    #                 .groupby(["kind","layer","unit"])["delta_total"].sum().rename("delta_attn_sum"))
    #     df_recs = df_recs.merge(comp_sum, how="left", left_on=["kind","layer","unit"], right_index=True)
    #     df_recs.loc[mask_attn & (df_recs["component"]=="total"), "delta_total"] = df_recs.loc[mask_attn & (df_recs["component"]=="total"), "delta_attn_sum"].fillna(0.0)
    #     df_recs.drop(columns=["delta_attn_sum"], inplace=True)

    return df_recs

def export_and_plot_topK(step: int, k: int = TOP_K):
    df = collect_annotated_neurons(step)

    # Two views: totals by neuron/head, for grad and for ΔW
    # 1) totals per structural unit (component == 'total')
    totals = df[df["component"] == "total"].copy()

    # Rank grad totals
    totals_grad = totals.copy()
    totals_grad["value"] = totals_grad["grad"]
    totals_grad = _concat_and_rank(totals_grad, "value")

    # Rank ΔW totals
    totals_delta = totals.copy()
    totals_delta["value"] = totals_delta["delta_total"]
    totals_delta = _concat_and_rank(totals_delta, "value")

    # 2) Export top-K combined table with identity strings
    def _ident(row):
        if row["kind"] == "MLP":
            return f"L{int(row['layer'])}/MLP:{int(row['unit'])}"
        else:
            return f"L{int(row['layer'])}/ATTN:head{int(row['unit'])}"

    out_rows = []

    for name, ranked in [("grad", totals_grad), ("delta", totals_delta)]:
        top = ranked.head(k).copy()
        top["who"] = top.apply(_ident, axis=1)
        top["energy_type"] = name
        out_rows.append(top[["energy_type","kind","layer","unit","who","value","rank","cum_frac","value_frac"]])

    out = pd.concat(out_rows, axis=0).reset_index(drop=True)
    out.to_csv(CSV_OUT, index=False)
    print(f"[CSV] wrote top-{k} neurons/heads with ranks → {CSV_OUT}")

    # ----- Plot 1: rank–cumulative scatter (color by kind) -----
    plt.figure(figsize=(6,4), dpi=200)
    for name, ranked, marker in [
        ("grad", totals_grad, "."),
        ("ΔW",  totals_delta, "x"),
    ]:
        color_map = np.where(ranked["kind"].values == "MLP", 0, 1)  # 0=MLP,1=ATTN
        # Normalize colors for a 2-color scatter (matplotlib auto palette is fine)
        plt.scatter(ranked["rank"].values[:k], ranked["cum_frac"].values[:k],
                    s=14, marker=marker, label=f"{name} (top{k})", alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Rank (log)"); plt.ylabel("Cumulative fraction"); plt.ylim(0,1)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(SCATTER_PNG, bbox_inches="tight"); plt.close()
    print(f"[PLOT] rank–cumulative scatter → {SCATTER_PNG}")

    # ----- Plot 2: bar chart of top-50 ΔW with labels -----
    top50 = totals_delta.head(min(50, len(totals_delta))).copy()
    if not top50.empty:
        top50["who"] = top50.apply(_ident, axis=1)
        plt.figure(figsize=(10,4), dpi=200)
        plt.bar(np.arange(len(top50)), top50["value"].values)
        plt.xticks(np.arange(len(top50)), top50["who"].values, rotation=90, fontsize=7)
        plt.ylabel("ΔW energy (L2²)"); plt.tight_layout()
        plt.savefig(BARS_PNG, bbox_inches="tight"); plt.close()
        print(f"[PLOT] top-50 ΔW bars → {BARS_PNG}")
    else:
        print("[PLOT] ΔW totals empty; skipped bar chart.")
        
# ================= ADDED =========================

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
        
    export_and_plot_topK(GLOBAL_STEP, TOP_K)

if __name__ == "__main__":
    main()