#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build per-layer timelines of row-wise weight deltas across checkpoints OR reuse
the NPZ deltas produced by analyze_weight_coldhot.py.

Per layer ℓ we stack rowwise deltas across transitions into H_ℓ ∈ ℝ^{T×R}:
  - T = number of transitions (N checkpoints - 1, or number of NPZ delta files)
  - R = rows (neurons/outputs) of the target linear (e.g., fc1.out_features)

Two modes:
  1) Checkpoint mode (default): load checkpoints and compute deltas.
  2) Reuse mode: --reuse_from_outdir points to analysis dir from
     analyze_weight_coldhot.py (reads weights/delta_{A->B}/*.npz).

Output under --outdir:
  by_layer/Lxx/
    mlp_fc1_timeline.{npz,png}
    q_proj_timeline.{npz,png}
    k_proj_timeline.{npz,png}
    v_proj_timeline.{npz,png}
    out_proj_timeline.{npz,png}
  transitions.json

Usage (reuse NPZ deltas):
  python analyze_coldhot_by_time.py \
    --reuse_from_outdir /pscratch/.../analysis \
    --outdir /pscratch/.../analysis_time \
    --which both --norm l2 --plot

Usage (compute from checkpoints, legacy):
  python analyze_coldhot_by_time.py \
    --ckpt_root /pscratch/.../runs/opt_fsdp \
    --ckpt_pattern 'checkpoint-epoch-*' \
    --include_final \
    --outdir /pscratch/.../analysis_time \
    --which both --norm l2 --plot
"""

import os, re, glob, json, argparse
from typing import Dict, List, Tuple, Optional
from glob import glob as _glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# -----------------------
# Utilities
# -----------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def extract_epoch_num(path: str) -> int:
    m = re.search(r"checkpoint-epoch-(\d+)$", path.rstrip("/"))
    return int(m.group(1)) if m else -1

def sorted_checkpoints(root: str, pattern: str, include_final: bool) -> List[str]:
    paths = sorted(glob.glob(os.path.join(root, pattern)), key=extract_epoch_num)
    if include_final:
        final = os.path.join(root, "checkpoint-final")
        if os.path.isdir(final):
            paths.append(final)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under {root} with pattern '{pattern}'.")
    return paths

def epoch_tag(path: str) -> str:
    return os.path.basename(path.rstrip("/"))

def plot_heat(mat: np.ndarray, title: str, outpath: str, cmap: str = "magma", log1p: bool = True):
    plt.figure(figsize=(11, 6))
    img = np.log1p(mat) if log1p else mat
    im = plt.imshow(img, aspect="auto", interpolation="nearest", cmap=cmap, origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Row index (neuron/output)")
    plt.ylabel("Transition index (time)")
    plt.title(title)
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------
# Discover modules (OPT-like)  [for checkpoint mode]
# -----------------------

def discover_mlp_fc1(model: nn.Module):
    fc1 = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            mo = re.match(r"model\\.decoder\\.layers\\.(\\d+)\\.fc1$", name)
            if mo:
                fc1.append((int(mo.group(1)), m))
    fc1.sort(key=lambda x: x[0])
    if not fc1:
        raise RuntimeError("No 'model.decoder.layers.{L}.fc1' modules found.")
    return fc1

def discover_mha_projs(model: nn.Module):
    projs = {k: [] for k in ["q_proj", "k_proj", "v_proj", "out_proj"]}
    pat = r"model\\.decoder\\.layers\\.(\\d+)\\.self_attn\\.(q_proj|k_proj|v_proj|out_proj)$"
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            mo = re.match(pat, name)
            if mo:
                lid, which = int(mo.group(1)), mo.group(2)
                projs[which].append((lid, m))
    for k in projs:
        projs[k].sort(key=lambda x: x[0])
    return projs

# -----------------------
# Row-wise deltas (for checkpoint mode)
# -----------------------

def row_change(Wa: torch.Tensor, Wb: torch.Tensor, norm: str) -> torch.Tensor:
    d = (Wb - Wa).abs()
    return d.sum(dim=1) if norm == "l1" else (d.square().sum(dim=1)).sqrt()

def mlp_delta_layerwise(A: AutoModelForCausalLM, B: AutoModelForCausalLM, norm: str) -> Dict[int, np.ndarray]:
    a_list = discover_mlp_fc1(A)
    b_list = discover_mlp_fc1(B)
    assert len(a_list) == len(b_list)
    out: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for (lid, la), (_, lb) in zip(a_list, b_list):
            Wa = la.weight.data.float().cpu()
            Wb = lb.weight.data.float().cpu()
            out[lid] = row_change(Wa, Wb, norm=norm).numpy()  # [R]
    return out

def mha_delta_layerwise(A: AutoModelForCausalLM, B: AutoModelForCausalLM, norm: str) -> Dict[str, Dict[int, np.ndarray]]:
    pa = discover_mha_projs(A)
    pb = discover_mha_projs(B)
    out: Dict[str, Dict[int, np.ndarray]] = {k: {} for k in ["q_proj","k_proj","v_proj","out_proj"]}
    with torch.no_grad():
        for k in ["q_proj","k_proj","v_proj","out_proj"]:
            la = pa[k]; lb = pb[k]
            if not la or not lb:
                continue
            for (lid, ma), (_, mb) in zip(la, lb):
                Wa = ma.weight.data.float().cpu()
                Wb = mb.weight.data.float().cpu()
                out[k][lid] = row_change(Wa, Wb, norm=norm).numpy()  # [R]
    return out

# -----------------------
# Build timelines from checkpoints  (legacy path)
# -----------------------

def build_timelines(ckpts: List[str], which: str, norm: str):
    transitions: List[str] = []
    tl_mlp: Dict[int, List[np.ndarray]] = {}
    tl_mha: Dict[str, Dict[int, List[np.ndarray]]] = {k: {} for k in ["q_proj","k_proj","v_proj","out_proj"]}

    for a, b in zip(ckpts[:-1], ckpts[1:]):
        ta, tb = epoch_tag(a), epoch_tag(b)
        trans_tag = f"{ta}->{tb}"
        transitions.append(trans_tag)

        A = AutoModelForCausalLM.from_pretrained(a, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        B = AutoModelForCausalLM.from_pretrained(b, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        if which in ("mlp","both"):
            d_mlp = mlp_delta_layerwise(A, B, norm=norm)
            for lid, rowvec in d_mlp.items():
                tl_mlp.setdefault(lid, []).append(rowvec[None, :])

        if which in ("mha","both"):
            d_mha = mha_delta_layerwise(A, B, norm=norm)
            for proj, dct in d_mha.items():
                for lid, rowvec in dct.items():
                    tl_mha[proj].setdefault(lid, []).append(rowvec[None, :])

        del A, B
        torch.cuda.empty_cache()

    tl_mlp_np: Optional[Dict[int, np.ndarray]] = None
    if which in ("mlp","both") and tl_mlp:
        tl_mlp_np = {lid: np.concatenate(chunks, axis=0) for lid, chunks in tl_mlp.items()}

    tl_mha_np: Optional[Dict[str, Dict[int, np.ndarray]]] = None
    if which in ("mha","both"):
        tl_mha_np = {}
        for proj in ["q_proj","k_proj","v_proj","out_proj"]:
            if tl_mha[proj]:
                tl_mha_np[proj] = {lid: np.concatenate(chunks, axis=0)
                                   for lid, chunks in tl_mha[proj].items()}

    return transitions, tl_mlp_np, tl_mha_np

# -----------------------
# Build timelines from existing NPZ deltas (REUSE PATH)
# -----------------------

def _find_transition_dirs(reuse_outdir: str) -> List[str]:
    root = os.path.join(reuse_outdir, "weights")
    return sorted(_glob(os.path.join(root, "delta_*")))

def _trans_tag_from_dir(d: str) -> str:
    base = os.path.basename(d.rstrip("/"))
    return base[len("delta_"):] if base.startswith("delta_") else base

def collect_transitions_from_npz(reuse_outdir: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Scan weights/delta_{A->B}/ and collect available per-transition NPZ paths.

    Returns:
      transitions: ordered list of trans_tag (alphabetical as a fallback)
      files_map: dict trans_tag -> dict(component -> filepath)
                 components: mlp_fc1, attn_q_proj, attn_k_proj, attn_v_proj, attn_out_proj
    """
    dirs = _find_transition_dirs(reuse_outdir)
    if not dirs:
        raise FileNotFoundError(f"No delta_* directories under {reuse_outdir}/weights")

    files_map: Dict[str, Dict[str, str]] = {}
    for d in dirs:
        tag = _trans_tag_from_dir(d)
        comp = {}
        # MLP
        mlp = os.path.join(d, "mlp_fc1_row_delta.npz")
        if os.path.exists(mlp):
            comp["mlp_fc1"] = mlp
        # MHA
        for k in ["q_proj","k_proj","v_proj","out_proj"]:
            p = os.path.join(d, f"attn_{k}_row_delta.npz")
            if os.path.exists(p):
                comp[f"attn_{k}"] = p
        if comp:
            files_map[tag] = comp

    transitions = sorted(files_map.keys())  # alphabetical; good enough if tags contain epoch numbers
    return transitions, files_map

def build_timelines_from_npz(reuse_outdir: str, which: str):
    transitions, files_map = collect_transitions_from_npz(reuse_outdir)

    tl_mlp_lists: Dict[int, List[np.ndarray]] = {}
    tl_mha_lists: Dict[str, Dict[int, List[np.ndarray]]] = {k: {} for k in ["q_proj","k_proj","v_proj","out_proj"]}

    # Iterate in time order; each NPZ contains delta [L, R]
    for trans in transitions:
        comp = files_map[trans]
        if which in ("mlp","both") and "mlp_fc1" in comp:
            data = np.load(comp["mlp_fc1"])
            M = data["delta"]  # [L, R]
            L, R = M.shape
            for lid in range(L):
                tl_mlp_lists.setdefault(lid, []).append(M[lid][None, :])

        if which in ("mha","both"):
            for k in ["q_proj","k_proj","v_proj","out_proj"]:
                key = f"attn_{k}"
                if key in comp:
                    data = np.load(comp[key])
                    M = data["delta"]  # [L, R]
                    L, R = M.shape
                    for lid in range(L):
                        tl_mha_lists[k].setdefault(lid, []).append(M[lid][None, :])

    tl_mlp_np = None
    if which in ("mlp","both") and tl_mlp_lists:
        tl_mlp_np = {lid: np.concatenate(v, axis=0) for lid, v in tl_mlp_lists.items()}

    tl_mha_np = None
    if which in ("mha","both"):
        tl_mha_np = {}
        for k in ["q_proj","k_proj","v_proj","out_proj"]:
            if tl_mha_lists[k]:
                tl_mha_np[k] = {lid: np.concatenate(v, axis=0) for lid, v in tl_mha_lists[k].items()}

    print(f"[*] Reused transitions: {len(transitions)}\n     Components: "
          f"{'MLP ' if tl_mlp_np else ''}"
          f"{'MHA(q/k/v/out)' if tl_mha_np else ''}")
    return transitions, tl_mlp_np, tl_mha_np

# -----------------------
# Save per-layer timelines + plots
# -----------------------

def save_timelines(outdir: str,
                   transitions: List[str],
                   tl_mlp: Dict[int, np.ndarray] = None,
                   tl_mha: Dict[str, Dict[int, np.ndarray]] = None,
                   plot: bool = True,
                   norm: str = "l2"):
    layers_root = ensure_dir(os.path.join(outdir, "by_layer"))
    with open(os.path.join(outdir, "transitions.json"), "w") as f:
        json.dump(transitions, f, indent=2)

    def _save_one(lid_dir: str, name: str, mat: np.ndarray, title: str):
        path_npz = os.path.join(lid_dir, f"{name}_timeline.npz")
        np.savez_compressed(path_npz, delta_timeline=mat, transitions=np.array(transitions, dtype=object))
        if plot:
            plot_heat(mat, f"{title} — norm={norm}", os.path.join(lid_dir, f"{name}_timeline.png"))

    if tl_mlp is not None:
        for lid, H in tl_mlp.items():
            lid_dir = ensure_dir(os.path.join(layers_root, f"L{lid:02d}"))
            _save_one(lid_dir, "mlp_fc1", H, f"L{lid:02d} Δ fc1 rows over time")

    if tl_mha is not None:
        for proj, per_layer in tl_mha.items():
            for lid, H in per_layer.items():
                lid_dir = ensure_dir(os.path.join(layers_root, f"L{lid:02d}"))
                _save_one(lid_dir, proj, H, f"L{lid:02d} Δ {proj} rows over time")

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    # Reuse path (preferred if you already ran analyze_weight_coldhot.py)
    ap.add_argument("--reuse_from_outdir", default=None,
                    help="Path to analysis outdir from analyze_weight_coldhot.py (reads weights/delta_*/ NPZs).")
    # Fallback checkpoint path (legacy)
    ap.add_argument("--ckpt_root", default=None, help="Root folder with checkpoints (HF layout).")
    ap.add_argument("--ckpt_pattern", default="checkpoint-epoch-*",
                    help="Glob for checkpoints to order.")
    ap.add_argument("--include_final", action="store_true",
                    help="Append checkpoint-final at the end if present.")
    # Common
    ap.add_argument("--outdir", required=True, help="Output directory for timelines and plots.")
    ap.add_argument("--which", choices=["mlp","mha","both"], default="both")
    ap.add_argument("--norm", choices=["l1","l2"], default="l2")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.reuse_from_outdir:
        # Build timelines from existing NPZ deltas (fast path)
        transitions, tl_mlp, tl_mha = build_timelines_from_npz(args.reuse_from_outdir, which=args.which)
    else:
        if not args.ckpt_root:
            raise ValueError("Either --reuse_from_outdir or --ckpt_root must be provided.")
        ckpts = sorted_checkpoints(args.ckpt_root, args.ckpt_pattern, args.include_final)
        print("[*] checkpoints:", [epoch_tag(c) for c in ckpts])
        transitions, tl_mlp, tl_mha = build_timelines(ckpts, which=args.which, norm=args.norm)

    save_timelines(args.outdir, transitions, tl_mlp, tl_mha, plot=args.plot, norm=args.norm)
    print(f"[✓] Done. Per-layer timelines written under: {os.path.join(args.outdir, 'by_layer')}")

if __name__ == "__main__":
    main()