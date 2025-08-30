#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timeline plot of hot/cold neurons across training transitions.

Reads NPZ masks from:
  <ANALYSIS_OUTDIR>/hotcold/<subdir>/<component>/hotcold_delta_{A}->{B}.npz

Where:
  subdir:
    - "mlp" with component "mlp_fc1"
    - "mha" with components "q_proj", "k_proj", "v_proj", "out_proj"

It constructs a matrix [T, R] for a chosen layer:
  T = #transitions (epoch_i -> epoch_{i+1})
  R = #rows (neurons or output channels)
Each cell is 1 if row was hot in that transition, else 0.

Features:
  - sort neurons by 'hot_freq' | 'first_hot' | 'variance'
  - optionally keep only top_k rows
  - render a raster timeline + per-transition hot-count line

Usage:
  python plot_coldhot_timelines.py \
    --analysis_dir /pscratch/.../analysis \
    --component mlp_fc1 \
    --layer 10 \
    --top_k 256 \
    --sort hot_freq \
    --out /pscratch/.../analysis/plots/mlp_fc1_L10_timeline.png
"""

import os, re, argparse, glob
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def parse_transition_tag(fname: str):
    # expects hotcold_delta_epochX->epochY.npz (or similar)
    m = re.search(r"hotcold_delta_(.+)->(.+)\.npz$", fname)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def epoch_num(tag: str):
    # try to extract integer trailing number from 'checkpoint-epoch-12' or 'epoch-12'
    m = re.search(r"(\d+)$", tag)
    return int(m.group(1)) if m else -1

def load_masks_for_component(analysis_dir: str, component: str):
    """
    Returns:
      transitions: list of (start_tag, end_tag) sorted by epoch number if possible
      masks: list of np.ndarray with shape [layers, rows] boolean
      thresholds: list of (hot_tau, cold_tau) per file (may be scalar or per-layer array)
    """
    if component == "mlp_fc1":
        comp_dir = os.path.join(analysis_dir, "hotcold", "mlp")
        # files live directly under comp_dir
        files = sorted(glob.glob(os.path.join(comp_dir, "hotcold_delta_*.npz")))
    else:
        # mha/<proj>/
        comp_dir = os.path.join(analysis_dir, "hotcold", "mha", component)
        files = sorted(glob.glob(os.path.join(comp_dir, "hotcold_delta_*.npz")))

    items = []
    for f in files:
        a, b = parse_transition_tag(os.path.basename(f))
        if a is None: 
            continue
        data = np.load(f)
        hot = data["hot"].astype(bool)
        hot_tau = data.get("hot_tau", None)
        cold_tau = data.get("cold_tau", None)
        items.append(((a,b), f, hot, hot_tau, cold_tau))

    if not items:
        raise FileNotFoundError(f"No hot/cold files found for component '{component}' under {comp_dir}")

    # Sort by start epoch number if available, else lexicographically
    def sort_key(it):
        (a,b), _, _, _, _ = it
        ea, eb = epoch_num(a), epoch_num(b)
        return (ea if ea >= 0 else a, eb if eb >= 0 else b)
    items.sort(key=sort_key)

    transitions = [it[0] for it in items]
    masks = [it[2] for it in items]
    taus = [(it[3], it[4]) for it in items]
    return transitions, masks, taus

def build_timeline(masks, layer: int):
    """
    masks: list of [L, R] bool
    Return: H [T, R] where H[t, r] = hot/not-hot of row r at transition t for layer 'layer'
    """
    # shape checks
    L = masks[0].shape[0]
    assert all(m.shape[0] == L for m in masks), "layer count mismatch across transitions"
    R = masks[0].shape[1]
    H = np.zeros((len(masks), R), dtype=np.int8)
    for t, M in enumerate(masks):
        H[t] = M[layer].astype(np.int8)
    return H  # [T, R]

def sort_rows(H: np.ndarray, mode: str):
    """
    H: [T, R] 0/1
    Returns reordered H and the permutation indices.
    """
    T, R = H.shape
    if mode == "hot_freq":
        score = H.sum(axis=0)  # how often hot
        order = np.argsort(-score)  # descending
    elif mode == "first_hot":
        # first transition index where row becomes hot; inf if never
        first = np.argmax(H > 0, axis=0)  # returns 0 if all 0s; need mask
        never = (H.sum(axis=0) == 0)
        first = first.astype(np.float64)
        first[never] = np.inf
        order = np.argsort(first)  # earlier first-hot on top, then never at bottom
    elif mode == "variance":
        score = H.var(axis=0)
        order = np.argsort(-score)
    else:
        order = np.arange(R)
    return H[:, order], order

def maybe_topk(H: np.ndarray, k: int):
    T, R = H.shape
    if k is None or k <= 0 or k >= R:
        return H
    # keep rows with highest hot frequency
    freq = H.sum(axis=0)
    keep = np.argsort(-freq)[:k]
    return H[:, keep]

def plot_timeline(H: np.ndarray, transitions, title: str, outpath: str, overlay_counts: bool = True):
    """
    H: [T, R] timeline matrix (time x rows)
    """
    T, R = H.shape
    fig_h = 6 if not overlay_counts else 7.5

    plt.figure(figsize=(max(10, T * 0.4), fig_h))
    # transpose so neurons on y axis, time on x axis
    im = plt.imshow(H.T, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.xlabel("Transition")
    plt.ylabel("Neuron / row (sorted)")
    plt.title(title)

    # x tick labels (sparse to reduce clutter)
    xs = np.arange(T)
    if T <= 20:
        xticks = xs
    else:
        step = max(1, T // 20)
        xticks = xs[::step]
    labels = [f"{a}->{b}" for (a,b) in transitions]
    plt.xticks(xticks, [labels[i] for i in xticks], rotation=45, ha="right")

    # optional overlay: hot count per transition as a line at the bottom axis
    if overlay_counts:
        ax2 = plt.gca().inset_axes([0.08, -0.28, 0.84, 0.22])  # [x0,y0,w,h] in axes fraction
        counts = H.sum(axis=1)  # per transition
        ax2.plot(np.arange(T), counts)
        ax2.set_xlim(0, T-1)
        ax2.set_ylabel("# hot")
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([labels[i] for i in xticks], rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Hot count per transition", fontsize=10)

    ensure_dir(os.path.dirname(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", required=True, help="folder that contains hotcold/ from analyze_weight_coldhot.py")
    ap.add_argument("--component", required=True,
                    choices=["mlp_fc1", "q_proj", "k_proj", "v_proj", "out_proj"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sort", default="hot_freq", choices=["hot_freq","first_hot","variance","none"])
    ap.add_argument("--top_k", type=int, default=256, help="keep only top-K rows after sorting (0=all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    transitions, masks, _ = load_masks_for_component(args.analysis_dir, args.component)

    # basic shape info for bounds checking
    L, R = masks[0].shape
    # if masks were saved as [layers, rows], we set L, R from the first mask directly
    L, R = masks[0].shape[0], masks[0].shape[1]
    if args.layer < 0 or args.layer >= L:
        raise ValueError(f"Layer {args.layer} is out of range [0, {L-1}] for component {args.component}")

    H = build_timeline(masks, args.layer)  # [T, R]
    H, order = sort_rows(H, args.sort)
    H = maybe_topk(H, args.top_k)

    title = f"{args.component} — layer {args.layer} — timeline (sorted={args.sort}, top_k={args.top_k})"
    plot_timeline(H, transitions, title, args.out, overlay_counts=True)
    print(f"[✓] wrote {args.out}")

if __name__ == "__main__":
    main()