#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot hot/cold neuron snapshots produced by finetune_opt_checkpoint.py.

Usage:
  python plot_coldhot.py --dir runs/opt13b_hotcold
  python plot_coldhot.py --dir runs/opt13b_hotcold --step 0000200 --tau 1e-5 --g0 0 --g1 1

Outputs:
  - <outdir>/plots/heatmap_G_group{g}_step{step}.png
  - <outdir>/plots/heatmap_A_group{g}_step{step}.png   (if --plot_A)
  - <outdir>/plots/hot_fraction_step{step}.png
"""

import os, re, argparse, glob
import numpy as np
import matplotlib.pyplot as plt

def latest_step_tag(outdir: str):
    files = sorted(glob.glob(os.path.join(outdir, "hotcold_step*.npz")))
    if not files:
        raise FileNotFoundError(f"No snapshots found under {outdir}")
    m = re.search(r"hotcold_step(\d+)\.npz$", files[-1])
    if not m:
        raise RuntimeError("Could not parse step from last file name.")
    return m.group(1)

def load_snapshot(outdir: str, step_tag: str):
    path = os.path.join(outdir, f"hotcold_step{step_tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot not found: {path}")
    return np.load(path, allow_pickle=True)

def plot_heatmap(mat, title, path, vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(mat, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Neurons")
    plt.ylabel("Layers (top=0)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with hotcold_step*.npz")
    ap.add_argument("--step", default=None, help="7-digit step tag; if omitted, use the latest")
    ap.add_argument("--tau", type=float, default=1e-5, help="hot threshold for H or G comparison")
    ap.add_argument("--g0", type=int, default=0, help="first group index")
    ap.add_argument("--g1", type=int, default=1, help="second group index")
    ap.add_argument("--plot_A", action="store_true", help="also plot activation EMA heatmaps")
    args = ap.parse_args()

    step_tag = args.step or latest_step_tag(args.dir)
    snap = load_snapshot(args.dir, step_tag)

    layers   = int(snap["layers"])
    neurons  = int(snap["neurons"])
    groups   = int(snap["groups"])
    G_list   = [snap["G"][i] for i in range(groups)]
    H_list   = [snap["H"][i] for i in range(groups)]
    A_list   = [snap["A"][i] for i in range(groups)]

    plots_dir = os.path.join(args.dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Heatmaps for G
    for g in range(groups):
        G = G_list[g]
        plot_heatmap(
            G,
            title=f"G (EMA |grad|), group {g}, step {int(snap['step'])}",
            path=os.path.join(plots_dir, f"heatmap_G_group{g}_step{step_tag}.png"),
            vmin=None, vmax=None
        )

    # Optional heatmaps for A
    if args.plot_A:
        for g in range(groups):
            A = A_list[g]
            plot_heatmap(
                A,
                title=f"A (EMA ReLU act), group {g}, step {int(snap['step'])}",
                path=os.path.join(plots_dir, f"heatmap_A_group{g}_step{step_tag}.png"),
                vmin=None, vmax=None
            )

    # Hot fraction per layer for each group (using G > tau as proxy)
    plt.figure(figsize=(10, 4))
    for g in range(groups):
        G = G_list[g]
        hot = (G > args.tau).astype(np.float32)
        frac = hot.mean(axis=1)  # per-layer fraction of hot neurons
        plt.plot(np.arange(layers), frac, label=f"group {g}")
    plt.xlabel("Layer")
    plt.ylabel(f"Frac hot (G > {args.tau:g})")
    plt.title(f"Hot neuron fraction per layer @ step {int(snap['step'])}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"hot_fraction_step{step_tag}.png")
    plt.savefig(out_path, dpi=200); plt.close()

    print(f"✓ Wrote plots to {plots_dir}")

if __name__ == "__main__":
    main()