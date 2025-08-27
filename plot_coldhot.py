#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot hot/cold diagnostics from snapshot .npz files produced by finetune_opt_checkpoint.py

Creates:
  - heat_grad_g{g}.png       : |grad| EMA heatmap per group
  - heat_selectivity.png     : S = (G0-G1)/(G0+G1) heatmap
  - hot_fraction_layer.png   : fraction of hot neurons per layer (fixed tau & percentile)
  - grad_hist_group0.png     : histogram (log scale) of |grad| EMA for group 0

Quick use:
  python plot_hotcold.py --dir runs/opt13b_hotcold --g0 0 --g1 1 --tau 1e-5
"""

import os, glob, numpy as np
import matplotlib.pyplot as plt

def load_latest(npz_dir):
    files = sorted(glob.glob(os.path.join(npz_dir, "hotcold_step*.npz")))
    if not files:
        raise FileNotFoundError(f"No snapshots found in {npz_dir}")
    return files[-1]

def plot_heat(mat, title, out_png):
    plt.figure(figsize=(12,5))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Neuron index")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main(npz_dir, g0=0, g1=1, tau=1e-5, perc=90):
    path = load_latest(npz_dir)
    snap = np.load(path, allow_pickle=True)
    G = snap["G"]  # object array of length groups
    layers = int(snap["layers"].item())
    neurons = int(snap["neurons"].item())
    groups = int(snap["groups"].item())
    step = int(snap["step"].item())
    print(f"Loaded {path}  step={step}  layers={layers}  neurons={neurons}  groups={groups}")

    # 1) Per-group |grad| heatmaps
    for gi in range(groups):
        Gi = G[gi]
        plot_heat(Gi, f"|grad| EMA (group {gi})", os.path.join(npz_dir, f"heat_grad_g{gi}.png"))

    # 2) Selectivity S
    if groups >= 2:
        G0, G1 = G[g0], G[g1]
        S = (G0 - G1) / (G0 + G1 + 1e-8)
        plot_heat(S, f"Selectivity S (group {g0} vs {g1})", os.path.join(npz_dir, "heat_selectivity.png"))

    # 3) Hot fraction per layer (group g0)
    Gg = G[g0]
    hot_frac_tau = (Gg > tau).mean(axis=1)  # fixed threshold
    thr = np.percentile(Gg, perc)
    hot_frac_p = (Gg > thr).mean(axis=1)    # percentile threshold

    x = np.arange(layers)
    plt.figure()
    plt.plot(x, hot_frac_tau, label=f"hot frac (> {tau:g})")
    plt.plot(x, hot_frac_p, label=f"hot frac (> p{perc})")
    plt.xlabel("Layer"); plt.ylabel("Fraction of hot neurons")
    plt.title(f"Hot fraction per layer (group {g0})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(npz_dir, "hot_fraction_layer.png"), dpi=150)
    plt.close()

    # 4) Distribution snapshot
    plt.figure()
    flat = Gg.flatten()
    plt.hist(flat, bins=100, log=True)
    plt.xlabel("|grad| EMA (group {g0})"); plt.ylabel("count (log)")
    plt.title("Per-neuron |grad| distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(npz_dir, "grad_hist_group0.png"), dpi=150)
    plt.close()

    print("Wrote plots to:", npz_dir)
    print("\nDecision guide:")
    print("• Heavy-tailed histogram + small hot fraction (e.g., ≤10–20%) ⇒ hot/cold exists.")
    print("• Selectivity heatmap with clear ± regions ⇒ groups specialize different neurons.")
    print("• Layerwise patterns (ridges) often strengthen in later layers.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="snapshot directory")
    ap.add_argument("--g0", type=int, default=0)
    ap.add_argument("--g1", type=int, default=1)
    ap.add_argument("--tau", type=float, default=1e-5)
    ap.add_argument("--perc", type=int, default=90)
    args = ap.parse_args()
    main(args.dir, args.g0, args.g1, args.tau, args.perc)