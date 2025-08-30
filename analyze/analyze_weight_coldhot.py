#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze row-wise weight deltas across checkpoints for:
  • MLP: fc1 (per-neuron rows)
  • MHA: q_proj, k_proj, v_proj, out_proj (per-output rows)

Outputs (under --outdir):
  weights/
    delta_{A}->{B}/
      mlp_fc1_row_delta.npz      # [L, H_mlp]
      attn_q_row_delta.npz       # [L, H_attn]  (if --which includes mha)
      attn_k_row_delta.npz
      attn_v_row_delta.npz
      attn_out_row_delta.npz
      *.png                      # optional heatmaps (log1p)
  hotcold/
    mlp/
      hotcold_delta_{A}->{B}.npz       # hot/cold boolean masks for that transition
    mha/
      <proj>/hotcold_delta_{A}->{B}.npz
    packed_mlp_fc1.npz                 # (optional) stacked timeline, if --pack_tensors
    packed_q_proj.npz                  # (optional) stacked timeline, if --pack_tensors
    ...
    hotcold_summary.csv                # row counts per layer across transitions
    hotcold_overlap.csv                # Jaccard & newly-hot/cooled stats
  manifest.json                        # list of transitions and files produced

Usage example:
  python analyze_weight_coldhot.py \
    --ckpt_root /pscratch/.../runs/opt17b_fsdp_gsm8k \
    --ckpt_pattern "checkpoint-epoch-*" \
    --include_final \
    --outdir /pscratch/.../runs/opt17b_fsdp_gsm8k/analysis \
    --which both --norm l2 \
    --hot_pct 95 --cold_pct 50 --per_layer_thresholds \
    --plot \
    --reuse_if_exists --pack_tensors
"""

import os, re, glob, argparse, csv, json
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# -----------------------
# Utilities
# -----------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def write_manifest(path: str, rows: List[Dict]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

def read_manifest(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

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

def plot_heat(mat: np.ndarray, title: str, outpath: str, cmap="magma"):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(np.log1p(mat), aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Row index (neuron/output)")
    plt.ylabel("Layer index")
    plt.title(title)
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------
# Discover modules (OPT)
# -----------------------

def discover_mlp_fc1(model: nn.Module) -> List[Tuple[int, nn.Linear]]:
    fc1 = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            mo = re.match(r"model\.decoder\.layers\.(\d+)\.fc1$", name)
            if mo:
                fc1.append((int(mo.group(1)), m))
    fc1.sort(key=lambda x: x[0])
    if not fc1:
        raise RuntimeError("No 'model.decoder.layers.{L}.fc1' modules found.")
    return fc1

def discover_mha_projs(model: nn.Module) -> Dict[str, List[Tuple[int, nn.Linear]]]:
    projs = {k: [] for k in ["q_proj", "k_proj", "v_proj", "out_proj"]}
    pat = r"model\.decoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"
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
# Row-wise deltas
# -----------------------

def row_change(Wa: torch.Tensor, Wb: torch.Tensor, norm: str) -> torch.Tensor:
    d = (Wb - Wa).abs()
    return d.sum(dim=1) if norm == "l1" else (d.square().sum(dim=1)).sqrt()

def compute_mlp_delta(A: AutoModelForCausalLM, B: AutoModelForCausalLM, norm: str) -> np.ndarray:
    fa = discover_mlp_fc1(A); fb = discover_mlp_fc1(B)
    assert len(fa) == len(fb)
    L = len(fa); H = fa[0][1].out_features
    for (_, la), (_, lb) in zip(fa, fb):
        assert la.out_features == lb.out_features and la.in_features == lb.in_features
    M = np.zeros((L, H), dtype=np.float32)
    with torch.no_grad():
        for (lid, la), (_, lb) in zip(fa, fb):
            Wa = la.weight.data.float().cpu()
            Wb = lb.weight.data.float().cpu()
            M[lid] = row_change(Wa, Wb, norm=norm).numpy()
    return M

def compute_mha_deltas(A: AutoModelForCausalLM, B: AutoModelForCausalLM, norm: str) -> Dict[str, np.ndarray]:
    pa = discover_mha_projs(A); pb = discover_mha_projs(B)
    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for k in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            fa, fb = pa[k], pb[k]
            if not fa or not fb:
                continue
            assert len(fa) == len(fb)
            L = len(fa); H = fa[0][1].out_features
            for (_, la), (_, lb) in zip(fa, fb):
                assert la.out_features == lb.out_features and la.in_features == lb.in_features
            M = np.zeros((L, H), dtype=np.float32)
            for (lid, la), (_, lb) in zip(fa, fb):
                Wa = la.weight.data.float().cpu()
                Wb = lb.weight.data.float().cpu()
                M[lid] = row_change(Wa, Wb, norm=norm).numpy()
            out[k] = M
    return out

# -----------------------
# Hot/Cold labeling
# -----------------------

def thresholds_from_percentile(M: np.ndarray, hot_pct: float, cold_pct: float, per_layer: bool):
    if per_layer:
        L = M.shape[0]
        hot = np.zeros((L,), dtype=np.float32)
        cold = np.zeros((L,), dtype=np.float32)
        for i in range(L):
            li = M[i]
            hot[i] = np.percentile(li, hot_pct)
            cold[i] = np.percentile(li, cold_pct)
        return hot, cold
    else:
        return np.percentile(M, hot_pct), np.percentile(M, cold_pct)

def apply_thresholds(M: np.ndarray, hot_tau, cold_tau, per_layer: bool):
    if per_layer:
        hot_mask = M >= hot_tau[:, None]
        cold_mask = M <= cold_tau[:, None]
    else:
        hot_mask = M >= hot_tau
        cold_mask = M <= cold_tau
    return hot_mask, cold_mask

# -----------------------
# Reuse / packing helpers
# -----------------------

def transition_dirs(outdir: str, trans_tag: str) -> Dict[str, str]:
    wt = ensure_dir(os.path.join(outdir, "weights", f"delta_{trans_tag}"))
    hc_mlp = ensure_dir(os.path.join(outdir, "hotcold", "mlp"))
    hc_mha = ensure_dir(os.path.join(outdir, "hotcold", "mha"))
    return {"weights": wt, "mlp": hc_mlp, "mha": hc_mha}

def transition_complete(outdir: str, trans_tag: str, which: str) -> bool:
    """Check whether required delta & hot/cold files exist for this transition."""
    dirs = transition_dirs(outdir, trans_tag)
    ok = True
    if which in ("mlp","both"):
        ok = ok and os.path.exists(os.path.join(dirs["weights"], "mlp_fc1_row_delta.npz"))
        ok = ok and os.path.exists(os.path.join(dirs["mlp"], f"hotcold_delta_{trans_tag}.npz"))
    if which in ("mha","both"):
        for k in ["q_proj","k_proj","v_proj","out_proj"]:
            ok = ok and os.path.exists(os.path.join(dirs["weights"], f"attn_{k}_row_delta.npz"))
            sub = os.path.join(dirs["mha"], k)
            ok = ok and os.path.exists(os.path.join(sub, f"hotcold_delta_{trans_tag}.npz"))
    return ok

def pack_component_masks(hotcold_dir: str, component: str, out_npz: str):
    """
    Read all per-transition hotcold files for a component and stack into one NPZ:
      transitions: array of strings [T]
      hot: bool array [T, L, R]
      cold: bool array [T, L, R]
      hot_tau, cold_tau: object arrays (per-transition arrays or scalars)
    """
    import glob as _glob
    if component == "mlp_fc1":
        comp_dir = os.path.join(hotcold_dir, "mlp")
        files = sorted(_glob.glob(os.path.join(comp_dir, "hotcold_delta_*.npz")))
    else:
        comp_dir = os.path.join(hotcold_dir, "mha", component.replace("attn_",""))
        files = sorted(_glob.glob(os.path.join(comp_dir, "hotcold_delta_*.npz")))
    if not files:
        return False

    transitions = []
    hot_list, cold_list = [], []
    hot_taus, cold_taus = [], []
    L, R = None, None
    for f in files:
        base = os.path.basename(f)
        m = re.search(r"hotcold_delta_(.+)\.npz$", base)
        trans = m.group(1) if m else base
        data = np.load(f, allow_pickle=True)
        hot = data["hot"].astype(bool)
        cold = data["cold"].astype(bool)
        if L is None:
            L, R = hot.shape[0], hot.shape[1]
        transitions.append(trans)
        hot_list.append(hot[None, ...])
        cold_list.append(cold[None, ...])
        hot_taus.append(data.get("hot_tau", None))
        cold_taus.append(data.get("cold_tau", None))
    hot_arr = np.concatenate(hot_list, axis=0)   # [T, L, R]
    cold_arr = np.concatenate(cold_list, axis=0) # [T, L, R]
    ensure_dir(os.path.dirname(out_npz))
    np.savez_compressed(out_npz,
                        transitions=np.array(transitions, dtype=object),
                        hot=hot_arr,
                        cold=cold_arr,
                        hot_tau=np.array(hot_taus, dtype=object),
                        cold_tau=np.array(cold_taus, dtype=object))
    return True

# -----------------------
# Main analysis
# -----------------------

def analyze(args):
    if args.which in ("mlp", "both"):
        print("analyze mlp")
        if args.plot:
            print("args plot")
    ckpts = sorted_checkpoints(args.ckpt_root, args.ckpt_pattern, args.include_final)
    tags = [epoch_tag(c) for c in ckpts]
    print("[*] checkpoints:", tags)

    weights_dir = ensure_dir(os.path.join(args.outdir, "weights"))
    hotcold_dir = ensure_dir(os.path.join(args.outdir, "hotcold"))

    # Summaries
    hc_rows = [["transition","component","layer","hot_count","cold_count","rows"]]
    overlap_rows = [["transition","component","layer","jaccard_hot","newly_hot","cooled_off","hot_t","hot_t1"]]

    prev_hot_masks = {}  # component -> layer -> boolean array (from previous transition)
    manifest_rows: List[Dict] = []

    for (a, b) in zip(ckpts[:-1], ckpts[1:]):
        ta, tb = epoch_tag(a), epoch_tag(b)
        trans_tag = f"{ta}->{tb}"
        print(f"[Δ] {trans_tag}")

        # Reuse if all outputs already exist
        if args.reuse_if_exists and transition_complete(args.outdir, trans_tag, args.which):
            print(f"[skip] {trans_tag} already computed; reusing on-disk results.")
            manifest_rows.append({"transition": trans_tag, "reused": True})
            # For overlap dynamics, we need the current hot masks to become "prev" for next iteration.
            # Load masks to update prev_hot_masks.
            if args.which in ("mlp","both"):
                mlp_mask_f = os.path.join(hotcold_dir, "mlp", f"hotcold_delta_{trans_tag}.npz")
                M = np.load(mlp_mask_f, allow_pickle=True)["hot"].astype(bool)
                prev_hot_masks["mlp_fc1"] = M
            if args.which in ("mha","both"):
                for k in ["q_proj","k_proj","v_proj","out_proj"]:
                    f = os.path.join(hotcold_dir, "mha", k, f"hotcold_delta_{trans_tag}.npz")
                    if os.path.exists(f):
                        M = np.load(f, allow_pickle=True)["hot"].astype(bool)
                        prev_hot_masks[f"attn_{k}"] = M
            continue

        # Load models on CPU
        A = AutoModelForCausalLM.from_pretrained(a, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        B = AutoModelForCausalLM.from_pretrained(b, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        manifest_entry = {"transition": trans_tag, "reused": False, "files": {}}

        # --- compute and save deltas ---
        comp_mats: Dict[str, np.ndarray] = {}

        if args.which in ("mlp", "both"):
            print("analyze mlp")
            M_mlp = compute_mlp_delta(A, B, norm=args.norm)
            outdir = ensure_dir(os.path.join(weights_dir, f"delta_{trans_tag}"))
            mlp_delta_path = os.path.join(outdir, "mlp_fc1_row_delta.npz")
            np.savez_compressed(mlp_delta_path,
                                delta=M_mlp, norm=args.norm, layers=M_mlp.shape[0], neurons=M_mlp.shape[1])
            manifest_entry["files"]["mlp_delta"] = mlp_delta_path
            print("- plotting mlp_fc1_row_delta.png")
            if args.plot:
                plot_heat(M_mlp, f"Δ fc1 rows ({args.norm}) — {trans_tag}",
                          os.path.join(outdir, "mlp_fc1_row_delta.png"))
            comp_mats["mlp_fc1"] = M_mlp

        if args.which in ("mha", "both"):
            mha = compute_mha_deltas(A, B, norm=args.norm)
            outdir = ensure_dir(os.path.join(weights_dir, f"delta_{trans_tag}"))
            for k, M in mha.items():
                attn_delta_path = os.path.join(outdir, f"attn_{k}_row_delta.npz")
                np.savez_compressed(attn_delta_path,
                                    delta=M, norm=args.norm, layers=M.shape[0], rows=M.shape[1])
                manifest_entry["files"][f"attn_{k}_delta"] = attn_delta_path
                if args.plot:
                    plot_heat(M, f"Δ {k} rows ({args.norm}) — {trans_tag}",
                              os.path.join(outdir, f"attn_{k}_row_delta.png"))
                comp_mats[f"attn_{k}"] = M

        # --- hot/cold per transition ---
        for comp, M in comp_mats.items():
            if args.per_layer_thresholds:
                hot_tau, cold_tau = thresholds_from_percentile(M, args.hot_pct, args.cold_pct, per_layer=True)
            else:
                hot_tau, cold_tau = thresholds_from_percentile(M, args.hot_pct, args.cold_pct, per_layer=False)
            hot_mask, cold_mask = apply_thresholds(M, hot_tau, cold_tau, args.per_layer_thresholds)

            # save masks
            sub = "mlp" if comp.startswith("mlp") else "mha"
            comp_dir = ensure_dir(os.path.join(hotcold_dir, sub))
            if sub == "mha":
                comp_dir = ensure_dir(os.path.join(comp_dir, comp.replace("attn_", "")))  # q_proj / ...
            mask_path = os.path.join(comp_dir, f"hotcold_delta_{trans_tag}.npz")
            np.savez_compressed(mask_path,
                                hot=hot_mask, cold=cold_mask, hot_tau=hot_tau, cold_tau=cold_tau,
                                component=comp, transition=trans_tag)
            # record in manifest
            key = "mlp_mask" if comp == "mlp_fc1" else f"{comp}_mask"
            manifest_entry["files"][key] = mask_path

            # counts
            L = M.shape[0]; R = M.shape[1]
            for lid in range(L):
                hc_rows.append([trans_tag, comp, lid, int(hot_mask[lid].sum()), int(cold_mask[lid].sum()), R])

            # overlap dynamics vs previous transition (same component)
            prev = prev_hot_masks.get(comp)
            if prev is not None and prev.shape == hot_mask.shape:
                for lid in range(L):
                    Aset = prev[lid].astype(bool)
                    Bset = hot_mask[lid].astype(bool)
                    inter = (Aset & Bset).sum()
                    union = (Aset | Bset).sum()
                    jacc = float(inter) / float(union) if union > 0 else 0.0
                    new_hot = int((~Aset & Bset).sum())
                    cooled  = int((Aset & ~Bset).sum())
                    overlap_rows.append([trans_tag, comp, lid, jacc, new_hot, cooled,
                                         int(Aset.sum()), int(Bset.sum())])
            prev_hot_masks[comp] = hot_mask.copy()

        manifest_rows.append(manifest_entry)

        # free memory
        del A, B
        torch.cuda.empty_cache()

    # write summaries
    with open(os.path.join(hotcold_dir, "hotcold_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(hc_rows)
    with open(os.path.join(hotcold_dir, "hotcold_overlap.csv"), "w", newline="") as f:
        csv.writer(f).writerows(overlap_rows)

    # write manifest
    write_manifest(os.path.join(args.outdir, args.manifest_name), manifest_rows)

    # optional packing into single files for fast plotting
    if args.pack_tensors:
        hc_root = os.path.join(args.outdir, "hotcold")
        if args.which in ("mlp","both"):
            pack_component_masks(hc_root, "mlp_fc1", os.path.join(hc_root, "packed_mlp_fc1.npz"))
        if args.which in ("mha","both"):
            for k in ["q_proj","k_proj","v_proj","out_proj"]:
                pack_component_masks(hc_root, f"attn_{k}", os.path.join(hc_root, f"packed_{k}.npz"))

    print(f"[✓] Done. Results under: {args.outdir}")

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--ckpt_pattern", default="checkpoint-epoch-*")
    ap.add_argument("--include_final", action="store_true")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--which", choices=["mlp","mha","both"], default="both")
    ap.add_argument("--norm", choices=["l1","l2"], default="l2")

    ap.add_argument("--hot_pct", type=float, default=95.0,
                    help="percentile threshold for 'hot' rows (per transition)")
    ap.add_argument("--cold_pct", type=float, default=50.0,
                    help="percentile threshold for 'cold' rows (per transition)")
    ap.add_argument("--per_layer_thresholds", action="store_true",
                    help="compute thresholds per layer (recommended)")

    ap.add_argument("--plot", action="store_true",
                    help="write log1p heatmaps for each delta matrix")

    # NEW: caching / packing
    ap.add_argument("--reuse_if_exists", action="store_true",
                    help="Skip recomputing a transition if delta/mask files already exist on disk")
    ap.add_argument("--pack_tensors", action="store_true",
                    help="Pack per-transition masks into consolidated NPZ per component for faster plotting")
    ap.add_argument("--manifest_name", default="manifest.json",
                    help="Name of the analysis manifest written under outdir")

    args = ap.parse_args()
    ensure_dir(args.outdir)
    analyze(args)

if __name__ == "__main__":
    main()