#!/usr/bin/env python3
"""
plot_grad_and_weight_deltas.py

Usage:
  python plot_grad_and_weight_deltas.py \
      --run_root /pscratch/sd/l/lsx/yyt_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca \
      --param_regex 'self_attn\\.k_proj\\.weight' \
      --layer 0 \
      --with_scatter

Notes:
- Expects:
  <run_root>/grad_dump/index.csv
  <run_root>/weight_dump/stepXXXXXX_pre/, stepXXXXXX_post/
- Aggregation:
  * Grad per step: sum over matched rows of sqrt(sum_g2)  (equivalently sqrt of sum_g2 if a single tensor).
  * Weight delta per step: L2 norm of (post - pre) for matched tensors, summed over matches.
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

# ---------- I/O helpers for checkpoints ----------

def _is_ckpt_dir(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    files = os.listdir(p)
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return True
    if any(fn.endswith(".safetensors") for fn in files) or any(fn.endswith(".bin") for fn in files):
        return True
    return False

def _load_safetensors_dir(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    state = {}
    idx_json = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
        state.update(load_file(os.path.join(ckpt_dir, "model.safetensors")))
        return state
    if os.path.exists(idx_json):
        with open(idx_json, "r") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        for shard in shards:
            state.update(load_file(os.path.join(ckpt_dir, shard)))
        return state
    # Fallback: load any *.safetensors present
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(".safetensors"):
            state.update(load_file(os.path.join(ckpt_dir, fn)))
    if not state:
        raise FileNotFoundError(f"No safetensors found in {ckpt_dir}")
    return state

def _load_bin_dir(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    single = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(single):
        return torch.load(single, map_location="cpu")
    state = {}
    shard_re = re.compile(r"^pytorch_model-\d{5}-of-\d{5}\.bin$")
    shards = [fn for fn in os.listdir(ckpt_dir) if shard_re.match(fn)]
    for fn in sorted(shards):
        part = torch.load(os.path.join(ckpt_dir, fn), map_location="cpu")
        state.update(part)
    if not state:
        raise FileNotFoundError(f"No bin shards found in {ckpt_dir}")
    return state

def load_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    files = os.listdir(ckpt_dir)
    if "model.safetensors" in files or any(fn.endswith(".safetensors") for fn in files):
        return _load_safetensors_dir(ckpt_dir)
    if "pytorch_model.bin" in files or any(fn.endswith(".bin") for fn in files):
        return _load_bin_dir(ckpt_dir)
    raise FileNotFoundError(f"No recognizable checkpoint files in {ckpt_dir}")

# ---------- Readers for grad index.csv and weight pairs ----------

def read_grad_index(grad_dump_dir: str) -> List[Dict[str, str]]:
    index_csv = os.path.join(grad_dump_dir, "index.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Missing {index_csv}")
    rows = []
    with open(index_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def find_available_steps(weight_dump_dir: str) -> List[int]:
    steps = []
    for name in os.listdir(weight_dump_dir):
        m = re.match(r"^step(\d{6})_pre$", name)
        if m:
            s = int(m.group(1))
            pre = os.path.join(weight_dump_dir, f"step{m.group(1)}_pre")
            post = os.path.join(weight_dump_dir, f"step{m.group(1)}_post")
            if _is_ckpt_dir(pre) and _is_ckpt_dir(post):
                steps.append(s)
    return sorted(steps)

# ---------- Aggregation ----------

def aggregate_grad_metric(
    grad_rows: List[Dict[str, str]],
    param_regex: re.Pattern,
    layer_filter: Optional[int],
) -> Dict[int, float]:
    """
    Returns: map step -> grad magnitude. We use sqrt(sum_g2) per row, sum across matched rows.
    """
    step_to_val = {}
    for row in grad_rows:
        try:
            step = int(row["global_step"])
        except Exception:
            continue
        lid = int(row["layer"]) if row.get("layer", "").isdigit() else None
        pname = row.get("param", "")
        submod = row.get("submodule", "")
        full_name = f"{submod}.{pname}" if submod else pname
        if layer_filter is not None and lid is not None and lid != layer_filter:
            continue
        if not param_regex.search(full_name):
            continue
        # sum_g2 was written as float string; take sqrt to get L2
        try:
            sum_g2 = float(row["sum_g2"])
        except Exception:
            continue
        l2 = np.sqrt(max(sum_g2, 0.0))
        step_to_val[step] = step_to_val.get(step, 0.0) + float(l2)
    return step_to_val

def aggregate_weight_delta_l2_for_step(
    weight_dump_dir: str,
    step: int,
    param_regex: re.Pattern,
    layer_filter: Optional[int],
) -> float:
    """
    Loads pre and post, computes sum of L2 norms for matched tensors (after optional layer filter).
    Layer filter is implemented by matching the 'model.layers.<i>.' prefix in the state_dict key.
    """
    tag = f"step{step:06d}"
    pre_dir = os.path.join(weight_dump_dir, f"{tag}_pre")
    post_dir = os.path.join(weight_dump_dir, f"{tag}_post")
    sd_pre = load_state_dict(pre_dir)
    sd_post = load_state_dict(post_dir)

    # Optional layer prefix builder
    layer_pref = None
    if layer_filter is not None:
        # Handle both zero-padded and non-padded forms commonly seen
        layer_pref_opts = [
            f"model.layers.{layer_filter}.",
            f"model.layers.{layer_filter:02d}.",
        ]
    else:
        layer_pref_opts = None

    total = 0.0
    for k, a in sd_pre.items():
        if k not in sd_post:
            continue
        if not isinstance(a, torch.Tensor) or not isinstance(sd_post[k], torch.Tensor):
            continue
        # Layer filter
        if layer_pref_opts is not None and not any(k.startswith(p) for p in layer_pref_opts):
            continue
        # Param name filter: apply regex to tail (strip prefix up to 'model.layers.<i>.')
        tail = k
        if layer_pref_opts is not None:
            for p in layer_pref_opts:
                if k.startswith(p):
                    tail = k[len(p):]
                    break
        if not param_regex.search(tail):
            continue
        b = sd_post[k]
        if a.shape != b.shape:
            continue
        diff = (a.detach().cpu().to(torch.float32) - b.detach().cpu().to(torch.float32))
        l2 = torch.linalg.vector_norm(diff).item()
        total += float(l2)
    return total

# ---------- Plotting ----------

def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def plot_series(steps: List[int], values: List[float], title: str, out_path: str, ylabel: str):
    plt.figure()
    plt.plot(steps, values, marker="o")
    plt.title(title)
    plt.xlabel("global_step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_scatter(xsteps: List[int], xvals: List[float], yvals: List[float], title: str, out_path: str):
    plt.figure()
    plt.scatter(xvals, yvals)
    plt.title(title)
    plt.xlabel("Grad magnitude (sum of L2)")
    plt.ylabel("Weight delta L2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Plot gradient magnitudes and corresponding weight deltas over steps.")
    ap.add_argument("--run_root", required=True, help="Run root (contains grad_dump/ and weight_dump/)")
    ap.add_argument("--param_regex", required=True, help="Regex for param tail (e.g., 'self_attn\\.k_proj\\.weight' or 'mlp\\.(up_proj|down_proj)\\.weight')")
    ap.add_argument("--layer", type=int, default=None, help="If set, restrict to this layer id (e.g., 0)")
    ap.add_argument("--with_scatter", action="store_true", help="Also produce a grad-vs-delta scatter")
    args = ap.parse_args()

    grad_dump_dir = os.path.join(args.run_root, "grad_dump")
    weight_dump_dir = os.path.join(args.run_root, "weight_dump")
    out_dir = os.path.join(args.run_root, "analysis_plots")
    ensure_out_dir(out_dir)

    param_re = re.compile(args.param_regex)

    # 1) Aggregate gradient magnitudes by step
    grad_rows = read_grad_index(grad_dump_dir)
    step_to_grad = aggregate_grad_metric(grad_rows, param_re, args.layer)

    # 2) Find steps that have both pre & post weights
    steps = find_available_steps(weight_dump_dir)

    # 3) For each such step, compute weight delta L2 for matched params
    step_to_wdelta = {}
    for s in steps:
        try:
            val = aggregate_weight_delta_l2_for_step(weight_dump_dir, s, param_re, args.layer)
            step_to_wdelta[s] = val
        except Exception as e:
            # Skip steps where load failed
            continue

    # 4) Align steps present in both series
    common_steps = sorted(set(step_to_grad.keys()) & set(step_to_wdelta.keys()))
    if not common_steps:
        print("No overlapping steps between gradient logs and weight checkpoints for the given filter.")
        return

    grad_vals = [step_to_grad[s] for s in common_steps]
    wdel_vals = [step_to_wdelta[s] for s in common_steps]

    # 5) Plot
    tag = args.param_regex.replace("\\", "")
    if args.layer is not None:
        tag = f"L{args.layer}_{tag}"

    plot_series(
        common_steps, grad_vals,
        title=f"Gradient magnitude over steps ({tag})",
        out_path=os.path.join(out_dir, f"grad_over_time_{tag}.png"),
        ylabel="sum of L2(grad) across matched tensors",
    )

    plot_series(
        common_steps, wdel_vals,
        title=f"Weight delta L2 over steps (post - pre) ({tag})",
        out_path=os.path.join(out_dir, f"wdelta_over_time_{tag}.png"),
        ylabel="sum of L2(post - pre) across matched tensors",
    )

    if args.with_scatter:
        plot_scatter(
            common_steps, grad_vals, wdel_vals,
            title=f"Grad vs Weight Delta (per step) ({tag})",
            out_path=os.path.join(out_dir, f"grad_vs_wdelta_{tag}.png"),
        )

    print("Wrote:")
    print(" ", os.path.join(out_dir, f"grad_over_time_{tag}.png"))
    print(" ", os.path.join(out_dir, f"wdelta_over_time_{tag}.png"))
    if args.with_scatter:
        print(" ", os.path.join(out_dir, f"grad_vs_wdelta_{tag}.png"))

if __name__ == "__main__":
    main()