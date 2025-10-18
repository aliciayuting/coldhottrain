#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track individual channel gradients over training steps.

This script:
  1) Loads gradients across multiple steps from grad_dump/index.csv
  2) Aggregates to channel-level gradient energy for each step
  3) Plots time series: x=step, y=gradient, one line per channel
  4) Optionally highlights top-k most active channels
"""

# ========================
# Config (edit these)
# ========================
GRAD_BASE_DIR   = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/gradient_timeseries"

# Which steps to analyze (set to None to auto-detect all available steps)
STEPS = None  # e.g., [100, 200, 300, 400, 500] or None for all
# STEPS = list(range(100, 1001, 100))  # Example: 100, 200, ..., 1000

# Visualization options
TOP_K_CHANNELS = 50        # Plot top-K most active channels (by max gradient)
BOTTOM_K_CHANNELS = 50     # Plot bottom-K least active channels (by max gradient)
INCLUDE_BIAS = True        # Match your callback settings
PLOT_LOG_SCALE = True      # Use log scale for y-axis (gradients can span many orders)
PLOT_ALL_CHANNELS = True   # If True, plot all channels (can be messy)
PLOT_TOP_K = True          # Plot top-K channels
PLOT_BOTTOM_K = True       # Plot bottom-K channels

# Channel type to analyze
CHANNEL_TYPE = "incoming"  # "incoming", "outgoing", or "both"

# ========================
# Script
# ========================
import os
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs(OUT_DIR, exist_ok=True)


def _load_any_tensor(path: str) -> torch.Tensor:
    """Load tensor from .pt, .pth, .bin, .npy, or .npz file."""
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
                            t = v
                            break
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


def load_grad_channels_for_step(grad_base_dir: str, step: int, 
                                 channel_type: str = "incoming") -> np.ndarray:
    """
    Load gradient data for a single step and aggregate to channel-level.
    
    Args:
        grad_base_dir: Base directory containing index.csv
        step: Training step number
        channel_type: "incoming" (up_proj columns), "outgoing" (down_proj rows), or "both"
    
    Returns:
        Array of per-channel gradient energies
    """
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    
    df = pd.read_csv(index_csv)
    rows = df[df["global_step"] == step]
    
    if rows.empty:
        warnings.warn(f"No entries for global_step={step} in index.csv")
        return np.array([])
    
    # Store per-layer channel energies
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
            warnings.warn(f"[grad] missing {f}")
            continue
        
        G = _load_any_tensor(f)
        
        if G.ndim != 2:
            if INCLUDE_BIAS and G.ndim == 1 and param.endswith(".bias"):
                pass
            else:
                continue
        
        if sub == "mlp":
            if param == "up_proj.weight":
                # Per-channel incoming: sum over output dim (columns)
                e = (G.to(torch.float32).pow(2).sum(dim=0)).cpu().numpy()
                _ensure_mlp(layer_id, "incoming", e.size)
                mlp_energy_per_layer[layer_id]["incoming"] += e
                
            elif param == "down_proj.weight":
                # Per-channel outgoing: sum over input dim (rows)
                e = (G.to(torch.float32).pow(2).sum(dim=1)).cpu().numpy()
                _ensure_mlp(layer_id, "outgoing", e.size)
                mlp_energy_per_layer[layer_id]["outgoing"] += e
    
    # Combine channels based on channel_type
    mlp_all = []
    for lid, d in mlp_energy_per_layer.items():
        inc = d.get("incoming", None)
        out = d.get("outgoing", None)
        
        if channel_type == "incoming" and inc is not None:
            mlp_all.append(inc)
        elif channel_type == "outgoing" and out is not None:
            mlp_all.append(out)
        elif channel_type == "both":
            if inc is not None and out is not None:
                mlp_all.append(inc + out)
            elif inc is not None:
                mlp_all.append(inc)
            elif out is not None:
                mlp_all.append(out)
    
    return np.concatenate(mlp_all, axis=0) if mlp_all else np.array([])


def load_all_steps_gradients(grad_base_dir: str, steps: List[int],
                             channel_type: str = "incoming") -> Tuple[List[int], np.ndarray]:
    """
    Load gradient data for multiple steps.
    
    Returns:
        steps: List of available steps
        gradient_matrix: Shape [num_steps, num_channels]
    """
    gradient_data = []
    valid_steps = []
    
    print(f"Loading gradients for {len(steps)} steps...")
    for step in steps:
        try:
            grad = load_grad_channels_for_step(grad_base_dir, step, channel_type)
            if grad.size > 0:
                gradient_data.append(grad)
                valid_steps.append(step)
                print(f"  Step {step}: {grad.size} channels, total L2²={grad.sum():.3e}")
            else:
                warnings.warn(f"No gradient data for step {step}")
        except Exception as e:
            warnings.warn(f"Failed to load step {step}: {e}")
    
    if not gradient_data:
        raise ValueError("No gradient data loaded for any step")
    
    # Check that all steps have same number of channels
    channel_counts = [g.size for g in gradient_data]
    if len(set(channel_counts)) > 1:
        warnings.warn(f"Inconsistent channel counts across steps: {set(channel_counts)}")
        # Pad or truncate to minimum size
        min_size = min(channel_counts)
        gradient_data = [g[:min_size] for g in gradient_data]
        print(f"Truncated to {min_size} channels for consistency")
    
    gradient_matrix = np.stack(gradient_data, axis=0)  # [num_steps, num_channels]
    return valid_steps, gradient_matrix


def detect_available_steps(grad_base_dir: str) -> List[int]:
    """Auto-detect available steps from index.csv."""
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    
    df = pd.read_csv(index_csv)
    steps = sorted(df["global_step"].unique())
    print(f"Detected {len(steps)} available steps: {steps[:10]}{'...' if len(steps) > 10 else ''}")
    return steps


def plot_gradient_timeseries(steps: List[int], gradient_matrix: np.ndarray,
                             top_k: int = 50, bottom_k: int = 50, 
                             log_scale: bool = True,
                             plot_all: bool = True, 
                             plot_top: bool = True,
                             plot_bottom: bool = True,
                             channel_type: str = "incoming"):
    """
    Plot gradient time series for channels.
    
    Args:
        steps: List of step numbers
        gradient_matrix: Shape [num_steps, num_channels]
        top_k: Number of top channels to highlight
        bottom_k: Number of bottom channels to highlight
        log_scale: Use log scale for y-axis
        plot_all: If True, plot all channels
        plot_top: If True, plot top-k channels
        plot_bottom: If True, plot bottom-k channels
        channel_type: Type of channels being plotted
    """
    num_steps, num_channels = gradient_matrix.shape
    
    # Find top-k and bottom-k channels by maximum gradient across all steps
    max_grads = gradient_matrix.max(axis=0)
    top_indices = np.argsort(max_grads)[::-1][:top_k]
    bottom_indices = np.argsort(max_grads)[:bottom_k]
    
    print(f"\nTop {top_k} channels by max gradient:")
    for i, idx in enumerate(top_indices[:10], 1):
        print(f"  {i}. Channel {idx}: max={max_grads[idx]:.3e}")
    
    print(f"\nBottom {bottom_k} channels by max gradient:")
    for i, idx in enumerate(bottom_indices[:10], 1):
        print(f"  {i}. Channel {idx}: max={max_grads[idx]:.3e}")
    
    # Plot 1: Top-k channels with individual lines
    if plot_top:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        for rank, idx in enumerate(top_indices):
            alpha = 1.0 if rank < 10 else 0.6 if rank < 30 else 0.3
            linewidth = 2.0 if rank < 10 else 1.2 if rank < 30 else 0.8
            label = f"Ch {idx}" if rank < 10 else None
            ax.plot(steps, gradient_matrix[:, idx], alpha=alpha, linewidth=linewidth, label=label)
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient L2² Energy", fontsize=12)
        ax.set_title(f"Top {top_k} Channel Gradients Over Time ({channel_type})", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if top_k <= 10:
            ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"gradient_timeseries_top{top_k}_{channel_type}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Top-{top_k} gradient time series -> {out_path}")
    
    # Plot 2: Bottom-k channels with individual lines
    if plot_bottom:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        for rank, idx in enumerate(bottom_indices):
            alpha = 1.0 if rank < 10 else 0.6 if rank < 30 else 0.3
            linewidth = 2.0 if rank < 10 else 1.2 if rank < 30 else 0.8
            label = f"Ch {idx}" if rank < 10 else None
            ax.plot(steps, gradient_matrix[:, idx], alpha=alpha, linewidth=linewidth, label=label)
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient L2² Energy", fontsize=12)
        ax.set_title(f"Bottom {bottom_k} Channel Gradients Over Time ({channel_type})", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if bottom_k <= 10:
            ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"gradient_timeseries_bottom{bottom_k}_{channel_type}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Bottom-{bottom_k} gradient time series -> {out_path}")
    
    # Plot 3: All channels (if requested) - uses transparency to show density
    if plot_all:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        for idx in range(num_channels):
            alpha = 0.02 if num_channels > 1000 else 0.1 if num_channels > 100 else 0.3
            ax.plot(steps, gradient_matrix[:, idx], alpha=alpha, linewidth=0.5, color='blue')
        
        # Overlay top channels in red
        for idx in top_indices[:5]:
            ax.plot(steps, gradient_matrix[:, idx], alpha=0.8, linewidth=2, 
                   color='red', label=f"Top Ch {idx}" if idx == top_indices[0] else "")
        
        # Overlay bottom channels in green
        for idx in bottom_indices[:5]:
            ax.plot(steps, gradient_matrix[:, idx], alpha=0.8, linewidth=2, 
                   color='green', label=f"Bottom Ch {idx}" if idx == bottom_indices[0] else "")
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient L2² Energy", fontsize=12)
        ax.set_title(f"All {num_channels} Channel Gradients ({channel_type})", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if len(top_indices) > 0 or len(bottom_indices) > 0:
            ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"gradient_timeseries_all_{channel_type}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] All channel gradient time series -> {out_path}")
    
    # Plot 4: Statistics over time (mean, median, percentiles)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    mean_grad = gradient_matrix.mean(axis=1)
    median_grad = np.median(gradient_matrix, axis=1)
    p90_grad = np.percentile(gradient_matrix, 90, axis=1)
    p99_grad = np.percentile(gradient_matrix, 99, axis=1)
    max_grad = gradient_matrix.max(axis=1)
    
    ax.plot(steps, mean_grad, label="Mean", linewidth=2)
    ax.plot(steps, median_grad, label="Median", linewidth=2)
    ax.plot(steps, p90_grad, label="90th percentile", linewidth=2)
    ax.plot(steps, p99_grad, label="99th percentile", linewidth=2)
    ax.plot(steps, max_grad, label="Max", linewidth=2, linestyle='--')
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Gradient L2² Energy", fontsize=12)
    ax.set_title(f"Gradient Statistics Over Time ({channel_type})", fontsize=14)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"gradient_stats_{channel_type}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Gradient statistics -> {out_path}")
    
    # Save data to CSV for further analysis
    csv_data = {
        "step": steps,
        "mean": mean_grad,
        "median": median_grad,
        "p90": p90_grad,
        "p99": p99_grad,
        "max": max_grad
    }
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(OUT_DIR, f"gradient_stats_{channel_type}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DATA] Gradient statistics saved to {csv_path}")
    
    # Save top-k channel trajectories
    if plot_top:
        top_channel_data = {"step": steps}
        for rank, idx in enumerate(top_indices, 1):
            top_channel_data[f"channel_{idx}"] = gradient_matrix[:, idx]
        df = pd.DataFrame(top_channel_data)
        csv_path = os.path.join(OUT_DIR, f"top{top_k}_channels_{channel_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[DATA] Top-{top_k} channel trajectories saved to {csv_path}")
    
    # Save bottom-k channel trajectories
    if plot_bottom:
        bottom_channel_data = {"step": steps}
        for rank, idx in enumerate(bottom_indices, 1):
            bottom_channel_data[f"channel_{idx}"] = gradient_matrix[:, idx]
        df = pd.DataFrame(bottom_channel_data)
        csv_path = os.path.join(OUT_DIR, f"bottom{bottom_k}_channels_{channel_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[DATA] Bottom-{bottom_k} channel trajectories saved to {csv_path}")


def main():
    # Detect or use specified steps
    if STEPS is None:
        steps = detect_available_steps(GRAD_BASE_DIR)
    else:
        steps = STEPS
    
    if not steps:
        raise ValueError("No steps to analyze")
    
    # Load gradient data across all steps
    valid_steps, gradient_matrix = load_all_steps_gradients(
        GRAD_BASE_DIR, steps, CHANNEL_TYPE
    )
    
    print(f"\nLoaded gradient matrix: {gradient_matrix.shape} (steps × channels)")
    print(f"Steps: {valid_steps}")
    
    # Plot time series
    plot_gradient_timeseries(
        valid_steps,
        gradient_matrix,
        top_k=TOP_K_CHANNELS,
        bottom_k=BOTTOM_K_CHANNELS,
        log_scale=PLOT_LOG_SCALE,
        plot_all=PLOT_ALL_CHANNELS,
        plot_top=PLOT_TOP_K,
        plot_bottom=PLOT_BOTTOM_K,
        channel_type=CHANNEL_TYPE
    )
    
    print(f"\n Analysis complete! Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()