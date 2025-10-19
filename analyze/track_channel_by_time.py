#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track individual channel gradients over training steps with normalization.

This script:
  1) Loads gradients across multiple steps from grad_dump/index.csv
  2) Aggregates to channel-level gradient using sqrt(L2)/dim normalization
  3) Supports flexible row/column aggregation for MLP and MHA matrices
  4) Plots time series for each matrix type and combined views

Usage:
  python script.py  # Uses default config values
  python script.py --grad_base_dir /path/to/grad_dump --mlp_dim row --mha_dim col
  python script.py --help  # Show all options
"""

# ========================
# Default Config
# ========================
DEFAULT_GRAD_BASE_DIR   = "/pscratch/sd/l/lsx/yyt_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
DEFAULT_OUT_DIR         = "/pscratch/sd/l/lsx/yyt_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/gradient_timeseries_normalized"
DEFAULT_TOP_K           = 50
DEFAULT_BOTTOM_K        = 50

# Which steps to analyze
DEFAULT_STEPS = None  # None = auto-detect all available steps

# Aggregation: 'row' or 'col' for each matrix type
DEFAULT_MLP_UP_DIM      = "row"    # up_proj: row = channel (d_hidden)
DEFAULT_MLP_DOWN_DIM    = "row"    # down_proj: col = channel (d_hidden)
DEFAULT_MLP_GATE_DIM    = "row"    # gate_proj: row = channel (d_hidden)
DEFAULT_MHA_Q_DIM       = "row"    # q_proj: row = head output
DEFAULT_MHA_K_DIM       = "row"    # k_proj: row = head output
DEFAULT_MHA_V_DIM       = "row"    # v_proj: row = head output
DEFAULT_MHA_O_DIM       = "col"    # o_proj: col = head input

# Visualization options
DEFAULT_INCLUDE_BIAS     = True
DEFAULT_PLOT_LOG_SCALE   = True
DEFAULT_PLOT_ALL         = True
DEFAULT_PLOT_TOP_K       = True
DEFAULT_PLOT_BOTTOM_K    = True
DEFAULT_PLOT_COMBINED    = True   # Plot all matrices together
DEFAULT_PLOT_INDIVIDUAL  = True   # Plot each matrix separately
DEFAULT_SPLIT_BY_LAYER   = True   # If True, create separate plots for each layer

# ========================
# Script
# ========================
import os
import sys
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


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


def aggregate_gradient(G: torch.Tensor, dim: str, normalize: bool = True) -> np.ndarray:
    """
    Aggregate gradient tensor along specified dimension with normalization.
    
    Args:
        G: Gradient tensor [rows, cols]
        dim: 'row' or 'col' - which dimension to aggregate to
        normalize: If True, use sqrt(L2) / sqrt(dim) normalization
    
    Returns:
        Array of per-row or per-col gradient values
    """
    if dim == "row":
        # Aggregate columns, return one value per row
        g_squared = G.pow(2).sum(dim=1)  # [rows]
        if normalize:
            n_cols = G.shape[1]
            result = torch.sqrt(g_squared) / np.sqrt(n_cols)
        else:
            result = torch.sqrt(g_squared)
    elif dim == "col":
        # Aggregate rows, return one value per column
        g_squared = G.pow(2).sum(dim=0)  # [cols]
        if normalize:
            n_rows = G.shape[0]
            result = torch.sqrt(g_squared) / np.sqrt(n_rows)
        else:
            result = torch.sqrt(g_squared)
    else:
        raise ValueError(f"dim must be 'row' or 'col', got {dim}")
    
    return result.cpu().numpy().astype(np.float64)


def load_grad_channels_for_step(
    grad_base_dir: str, 
    step: int,
    mlp_up_dim: str = "row",
    mlp_down_dim: str = "col", 
    mlp_gate_dim: str = "row",
    mha_q_dim: str = "row",
    mha_k_dim: str = "row",
    mha_v_dim: str = "row",
    mha_o_dim: str = "col",
    include_bias: bool = True,
    split_by_layer: bool = True
) -> Dict[str, Tuple[np.ndarray, List[Tuple[int, str]]]]:
    """
    Load gradient data for a single step and aggregate by matrix type.
    
    Args:
        split_by_layer: If True, create separate entries for each layer (e.g., mlp_up_layer0, mlp_up_layer1)
                       If False, concatenate all layers together (e.g., mlp_up)
    
    Returns:
        Dictionary mapping matrix name to (gradient_array, channel_ids)
    """
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    
    df = pd.read_csv(index_csv)
    rows = df[df["global_step"] == step]
    
    if rows.empty:
        warnings.warn(f"No entries for global_step={step} in index.csv")
        return {}
    
    # Store per-layer, per-matrix gradients
    matrix_data = defaultdict(lambda: defaultdict(list))  # [matrix_name][layer_id] -> list of arrays
    
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
            if include_bias and G.ndim == 1 and param.endswith(".bias"):
                pass  # Could handle bias separately if needed
            else:
                continue
        
        # Process MLP matrices
        if sub == "mlp":
            if param == "up_proj.weight":
                e = aggregate_gradient(G, mlp_up_dim)
                matrix_data["mlp_up"][layer_id].append(e)
            elif param == "down_proj.weight":
                e = aggregate_gradient(G, mlp_down_dim)
                matrix_data["mlp_down"][layer_id].append(e)
            elif param == "gate_proj.weight":
                e = aggregate_gradient(G, mlp_gate_dim)
                matrix_data["mlp_gate"][layer_id].append(e)
        
        # Process MHA matrices
        elif sub == "self_attn":
            if param == "q_proj.weight":
                e = aggregate_gradient(G, mha_q_dim)
                matrix_data["mha_q"][layer_id].append(e)
            elif param == "k_proj.weight":
                e = aggregate_gradient(G, mha_k_dim)
                matrix_data["mha_k"][layer_id].append(e)
            elif param == "v_proj.weight":
                e = aggregate_gradient(G, mha_v_dim)
                matrix_data["mha_v"][layer_id].append(e)
            elif param == "o_proj.weight":
                e = aggregate_gradient(G, mha_o_dim)
                matrix_data["mha_o"][layer_id].append(e)
    
    # Consolidate into arrays with channel IDs
    results = {}
    
    if split_by_layer:
        # Create separate entry for each layer
        for matrix_type, layer_dict in matrix_data.items():
            for layer_id in sorted(layer_dict.keys()):
                arrays = layer_dict[layer_id]
                if arrays:
                    # Sum if multiple arrays per layer (shouldn't happen, but be safe)
                    combined = sum(arrays) if len(arrays) > 1 else arrays[0]
                    
                    # Create channel IDs
                    channel_ids = []
                    for ch_idx in range(combined.size):
                        channel_ids.append((layer_id, f"{matrix_type}_layer{layer_id}_ch{ch_idx}"))
                    
                    matrix_name = f"{matrix_type}_layer{layer_id:02d}"
                    results[matrix_name] = (combined, channel_ids)
                    print(f"  {matrix_name}: {combined.size} channels")
    else:
        # Concatenate all layers together (original behavior)
        for matrix_type, layer_dict in matrix_data.items():
            all_arrays = []
            channel_ids = []
            
            for layer_id in sorted(layer_dict.keys()):
                arrays = layer_dict[layer_id]
                if arrays:
                    # Sum if multiple arrays per layer (shouldn't happen, but be safe)
                    combined = sum(arrays) if len(arrays) > 1 else arrays[0]
                    all_arrays.append(combined)
                    
                    # Create channel IDs
                    for ch_idx in range(combined.size):
                        channel_ids.append((layer_id, f"{matrix_type}_{ch_idx}"))
            
            if all_arrays:
                gradient_array = np.concatenate(all_arrays, axis=0)
                results[matrix_type] = (gradient_array, channel_ids)
                print(f"  {matrix_type}: {len(layer_dict)} layers, {gradient_array.size} channels")
    
    return results


def load_all_steps_gradients(
    grad_base_dir: str, 
    steps: List[int],
    mlp_up_dim: str = "row",
    mlp_down_dim: str = "col",
    mlp_gate_dim: str = "row",
    mha_q_dim: str = "row",
    mha_k_dim: str = "row",
    mha_v_dim: str = "row",
    mha_o_dim: str = "col",
    include_bias: bool = True,
    split_by_layer: bool = True
) -> Dict[str, Tuple[List[int], np.ndarray]]:
    """
    Load gradient data for multiple steps, organized by matrix type.
    
    Args:
        split_by_layer: If True, create separate entries for each layer
    
    Returns:
        Dictionary mapping matrix name to (valid_steps, gradient_matrix)
        where gradient_matrix has shape [num_steps, num_channels]
    """
    all_matrix_data = defaultdict(list)  # [matrix_name] -> list of (step, gradient_array, channel_ids)
    
    print(f"Loading gradients for {len(steps)} steps...")
    for step in steps:
        try:
            print(f"\nStep {step}:")
            step_results = load_grad_channels_for_step(
                grad_base_dir, step, mlp_up_dim, mlp_down_dim, mlp_gate_dim,
                mha_q_dim, mha_k_dim, mha_v_dim, mha_o_dim, include_bias, split_by_layer
            )
            
            for matrix_name, (grad_array, channel_ids) in step_results.items():
                all_matrix_data[matrix_name].append((step, grad_array, channel_ids))
        
        except Exception as e:
            warnings.warn(f"Failed to load step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    # Consolidate into matrices
    results = {}
    for matrix_name, step_data in all_matrix_data.items():
        if not step_data:
            continue
        
        # Extract steps and validate consistency
        valid_steps = [s for s, _, _ in step_data]
        channel_id_reference = step_data[0][2]
        gradient_arrays = []
        
        for step, grad_array, channel_ids in step_data:
            # Validate channel consistency
            if len(channel_ids) != len(channel_id_reference):
                warnings.warn(f"{matrix_name}: Step {step} has {len(channel_ids)} channels, "
                            f"expected {len(channel_id_reference)}. Skipping.")
                continue
            
            gradient_arrays.append(grad_array)
        
        if gradient_arrays:
            gradient_matrix = np.stack(gradient_arrays, axis=0)
            results[matrix_name] = (valid_steps, gradient_matrix)
            print(f"\n✓ {matrix_name}: {len(valid_steps)} steps × {gradient_matrix.shape[1]} channels")
    
    return results


def detect_available_steps(grad_base_dir: str) -> List[int]:
    """Auto-detect available steps from index.csv."""
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    
    df = pd.read_csv(index_csv)
    steps = sorted(df["global_step"].unique())
    print(f"Detected {len(steps)} available steps: {steps[:10]}{'...' if len(steps) > 10 else ''}")
    return steps


def plot_matrix_timeseries(
    matrix_name: str,
    steps: List[int],
    gradient_matrix: np.ndarray,
    out_dir: str,
    top_k: int = 50,
    bottom_k: int = 50,
    log_scale: bool = True,
    plot_all: bool = True,
    plot_top: bool = True,
    plot_bottom: bool = True
):
    """Plot time series for a single matrix type."""
    num_steps, num_channels = gradient_matrix.shape
    max_grads = gradient_matrix.max(axis=0)
    top_indices = np.argsort(max_grads)[::-1][:top_k]
    bottom_indices = np.argsort(max_grads)[:bottom_k]
    
    # Create matrix-specific subdirectory
    matrix_dir = os.path.join(out_dir, matrix_name)
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Top-k plot
    if plot_top and top_k > 0:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        for rank, idx in enumerate(top_indices):
            alpha = 1.0 if rank < 10 else 0.6 if rank < 30 else 0.3
            linewidth = 2.0 if rank < 10 else 1.2 if rank < 30 else 0.8
            label = f"Ch {idx}" if rank < 10 else None
            ax.plot(steps, gradient_matrix[:, idx], alpha=alpha, linewidth=linewidth, label=label)
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient (√L²/√dim)", fontsize=12)
        ax.set_title(f"Top {top_k} Channels: {matrix_name}", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if top_k <= 10:
            ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        out_path = os.path.join(matrix_dir, f"{matrix_name}_top{top_k}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {out_path}")
    
    # Bottom-k plot
    if plot_bottom and bottom_k > 0:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        for rank, idx in enumerate(bottom_indices):
            alpha = 1.0 if rank < 10 else 0.6 if rank < 30 else 0.3
            linewidth = 2.0 if rank < 10 else 1.2 if rank < 30 else 0.8
            label = f"Ch {idx}" if rank < 10 else None
            ax.plot(steps, gradient_matrix[:, idx], alpha=alpha, linewidth=linewidth, label=label)
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient (√L²/√dim)", fontsize=12)
        ax.set_title(f"Bottom {bottom_k} Channels: {matrix_name}", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if bottom_k <= 10:
            ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        out_path = os.path.join(matrix_dir, f"{matrix_name}_bottom{bottom_k}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {out_path}")
    
    # All channels plot
    if plot_all:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        alpha_val = 0.02 if num_channels > 1000 else 0.1 if num_channels > 100 else 0.3
        
        for idx in range(num_channels):
            ax.plot(steps, gradient_matrix[:, idx], alpha=alpha_val, linewidth=0.5, color='blue')
        
        # Overlay top and bottom
        for idx in top_indices[:5]:
            ax.plot(steps, gradient_matrix[:, idx], alpha=0.8, linewidth=2, color='red')
        for idx in bottom_indices[:5]:
            ax.plot(steps, gradient_matrix[:, idx], alpha=0.8, linewidth=2, color='green')
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient (√L²/√dim)", fontsize=12)
        ax.set_title(f"All {num_channels} Channels: {matrix_name}", fontsize=14)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(matrix_dir, f"{matrix_name}_all.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {out_path}")
    
    # Statistics plot
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
    ax.set_ylabel("Gradient (√L²/√dim)", fontsize=12)
    ax.set_title(f"Statistics: {matrix_name}", fontsize=14)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    plt.tight_layout()
    out_path = os.path.join(matrix_dir, f"{matrix_name}_stats.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")
    
    # Save CSV data for this matrix
    csv_data = {
        "step": steps,
        "mean": mean_grad,
        "median": median_grad,
        "p90": p90_grad,
        "p99": p99_grad,
        "max": max_grad
    }
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(matrix_dir, f"{matrix_name}_stats.csv")
    df.to_csv(csv_path, index=False)
    
    # Save top-k channel trajectories
    if plot_top:
        top_channel_data = {"step": steps}
        for rank, idx in enumerate(top_indices, 1):
            top_channel_data[f"channel_{idx}"] = gradient_matrix[:, idx]
        df = pd.DataFrame(top_channel_data)
        csv_path = os.path.join(matrix_dir, f"{matrix_name}_top{top_k}_channels.csv")
        df.to_csv(csv_path, index=False)
    
    # Save bottom-k channel trajectories
    if plot_bottom:
        bottom_channel_data = {"step": steps}
        for rank, idx in enumerate(bottom_indices, 1):
            bottom_channel_data[f"channel_{idx}"] = gradient_matrix[:, idx]
        df = pd.DataFrame(bottom_channel_data)
        csv_path = os.path.join(matrix_dir, f"{matrix_name}_bottom{bottom_k}_channels.csv")
        df.to_csv(csv_path, index=False)


def plot_combined_timeseries(
    all_matrices: Dict[str, Tuple[List[int], np.ndarray]],
    out_dir: str,
    log_scale: bool = True
):
    """Plot combined view of all matrix types."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    
    colors = {
        'mlp_up': '#e41a1c',
        'mlp_down': '#377eb8',
        'mlp_gate': '#4daf4a',
        'mha_q': '#984ea3',
        'mha_k': '#ff7f00',
        'mha_v': '#ffff33',
        'mha_o': '#a65628'
    }
    
    for matrix_name, (steps, gradient_matrix) in all_matrices.items():
        mean_grad = gradient_matrix.mean(axis=1)
        max_grad = gradient_matrix.max(axis=1)
        
        color = colors.get(matrix_name, 'black')
        ax.plot(steps, mean_grad, label=f"{matrix_name} (mean)", 
                linewidth=2, color=color, alpha=0.7)
        ax.plot(steps, max_grad, label=f"{matrix_name} (max)", 
                linewidth=1.5, linestyle='--', color=color, alpha=0.5)
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Gradient (√L²/√dim)", fontsize=12)
    ax.set_title("Combined View: All Matrix Types", fontsize=14)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "combined_all_matrices.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Combined view -> {out_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze normalized channel gradient changes over training steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--grad_base_dir", type=str, default=DEFAULT_GRAD_BASE_DIR,
                       help="Base directory containing grad_dump/index.csv")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                       help="Output directory for plots and CSV files")
    
    # Aggregation dimensions
    parser.add_argument("--mlp_up_dim", type=str, default=DEFAULT_MLP_UP_DIM, choices=["row", "col"],
                       help="Dimension to aggregate up_proj (row=channel)")
    parser.add_argument("--mlp_down_dim", type=str, default=DEFAULT_MLP_DOWN_DIM, choices=["row", "col"],
                       help="Dimension to aggregate down_proj (col=channel)")
    parser.add_argument("--mlp_gate_dim", type=str, default=DEFAULT_MLP_GATE_DIM, choices=["row", "col"],
                       help="Dimension to aggregate gate_proj (row=channel)")
    parser.add_argument("--mha_q_dim", type=str, default=DEFAULT_MHA_Q_DIM, choices=["row", "col"],
                       help="Dimension to aggregate q_proj")
    parser.add_argument("--mha_k_dim", type=str, default=DEFAULT_MHA_K_DIM, choices=["row", "col"],
                       help="Dimension to aggregate k_proj")
    parser.add_argument("--mha_v_dim", type=str, default=DEFAULT_MHA_V_DIM, choices=["row", "col"],
                       help="Dimension to aggregate v_proj")
    parser.add_argument("--mha_o_dim", type=str, default=DEFAULT_MHA_O_DIM, choices=["row", "col"],
                       help="Dimension to aggregate o_proj")
    
    # Channel selection
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                       help="Number of top channels to plot")
    parser.add_argument("--bottom_k", type=int, default=DEFAULT_BOTTOM_K,
                       help="Number of bottom channels to plot")
    
    # Steps
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                       help="Specific steps to analyze")
    parser.add_argument("--step_range", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                       help="Generate step range")
    
    # Plot options
    parser.add_argument("--plot_all", action="store_true", default=DEFAULT_PLOT_ALL)
    parser.add_argument("--no_plot_all", dest="plot_all", action="store_false")
    parser.add_argument("--plot_top", action="store_true", default=DEFAULT_PLOT_TOP_K)
    parser.add_argument("--no_plot_top", dest="plot_top", action="store_false")
    parser.add_argument("--plot_bottom", action="store_true", default=DEFAULT_PLOT_BOTTOM_K)
    parser.add_argument("--no_plot_bottom", dest="plot_bottom", action="store_false")
    parser.add_argument("--plot_combined", action="store_true", default=DEFAULT_PLOT_COMBINED)
    parser.add_argument("--no_plot_combined", dest="plot_combined", action="store_false")
    parser.add_argument("--plot_individual", action="store_true", default=DEFAULT_PLOT_INDIVIDUAL)
    parser.add_argument("--no_plot_individual", dest="plot_individual", action="store_false")
    parser.add_argument("--log_scale", action="store_true", default=DEFAULT_PLOT_LOG_SCALE)
    parser.add_argument("--no_log_scale", dest="log_scale", action="store_false")
    parser.add_argument("--include_bias", action="store_true", default=DEFAULT_INCLUDE_BIAS)
    parser.add_argument("--no_include_bias", dest="include_bias", action="store_false")
    
    # Layer splitting
    parser.add_argument("--split_by_layer", action="store_true", default=DEFAULT_SPLIT_BY_LAYER,
                       help="Create separate plots for each layer")
    parser.add_argument("--no_split_by_layer", dest="split_by_layer", action="store_false",
                       help="Combine all layers together")
    
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print("Normalized Channel Gradient Time Series Analysis")
    print("=" * 70)
    print(f"Gradient directory: {args.grad_base_dir}")
    print(f"Output directory:   {args.out_dir}")
    print(f"\nAggregation dimensions:")
    print(f"  MLP up_proj:   {args.mlp_up_dim}")
    print(f"  MLP down_proj: {args.mlp_down_dim}")
    print(f"  MLP gate_proj: {args.mlp_gate_dim}")
    print(f"  MHA q_proj:    {args.mha_q_dim}")
    print(f"  MHA k_proj:    {args.mha_k_dim}")
    print(f"  MHA v_proj:    {args.mha_v_dim}")
    print(f"  MHA o_proj:    {args.mha_o_dim}")
    print(f"\nSplit by layer:     {args.split_by_layer}")
    print("=" * 70)
    
    # Determine steps
    if args.step_range:
        steps = list(range(args.step_range[0], args.step_range[1], args.step_range[2]))
    elif args.steps:
        steps = args.steps
    else:
        steps = detect_available_steps(args.grad_base_dir)
    
    if not steps:
        raise ValueError("No steps to analyze")
    
    # Load gradient data
    all_matrices = load_all_steps_gradients(
        args.grad_base_dir, steps,
        args.mlp_up_dim, args.mlp_down_dim, args.mlp_gate_dim,
        args.mha_q_dim, args.mha_k_dim, args.mha_v_dim, args.mha_o_dim,
        args.include_bias, args.split_by_layer
    )
    
    if not all_matrices:
        raise ValueError("No gradient data loaded")
    
    # Plot individual matrices
    if args.plot_individual:
        print("\n" + "=" * 70)
        print("Plotting individual matrices...")
        print("=" * 70)
        for matrix_name, (valid_steps, gradient_matrix) in all_matrices.items():
            print(f"\nProcessing {matrix_name}...")
            plot_matrix_timeseries(
                matrix_name, valid_steps, gradient_matrix, args.out_dir,
                args.top_k, args.bottom_k, args.log_scale,
                args.plot_all, args.plot_top, args.plot_bottom
            )
    
    # Plot combined view
    if args.plot_combined:
        print("\n" + "=" * 70)
        print("Plotting combined view...")
        print("=" * 70)
        plot_combined_timeseries(all_matrices, args.out_dir, args.log_scale)
    
    print(f"\n✓ Analysis complete! Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()