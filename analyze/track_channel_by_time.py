#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track individual channel gradients over training steps.

This script:
  1) Loads gradients across multiple steps from grad_dump/index.csv
  2) Aggregates to channel-level gradient energy for each step
  3) Plots time series: x=step, y=gradient, one line per channel
  4) Optionally highlights top-k most active channels

Usage:
  python script.py  # Uses default config values below
  python script.py --grad_base_dir /path/to/grad_dump --top_k 100
  python script.py --help  # Show all options
"""

# ========================
# Default Config (can be overridden by command-line args)
# ========================
DEFAULT_GRAD_BASE_DIR   = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
DEFAULT_OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/gradient_timeseries"
DEFAULT_TOP_K           = 50
DEFAULT_BOTTOM_K        = 50
DEFAULT_CHANNEL_TYPE    = "incoming"  # "incoming", "outgoing", or "both"

# Which steps to analyze (set to None to auto-detect all available steps)
DEFAULT_STEPS = None  # e.g., [100, 200, 300, 400, 500] or None for all
# DEFAULT_STEPS = list(range(100, 1001, 100))  # Example: 100, 200, ..., 1000

# Visualization options
DEFAULT_INCLUDE_BIAS     = True   # Match your callback settings
DEFAULT_PLOT_LOG_SCALE   = True   # Use log scale for y-axis
DEFAULT_PLOT_ALL         = True   # Plot all channels
DEFAULT_PLOT_TOP_K       = True   # Plot top-K channels
DEFAULT_PLOT_BOTTOM_K    = True   # Plot bottom-K channels

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


def _infer_attention_heads(q_proj_weight: torch.Tensor, model_dim: int) -> Tuple[int, int]:
    """
    Infer number of attention heads and head dimension from q_proj weight.
    
    Args:
        q_proj_weight: Weight tensor of shape [n_heads * d_head, d_model]
        model_dim: Model dimension (d_model)
    
    Returns:
        (n_heads, d_head) tuple
    """
    out_dim = q_proj_weight.shape[0]
    
    # Try common head configurations
    for n_heads in [8, 12, 16, 20, 24, 32, 40, 48, 64]:
        if out_dim % n_heads == 0:
            d_head = out_dim // n_heads
            if 32 <= d_head <= 256:  # Reasonable head dimension range
                return n_heads, d_head
    
    # Fallback: assume d_head = 64 or 128
    for d_head in [64, 128]:
        if out_dim % d_head == 0:
            n_heads = out_dim // d_head
            return n_heads, d_head
    
    raise RuntimeError(f"Cannot infer attention heads from q_proj shape {tuple(q_proj_weight.shape)}")


def load_grad_channels_for_step(grad_base_dir: str, step: int, 
                                 channel_type: str = "incoming",
                                 include_bias: bool = True) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Load gradient data for a single step and aggregate to channel-level.
    
    Args:
        grad_base_dir: Base directory containing index.csv
        step: Training step number
        channel_type: "incoming", "outgoing", "both", "attention", or "mlp_and_attention"
        include_bias: Whether to include bias terms
    
    Returns:
        Array of per-channel gradient energies
        List of (layer_id, channel_key) tuples identifying each channel
    """
    index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(index_csv)
    
    df = pd.read_csv(index_csv)
    rows = df[df["global_step"] == step]
    
    if rows.empty:
        warnings.warn(f"No entries for global_step={step} in index.csv")
        return np.array([]), []
    
    # Store per-layer channel energies with ordering
    mlp_energy_per_layer: Dict[int, Dict[str, np.ndarray]] = {}
    attn_energy_per_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def _ensure_mlp(layer_id, key, size):
        d = mlp_energy_per_layer.setdefault(layer_id, {})
        if key not in d:
            d[key] = np.zeros(size, dtype=np.float64)
    
    def _ensure_attn(layer_id, key):
        d = attn_energy_per_layer.setdefault(layer_id, {})
        if key not in d:
            d[key] = None
    
    # First pass: collect all gradients
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
                pass
            else:
                continue
        
        # MLP processing
        if sub == "mlp":
            if param == "up_proj.weight":
                e = (G.to(torch.float32).pow(2).sum(dim=0)).cpu().numpy()
                _ensure_mlp(layer_id, "incoming", e.size)
                mlp_energy_per_layer[layer_id]["incoming"] += e
                
            elif param == "down_proj.weight":
                e = (G.to(torch.float32).pow(2).sum(dim=1)).cpu().numpy()
                _ensure_mlp(layer_id, "outgoing", e.size)
                mlp_energy_per_layer[layer_id]["outgoing"] += e
        
        # Attention processing
        elif sub == "self_attn":
            _ensure_attn(layer_id, param)
            if attn_energy_per_layer[layer_id][param] is None:
                attn_energy_per_layer[layer_id][param] = G.pow(2)
            else:
                attn_energy_per_layer[layer_id][param] += G.pow(2)
    
    # Process results based on channel_type
    result_arrays = []
    channel_ids = []
    
    # MLP channels
    if channel_type in ["incoming", "outgoing", "both", "mlp_and_attention"]:
        for lid in sorted(mlp_energy_per_layer.keys()):
            d = mlp_energy_per_layer[lid]
            inc = d.get("incoming", None)
            out = d.get("outgoing", None)
            
            if channel_type == "incoming" and inc is not None:
                result_arrays.append(inc)
                for ch_idx in range(inc.size):
                    channel_ids.append((lid, f"mlp_incoming_{ch_idx}"))
                    
            elif channel_type == "outgoing" and out is not None:
                result_arrays.append(out)
                for ch_idx in range(out.size):
                    channel_ids.append((lid, f"mlp_outgoing_{ch_idx}"))
                    
            elif channel_type in ["both", "mlp_and_attention"]:
                if inc is not None and out is not None:
                    combined = inc + out
                    result_arrays.append(combined)
                    for ch_idx in range(combined.size):
                        channel_ids.append((lid, f"mlp_both_{ch_idx}"))
                elif inc is not None:
                    result_arrays.append(inc)
                    for ch_idx in range(inc.size):
                        channel_ids.append((lid, f"mlp_incoming_{ch_idx}"))
                elif out is not None:
                    result_arrays.append(out)
                    for ch_idx in range(out.size):
                        channel_ids.append((lid, f"mlp_outgoing_{ch_idx}"))
    
    # Attention heads
    if channel_type in ["attention", "mlp_and_attention"]:
        for lid in sorted(attn_energy_per_layer.keys()):
            d = attn_energy_per_layer[lid]
            
            # Get q_proj to infer head structure
            q_proj_grad = d.get("q_proj.weight", None)
            k_proj_grad = d.get("k_proj.weight", None)
            v_proj_grad = d.get("v_proj.weight", None)
            o_proj_grad = d.get("o_proj.weight", None)
            
            if q_proj_grad is None:
                continue
            
            # Infer number of heads and head dimension
            try:
                n_heads, d_head = _infer_attention_heads(q_proj_grad, q_proj_grad.shape[1])
            except RuntimeError as e:
                warnings.warn(f"Layer {lid}: {e}")
                continue
            
            # Aggregate per-head gradient energy
            per_head_energy = np.zeros(n_heads, dtype=np.float64)
            
            # Q, K, V projections: [n_heads * d_head, d_model]
            # Split by heads and aggregate
            for proj_name, proj_grad in [("q_proj.weight", q_proj_grad), 
                                          ("k_proj.weight", k_proj_grad),
                                          ("v_proj.weight", v_proj_grad)]:
                if proj_grad is not None:
                    # Reshape to [n_heads, d_head, d_model] and sum over d_head and d_model
                    proj_reshaped = proj_grad.view(n_heads, d_head, -1)
                    head_energy = proj_reshaped.sum(dim=[1, 2]).cpu().numpy()
                    per_head_energy += head_energy
            
            # O projection: [d_model, n_heads * d_head]
            # This maps from heads back to model dimension
            if o_proj_grad is not None:
                # Reshape to [d_model, n_heads, d_head] and sum over d_model and d_head
                o_reshaped = o_proj_grad.view(-1, n_heads, d_head)
                head_energy = o_reshaped.sum(dim=[0, 2]).cpu().numpy()
                per_head_energy += head_energy
            
            result_arrays.append(per_head_energy)
            for head_idx in range(n_heads):
                channel_ids.append((lid, f"attn_head_{head_idx}"))
    
    gradient_array = np.concatenate(result_arrays, axis=0) if result_arrays else np.array([])
    return gradient_array, channel_ids


def load_all_steps_gradients(grad_base_dir: str, steps: List[int],
                             channel_type: str = "incoming",
                             include_bias: bool = True) -> Tuple[List[int], np.ndarray]:
    """
    Load gradient data for multiple steps.
    
    Returns:
        steps: List of available steps
        gradient_matrix: Shape [num_steps, num_channels]
    """
    gradient_data = []
    valid_steps = []
    channel_id_reference = None  # Store reference channel ordering from first step
    
    print(f"Loading gradients for {len(steps)} steps...")
    for step_idx, step in enumerate(steps):
        try:
            grad, channel_ids = load_grad_channels_for_step(grad_base_dir, step, channel_type, include_bias)
            
            if grad.size == 0:
                warnings.warn(f"No gradient data for step {step}")
                continue
            
            # Validate channel consistency across steps
            if step_idx == 0:
                # First valid step - use as reference
                channel_id_reference = channel_ids
                print(f"  Step {step} (reference): {grad.size} channels, total L2²={grad.sum():.3e}")
            else:
                # Subsequent steps - verify they match reference
                if len(channel_ids) != len(channel_id_reference):
                    raise ValueError(
                        f"Step {step} has {len(channel_ids)} channels, "
                        f"but reference step has {len(channel_id_reference)} channels. "
                        f"Channel count mismatch!"
                    )
                
                # Verify channel IDs match (same layer and type)
                mismatches = []
                for i, (ref_id, curr_id) in enumerate(zip(channel_id_reference, channel_ids)):
                    if ref_id != curr_id:
                        mismatches.append(f"  Index {i}: expected {ref_id}, got {curr_id}")
                
                if mismatches:
                    error_msg = f"Step {step} has channel ordering mismatch:\n" + "\n".join(mismatches[:5])
                    if len(mismatches) > 5:
                        error_msg += f"\n  ... and {len(mismatches) - 5} more mismatches"
                    raise ValueError(error_msg)
                
                print(f"  Step {step}: {grad.size} channels, total L2²={grad.sum():.3e} ✓")
            
            gradient_data.append(grad)
            valid_steps.append(step)
            
        except Exception as e:
            warnings.warn(f"Failed to load step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    if not gradient_data:
        raise ValueError("No gradient data loaded for any step")
    
    # Final consistency check
    channel_counts = [g.size for g in gradient_data]
    if len(set(channel_counts)) > 1:
        raise ValueError(
            f"CRITICAL: Inconsistent channel counts detected across steps: {set(channel_counts)}. "
            f"This should not happen after validation. Please check your data."
        )
    
    gradient_matrix = np.stack(gradient_data, axis=0)  # [num_steps, num_channels]
    print(f"\n✓ Loaded {len(valid_steps)} steps with consistent {gradient_matrix.shape[1]} channels")
    
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
                             out_dir: str,
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
        out_dir: Output directory for plots and data
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
        out_path = os.path.join(out_dir, f"gradient_timeseries_top{top_k}_{channel_type}.png")
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
        out_path = os.path.join(out_dir, f"gradient_timeseries_bottom{bottom_k}_{channel_type}.png")
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
        out_path = os.path.join(out_dir, f"gradient_timeseries_all_{channel_type}.png")
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
    out_path = os.path.join(out_dir, f"gradient_stats_{channel_type}.png")
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
    csv_path = os.path.join(out_dir, f"gradient_stats_{channel_type}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DATA] Gradient statistics saved to {csv_path}")
    
    # Save top-k channel trajectories
    if plot_top:
        top_channel_data = {"step": steps}
        for rank, idx in enumerate(top_indices, 1):
            top_channel_data[f"channel_{idx}"] = gradient_matrix[:, idx]
        df = pd.DataFrame(top_channel_data)
        csv_path = os.path.join(out_dir, f"top{top_k}_channels_{channel_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[DATA] Top-{top_k} channel trajectories saved to {csv_path}")
    
    # Save bottom-k channel trajectories
    if plot_bottom:
        bottom_channel_data = {"step": steps}
        for rank, idx in enumerate(bottom_indices, 1):
            bottom_channel_data[f"channel_{idx}"] = gradient_matrix[:, idx]
        df = pd.DataFrame(bottom_channel_data)
        csv_path = os.path.join(out_dir, f"bottom{bottom_k}_channels_{channel_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[DATA] Bottom-{bottom_k} channel trajectories saved to {csv_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze channel gradient changes over training steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required/primary arguments
    parser.add_argument(
        "--grad_base_dir",
        type=str,
        default=DEFAULT_GRAD_BASE_DIR,
        help="Base directory containing grad_dump/index.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for plots and CSV files"
    )
    
    # Channel selection
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top channels to plot"
    )
    parser.add_argument(
        "--bottom_k",
        type=int,
        default=DEFAULT_BOTTOM_K,
        help="Number of bottom channels to plot"
    )
    parser.add_argument(
        "--channel_type",
        type=str,
        default=DEFAULT_CHANNEL_TYPE,
        choices=["incoming", "outgoing", "both", "attention", "mlp_and_attention"],
        help="Type of channels to analyze"
    )
    
    # Steps to analyze
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Specific steps to analyze (e.g., --steps 100 200 300). If not provided, auto-detects all steps"
    )
    parser.add_argument(
        "--step_range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Generate step range (e.g., --step_range 100 1000 100 for 100,200,...,900)"
    )
    
    # Plot options
    parser.add_argument(
        "--plot_all",
        action="store_true",
        default=DEFAULT_PLOT_ALL,
        help="Plot all channels with transparency"
    )
    parser.add_argument(
        "--no_plot_all",
        dest="plot_all",
        action="store_false",
        help="Don't plot all channels"
    )
    parser.add_argument(
        "--plot_top",
        action="store_true",
        default=DEFAULT_PLOT_TOP_K,
        help="Plot top-k channels"
    )
    parser.add_argument(
        "--no_plot_top",
        dest="plot_top",
        action="store_false",
        help="Don't plot top-k channels"
    )
    parser.add_argument(
        "--plot_bottom",
        action="store_true",
        default=DEFAULT_PLOT_BOTTOM_K,
        help="Plot bottom-k channels"
    )
    parser.add_argument(
        "--no_plot_bottom",
        dest="plot_bottom",
        action="store_false",
        help="Don't plot bottom-k channels"
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        default=DEFAULT_PLOT_LOG_SCALE,
        help="Use log scale for y-axis"
    )
    parser.add_argument(
        "--no_log_scale",
        dest="log_scale",
        action="store_false",
        help="Use linear scale for y-axis"
    )
    
    # Other options
    parser.add_argument(
        "--include_bias",
        action="store_true",
        default=DEFAULT_INCLUDE_BIAS,
        help="Include bias terms in gradient calculation"
    )
    parser.add_argument(
        "--no_include_bias",
        dest="include_bias",
        action="store_false",
        help="Exclude bias terms from gradient calculation"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 60)
    print("Channel Gradient Time Series Analysis")
    print("=" * 60)
    print(f"Gradient directory: {args.grad_base_dir}")
    print(f"Output directory:   {args.out_dir}")
    print(f"Channel type:       {args.channel_type}")
    print(f"Top-K:              {args.top_k}")
    print(f"Bottom-K:           {args.bottom_k}")
    print(f"Plot all:           {args.plot_all}")
    print(f"Plot top:           {args.plot_top}")
    print(f"Plot bottom:        {args.plot_bottom}")
    print(f"Log scale:          {args.log_scale}")
    print(f"Include bias:       {args.include_bias}")
    print("=" * 60)
    
    # Determine steps to analyze
    if args.step_range:
        steps = list(range(args.step_range[0], args.step_range[1], args.step_range[2]))
        print(f"Using step range: {steps[:5]}{'...' if len(steps) > 5 else ''}")
    elif args.steps:
        steps = args.steps
        print(f"Using specified steps: {steps}")
    else:
        steps = detect_available_steps(args.grad_base_dir)
    
    if not steps:
        raise ValueError("No steps to analyze")
    
    # Load gradient data across all steps
    valid_steps, gradient_matrix = load_all_steps_gradients(
        args.grad_base_dir, 
        steps, 
        args.channel_type,
        args.include_bias
    )
    
    print(f"\nLoaded gradient matrix: {gradient_matrix.shape} (steps × channels)")
    print(f"Steps: {valid_steps}")
    
    # Plot time series
    plot_gradient_timeseries(
        valid_steps,
        gradient_matrix,
        args.out_dir,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        log_scale=args.log_scale,
        plot_all=args.plot_all,
        plot_top=args.plot_top,
        plot_bottom=args.plot_bottom,
        channel_type=args.channel_type
    )
    
    print(f"\n✓ Analysis complete! Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()