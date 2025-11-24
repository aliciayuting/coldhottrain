import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

# PATH = sys.argv[1] if len(sys.argv) > 1 else '/share/desa/nfs02/cold/runs/glue/statedump/runs/roberta-base-fp-seed0-lr5e-5-bs32-qvclassifier-fused-false/'
PATH = sys.argv[1] if len(sys.argv) > 1 else '/share/desa/nfs02/shouxu/cold/runs/glue/mnli/runs/roberta-base-fp-seed0-lr2e-5-bs32-epochs10-qvclassifier_false-DUMP-2'
FULL_FT = False
BY_HEAD = True
EVERY_NTH = 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "10epoch_gradient_norms")
DATA_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "stats")
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "plots")

from transformers.trainer import get_parameter_names

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)

def default_stats_entry():
    return {
        "iterations": [],
        "row_norms": [],
        "col_norms": [],
        "head_info": None,
        "head_info_initialized": False,
    }

def is_attention_qkv_param(param_name: str) -> bool:
    lowered = param_name.lower()
    keywords = (
        "query",
        "key",
        "value",
    )
    return any(word in lowered for word in keywords)

def build_head_split_info(param_name, param_shape, config):
    if not BY_HEAD or config is None:
        return None
    if not is_attention_qkv_param(param_name):
        return None
    if param_shape is None or len(param_shape) < 2:
        return None
    num_heads = getattr(config, "num_attention_heads", None)
    if not num_heads:
        return None
    rows = param_shape[0]
    cols = param_shape[1]
    if rows % num_heads != 0:
        return None
    head_dim = rows // num_heads
    row_head_ids = [row // head_dim for row in range(rows)]
    column_line_head_ids = []
    for _ in range(cols):
        column_line_head_ids.extend(range(num_heads))
    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_columns": cols,
        "row_head_ids": row_head_ids,
        "column_line_head_ids": column_line_head_ids,
    }

def get_head_colors(num_heads):
    cmap = plt.cm.get_cmap("tab20", num_heads)
    return [cmap(i) for i in range(num_heads)]

def map_grad_filename_to_param_name(filename):
    """
    Map gradient filename to RoBERTa parameter name.
    E.g., 'L00_self_attn_q_proj_weight.npy' -> 'roberta.encoder.layer.0.attention.self.query.weight'
    """
    # Parse the filename - corrected pattern
    # Pattern: L{layer}_self_attn_{proj}_proj_{type}.npy
    match = re.match(r'L(\d+)_self_attn_([qkvo])_proj_(weight|bias)\.npy', filename)
    if not match:
        # Try MLP pattern: L{layer}_mlp_{proj}_proj_{type}.npy
        match = re.match(r'L(\d+)_mlp_(\w+)_proj_(weight|bias)\.npy', filename)
        if not match:
            return None
        
        layer_id = int(match.group(1))
        proj_name = match.group(2)  # e.g., 'gate', 'up', 'down'
        param_type = match.group(3)  # 'weight' or 'bias'
        
        # Map MLP projections
        if proj_name == 'gate':
            return None  # RoBERTa doesn't have gate projection
        elif proj_name == 'up':
            return f"roberta.encoder.layer.{layer_id}.intermediate.dense.{param_type}"
        elif proj_name == 'down':
            return f"roberta.encoder.layer.{layer_id}.output.dense.{param_type}"
        return None
    
    # Attention projection
    layer_id = int(match.group(1))
    proj_letter = match.group(2)  # 'q', 'k', 'v', or 'o'
    param_type = match.group(3)  # 'weight' or 'bias'
    
    # Map projection letter to RoBERTa naming
    proj_map = {
        'q': 'query',
        'k': 'key',
        'v': 'value',
        'o': 'dense',  # output projection
    }
    
    if proj_letter not in proj_map:
        return None
    
    roberta_proj = proj_map[proj_letter]
    
    if roberta_proj == 'dense':
        # Output projection is in attention.output.dense
        return f"roberta.encoder.layer.{layer_id}.attention.output.{roberta_proj}.{param_type}"
    else:
        # Q, K, V are in attention.self
        return f"roberta.encoder.layer.{layer_id}.attention.self.{roberta_proj}.{param_type}"

def get_model_config(checkpoint_path):
    """Get model config from a checkpoint."""
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        # Try to find any checkpoint directory
        for entry in os.listdir(PATH):
            if "checkpoint" in entry:
                config_path = os.path.join(PATH, entry, "config.json")
                if os.path.exists(config_path):
                    break
    
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self, config_dict):
            for k, v in config_dict.items():
                setattr(self, k, v)
    
    return SimpleConfig(config_dict)

def compute_row_column_norms(tensor: torch.Tensor, head_info=None, select_first_n=None):
    tensor = tensor.detach().float()
    if tensor.ndim == 0:
        value = tensor.abs().reshape(1)
        return value, value.clone()

    row_view = tensor.reshape(tensor.shape[0], -1)
    row_norms = torch.linalg.norm(row_view, dim=1)

    if tensor.ndim >= 2:
        col_view = tensor.permute(1, 0, *range(2, tensor.ndim)).reshape(tensor.shape[1], -1)
        col_norms = torch.linalg.norm(col_view, dim=1)
    else:
        col_norms = row_norms.clone()
    
    if select_first_n is not None:
        row_norms = row_norms[:select_first_n]
        col_norms = col_norms[:select_first_n]
    
    return row_norms, col_norms

def plot_line_stack(axis, x_values, line_stack, colors=None):
    if line_stack.ndim == 1:
        line_stack = line_stack[:, None]
    if colors is None:
        axis.plot(x_values, line_stack, alpha=0.4)
        return
    num_lines = line_stack.shape[1]
    if len(colors) != num_lines:
        axis.plot(x_values, line_stack, alpha=0.4)
        return
    for line_idx in range(num_lines):
        axis.plot(x_values, line_stack[:, line_idx], alpha=0.4, color=colors[line_idx])

def plot_param_stats(param_name, iterations, row_stack, col_stack, head_info=None):
    if iterations.numel() == 0:
        return
    iterations_np = iterations.numpy()
    row_np = row_stack.numpy()
    col_np = col_stack.numpy()
    if row_np.ndim == 1:
        row_np = row_np[:, None]
    if col_np.ndim == 1:
        col_np = col_np[:, None]
    row_colors = None
    col_colors = None
    if head_info:
        palette = get_head_colors(head_info["num_heads"])
        row_head_ids = head_info.get("row_head_ids")
        column_head_ids = head_info.get("column_line_head_ids")
        if row_head_ids and len(row_head_ids) == row_np.shape[1]:
            row_colors = [palette[idx] for idx in row_head_ids]
        if column_head_ids and len(column_head_ids) == col_np.shape[1]:
            col_colors = [palette[idx] for idx in column_head_ids]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plot_line_stack(axes[0], iterations_np, row_np, row_colors)
    axes[0].set_ylabel("Row L2 norm")
    axes[0].set_title(f"{param_name} row norms (gradients)")
    axes[0].grid(True, alpha=0.2)
    plot_line_stack(axes[1], iterations_np, col_np, col_colors)
    axes[1].set_ylabel("Column L2 norm")
    axes[1].set_xlabel("Step")
    axes[1].set_title(f"{param_name} column norms (gradients)")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{slugify(param_name)}.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

def collect_grad_dump_dirs():
    """Collect gradient dump directories."""
    grad_dumps_dir = os.path.join(PATH, "grad_dumps")
    if not os.path.exists(grad_dumps_dir):
        raise ValueError(f"Gradient dumps directory not found: {grad_dumps_dir}")
    
    grad_dirs = []
    for entry in os.listdir(grad_dumps_dir):
        full_path = os.path.join(grad_dumps_dir, entry)
        if not os.path.isdir(full_path) or not entry.startswith("step"):
            continue
        try:
            step_num = int(entry.replace("step", ""))
        except ValueError:
            continue
        grad_dirs.append((step_num, entry))
    
    grad_dirs.sort(key=lambda x: x[0])
    return grad_dirs

def main():
    ensure_dir(DATA_OUTPUT_DIR)
    ensure_dir(PLOT_OUTPUT_DIR)

    grad_dirs = collect_grad_dump_dirs()
    if not grad_dirs:
        raise ValueError(f"No gradient dump directories found in {PATH}/grad_dumps")
    
    print(f"Found {len(grad_dirs)} gradient dump directories")

    model_config = None
    param_shapes = {}
    stats = defaultdict(default_stats_entry)

    for step_num, dir_name in grad_dirs:
        if step_num % EVERY_NTH != 0:
            continue
        
        grad_dir = os.path.join(PATH, "grad_dumps", dir_name)
        # print(f"Processing {dir_name} (step {step_num})")
        
        # Get model config from first checkpoint if not loaded
        if model_config is None:
            model_config = get_model_config(PATH)
        
        # Load all gradient files in this directory
        for filename in os.listdir(grad_dir):
            if not filename.endswith('.npy'):
                continue
            
            # Map filename to parameter name
            param_name = map_grad_filename_to_param_name(filename)
            if param_name is None:
                continue
            
            # Filter based on trainable params (query, value, classifier)
            if not ("query" in param_name or "value" in param_name or "classifier" in param_name or FULL_FT):
                continue
            
            # Load gradient
            grad_path = os.path.join(grad_dir, filename)
            grad = torch.from_numpy(np.load(grad_path))
            # Store param shape for head info initialization
            if param_name not in param_shapes:
                param_shapes[param_name] = tuple(grad.shape)
            
            entry = stats[param_name]
            if not entry["head_info_initialized"]:
                entry["head_info"] = build_head_split_info(
                    param_name, param_shapes[param_name], model_config
                )
                entry["head_info_initialized"] = True
            head_info = entry["head_info"]

            # Compute row and column norms
            row_norms, col_norms = compute_row_column_norms(grad, head_info)
            entry["iterations"].append(step_num)
            entry["row_norms"].append(row_norms.cpu())
            entry["col_norms"].append(col_norms.cpu())
            print(f"Processed {param_name} shape row_norms {row_norms.shape} col_norms {col_norms.shape} at step {step_num}")

    print(f"Collected gradients for parameters: {list(stats.keys())}")
    
    # Save and plot results
    for param_name, data in stats.items():
        if not data["iterations"]:
            continue
        iterations = torch.tensor(data["iterations"], dtype=torch.long)
        row_stack = torch.stack(data["row_norms"])
        col_stack = torch.stack(data["col_norms"])
        sort_idx = torch.argsort(iterations)
        iterations = iterations[sort_idx]
        row_stack = row_stack[sort_idx]
        col_stack = col_stack[sort_idx]

        head_info = data.get("head_info")
        payload = {
            "param_name": param_name,
            "iterations": iterations,
            "row_norms": row_stack,
            "col_norms": col_stack,
        }
        if head_info:
            payload["head_info"] = head_info
        stats_path = os.path.join(DATA_OUTPUT_DIR, f"{slugify(param_name)}.pt")
        torch.save(payload, stats_path)
        print(f"Saved stats for {param_name} to {stats_path}")

        plot_param_stats(param_name, iterations, row_stack, col_stack, head_info=head_info)

if __name__ == "__main__":
    main()