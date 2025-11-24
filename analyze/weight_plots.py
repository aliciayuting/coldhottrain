import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification
from safetensors.torch import load_file

# PATH = sys.argv[1] if len(sys.argv) > 1 else '/share/desa/nfs02/yy354/runs/glue/statedump/runs/roberta-base-fp-seed0-lr5e-5-bs32-qvclassifier-fused-false/'
PATH = sys.argv[1] if len(sys.argv) > 1 else '/share/desa/nfs02/shouxu/cold/runs/glue/mnli/runs/roberta-base-fp-seed0-lr2e-5-bs32-epochs10-qvclassifier_false-DUMP-2'
FULL_FT = False
BY_HEAD = True
EVERY_NTH = 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "10epoch_weight_changes_choose15")
DATA_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "stats")
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "plots")


from transformers.trainer import get_parameter_names  # same helper Trainer uses

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
        # "q_proj",
        # "k_proj",
        # "v_proj",
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

def get_trainable_params(checkpoint_path):
    """Get the set of trainable parameter names based on the filtering logic."""
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    num_labels = len(config.get("id2label", {}))
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=num_labels
    )

    decay = get_parameter_names(model, [torch.nn.LayerNorm])
    decay = [n for n in decay if "bias" not in n]

    trainable_params = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply same filtering logic as original script
        if "query" in name or "value" in name or "classifier" in name or FULL_FT:
            trainable_params.add(name)

    return trainable_params, model.config


def compute_row_column_norms(tensor: torch.Tensor, head_info=None):
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
    return row_norms[:5], col_norms[:5]  # Choose first 5 columns for plotting


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
    axes[0].set_title(f"{param_name} row norms (weight changes)")
    axes[0].grid(True, alpha=0.2)
    plot_line_stack(axes[1], iterations_np, col_np, col_colors)
    axes[1].set_ylabel("Column L2 norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title(f"{param_name} column norms (weight changes)")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{slugify(param_name)}.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def collect_checkpoint_dirs():
    checkpoint_dirs = []
    for entry in os.listdir(PATH):
        full_path = os.path.join(PATH, entry)
        if not os.path.isdir(full_path) or "checkpoint" not in entry:
            continue
        try:
            iter_num = int(entry.split("-")[-1])
        except ValueError:
            continue
        checkpoint_dirs.append((iter_num, entry))
    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs


def main():
    ensure_dir(DATA_OUTPUT_DIR)
    ensure_dir(PLOT_OUTPUT_DIR)

    checkpoint_dirs = collect_checkpoint_dirs()
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {PATH}")

    trainable_params = None
    model_config = None
    param_shapes = {}
    stats = defaultdict(default_stats_entry)

    # We need pairs of consecutive checkpoints to compute weight changes
    for i in range(len(checkpoint_dirs) - 1):
        iter_num_curr, dir_name_curr = checkpoint_dirs[i]
        iter_num_next, dir_name_next = checkpoint_dirs[i + 1]
        
        # Use the iteration number of the later checkpoint as the label
        iter_num = iter_num_next
        
        if iter_num % EVERY_NTH != 0:
            continue
        
        checkpoint_path_curr = os.path.join(PATH, dir_name_curr)
        checkpoint_path_next = os.path.join(PATH, dir_name_next)
        
        
        # Get trainable params from first checkpoint
        if trainable_params is None:
            trainable_params, model_config = get_trainable_params(checkpoint_path_curr)
        
        # Load weights from both checkpoints
        weights_curr = load_file(os.path.join(checkpoint_path_curr, "model.safetensors"))
        weights_next = load_file(os.path.join(checkpoint_path_next, "model.safetensors"))
        
        # Compute weight changes for trainable params
        for param_name in trainable_params:
            if param_name not in weights_curr or param_name not in weights_next:
                continue
            weight_change = weights_next[param_name] - weights_curr[param_name]

            # Store param shape for head info initialization
            if param_name not in param_shapes:
                param_shapes[param_name] = tuple(weight_change.shape)
            
            entry = stats[param_name]
            if not entry["head_info_initialized"]:
                entry["head_info"] = build_head_split_info(
                    param_name, param_shapes[param_name], model_config
                )
                entry["head_info_initialized"] = True
            head_info = entry["head_info"]

            row_norms, col_norms = compute_row_column_norms(weight_change, head_info)
            entry["iterations"].append(iter_num)
            entry["row_norms"].append(row_norms.cpu())
            entry["col_norms"].append(col_norms.cpu())
    print(f"stats keys : {list(stats.keys())}") 
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