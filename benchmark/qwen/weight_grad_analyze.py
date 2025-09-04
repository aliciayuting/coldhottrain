# --- Notebook config (edit these) ---
GRAD_BASE_DIR  = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/grad_dump"
GLOBAL_STEP    = 200          # e.g., 200 -> reads step000200 grads from index.csv
WEIGHT_ROOT    = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/weight_dump"
WEIGHT_STEP_TAG= "step000200_pre"   # e.g., step000200_pre / step000200_post / step000200_next_pre
OUT_DIR        = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/plots"

# Controls
SAMPLE_FRAC    = 1.0          # set <1.0 (e.g., 0.25) for uniform subsampling to save RAM
TOP_P          = 0.01         # annotate top 1%
INCLUDE_PATTERNS = None       # e.g., [r"self_attn\..*\.weight", r"mlp\..*\.weight"]
EXCLUDE_PATTERNS = None       # e.g., [r"lm_head", r"embed"]
INCLUDE_BIAS     = False      # set True to include *.bias in weights curve

# --- Imports ---
import os, re, gc, math, warnings, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Optional safetensors support for HF checkpoints
try:
    from safetensors.torch import load_file as safetensors_load
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False

# ------------------------------
# Utilities
# ------------------------------
def _load_any_tensor(path: str) -> torch.Tensor:
    """
    Load a tensor from path (supports torch .pt/.pth/.bin and numpy .npy/.npz).
    Returns a CPU float32 tensor.
    """
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
                            t = v; break
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

def _tensor_to_sampled_g2_1d(t: torch.Tensor, sample_frac: float) -> np.ndarray:
    """
    Convert a tensor to a 1D numpy array of squared magnitudes (g^2 or w^2),
    with optional uniform subsampling.
    """
    t = t.reshape(-1)
    if sample_frac < 1.0:
        n = t.numel()
        k = max(1, int(math.ceil(n * sample_frac)))
        idx = torch.randperm(n)[:k]
        t = t[idx]
    g2 = (t**2).to(dtype=torch.float32).cpu().numpy()
    del t
    return g2

def _make_curve_and_plot(values_sq: np.ndarray,
                         label: str,
                         save_path: str,
                         top_p: float = 0.01,
                         figsize=(5.0, 4.0),
                         dpi=200):
    """
    Build Lorenz-style cumulative curve and save a plot.
    """
    if values_sq.size == 0:
        raise RuntimeError("No values provided to plot.")

    order = np.argsort(values_sq)[::-1]
    sorted_vals = values_sq[order]
    del order; gc.collect()

    N = sorted_vals.size
    x = (np.arange(1, N + 1, dtype=np.float64)) / float(N)
    cum = np.cumsum(sorted_vals, dtype=np.float64)
    total = cum[-1]
    y = cum / total

    k_top = max(1, int(math.ceil(top_p * N)))
    y_at_top = y[k_top - 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, y, linewidth=2)

    # Guides
    ax.axvline(top_p, linestyle="--", linewidth=1.5)
    ax.axhline(y_at_top, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Proportion of Elements", fontsize=12)
    ax.set_ylabel("Cumulative L2 Norm\u00b2", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)

    lbl = f"Top {int(top_p*100)}%\n{y_at_top*100:.1f}% of L2\u00b2"
    ax.legend([label], loc="lower right", frameon=True)
    ax.text(top_p + 0.002, min(0.98, y_at_top + 0.02), lbl, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {"N": int(N), "top_p": float(top_p), "capture": float(y_at_top), "save_path": save_path}

# ------------------------------
# Gradients: read from index.csv
# ------------------------------
def load_grad_sq_from_index(grad_base_dir: str,
                            global_step: int,
                            sample_frac: float = 1.0,
                            index_csv: str | None = None) -> np.ndarray:
    if index_csv is None:
        index_csv = os.path.join(grad_base_dir, "index.csv")
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(f"Missing index.csv: {index_csv}")

    df = pd.read_csv(index_csv)
    if "global_step" not in df.columns or "file" not in df.columns:
        raise ValueError("index.csv must contain columns 'global_step' and 'file'.")

    rows = df[df["global_step"] == global_step]
    if rows.empty:
        raise ValueError(f"No entries for global_step={global_step} in {index_csv}")

    file_paths = [os.path.join(grad_base_dir, f) for f in rows["file"].tolist()]

    chunks = []
    total_elems = 0
    for p in file_paths:
        if not os.path.isfile(p):
            warnings.warn(f"[grad] Missing file: {p}")
            continue
        t = _load_any_tensor(p)
        g2 = _tensor_to_sampled_g2_1d(t, sample_frac)
        chunks.append(g2)
        total_elems += g2.size

    if total_elems == 0:
        raise RuntimeError("No gradients loaded (all files missing or empty).")

    g2_all = np.concatenate(chunks, axis=0)
    del chunks; gc.collect()
    return g2_all

# ------------------------------
# Weights: read from HF checkpoint folder
# ------------------------------
def _param_name_passes(name: str,
                       include_patterns,
                       exclude_patterns,
                       include_bias: bool) -> bool:
    if not include_bias:
        if name.endswith(".bias") or name.split(".")[-1] == "bias":
            return False

    def _match_any(patterns, s):
        return any(re.search(p, s) for p in patterns) if patterns else False

    if include_patterns and not _match_any(include_patterns, name):
        return False
    if exclude_patterns and _match_any(exclude_patterns, name):
        return False
    return True

def _gather_weight_files(ckpt_dir: str) -> list[str]:
    safes = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if len(safes) > 0:
        return safes
    pt_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(pt_bin):
        return [pt_bin]
    raise FileNotFoundError(
        f"No weight files found in {ckpt_dir} (looked for *.safetensors or pytorch_model.bin)")

def load_weight_sq_from_checkpoint(weight_root: str,
                                   step_tag: str,
                                   sample_frac: float = 1.0,
                                   include_patterns=None,
                                   exclude_patterns=None,
                                   include_bias: bool = False) -> np.ndarray:
    ckpt_dir = os.path.join(weight_root, step_tag)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir does not exist: {ckpt_dir}")

    files = _gather_weight_files(ckpt_dir)
    chunks = []
    total_elems = 0

    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".safetensors":
            if not _HAVE_SAFETENSORS:
                raise RuntimeError("safetensors not installed but .safetensors files are present. `pip install safetensors`")
            sd = safetensors_load(f, device="cpu")  # dict[name] -> tensor
            for name, t in sd.items():
                if not _param_name_passes(name, include_patterns, exclude_patterns, include_bias):
                    continue
                t = t.detach().to(dtype=torch.float32, device="cpu")
                w2 = _tensor_to_sampled_g2_1d(t, sample_frac)
                chunks.append(w2)
                total_elems += w2.size
            del sd
        elif ext in [".bin", ".pt", ".pth"]:
            obj = torch.load(f, map_location="cpu")
            if not isinstance(obj, dict):
                raise RuntimeError(f"Unexpected object in {f}: type={type(obj)} (expected state_dict dict)")
            for name, t in obj.items():
                if not isinstance(t, torch.Tensor):
                    continue
                if not _param_name_passes(name, include_patterns, exclude_patterns, include_bias):
                    continue
                t = t.detach().to(dtype=torch.float32, device="cpu")
                w2 = _tensor_to_sampled_g2_1d(t, sample_frac)
                chunks.append(w2)
                total_elems += w2.size
            del obj
        else:
            warnings.warn(f"Skipping unsupported weight file: {f}")

    if total_elems == 0:
        raise RuntimeError("No weights loaded (filters too strict or empty checkpoint).")

    w2_all = np.concatenate(chunks, axis=0)
    del chunks; gc.collect()
    return w2_all

# ------------------------------
# Run: GRAD curve
# ------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

grad_sq = load_grad_sq_from_index(
    grad_base_dir=GRAD_BASE_DIR,
    global_step=GLOBAL_STEP,
    sample_frac=SAMPLE_FRAC
)
grad_save_path = os.path.join(OUT_DIR, f"grad_curve_step{GLOBAL_STEP:06d}.png")
grad_stats = _make_curve_and_plot(
    grad_sq,
    label=f"Gradients @ step {GLOBAL_STEP}",
    save_path=grad_save_path,
    top_p=TOP_P
)
print(f"[GRAD] N={grad_stats['N']}  top{int(TOP_P*100)}% captures {grad_stats['capture']*100:.2f}%  -> {grad_stats['save_path']}")

# ------------------------------
# Run: WEIGHT curve
# ------------------------------
weight_sq = load_weight_sq_from_checkpoint(
    weight_root=WEIGHT_ROOT,
    step_tag=WEIGHT_STEP_TAG,
    sample_frac=SAMPLE_FRAC,
    include_patterns=INCLUDE_PATTERNS,
    exclude_patterns=EXCLUDE_PATTERNS,
    include_bias=INCLUDE_BIAS
)
weight_save_path = os.path.join(OUT_DIR, f"weight_curve_{WEIGHT_STEP_TAG}.png")
weight_stats = _make_curve_and_plot(
    weight_sq,
    label=f"Weights @ {WEIGHT_STEP_TAG}",
    save_path=weight_save_path,
    top_p=TOP_P
)
print(f"[WEIGHT] N={weight_stats['N']}  top{int(TOP_P*100)}% captures {weight_stats['capture']*100:.2f}%  -> {weight_stats['save_path']}")