#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Dict, Tuple

def load_state_dict(ckpt_dir: str) -> Dict[str, object]:
    """
    Load a state_dict from a HF checkpoint directory that contains either
    model.safetensors or pytorch_model.bin.
    """
    st_path_sft = os.path.join(ckpt_dir, "model.safetensors")
    st_path_bin = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(st_path_sft):
        from safetensors.torch import load_file as sft_load_file
        return sft_load_file(st_path_sft)
    elif os.path.exists(st_path_bin):
        import torch
        return torch.load(st_path_bin, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {ckpt_dir}"
        )

def iter_ckpt_dirs(root: str):
    """
    Yield checkpoint directories. If root looks like a single checkpoint
    (contains model.safetensors / pytorch_model.bin), yield root itself;
    otherwise look for step* subdirectories.
    """
    has_sft = os.path.exists(os.path.join(root, "model.safetensors"))
    has_bin = os.path.exists(os.path.join(root, "pytorch_model.bin"))
    if has_sft or has_bin:
        yield root
        return
    # Otherwise, scan step directories
    for name in sorted(os.listdir(root)):
        if name.startswith("step") and os.path.isdir(os.path.join(root, name)):
            yield os.path.join(root, name)

def main():
    ap = argparse.ArgumentParser(
        description="Print tensor name and shape from HF checkpoint(s)."
    )
    ap.add_argument("path", help="Path to a checkpoint dir (e.g., weight_dump/step000100_post) or weight_dump root")
    ap.add_argument("--filter", help="Regex to filter parameter names (e.g. 'mlp|self_attn')", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Limit how many tensors to print per checkpoint (0 = no limit)")
    ap.add_argument("--summary", action="store_true", help="Only print a compact summary per checkpoint")
    args = ap.parse_args()

    name_re = re.compile(args.filter) if args.filter else None

    any_printed = False
    for ckpt_dir in iter_ckpt_dirs(args.path):
        print(f"\n=== {ckpt_dir} ===")
        try:
            sd = load_state_dict(ckpt_dir)
        except Exception as e:
            print(f"  [WARN] Skip {ckpt_dir}: {e}")
            continue

        total = 0
        total_elems = 0
        printed = 0

        # Build a compact per-prefix summary if requested
        prefix_counts: Dict[str, Tuple[int, int]] = {}  # prefix -> (num tensors, total elems)

        for k, v in sd.items():
            # v is a torch.Tensor (for both safetensors and bin)
            try:
                shape = tuple(v.shape)
                numel = int(v.numel())
                dtype = str(v.dtype)
            except Exception:
                # If something unexpected is inside the state_dict
                continue

            if name_re and not name_re.search(k):
                continue

            total += 1
            total_elems += numel

            if args.summary:
                # Group by top-2 path components (e.g., 'model.layers.00.self_attn')
                parts = k.split(".")
                prefix = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts)
                n, e = prefix_counts.get(prefix, (0, 0))
                prefix_counts[prefix] = (n + 1, e + numel)
            else:
                if args.limit == 0 or printed < args.limit:
                    print(f"  {k:60s}  shape={shape}  dtype={dtype}  numel={numel}")
                    printed += 1

        if args.summary:
            # Print summary by prefix
            for pref in sorted(prefix_counts.keys()):
                n, e = prefix_counts[pref]
                print(f"  [{pref}]  tensors={n:5d}  total_elems={e}")
        print(f"  --- totals --- tensors={total}  total_elems={total_elems}")
        any_printed = True

    if not any_printed:
        print("[INFO] No checkpoints found or nothing matched your filter.")

if __name__ == "__main__":
    main()