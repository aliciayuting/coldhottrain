#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple

import torch

def _is_ckpt_dir(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    files = os.listdir(p)
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return True
    # sharded cases
    if any(fn.endswith(".safetensors") for fn in files) or any(fn.endswith(".bin") for fn in files):
        return True
    return False

def _load_safetensors_dir(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load single or sharded safetensors."""
    from safetensors.torch import load_file
    idx_json = os.path.join(ckpt_dir, "model.safetensors.index.json")
    state = {}
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
    # fallback: load all *.safetensors files
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(".safetensors"):
            state.update(load_file(os.path.join(ckpt_dir, fn)))
    if not state:
        raise FileNotFoundError(f"No safetensors found in {ckpt_dir}")
    return state

def _load_bin_dir(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load single or sharded PyTorch bin files."""
    single = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(single):
        return torch.load(single, map_location="cpu")
    # sharded: pytorch_model-00001-of-XXXXX.bin
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
    if not _is_ckpt_dir(ckpt_dir):
        raise FileNotFoundError(f"Not a checkpoint dir: {ckpt_dir}")
    files = os.listdir(ckpt_dir)
    if "model.safetensors" in files or any(fn.endswith(".safetensors") for fn in files):
        return _load_safetensors_dir(ckpt_dir)
    if "pytorch_model.bin" in files or any(fn.endswith(".bin") for fn in files):
        return _load_bin_dir(ckpt_dir)
    raise FileNotFoundError(f"No recognizable checkpoint files in {ckpt_dir}")

def find_step_dirs(weight_dump_root: str, step: int) -> Tuple[str, str]:
    step_tag = f"step{step:06d}"
    pre = os.path.join(weight_dump_root, f"{step_tag}_pre")
    post = os.path.join(weight_dump_root, f"{step_tag}_post")
    if not _is_ckpt_dir(pre):
        raise FileNotFoundError(f"Missing pre dir: {pre}")
    if not _is_ckpt_dir(post):
        raise FileNotFoundError(f"Missing post dir: {post}")
    return pre, post

def compare_tensors(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    a32 = a.detach().cpu().to(torch.float32)
    b32 = b.detach().cpu().to(torch.float32)
    diff = (a32 - b32)
    absdiff = diff.abs()
    l2 = torch.linalg.vector_norm(diff).item()
    base = torch.linalg.vector_norm(a32).item()
    rel_l2 = (l2 / base) if base > 0 else float("inf") if l2 > 0 else 0.0
    mean_abs = absdiff.mean().item()
    max_abs = absdiff.max().item()
    # cosine similarity
    denom = (torch.linalg.vector_norm(a32) * torch.linalg.vector_norm(b32)).item()
    cos = (a32.flatten() @ b32.flatten()).item() / denom if denom > 0 else float("nan")
    # equality within tolerance
    equal = torch.allclose(a32, b32, atol=atol, rtol=rtol)
    # accumulated difference (sum of |diff|)
    accu = absdiff.sum().item()
    return {
        "equal": bool(equal),
        "l2": l2,
        "rel_l2": rel_l2,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "cosine": cos,
        "accu_abs_sum": accu,
        "numel": a32.numel(),
    }

def main():
    ap = argparse.ArgumentParser(description="Compare pre vs post weights for a given step under weight_dump.")
    ap.add_argument("weight_dump_root", help="Path to weight_dump/")
    ap.add_argument("step", type=int, help="Step number (e.g., 800)")
    ap.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for equality")
    ap.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for equality")
    ap.add_argument("--filter", default=None, help="Regex to restrict parameter names (e.g., 'self_attn|mlp')")
    ap.add_argument("--topk", type=int, default=10, help="Show top-K tensors by L2 diff")
    args = ap.parse_args()

    pre_dir, post_dir = find_step_dirs(args.weight_dump_root, args.step)
    print(f"[info] comparing:\n  pre : {pre_dir}\n  post: {post_dir}\n  tol : atol={args.atol} rtol={args.rtol}")

    sd_pre = load_state_dict(pre_dir)
    sd_post = load_state_dict(post_dir)

    name_re = re.compile(args.filter) if args.filter else None

    common_keys = sorted(set(sd_pre.keys()) & set(sd_post.keys()))
    if name_re:
        common_keys = [k for k in common_keys if name_re.search(k)]

    if not common_keys:
        print("[warn] no overlapping parameter keys between pre and post (after filter).")
        sys.exit(0)

    totals = {
        "equal_count": 0,
        "total_tensors": 0,
        "sum_accu": 0.0,
        "sum_l2": 0.0,
        "sum_numel": 0,
    }
    rows = []

    for k in common_keys:
        a = sd_pre[k]
        b = sd_post[k]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            continue
        if a.shape != b.shape:
            print(f"[mismatch shape] {k}: pre{tuple(a.shape)} vs post{tuple(b.shape)}")
            continue
        m = compare_tensors(a, b, args.atol, args.rtol)
        totals["total_tensors"] += 1
        totals["sum_accu"] += m["accu_abs_sum"]
        totals["sum_l2"] += m["l2"]
        totals["sum_numel"] += m["numel"]
        if m["equal"]:
            totals["equal_count"] += 1
        rows.append((k, m))

    # Summary
    print("\n=== SUMMARY ===")
    print(f"tensors compared     : {totals['total_tensors']}")
    print(f"tensors equal (tol)  : {totals['equal_count']}  ({totals['equal_count']/max(1,totals['total_tensors']):.2%})")
    print(f"accumulated |diff|   : {totals['sum_accu']:.6f}")
    print(f"total L2(diff)       : {totals['sum_l2']:.6f}")
    avg_accu_per_elem = totals["sum_accu"] / max(1, totals["sum_numel"])
    print(f"mean |diff| per elem : {avg_accu_per_elem:.6e}")

    # Top-K by L2 difference
    rows.sort(key=lambda kv: kv[1]["l2"], reverse=True)
    topk = rows[: max(0, args.topk)]
    if topk:
        print(f"\n=== TOP {len(topk)} by L2(diff) ===")
        for k, m in topk:
            print(f"{k:60s}  l2={m['l2']:.6f}  rel_l2={m['rel_l2']:.3e}  "
                  f"mean|d|={m['mean_abs']:.3e}  max|d|={m['max_abs']:.3e}  "
                  f"cos={m['cosine']:.6f}  accu={m['accu_abs_sum']:.6f}  equal={m['equal']}")

if __name__ == "__main__":
    main()