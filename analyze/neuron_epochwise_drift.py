#!/usr/bin/env python3
import os, json, csv, argparse, torch
from typing import Dict, List

torch.set_grad_enabled(False)

# ----------------------- Operators ----------------------- #
ATTN_OPS = ["q_proj", "k_proj", "v_proj", "out_proj"]
MLP_OPS  = ["fc1", "fc2"]
LN_OPS   = ["self_attn_layer_norm", "final_layer_norm"]
DEFAULT_OPS = ATTN_OPS + MLP_OPS + LN_OPS

def op_to_param_key(layer_idx: int, op: str) -> str:
    if op in ATTN_OPS:
        return f"model.decoder.layers.{layer_idx}.self_attn.{op}.weight"
    elif op in MLP_OPS or op in LN_OPS:
        return f"model.decoder.layers.{layer_idx}.{op}.weight"
    raise ValueError(f"Unknown operator: {op}")

# ------------------- Safetensors batching ------------------- #
HAS_SAFE = False
try:
    from safetensors import safe_open
    HAS_SAFE = True
except Exception:
    HAS_SAFE = False

def _group_keys_by_safetensors_shard(ckpt_dir: str, keys: List[str]) -> Dict[str, List[str]]:
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError
    with open(index_path, "r") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    groups: Dict[str, List[str]] = {}
    for k in keys:
        rel = weight_map.get(k)
        if rel is None:
            raise KeyError(f"Key not found in safetensors index: {k}")
        groups.setdefault(rel, []).append(k)
    return groups

def _load_many_safetensors(ckpt_dir: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    if not HAS_SAFE:
        raise RuntimeError("Please `pip install safetensors` to read sharded safetensors efficiently.")
    groups = _group_keys_by_safetensors_shard(ckpt_dir, keys)
    out: Dict[str, torch.Tensor] = {}
    for shard_rel, klist in groups.items():
        shard_path = os.path.join(ckpt_dir, shard_rel)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in klist:
                out[k] = f.get_tensor(k).float().cpu()
    return out

# ---------------------- .bin batching ---------------------- #
def _group_keys_by_bin_shard(ckpt_dir: str, keys: List[str]) -> Dict[str, List[str]]:
    index_path = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if not os.path.isfile(index_path):
        single = os.path.join(ckpt_dir, "pytorch_model.bin")
        if not os.path.isfile(single):
            raise FileNotFoundError("No .bin checkpoints found")
        return {"pytorch_model.bin": keys}
    with open(index_path, "r") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    groups: Dict[str, List[str]] = {}
    for k in keys:
        rel = weight_map.get(k)
        if rel is None:
            raise KeyError(f"Key not found in .bin index: {k}")
        groups.setdefault(rel, []).append(k)
    return groups

def _load_many_bin(ckpt_dir: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    groups = _group_keys_by_bin_shard(ckpt_dir, keys)
    out: Dict[str, torch.Tensor] = {}
    for shard_rel, klist in groups.items():
        shard_path = os.path.join(ckpt_dir, shard_rel)
        state = torch.load(shard_path, map_location="cpu")
        for k in klist:
            if k not in state:
                raise KeyError(f"Key {k} not found in shard {shard_rel}")
            out[k] = state[k].float().cpu()
        del state
    return out

# ---------------------- Layer loader ---------------------- #
def load_layer_tensors(ckpt_dir: str, layer: int, ops: List[str]) -> Dict[str, torch.Tensor]:
    keys = [op_to_param_key(layer, op) for op in ops]
    # Prefer safetensors if present
    try:
        return _load_many_safetensors(ckpt_dir, keys)
    except FileNotFoundError:
        return _load_many_bin(ckpt_dir, keys)

# ---------------------- Math helpers ---------------------- #
def per_column_l2_abs_change(W_prev: torch.Tensor, W_cur: torch.Tensor) -> torch.Tensor:
    if W_prev.shape != W_cur.shape:
        raise ValueError(f"Shape mismatch: prev{tuple(W_prev.shape)} vs cur{tuple(W_cur.shape)}")
    if W_prev.ndim == 2:
        diff = (W_cur - W_prev)                 # [out, in]
        return diff.pow(2).sum(dim=0).sqrt()    # [in]
    if W_prev.ndim == 1:
        return (W_cur - W_prev).abs()           # [hidden_size]
    raise ValueError(f"Unsupported ndim={W_prev.ndim}")

# ------------------------- Runner ------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Consecutive-epoch per-column L2 changes for one layer.")
    ap.add_argument("--base", required=True, help="Base dir containing checkpoint folders")
    ap.add_argument("--layer", type=int, required=True, help="Layer index (0-based)")
    ap.add_argument("--e-start", type=int, required=True, help="First epoch (inclusive), e.g., 1")
    ap.add_argument("--e-end", type=int, required=True, help="Last epoch (inclusive), e.g., 50")
    ap.add_argument("--pattern", default="checkpoint-epoch-{e}",
                    help="Checkpoint dir name pattern with {e} placeholder (default: checkpoint-epoch-{e}). "
                         "Use 'checkpoint-{e}' if that's your layout.")
    ap.add_argument("--ops", default=",".join(DEFAULT_OPS),
                    help="Comma-separated ops to include "
                         "(default: q_proj,k_proj,v_proj,out_proj,fc1,fc2,self_attn_layer_norm,final_layer_norm)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--low-mem", action="store_true",
                    help="Load one operator at a time (slower but < 200MB). Default loads all ops for an epoch (~0.6–1.2GB for OPT-6.7B).")
    args = ap.parse_args()

    ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    if args.e_end <= args.e_start:
        raise SystemExit("--e-end must be > --e-start")

    # Prepare CSV
    fieldnames = ["epoch_from","epoch_to","layer","operator","param_key","dim","index",
                  "l2_abs_change","out_features","in_features"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    # Helper to build checkpoint dir path
    def ck(e): return os.path.join(args.base, args.pattern.format(e=e))

    # Sanity
    if not os.path.isdir(ck(args.e_start)) or not os.path.isdir(ck(args.e_start+1)):
        raise FileNotFoundError("Missing starting checkpoints. Check --base and --pattern.")

    if args.low_mem:
        # Stream operator-by-operator for each epoch pair (minimal RAM)
        for e in range(args.e_start, args.e_end):
            ck_e, ck_ep1 = ck(e), ck(e+1)
            print(f"[pair] epochs {e} -> {e+1}")
            for op in ops:
                key = op_to_param_key(args.layer, op)
                # Load only this op from each checkpoint
                tensors_e   = load_layer_tensors(ck_e,   args.layer, [op])
                tensors_ep1 = load_layer_tensors(ck_ep1, args.layer, [op])
                W_e   = tensors_e[key]
                W_ep1 = tensors_ep1[key]
                changes = per_column_l2_abs_change(W_e, W_ep1)

                if W_e.ndim == 2:
                    out_features, in_features = W_e.shape
                    for j in range(in_features):
                        w.writerow({
                            "epoch_from": e, "epoch_to": e+1, "layer": args.layer,
                            "operator": op, "param_key": key, "dim": "column", "index": j,
                            "l2_abs_change": float(changes[j].item()),
                            "out_features": out_features, "in_features": in_features,
                        })
                else:
                    size = W_e.shape[0]
                    for j in range(size):
                        w.writerow({
                            "epoch_from": e, "epoch_to": e+1, "layer": args.layer,
                            "operator": op, "param_key": key, "dim": "element", "index": j,
                            "l2_abs_change": float(changes[j].item()),
                            "out_features": 1, "in_features": size,
                        })
                # free ASAP
                del tensors_e, tensors_ep1, W_e, W_ep1, changes
    else:
        # Rolling cache of all ops for two consecutive epochs (faster; more RAM)
        prev_tensors = None
        for e in range(args.e_start, args.e_end):
            ck_e, ck_ep1 = ck(e), ck(e+1)
            print(f"[pair] epochs {e} -> {e+1}")
            if prev_tensors is None:
                prev_tensors = load_layer_tensors(ck_e, args.layer, ops)
            next_tensors = load_layer_tensors(ck_ep1, args.layer, ops)

            for op in ops:
                key = op_to_param_key(args.layer, op)
                W_e   = prev_tensors[key]
                W_ep1 = next_tensors[key]
                changes = per_column_l2_abs_change(W_e, W_ep1)

                if W_e.ndim == 2:
                    out_features, in_features = W_e.shape
                    for j in range(in_features):
                        w.writerow({
                            "epoch_from": e, "epoch_to": e+1, "layer": args.layer,
                            "operator": op, "param_key": key, "dim": "column", "index": j,
                            "l2_abs_change": float(changes[j].item()),
                            "out_features": out_features, "in_features": in_features,
                        })
                else:
                    size = W_e.shape[0]
                    for j in range(size):
                        w.writerow({
                            "epoch_from": e, "epoch_to": e+1, "layer": args.layer,
                            "operator": op, "param_key": key, "dim": "element", "index": j,
                            "l2_abs_change": float(changes[j].item()),
                            "out_features": 1, "in_features": size,
                        })
            # roll the window
            prev_tensors = next_tensors

    f.close()
    print(f"[OK] Wrote pairwise drift rows to: {args.out}")

if __name__ == "__main__":
    main()


'''
python neuron_epochwise_drift.py \
    --base /pscratch/sd/l/lsx/runs/opt67b_fsdp_gsm8k \
    --layer 0 \
    --e-start 1 \
    --e-end 2 \
    --pattern checkpoint-epoch-{e} \
    --out test.csv

'''