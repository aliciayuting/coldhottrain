#!/usr/bin/env python3
import os
import re
import sys
from typing import Dict

def load_state_dict(ckpt_dir: str) -> Dict[str, "Tensor"]:
    sft = os.path.join(ckpt_dir, "model.safetensors")
    binf = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(sft):
        from safetensors.torch import load_file
        return load_file(sft)
    if os.path.exists(binf):
        import torch
        return torch.load(binf, map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {ckpt_dir}")

def iter_ckpt_dirs(root: str):
    """Yield checkpoint dirs. If root itself is a checkpoint, yield it; else scan step* subdirs."""
    if os.path.exists(os.path.join(root, "model.safetensors")) or os.path.exists(os.path.join(root, "pytorch_model.bin")):
        yield root
        return
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if name.startswith("step") and os.path.isdir(p):
            if (os.path.exists(os.path.join(p, "model.safetensors"))
                or os.path.exists(os.path.join(p, "pytorch_model.bin"))):
                yield p

def find_first_layer_wk_key(keys):
    """Prefer common layer-0 WK names; else smallest layer index that matches."""
    candidates = [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.00.self_attn.k_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.self_attn.Wk.weight",
        "model.layers.0.attention.Wk.weight",
    ]
    keyset = set(keys)
    for c in candidates:
        if c in keyset:
            return c
    pat = re.compile(r"^model\.layers\.(\d+)\.(?:self_attn|attention)\.(?:k_proj|Wk)\.weight$")
    best = None
    best_layer = 10**9
    for k in keys:
        m = pat.match(k)
        if m:
            layer = int(m.group(1))
            if layer < best_layer:
                best_layer = layer
                best = k
    return best

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_wk_first10.py <checkpoint_dir or weight_dump_root>", file=sys.stderr)
        sys.exit(1)

    root = sys.argv[1]
    any_found = False
    for ckpt in iter_ckpt_dirs(root):
        any_found = True
        try:
            sd = load_state_dict(ckpt)
        except Exception as e:
            print(f"[WARN] {ckpt}: {e}")
            continue

        wk_key = find_first_layer_wk_key(sd.keys())
        if wk_key is None:
            print(f"{ckpt}: WK key not found")
            continue

        t = sd[wk_key]
        try:
            # Avoid importing torch unless we have to—safetensors returns torch tensors anyway.
            import torch  # local import to keep script lightweight
            flat10 = t.reshape(-1)[:100].to(torch.float32).tolist()
            print(f"{ckpt}: {wk_key} shape={tuple(t.shape)} dtype={t.dtype} first10={flat10}")
        except Exception as e:
            print(f"{ckpt}: {wk_key} <unable to read> reason={e}")

    if not any_found:
        print("[INFO] No checkpoints found at the given path.")

if __name__ == "__main__":
    main()