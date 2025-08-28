#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze OPT checkpoints: weight deltas and activation (hot/cold) changes

Outputs (under --outdir):
  weights/
    delta_{A}->{B}/
      fc1_row_delta.npz  # [layers, neurons] L1/L2 row change for fc1.weight
      fc1_row_delta.png  # heatmap (log1p)
  activations/
    act_epoch{E}.npz     # A[g][layer, neuron] or A_all[layer, neuron]
    act_epoch{E}.png     # heatmap (log1p, per-layer norm/clip options)
    delta_{A}->{B}.png   # activation deltas (bwr, symmetric, clipped)
    timeseries_layer{L}.png  # optional top-k neuron curves across epochs

Usage (common):
  python analyze_checkpoints.py \
    --ckpt_root /path/to/runs/opt17b_fsdp_gsm8k \
    --ckpt_pattern checkpoint-epoch-* \
    --include_final \
    --outdir /path/to/runs/opt17b_fsdp_gsm8k/analysis \
    --samples 256 --seq_len 512 \
    --per_group \
    --clip_pct 1 99 --log1p --per_layer_norm

Tip: run with small --samples first (128–512) to keep it quick.
"""

import os, re, glob, math, argparse, shutil
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# -----------------------
# Utilities
# -----------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def extract_epoch_num(path: str) -> int:
    # matches .../checkpoint-epoch-3, returns 3
    m = re.search(r"checkpoint-epoch-(\d+)$", path.rstrip("/"))
    return int(m.group(1)) if m else -1

def sorted_checkpoints(root: str, pattern: str, include_final: bool) -> List[str]:
    paths = sorted(glob.glob(os.path.join(root, pattern)), key=extract_epoch_num)
    if include_final:
        final = os.path.join(root, "checkpoint-final")
        if os.path.isdir(final):
            paths.append(final)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under {root} with pattern '{pattern}'.")
    return paths

def discover_fc1(model: nn.Module) -> List[Tuple[int, nn.Linear]]:
    fc1 = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            mo = re.match(r"model\.decoder\.layers\.(\d+)\.fc1$", name)
            if mo:
                fc1.append((int(mo.group(1)), m))
    fc1.sort(key=lambda x: x[0])
    if not fc1:
        raise RuntimeError("Could not find any 'model.decoder.layers.{L}.fc1' modules.")
    return fc1

def percentile_clip(x: np.ndarray, lo=None, hi=None):
    if lo is None and hi is None:
        return x, None, None
    a = x.copy()
    if lo is not None:
        vlo = np.percentile(a, lo)
        a = np.maximum(a, vlo)
    if hi is not None:
        vhi = np.percentile(a, hi)
        a = np.minimum(a, vhi)
    return a, lo, hi

def per_layer_minmax(x: np.ndarray):
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    den = np.maximum(mx - mn, 1e-12)
    return (x - mn) / den

def plot_heat(mat: np.ndarray, title: str, outpath: str,
              cmap="magma", vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest",
                    cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Neuron index")
    plt.ylabel("Layer index")
    plt.title(title)
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------
# Weight deltas (fc1 rows)
# -----------------------

def row_change(Wa: torch.Tensor, Wb: torch.Tensor, norm: str) -> torch.Tensor:
    # Wa, Wb: [out, in]
    d = (Wb - Wa).abs()
    if norm == "l1":
        return d.sum(dim=1)  # [out]
    else:
        return (d.square().sum(dim=1)).sqrt()  # [out], L2

def analyze_weight_deltas(ckpts: List[str], out_root: str, norm: str):
    print("[weights] computing fc1 row deltas ...")
    for a, b in zip(ckpts[:-1], ckpts[1:]):
        tag = f"{os.path.basename(a)}->{os.path.basename(b)}"
        outdir = ensure_dir(os.path.join(out_root, "weights", f"delta_{tag}"))
        print(f"  - {tag}")

        A = AutoModelForCausalLM.from_pretrained(a, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        B = AutoModelForCausalLM.from_pretrained(b, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

        fa = discover_fc1(A); fb = discover_fc1(B)
        assert len(fa) == len(fb), "Layer count mismatch across checkpoints."
        layers = len(fa)
        H = fa[0][1].out_features
        for (_, la), (_, lb) in zip(fa, fb):
            assert la.out_features == lb.out_features and la.in_features == lb.in_features, "fc1 shape mismatch."

        M = np.zeros((layers, H), dtype=np.float32)
        with torch.no_grad():
            for (lid, la), (_, lb) in zip(fa, fb):
                Wa = la.weight.data.float().cpu()
                Wb = lb.weight.data.float().cpu()
                M[lid] = row_change(Wa, Wb, norm=norm).numpy()

        np.savez_compressed(os.path.join(outdir, "fc1_row_delta.npz"),
                            delta=M, norm=norm, layers=layers, neurons=H,
                            ckpt_a=a, ckpt_b=b)

        # plot with log1p for contrast
        plot_heat(np.log1p(M), f"Δfc1.row ({norm})  {tag}",
                  os.path.join(outdir, "fc1_row_delta.png"), cmap="magma")


# -----------------------
# Activations per ckpt
# -----------------------

def extract_tail(ans: str) -> str:
    return ans.split("####")[-1].strip() if "####" in ans else ans.strip()

def group_q(q: str) -> int:
    kw = ["sum","add","+","minus","-","*","times","product","difference",
          "minutes","hours","speed","rate","percent","percentage","discount",
          "ratio","average","per","each","total","cost","price","profit","loss"]
    ql = q.lower()
    return 0 if any(k in ql for k in kw) else 1

def build_text(q, a): return f"Question:\n{q}\n\nAnswer:\n{a}\n"

def load_probe(samples: int, seed: int) -> Tuple[List[str], List[int]]:
    ds = load_dataset("openai/gsm8k", "main")
    tr = ds["train"].shuffle(seed=seed).select(range(samples))
    texts, groups = [], []
    for ex in tr:
        q = ex["question"].strip(); a = extract_tail(ex["answer"])
        texts.append(build_text(q, a))
        groups.append(group_q(q))
    return texts, groups

def probe_activations(ckpt: str, out_root: str, samples: int, seq_len: int,
                      per_group: bool, clip_pct: Tuple[float, float],
                      log1p: bool, per_layer_norm: bool, device: str):
    print(f"[activations] probing {os.path.basename(ckpt)} ...")
    outdir = ensure_dir(os.path.join(out_root, "activations"))

    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    fc1 = discover_fc1(model)
    L = len(fc1); H = fc1[0][1].out_features

    if per_group:
        A = [np.zeros((L, H), dtype=np.float64) for _ in range(2)]
        C = [np.zeros((L,),   dtype=np.int64)   for _ in range(2)]
    else:
        A = [np.zeros((L, H), dtype=np.float64)]
        C = [np.zeros((L,),   dtype=np.int64)]

    hooks = []
    def make_hook(lid):
        def hook(mod, inp, out):
            y = torch.relu(out)          # [B,T,H]
            row = y.float().mean(dim=(0,1))  # [H]
            r = row.detach().cpu().numpy()
            if per_group:
                g = hook.curr_group
                A[g][lid] += r; C[g][lid] += 1
            else:
                A[0][lid] += r; C[0][lid] += 1
        return hook

    for lid, m in fc1:
        h = m.register_forward_hook(make_hook(lid))
        hooks.append(h)

    texts, groups = load_probe(samples=samples, seed=1234)
    B = 8
    with torch.no_grad():
        for i in range(0, len(texts), B):
            chunk = texts[i:i+B]
            batch = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=seq_len).to(device)
            if per_group:
                g = groups[i]  # approximate; if you want strict per-group batches, pre-split texts by group
                for h in hooks: h.curr_group = g
            _ = model(**batch)

    for h in hooks: h.remove()

    # Save and plot
    tag = os.path.basename(ckpt)
    if per_group:
        for g in (0, 1):
            Ag = A[g] / np.maximum(C[g][:, None], 1)
            np.savez_compressed(os.path.join(outdir, f"act_{tag}_group{g}.npz"),
                                A=Ag, layers=L, neurons=H, group=g)
            M = Ag.copy()
            if log1p: M = np.log1p(M)
            if clip_pct[0] is not None or clip_pct[1] is not None:
                M, _, _ = percentile_clip(M, clip_pct[0], clip_pct[1])
            if per_layer_norm: M = per_layer_minmax(M)
            plot_heat(M, f"Mean ReLU(fc1) — {tag} (group {g})",
                      os.path.join(outdir, f"act_{tag}_group{g}.png"), cmap="magma")
    else:
        A0 = A[0] / np.maximum(C[0][:, None], 1)
        np.savez_compressed(os.path.join(outdir, f"act_{tag}.npz"),
                            A=A0, layers=L, neurons=H)
        M = A0.copy()
        if log1p: M = np.log1p(M)
        if clip_pct[0] is not None or clip_pct[1] is not None:
            M, _, _ = percentile_clip(M, clip_pct[0], clip_pct[1])
        if per_layer_norm: M = per_layer_minmax(M)
        plot_heat(M, f"Mean ReLU(fc1) — {tag}",
                  os.path.join(outdir, f"act_{tag}.png"), cmap="magma")

def analyze_activation_deltas(ckpts: List[str], out_root: str,
                              per_group: bool, log1p_delta: bool, clip_pct: Tuple[float, float]):
    print("[activations] computing deltas between checkpoints ...")
    base = os.path.join(out_root, "activations")
    for a, b in zip(ckpts[:-1], ckpts[1:]):
        tag = f"{os.path.basename(a)}->{os.path.basename(b)}"
        print(f"  - {tag}")
        if per_group:
            for g in (0, 1):
                pa = os.path.join(base, f"act_{os.path.basename(a)}_group{g}.npz")
                pb = os.path.join(base, f"act_{os.path.basename(b)}_group{g}.npz")
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    print(f"    skip group {g}: {pa} or {pb} missing")
                    continue
                A1 = np.load(pa)["A"]; A2 = np.load(pb)["A"]
                D = A2 - A1
                if log1p_delta:
                    sign = np.sign(D)
                    D = sign * np.log1p(np.abs(D))
                vmax = np.percentile(np.abs(D), clip_pct[1] if clip_pct[1] is not None else 100.0)
                vmax = max(vmax, 1e-12)
                vmin = -vmax
                plot_heat(D, f"Δ Mean ReLU(fc1) — {tag} (group {g})",
                          os.path.join(base, f"act_delta_{tag}_group{g}.png"),
                          cmap="bwr", vmin=vmin, vmax=vmax)
        else:
            pa = os.path.join(base, f"act_{os.path.basename(a)}.npz")
            pb = os.path.join(base, f"act_{os.path.basename(b)}.npz")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                print(f"    skip: {pa} or {pb} missing")
                continue
            A1 = np.load(pa)["A"]; A2 = np.load(pb)["A"]
            D = A2 - A1
            if log1p_delta:
                sign = np.sign(D)
                D = sign * np.log1p(np.abs(D))
            vmax = np.percentile(np.abs(D), clip_pct[1] if clip_pct[1] is not None else 100.0)
            vmax = max(vmax, 1e-12)
            vmin = -vmax
            plot_heat(D, f"Δ Mean ReLU(fc1) — {tag}",
                      os.path.join(base, f"act_delta_{tag}.png"),
                      cmap="bwr", vmin=vmin, vmax=vmax)

def plot_timeseries(ckpts: List[str], out_root: str, layer: int, topk: int, per_group: bool):
    base = os.path.join(out_root, "activations")
    # build a time series per neuron for the chosen layer based on already-saved A
    def loadA(tag):
        if per_group:
            # default to group 0; feel free to adapt
            p = os.path.join(base, f"act_{tag}_group0.npz")
        else:
            p = os.path.join(base, f"act_{tag}.npz")
        return np.load(p)["A"]
    tags = [os.path.basename(x) for x in ckpts]
    As = [loadA(t) for t in tags if os.path.exists(os.path.join(base, f"act_{t}_group0.npz" if per_group else f"act_{t}.npz"))]
    if not As:
        print("[timeseries] activation files not found; run probe first.")
        return
    steps = list(range(len(As)))
    series = np.stack([A[layer] for A in As], axis=0)  # [T, neurons]
    mean_over_time = series.mean(axis=0)
    idx = np.argsort(mean_over_time)[-topk:]
    plt.figure(figsize=(10, 6))
    for j in idx:
        plt.plot(steps, series[:, j], alpha=0.85)
    plt.xlabel("Checkpoint index")
    plt.ylabel(f"Mean ReLU(fc1) — layer {layer} (group {0 if per_group else 'all'})")
    plt.title(f"Top-{topk} neurons over checkpoints")
    ensure_dir(base)
    plt.savefig(os.path.join(base, f"timeseries_layer{layer}_top{topk}.png"), dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True, help="Folder containing checkpoint-epoch-* dirs")
    ap.add_argument("--ckpt_pattern", default="checkpoint-epoch-*", help="Glob under ckpt_root")
    ap.add_argument("--include_final", action="store_true", help="Also include checkpoint-final if present")
    ap.add_argument("--outdir", required=True, help="Where to write analysis outputs")

    # weights
    ap.add_argument("--weight_norm", choices=["l1","l2"], default="l2")

    # activations (probe)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--per_group", action="store_true")

    # plotting controls
    ap.add_argument("--clip_pct", nargs=2, type=float, default=[1.0, 99.0], help="percentile clip for ABS/activations")
    ap.add_argument("--log1p", action="store_true", help="log1p absolute activations before plotting")
    ap.add_argument("--log1p_delta", action="store_true", help="log1p on activation deltas (signed)")
    ap.add_argument("--per_layer_norm", action="store_true")

    # optional timeseries
    ap.add_argument("--timeseries", action="store_true")
    ap.add_argument("--ts_layer", type=int, default=0)
    ap.add_argument("--ts_topk", type=int, default=16)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted_checkpoints(args.ckpt_root, args.ckpt_pattern, include_final=args.include_final)
    print("[*] checkpoints:", [os.path.basename(p) for p in ckpts])

    ensure_dir(args.outdir)

    # 1) Weight deltas
    analyze_weight_deltas(ckpts, args.outdir, norm=args.weight_norm)

    # 2) Activations (per checkpoint)
    for ck in ckpts:
        probe_activations(
            ckpt=ck, out_root=args.outdir, samples=args.samples, seq_len=args.seq_len,
            per_group=args.per_group, clip_pct=(args.clip_pct[0], args.clip_pct[1]),
            log1p=args.log1p, per_layer_norm=args.per_layer_norm, device=device
        )

    # 3) Activation deltas across checkpoints
    analyze_activation_deltas(ckpts, args.outdir, per_group=args.per_group,
                              log1p_delta=args.log1p_delta, clip_pct=(args.clip_pct[0], args.clip_pct[1]))

    # 4) Optional time series
    if args.timeseries:
        plot_timeseries(ckpts, args.outdir, layer=args.ts_layer, topk=args.ts_topk, per_group=args.per_group)

    print("[✓] Analysis complete:", args.outdir)


if __name__ == "__main__":
    main()