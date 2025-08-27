#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune OPT-13B on GSM8K and checkpoint per-neuron hot/cold stats.
Writes compressed .npz snapshots containing:
  - G[g]: EMA of |grad| per neuron (fc1 rows), shape [layers, neurons]
  - H[g]: hit counts (> threshold), same shape
  - A[g]: EMA of activations ReLU(fc1(x)), same shape

Usage (single node, multiple GPUs OK):
  python finetune_opt_checkpoint.py --model facebook/opt-13b --epochs 1 \
    --seq_len 2048 --per_device_batch 1 --grad_accum 8 \
    --unfreeze_n 12 --targets mlp,mha \
    --max_train_samples 2000 \
    --log_every 50 --snap_every 200 \
    --outdir runs/opt13b_hotcold

Dependencies:
  python 3.10, torch 2.2/2.3, transformers >= 4.40, datasets, tqdm, numpy, matplotlib (for later plotting)
"""

import os, re, math, argparse, random
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_dist(): return dist.is_available() and dist.is_initialized()
def rank():    return dist.get_rank() if is_dist() else 0
def is_main(): return rank() == 0

# -------------------------
# Data: GSM8K (train) + simple 2-way grouping
# -------------------------
def extract_gsm_answer_tail(ans: str) -> str:
    return ans.split("####")[-1].strip() if "####" in ans else ans.strip()

def group_gsm_question(q: str) -> int:
    # Group 0: arithmetic/rate/percent-ish; Group 1: other
    kw = [
        "sum","add","+","minus","-","*","times","product","difference",
        "minutes","hours","speed","rate","percent","percentage","discount",
        "ratio","average","per","each","total","cost","price","profit","loss"
    ]
    ql = q.lower()
    return 0 if any(k in ql for k in kw) else 1

def build_text(q: str, a: str) -> str:
    return f"Question:\n{q}\n\nAnswer:\n{a}\n"

def load_gsm8k_examples(max_train_samples: int = 0, seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]
    if max_train_samples:
        train = train.shuffle(seed).select(range(min(max_train_samples, len(train))))
    examples = []
    for ex in train:
        q = ex["question"].strip()
        a = extract_gsm_answer_tail(ex["answer"])
        grp = group_gsm_question(q)   # 0 or 1
        examples.append({"text": build_text(q, a), "group": grp})
    return examples

def make_collate_fn(tokenizer, max_seq_len: int):
    eos = tokenizer.eos_token or tokenizer.pad_token
    def collate(batch):
        texts = [b["text"] + eos for b in batch]
        enc = tokenizer(
            texts, return_tensors="pt", truncation=True, padding="longest",
            max_length=max_seq_len
        )
        enc["labels"] = enc["input_ids"].clone()
        enc["group"] = torch.tensor([b["group"] for b in batch], dtype=torch.long)
        return enc
    return collate

# -------------------------
# OPT-13B module discovery & unfreeze
# -------------------------
def discover_opt_fc1(model: nn.Module) -> List[Tuple[int,str,nn.Linear]]:
    fc1_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = re.match(r"model\.decoder\.layers\.(\d+)\.fc1$", name)
            if m:
                fc1_modules.append((int(m.group(1)), name, mod))
    fc1_modules.sort(key=lambda x: x[0])
    return fc1_modules

def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)

def unfreeze_last_n(m: nn.Module, targets: str, last_n: int):
    want_mlp = "mlp" in targets
    want_mha = "mha" in targets

    # find number of layers
    max_layer = -1
    for name, _ in m.named_modules():
        mm = re.match(r"model\.decoder\.layers\.(\d+)", name)
        if mm: max_layer = max(max_layer, int(mm.group(1)))
    total_layers = max_layer + 1
    low = max(0, total_layers - last_n)
    hi  = total_layers

    for name, mod in m.named_modules():
        mm = re.match(r"model\.decoder\.layers\.(\d+)\.(.*)$", name)
        if not mm: continue
        lid = int(mm.group(1))
        if not (low <= lid < hi): continue
        if want_mlp and (name.endswith(".fc1") or name.endswith(".fc2")):
            if isinstance(mod, nn.Linear):
                if mod.weight is not None: mod.weight.requires_grad_(True)
                if mod.bias   is not None: mod.bias.requires_grad_(True)
        if want_mha and (".self_attn." in name):
            if isinstance(mod, nn.Linear):
                if mod.weight is not None: mod.weight.requires_grad_(True)
                if mod.bias   is not None: mod.bias.requires_grad_(True)

# -------------------------
# Hot/Cold logger (per-neuron stats)
# -------------------------
class HotColdLogger:
    def __init__(self, model, groups: int, ema_alpha: float,
                 hit_t: float, act_ema_alpha: float, outdir: str):
        self.groups = groups
        self.ema_alpha = ema_alpha
        self.hit_t = hit_t
        self.act_ema_alpha = act_ema_alpha
        self.outdir = outdir

        self.fc1_modules = discover_opt_fc1(model)
        assert self.fc1_modules, "No OPT fc1 modules found."
        self.num_layers = self.fc1_modules[-1][0] + 1
        self.neurons = self.fc1_modules[0][2].out_features
        for _, _, m in self.fc1_modules:
            assert m.out_features == self.neurons, "Varying FFN sizes not supported."

        device = next(model.parameters()).device
        self.G = [torch.zeros((self.num_layers, self.neurons), dtype=torch.float32, device=device)
                  for _ in range(groups)]
        self.H = [torch.zeros((self.num_layers, self.neurons), dtype=torch.int32,   device=device)
                  for _ in range(groups)]
        self.A = [torch.zeros((self.num_layers, self.neurons), dtype=torch.float32, device=device)
                  for _ in range(groups)]

        self._curr_group = torch.tensor(0, device=device)

        # Register gradient & forward hooks
        for lid, _, mod in self.fc1_modules:
            mod.weight.register_hook(self._make_grad_hook(lid))
            mod.register_forward_hook(self._make_forward_hook(lid))

    def set_group(self, g: int): self._curr_group.fill_(int(g))
    def get_group(self) -> int:  return int(self._curr_group.item())

    def _make_grad_hook(self, layer_idx: int):
        def _hook(grad: torch.Tensor):
            # grad shape: [out_features, in_features]
            with torch.no_grad():
                g = self.get_group()
                row = grad.abs().mean(dim=1).to(torch.float32)  # [neurons]
                self.G[g][layer_idx].mul_(1 - self.ema_alpha).add_(self.ema_alpha * row)
                self.H[g][layer_idx] += (row > self.hit_t).to(torch.int32)
        return _hook

    def _make_forward_hook(self, layer_idx: int):
        def _fwd(module, inp, out):
            # out is fc1(x); apply ReLU to mirror OPT’s FFN
            with torch.no_grad():
                g = self.get_group()
                y = torch.relu(out)
                if y.dim() == 3:  # [B,T,H]
                    row = y.abs().mean(dim=(0,1)).to(torch.float32)
                else:             # [*,H]
                    row = y.abs().mean(dim=0).to(torch.float32)
                self.A[g][layer_idx].mul_(1 - self.act_ema_alpha).add_(self.act_ema_alpha * row)
        return _fwd

    def reduce_stats(self):
        if not is_dist(): return
        for g in range(self.groups):
            dist.all_reduce(self.G[g], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.H[g], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.A[g], op=dist.ReduceOp.SUM)

    def save_snapshot(self, step: int):
        if not is_main(): return
        os.makedirs(self.outdir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.outdir, f"hotcold_step{step:07d}.npz"),
            step=step, layers=self.num_layers, neurons=self.neurons, groups=self.groups,
            G=[t.detach().cpu().numpy() for t in self.G],
            H=[t.detach().cpu().numpy() for t in self.H],
            A=[t.detach().cpu().numpy() for t in self.A],
        )

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="facebook/opt-13b",
                    help="HF model id; default OPT-13B (ReLU FFN)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--per_device_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--snap_every", type=int, default=200)
    ap.add_argument("--max_train_samples", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="runs/opt13b_hotcold")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--ema_alpha", type=float, default=0.1)
    ap.add_argument("--hit_t", type=float, default=1e-5)
    ap.add_argument("--act_ema_alpha", type=float, default=0.1)
    ap.add_argument("--unfreeze_n", type=int, default=12,
                    help="unfreeze last N decoder layers")
    ap.add_argument("--targets", type=str, default="mlp,mha",
                    help="comma list among: mlp, mha")
    args = ap.parse_args()

    set_seed(args.seed)

    # Initialize distributed if launched via torchrun/sbatch
    if "RANK" in os.environ or "SLURM_PROCID" in os.environ:
        dist.init_process_group(backend="nccl")

    if is_main():
        os.makedirs(args.outdir, exist_ok=True)
        print("Loading GSM8K...")

    # 1) Data
    examples = load_gsm8k_examples(max_train_samples=args.max_train_samples, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    collate = make_collate_fn(tokenizer, args.seq_len)
    loader = DataLoader(examples, batch_size=args.per_device_batch,
                        shuffle=True, collate_fn=collate, drop_last=False)

    # 2) Model
    if is_main():
        print(f"Loading model: {args.model} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.gradient_checkpointing_enable()

    # 3) Freeze all, then unfreeze last-N layers for requested targets
    freeze_all(model)
    unfreeze_last_n(model, targets=args.targets, last_n=args.unfreeze_n)

    # 4) Optimizer & schedule
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    total_steps = args.epochs * math.ceil(len(loader) / max(1, args.grad_accum))
    warmup_steps = max(10, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5) Logger (hot/cold stats)
    logger = HotColdLogger(
        model=model, groups=args.groups, ema_alpha=args.ema_alpha,
        hit_t=args.hit_t, act_ema_alpha=args.act_ema_alpha, outdir=args.outdir
    )

    device = next(model.parameters()).device
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_main(),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in pbar:
            logger.set_group(int(batch["group"][0]))  # one group per microbatch

            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                loss = out.loss / args.grad_accum

            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (global_step + 1) % args.log_every == 0:
                logger.reduce_stats()

            if (global_step + 1) % args.snap_every == 0:
                logger.save_snapshot(global_step + 1)

            global_step += 1
            if is_main():
                pbar.set_postfix({"loss": f"{out.loss.item():.4f}", "gs": global_step})

    # Final snapshot no matter what
    logger.save_snapshot(global_step)
    if is_main():
        print("Done. Snapshots in:", args.outdir)

if __name__ == "__main__":
    main()