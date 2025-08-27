#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, math, argparse, json, random
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
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main():
    return get_rank() == 0

# -------------------------
# Data
# -------------------------
def extract_gsm_answer_tail(ans: str) -> str:
    return ans.split("####")[-1].strip() if "####" in ans else ans.strip()

def simple_group_gsm(q: str) -> int:
    # Group 0: arithmetic/rates/percent; Group 1: other
    kw = [
        "sum","add","+","minus","-","*","times","product","difference",
        "minutes","hours","speed","rate","percent","percentage","discount",
        "ratio","average","per","each","total","cost","price","profit","loss"
    ]
    ql = q.lower()
    return 0 if any(k in ql for k in kw) else 1

def build_text(q: str, a: str) -> str:
    return f"Question:\n{q}\n\nAnswer:\n{a}\n"

def load_gsm8k(max_train_samples: int = 0, seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]
    if max_train_samples:
        train = train.shuffle(seed).select(range(min(max_train_samples, len(train))))
    examples = []
    for ex in train:
        q = ex["question"].strip()
        a = extract_gsm_answer_tail(ex["answer"])
        grp = simple_group_gsm(q)   # 0 or 1
        examples.append({"text": build_text(q, a), "group": grp})
    return examples

# -------------------------
# Collate
# -------------------------
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
# Model surgery (OPT-13B)
# -------------------------
def discover_opt_ffn_fc1(model: nn.Module) -> List[Tuple[int,str,nn.Linear]]:
    """
    Find OPT decoder FFN fc1 modules.
    Returns list of (layer_idx, name, module).
    """
    fc1_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = re.match(r"model\.decoder\.layers\.(\d+)\.fc1$", name)
            if m:
                lid = int(m.group(1))
                fc1_modules.append((lid, name, mod))
    fc1_modules.sort(key=lambda x: x[0])
    return fc1_modules

def discover_opt_attention_projs(model: nn.Module):
    """Return dict layer_idx -> list of (name, module) for attention {q,k,v,o}_proj."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = re.match(r"model\.decoder\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|out_proj))$", name)
            if m:
                lid = int(m.group(1))
                out.setdefault(lid, []).append((name, mod))
    return out

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze_last_n(model: nn.Module, targets: str, last_n: int):
    """
    targets: "mlp", "mha", or "mha,mlp"
    last_n: last N decoder layers
    """
    want_mlp = "mlp" in targets
    want_mha = "mha" in targets

    # OPT layers total:
    # discover total from module names
    max_layer = -1
    for name, _ in model.named_modules():
        m = re.match(r"model\.decoder\.layers\.(\d+)", name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    total_layers = max_layer + 1

    low = max(0, total_layers - last_n)
    hi = total_layers

    for name, mod in model.named_modules():
        m = re.match(r"model\.decoder\.layers\.(\d+)\.(.*)$", name)
        if not m:
            continue
        lid = int(m.group(1))
        if not (low <= lid < hi):
            continue
        if want_mlp and (name.endswith(".fc1") or name.endswith(".fc2")):
            if isinstance(mod, nn.Linear):
                if mod.weight is not None: mod.weight.requires_grad_(True)
                if mod.bias   is not None: mod.bias.requires_grad_(True)
        if want_mha and (".self_attn." in name):
            if isinstance(mod, nn.Linear):
                if mod.weight is not None: mod.weight.requires_grad_(True)
                if mod.bias   is not None: mod.bias.requires_grad_(True)

# -------------------------
# Hot/Cold logging hooks
# -------------------------
class HotColdLogger:
    """
    Tracks per-group, per-layer, per-neuron (fc1 rows) gradient L1 EMA,
    hit counts, and optional activation EMA (ReLU(fc1(x))).
    """
    def __init__(self, model, groups: int, ema_alpha: float,
                 hit_t: float, act_ema_alpha: float,
                 outdir: str):
        self.groups = groups
        self.ema_alpha = ema_alpha
        self.hit_t = hit_t
        self.act_ema_alpha = act_ema_alpha
        self.outdir = outdir

        self.fc1_modules = discover_opt_ffn_fc1(model)
        assert self.fc1_modules, "No OPT fc1 modules found."
        self.num_layers = self.fc1_modules[-1][0] + 1
        self.neurons = self.fc1_modules[0][2].out_features
        for _, _, m in self.fc1_modules:
            assert m.out_features == self.neurons, "Varying FFN sizes not supported."

        device = next(model.parameters()).device
        # Accumulators (on device for speed; we all-reduce in-place)
        self.G = [torch.zeros((self.num_layers, self.neurons), dtype=torch.float32, device=device)
                  for _ in range(groups)]
        self.H = [torch.zeros((self.num_layers, self.neurons), dtype=torch.int32, device=device)
                  for _ in range(groups)]
        self.A = [torch.zeros((self.num_layers, self.neurons), dtype=torch.float32, device=device)
                  for _ in range(groups)]

        # "current group" per microbatch
        self._curr_group = torch.tensor(0, device=device)

        # Register hooks
        for lid, _, mod in self.fc1_modules:
            mod.weight.register_hook(self._make_grad_hook(lid))
            mod.register_forward_hook(self._make_forward_hook(lid))

    def set_group(self, g: int):
        self._curr_group.fill_(int(g))

    def get_group(self) -> int:
        return int(self._curr_group.item())

    def _make_grad_hook(self, layer_idx: int):
        def _hook(grad: torch.Tensor):
            # grad: [out_features, in_features]
            with torch.no_grad():
                g = self.get_group()
                row = grad.abs().mean(dim=1).to(torch.float32)  # [neurons]
                self.G[g][layer_idx].mul_(1 - self.ema_alpha).add_(self.ema_alpha * row)
                self.H[g][layer_idx] += (row > self.hit_t).to(torch.int32)
        return _hook

    def _make_forward_hook(self, layer_idx: int):
        def _fwd(module, inp, out):
            # out is fc1(x); apply ReLU to mirror OPT FFN
            with torch.no_grad():
                g = self.get_group()
                y = torch.relu(out)
                if y.dim() == 3:
                    row = y.abs().mean(dim=(0,1)).to(torch.float32)  # [H]
                else:
                    row = y.abs().mean(dim=0).to(torch.float32)      # [H]
                self.A[g][layer_idx].mul_(1 - self.act_ema_alpha).add_(self.act_ema_alpha * row)
        return _fwd

    def reduce_stats(self):
        if not is_dist():
            return
        for g in range(self.groups):
            dist.all_reduce(self.G[g], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.H[g], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.A[g], op=dist.ReduceOp.SUM)

    def save_snapshot(self, step: int):
        if not is_main():
            return
        snap = {
            "step": step,
            "layers": self.num_layers,
            "neurons": self.neurons,
            "groups": self.groups,
            "G": [t.detach().cpu().numpy() for t in self.G],
            "H": [t.detach().cpu().numpy() for t in self.H],
            "A": [t.detach().cpu().numpy() for t in self.A],
        }
        os.makedirs(self.outdir, exist_ok=True)
        np.savez_compressed(os.path.join(self.outdir, f"hotcold_step{step:07d}.npz"), **snap)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="facebook/opt-13b",
                    help="HF model id; default is OPT-13B (ReLU FFN)")
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
    ap.add_argument("--max_train_samples", type=int, default=2000,
                    help="Subset GSM8K for a quick run; set 0 to use all")
    ap.add_argument("--outdir", type=str, default="runs/opt13b_hotcold")
    ap.add_argument("--groups", type=int, default=2, help="number of groups (we use 2 for GSM8K)")
    ap.add_argument("--ema_alpha", type=float, default=0.1)
    ap.add_argument("--hit_t", type=float, default=1e-5)
    ap.add_argument("--act_ema_alpha", type=float, default=0.1)
    ap.add_argument("--unfreeze_scope", type=str, default="last_n_layers",
                    choices=["last_n_layers"])
    ap.add_argument("--unfreeze_n", type=int, default=12)
    ap.add_argument("--targets", type=str, default="mlp,mha",
                    help="comma list: mlp, mha")
    args = ap.parse_args()

    set_seed(args.seed)

    # Distributed init if launched with torchrun
    if "RANK" in os.environ or "SLURM_PROCID" in os.environ:
        dist.init_process_group(backend="nccl")

    if is_main():
        os.makedirs(args.outdir, exist_ok=True)

    # 1) Data (GSM8K train only for now)
    if is_main():
        print("Loading GSM8K...")
    examples = load_gsm8k(max_train_samples=args.max_train_samples, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    collate = make_collate_fn(tokenizer, args.seq_len)
    # Shard data manually for DDP if desired, but simplest is to rely on DataLoader shuffle
    loader = DataLoader(examples, batch_size=args.per_device_batch,
                        shuffle=True, collate_fn=collate, drop_last=False)

    # 2) Model
    if is_main():
        print(f"Loading model: {args.model} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",           # shards across visible GPUs on a single node
    )
    model.gradient_checkpointing_enable()

    # 3) Freeze all, then unfreeze last N layers of selected targets
    freeze_all(model)
    unfreeze_last_n(model, targets=args.targets, last_n=args.unfreeze_n)

    # 4) Optimizer/scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    total_steps = args.epochs * math.ceil(len(loader) / max(1, args.grad_accum))
    warmup = max(10, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    # 5) Hot/Cold Logger
    logger = HotColdLogger(
        model=model,
        groups=args.groups,
        ema_alpha=args.ema_alpha,
        hit_t=args.hit_t,
        act_ema_alpha=args.act_ema_alpha,
        outdir=args.outdir
    )

    device = next(model.parameters()).device
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_main(),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in pbar:
            # single-group per microbatch (batch size often 1 for long seqs)
            logger.set_group(int(batch["group"][0]))

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

    if is_main():
        print("Done. Snapshots written to:", args.outdir)

if __name__ == "__main__":
    main()