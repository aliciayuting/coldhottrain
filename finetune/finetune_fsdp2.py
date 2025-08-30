#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FSDP training of OPT-1.7B on GSM8K (openai/gsm8k), 4 GPUs.

Features
    - PyTorch FSDP (FULL_SHARD ~ ZeRO-3) with bf16 mixed precision
    - Gradient checkpointing
    - DistributedSampler
    - Simple causal-LM loss on "Question ... Answer ..." strings
    - Full-state checkpoint save (rank0-only) compatible with HF `from_pretrained`
    - Flexible snapshot scheduling (micro/epoch/exponential) for hot/cold stats

Usage
    torchrun --standalone --nproc_per_node=4 finetune_fsdp2.py \
        --model facebook/opt-1.7b \
        --epochs 50 --seq_len 1024 \
        --per_device_batch 1 --grad_accum 8 \
        --lr 2e-5 --wd 0.05 --warmup_ratio 0.03 \
        --snap_strategy exp --snap_exp_start 200 --snap_exp_growth 1.5 --snap_exp_max 5000 \
        --outdir ./runs/opt17b_fsdp_gsm8k
"""

import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

import functools
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main() -> bool:
    return rank() == 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


# -------------------------
# Data (GSM8K)
# -------------------------

def group_gsm_question(q: str) -> int:
    """Group 0: arithmetic/rate/percent-ish; Group 1: other."""
    kw = [
        "sum","add","+","minus","-","*","times","product","difference",
        "minutes","hours","speed","rate","percent","percentage","discount",
        "ratio","average","per","each","total","cost","price","profit","loss"
    ]
    ql = q.lower()
    return 0 if any(k in ql for k in kw) else 1


def extract_gsm_answer_tail(ans: str) -> str:
    # GSM8K answers end with "#### 1234"
    return ans.split("####")[-1].strip() if "####" in ans else ans.strip()


def build_text(q: str, a: str) -> str:
    return f"Question:\n{q}\n\nAnswer:\n{a}\n"


def load_gsm8k_texts(max_train_samples: int = 0, seed: int = 42) -> List[Dict[str, str]]:
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]
    if max_train_samples:
        train = train.shuffle(seed=seed).select(range(min(max_train_samples, len(train))))
    items: List[Dict[str, str]] = []
    for ex in train:
        q = ex["question"].strip()
        a = extract_gsm_answer_tail(ex["answer"])
        items.append({
            "text": build_text(q, a),
            "group": group_gsm_question(q),
        })
    return items


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_seq_len: int

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token or self.tokenizer.pad_token or ""
        texts = [b["text"] + eos for b in batch]
        groups = [int(b["group"]) for b in batch]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.max_seq_len,
        )
        enc["labels"] = enc["input_ids"].clone()
        enc["group"] = torch.tensor(groups, dtype=torch.long)
        return enc


# -------------------------
# Model helpers
# -------------------------

def enable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_last_n_decoder_layers(model: nn.Module, n: int) -> None:
    # OPT layers live under model.decoder.layers.{L}.*
    # Unfreeze the last n decoder layers
    max_layer = -1
    for name, _ in model.named_modules():
        if name.startswith("model.decoder.layers."):
            try:
                lid = int(name.split(".")[3])
                max_layer = max(max_layer, lid)
            except Exception:
                pass
    if max_layer < 0 or n <= 0:
        return
    low, hi = max(0, max_layer + 1 - n), max_layer + 1
    for name, module in model.named_modules():
        if name.startswith("model.decoder.layers."):
            try:
                lid = int(name.split(".")[3])
            except Exception:
                continue
            if low <= lid < hi:
                for p in module.parameters(recurse=True):
                    p.requires_grad_(True)


# -------------------------
# Hot/Cold neuron logging (per-layer fc1 rows)
# -------------------------

def discover_opt_fc1(model: nn.Module):
    fc1_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = re.match(r"model\.decoder\.layers\.(\d+)\.fc1$", name)
            if m:
                fc1_modules.append((int(m.group(1)), name, mod))
    fc1_modules.sort(key=lambda x: x[0])
    return fc1_modules

class HotColdLogger:
    def __init__(self, model: nn.Module, groups: int, ema_alpha: float,
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

        # Allocate on CPU first; move lazily on first hook call to match module's device
        self._buffers_device = torch.device("cpu")
        self.G = [torch.zeros((self.num_layers, self.neurons), dtype=torch.float32, device=self._buffers_device)
                  for _ in range(groups)]
        self.H = [torch.zeros((self.num_layers, self.neurons), dtype=torch.int32,   device=self._buffers_device)
                  for _ in range(groups)]
        self.A = [torch.zeros((self.num_layers, self.neurons), dtype=torch.float32, device=self._buffers_device)
                  for _ in range(groups)]
        self._curr_group = torch.tensor(0, device=self._buffers_device)

        # Register hooks on fc1 immediately (before FSDP wrapping)
        for lid, _, mod in self.fc1_modules:
            if hasattr(mod, 'weight') and isinstance(mod.weight, torch.Tensor):
                mod.weight.register_hook(self._make_grad_hook(lid))
            mod.register_forward_hook(self._make_forward_hook(lid))
    def _ensure_device(self, device: torch.device):
        """Move internal EMA/Hit buffers to the given device exactly once."""
        if device == self._buffers_device:
            return
        # Move group selector
        self._curr_group = self._curr_group.to(device)
        # Move stats buffers
        for g in range(self.groups):
            self.G[g] = self.G[g].to(device)
            self.H[g] = self.H[g].to(device)
            self.A[g] = self.A[g].to(device)
        self._buffers_device = device

    def set_group(self, g: int):
        self._curr_group.fill_(int(g))

    def get_group(self) -> int:
        return int(self._curr_group.item())

    def _make_grad_hook(self, layer_idx: int):
        def _hook(grad: torch.Tensor):
            with torch.no_grad():
                # Ensure buffers are on the grad's device (e.g., cuda:local_rank)
                self._ensure_device(grad.device)
                g = self.get_group()
                row = grad.abs().mean(dim=1).to(torch.float32)
                self.G[g][layer_idx].mul_(1 - self.ema_alpha).add_(self.ema_alpha * row)
                self.H[g][layer_idx] += (row > self.hit_t).to(torch.int32)
        return _hook

    def _make_forward_hook(self, layer_idx: int):
        def _fwd(module, inp, out):
            with torch.no_grad():
                # out resides on the active module's device; move buffers if needed
                self._ensure_device(out.device)
                g = self.get_group()
                y = torch.relu(out)
                if y.dim() == 3:
                    row = y.abs().mean(dim=(0,1)).to(torch.float32)
                else:
                    row = y.abs().mean(dim=0).to(torch.float32)
                self.A[g][layer_idx].mul_(1 - self.act_ema_alpha).add_(self.act_ema_alpha * row)
        return _fwd

    def reduce_stats(self):
        if not is_dist():
            return
        # Ensure buffers are on CUDA for NCCL all_reduce (some ranks may not have fired hooks yet)
        if self._buffers_device.type != "cuda":
            try:
                dev = torch.device(f"cuda:{local_rank()}")
            except Exception:
                dev = torch.device("cuda")
            self._ensure_device(dev)
        for g in range(self.groups):
            dist.all_reduce(self.G[g], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.H[g], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.A[g], op=dist.ReduceOp.SUM)

    def save_snapshot(self, step: int, epoch: int = None, tag_snapshots: bool = False):
        if not is_main():
            return
        os.makedirs(self.outdir, exist_ok=True)
        # filename tag
        if tag_snapshots and epoch is not None:
            fname = f"hotcold_ep{epoch:04d}_step{step:08d}.npz"
        else:
            fname = f"hotcold_step{step:08d}.npz"
        np.savez_compressed(
            os.path.join(self.outdir, fname),
            step=step,
            epoch=( -1 if epoch is None else int(epoch) ),
            layers=self.num_layers,
            neurons=self.neurons,
            groups=self.groups,
            G=[t.detach().cpu().numpy() for t in self.G],
            H=[t.detach().cpu().numpy() for t in self.H],
            A=[t.detach().cpu().numpy() for t in self.A],
        )


# -------------------------
# Snapshot Scheduler
# -------------------------

class SnapshotScheduler:
    """
    Decide when to call hot/cold snapshot based on:
      - micro: every N optimizer steps
      - epoch: every N epochs
      - exp  : exponentially increasing step interval (start, growth, max)
    """
    def __init__(self, mode:str, every_steps:int, every_epochs:int,
                 exp_start:int, exp_growth:float, exp_max:int):
        self.mode = mode
        self.every_steps = max(1, int(every_steps))
        self.every_epochs = max(1, int(every_epochs))
        self.exp_growth = max(1.0, float(exp_growth))
        self.exp_max = max(1, int(exp_max))
        # internal state
        self.next_step = None
        if mode == "exp":
            self.next_step = int(exp_start)

    def should_snapshot(self, epoch_idx:int, optimizer_step:int) -> bool:
        """
        epoch_idx: 0-based epoch index
        optimizer_step: global count of completed optimizer steps (post-step)
        """
        if self.mode == "micro":
            return (optimizer_step % self.every_steps) == 0
        elif self.mode == "epoch":
            # only snapshot at end-of-epoch; caller should pass in post-epoch
            return ((epoch_idx + 1) % self.every_epochs) == 0
        elif self.mode == "exp":
            if self.next_step is None:
                return False
            return optimizer_step >= self.next_step
        else:
            return False

    def advance_if_needed(self, optimizer_step:int):
        if self.mode != "exp":
            return
        if self.next_step is None:
            return
        while optimizer_step >= self.next_step:
            interval = int(min(self.exp_max, max(1, round((self.next_step if self.next_step > 0 else 1) * self.exp_growth))))
            # if next_step == 0 (unlikely), set to exp_start again
            interval = max(interval, 1)
            self.next_step += interval


# -------------------------
# Training
# -------------------------

def main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.7b")
    parser.add_argument("--outdir", type=str, default="./runs/opt17b_fsdp_gsm8k")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_samples", type=int, default=2000)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--snap_every", type=int, default=200)
    parser.add_argument("--snap_strategy", choices=["micro","epoch","exp"], default="micro",
                        help="micro: every N optimizer steps; epoch: every N epochs; exp: exponentially increasing step interval")
    parser.add_argument("--snap_every_epochs", type=int, default=1,
                        help="if --snap_strategy=epoch, snapshot every N epochs")
    parser.add_argument("--snap_exp_start", type=int, default=200,
                        help="if --snap_strategy=exp, initial step interval")
    parser.add_argument("--snap_exp_growth", type=float, default=1.5,
                        help="if --snap_strategy=exp, multiply interval by this after each snapshot")
    parser.add_argument("--snap_exp_max", type=int, default=5000,
                        help="if --snap_strategy=exp, cap the interval to this many steps")
    parser.add_argument("--tag_snapshots", action="store_true",
                        help="add epoch/step tags to snapshot filenames")
    parser.add_argument("--loss_log_every", type=int, default=50,
                        help="write train loss every N optimizer steps to CSV on rank0")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--hit_t", type=float, default=1e-5)
    parser.add_argument("--act_ema_alpha", type=float, default=0.1)

    parser.add_argument("--unfreeze_last_n", type=int, default=12)  # optional LoRA-ish partial finetune
    parser.add_argument("--clip_norm", type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    set_seed(args.seed)

    # Init distributed + device
    if "RANK" in os.environ or "SLURM_PROCID" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank())

    if is_main():
        print(f"[rank {rank()}] Loading dataset...")
    texts = load_gsm8k_texts(max_train_samples=args.max_train_samples, seed=args.seed)

    if is_main():
        print(f"[rank {rank()}] Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate = Collator(tokenizer, args.seq_len)

    # Dataset / Sampler / Loader
    if is_dist():
        sampler = DistributedSampler(texts, shuffle=True, drop_last=False)
        loader = DataLoader(
            texts,
            batch_size=args.per_device_batch,
            sampler=sampler,
            collate_fn=collate,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            texts,
            batch_size=args.per_device_batch,
            shuffle=True,
            collate_fn=collate,
            pin_memory=True,
        )

    # Load model on CPU; FSDP moves shards to GPUs
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # GC and cache
    model.config.use_cache = False          # silence the warning and avoid incompatibility
    model.gradient_checkpointing_enable()

    # EITHER: train all params (simple & robust)
    for p in model.parameters():
        p.requires_grad_(True)

    # OR: freeze then unfreeze last N decoder layers (if you really want partial finetune)
    # freeze_all(model)
    # unfreeze_last_n_decoder_layers(model, args.unfreeze_last_n)
    # ^ double-check this function actually finds your layers for opt-1.7b

    # Hot/Cold logger: register hooks BEFORE FSDP wrap
    hotcold = HotColdLogger(
        model=model,
        groups=args.groups,
        ema_alpha=args.ema_alpha,
        hit_t=args.hit_t,
        act_ema_alpha=args.act_ema_alpha,
        outdir=args.outdir,
    )

    # Snapshot scheduler
    snap_sched = SnapshotScheduler(
        mode=args.snap_strategy,
        every_steps=args.snap_every,
        every_epochs=args.snap_every_epochs,
        exp_start=args.snap_exp_start,
        exp_growth=args.snap_exp_growth,
        exp_max=args.snap_exp_max,
    )

    # Loss CSV (rank0 only)
    loss_csv_path = os.path.join(args.outdir, "train_loss.csv")
    if is_main() and not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f:
            f.write("epoch,optimizer_step,micro_id,loss\n")

    # ---- FSDP setup (ZeRO-3 equivalent) ----
    mp_conf = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    # # Wrap large submodules (typically decoder blocks) to improve memory/overlap
    # auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=10_000_000)

    # model = FSDP(
    #     model,
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     mixed_precision=mp_conf,
    #     auto_wrap_policy=auto_wrap_policy,
    #     device_id=local_rank(),
    #     use_orig_params=True,
    # )
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={OPTDecoderLayer},
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_conf,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank(),
        use_orig_params=True,
    )

    # Optimizer / Scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(10, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train
    global_step = 0
    optimizer_step = 0
    model.train()
    if is_dist():
        sampler.set_epoch(0)

    for epoch in range(args.epochs):
        if is_dist():
            sampler.set_epoch(epoch)

        pbar = tqdm(total=len(loader), disable=not is_main(), desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            if "group" in batch:
                hotcold.set_group(int(batch["group"][0].item()))
            # Move to local device
            batch = {k: v.cuda(local_rank(), non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if args.clip_norm and args.clip_norm > 0:
                    # Clip only the parameters that require grad
                    grads = [p for p in model.parameters() if p.requires_grad]
                    nn.utils.clip_grad_norm_(grads, args.clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                optimizer_step += 1

                # Periodic distributed reduce (DP only) keyed by micro id
                micro_id = epoch * len(loader) + step + 1
                if micro_id % args.log_every == 0:
                    hotcold.reduce_stats()

                # Snapshot based on scheduler (optimizer step granularity)
                if snap_sched.should_snapshot(epoch_idx=epoch, optimizer_step=optimizer_step):
                    hotcold.save_snapshot(step=optimizer_step, epoch=epoch, tag_snapshots=args.tag_snapshots)
                    snap_sched.advance_if_needed(optimizer_step)

                # Periodic loss CSV on rank0
                if is_main() and (optimizer_step % max(1, args.loss_log_every) == 0):
                    with open(loss_csv_path, "a") as f:
                        f.write(f"{epoch},{optimizer_step},{micro_id},{out.loss.item():.6f}\n")

            if is_main():
                pbar.set_postfix({"loss": f"{out.loss.item():.4f}", "gs": global_step})
                pbar.update(1)

        # End-of-epoch snapshot if using epoch strategy
        if snap_sched.should_snapshot(epoch_idx=epoch, optimizer_step=optimizer_step) and args.snap_strategy == "epoch":
            hotcold.save_snapshot(step=optimizer_step, epoch=epoch, tag_snapshots=args.tag_snapshots)

        if is_main():
            pbar.close()

    # Final snapshot regardless of cadence
    hotcold.save_snapshot(step=optimizer_step, epoch=args.epochs-1, tag_snapshots=args.tag_snapshots)

    # ---- Save full (gathered) HF checkpoint on rank 0 ----
    if is_dist():
        dist.barrier()

    save_dir = os.path.join(args.outdir, "checkpoint-final")
    if is_main():
        os.makedirs(save_dir, exist_ok=True)

    # Gather full-state on rank0 CPU and save in HF format
    cpu_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cpu_cfg):
        full_state = model.state_dict()

    if is_main():
        # Use HF save_pretrained with an explicit state_dict (works with FSDP)
        # Reload a fresh model config to bind save_pretrained; avoids FSDP wrapper in object
        base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        base_model.save_pretrained(save_dir, state_dict=full_state, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        print(f"[rank 0] Saved checkpoint to: {save_dir}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
        


if __name__ == "__main__":
    main()