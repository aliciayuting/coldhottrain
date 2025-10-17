#!/usr/bin/env python3
# Train a LoRA adapter for Qwen2.5-0.5B on GLUE (SST-2 or MNLI)

'''
python3 lora_finetune_qwen_glue.py \
  --task_name mnli \
  --output_dir qwen25_mnli_lora_adapter \
  --lora_r 8
'''

import os
import argparse
from typing import Dict, Any
import numpy as np

import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import logging 
from probe2 import *
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train a LoRA adapter for Qwen2.5-0.5B on GLUE")
    p.add_argument("--task_name", type=str, default="sst2", choices=["sst2", "mnli"],
                   help="GLUE task to fine-tune on.")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                   help="Base model to fine-tune.")
    p.add_argument("--output_dir", type=str, default="qwen25_lora_glue_adapter",
                   help="Where to save the LoRA adapter.")
    p.add_argument("--use_qlora", type=str, default="false",
                   help="If 'true', load base in 4-bit and prepare for k-bit training.")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16, help="Per-device train/eval batch size.")
    p.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    # p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    # p.add_argument("--save_steps", type=int, default=200)
    # p.add_argument("--seed", type=int, default=42)

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16)
    p.add_argument("--lora_dropout", type=float, default=0) #0.05

    # Mixed precision
    p.add_argument("--fp16", action="store_true", help="Force FP16 (overrides bf16 if both set).")
    p.add_argument("--bf16", action="store_true", help="Use BF16 if available.")
    return p.parse_args()

def str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "t", "yes", "y"}

# ---------- Data ----------
def get_num_labels(task: str) -> int:
    return 3 if task == "mnli" else 2

def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def tokenizers_for_task(task: str, tok, max_len: int):
    if task == "sst2":
        def tok_fn(ex):
            return tok(ex["sentence"], truncation=True, max_length=max_len)
        remove_cols = ["sentence", "label", "idx"]
    else:  # mnli
        def tok_fn(ex):
            return tok(ex["premise"], ex["hypothesis"], truncation=True, max_length=max_len)
        remove_cols = ["premise", "hypothesis", "label", "idx"]

    return tok_fn, remove_cols

# ---------- Metrics ----------
def make_compute_metrics(task: str):
    # For both SST-2 and MNLI, accuracy is a good primary metric
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy 
        # preds = eval_pred.predictions
        # if isinstance(preds, tuple):
        #     preds = preds[0]
        # y_pred = preds.argmax(axis=-1)
        # y_true = eval_pred.label_ids
        # return accuracy_metric.compute(predictions=y_pred, references=y_true)

    return compute_metrics

# ---------- Model init (with optional QLoRA) ----------
def load_base_model(args, tok, num_labels: int):
    cfg = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        pad_token_id=tok.pad_token_id,
    )

    use_qlora = str2bool(args.use_qlora) if isinstance(args.use_qlora, str) else args.use_qlora
    low_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    if use_qlora:
        from transformers import BitsAndBytesConfig
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
            else torch.float16
        )
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=cfg, quantization_config=bnb_cfg, device_map="auto", torch_dtype=low_dtype
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    else:
        base = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=cfg,torch_dtype=low_dtype)

    base.resize_token_embeddings(len(tok))
    base.config.pad_token_id = tok.pad_token_id
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.pad_token_id = tok.pad_token_id

    return base

def wrap_with_lora(base, args):
    # Typical Qwen target modules for LoRA on seq cls
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora_cfg)
    # Show trainable params for sanity
    model.print_trainable_parameters()
    return model

# ---------- Main ----------
def main():
    args = parse_args()
    # torch.manual_seed(args.seed)
    SCRATCH_PREFIX = "/pscratch/sd/l/lsx/lora"
    # ensure output_dir always lives under this directory
    if not args.output_dir.startswith(SCRATCH_PREFIX):
        args.output_dir = os.path.join(SCRATCH_PREFIX, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    tok = build_tokenizer(args.model_name)
    ds = load_dataset("nyu-mll/glue", args.task_name)
    num_labels = get_num_labels(args.task_name)

    tok_fn, remove_cols = tokenizers_for_task(args.task_name, tok, args.max_len)
    ds_tok = ds.map(tok_fn, batched=True)
    if "label" in ds_tok["train"].column_names:
        ds_tok = ds_tok.rename_column("label", "labels")
    # Keep label + model inputs only (HF Trainer handles "label")
    cols_to_remove = [c for c in remove_cols if c in ds_tok["train"].column_names]
    if cols_to_remove:
        ds_tok = ds_tok.remove_columns(cols_to_remove)

    # Splits
    if args.task_name == "mnli":
        eval_split = "validation_matched"
        eval_dataset = ds_tok[eval_split]
        # (You could also evaluate mismatched separately if desired)
    else:
        eval_dataset = ds_tok["validation"]
    train_dataset = ds_tok["train"]

    collator = DataCollatorWithPadding(
        tokenizer=tok, pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # Model
    base = load_base_model(args, tok, num_labels)
    model = wrap_with_lora(base, args)

    # Precision
    use_bf16_hw = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    fp16 = args.fp16 or (not args.bf16 and not use_bf16_hw)  # default to fp16 on older GPUs
    bf16 = args.bf16 or (use_bf16_hw and not args.fp16)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_checkpointing=True,
        # warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        ddp_find_unused_parameters=False,
        # save_steps=args.save_steps,
        # save_total_limit=2,
        # greater_is_better=True,
        fp16=fp16,
         bf16=bf16,
        # report_to="none",
        # seed=args.seed,
    )

    compute_metrics = make_compute_metrics(args.task_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    ram_cb = VramBreakdownCallback()

    #trainer.add_callback(probe_cb)
    trainer.add_callback(ram_cb)

    def log_memory_stats():
        """Log current GPU memory statistics"""
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Max Allocated: {max_allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    log_memory_stats()

    # Train
    trainer.train()
    log_memory_stats()
    # Save ONLY the LoRA adapter
    model.save_pretrained(args.output_dir)
    # Optionally keep tokenizer/config alongside (useful for later)
    tok.save_pretrained(args.output_dir)

    print(f"\nLoRA adapter saved to: {args.output_dir}")
    print("You can later load it with:\n"
          "  base = AutoModelForSequenceClassification.from_pretrained('Qwen/Qwen2.5-0.5B', config=cfg)\n"
          "  model = PeftModel.from_pretrained(base, '<adapter_dir>')")

if __name__ == "__main__":
    main()