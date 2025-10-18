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
from torch.utils.data import DataLoader


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
    p.add_argument("--num_epochs", type=float, default=1.0)  # Increased from 1.0
    p.add_argument("--learning_rate", type=float, default=5e-4)  # Increase from 1e-4
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)  # Smaller warmup
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=500)  # More frequent eval
    p.add_argument("--max_grad_norm", type=float, default=1.0)  # Less aggressive clipping

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=8)  # Back to 8, simpler is better
    p.add_argument("--lora_alpha", type=float, default=32)  # Match with r for scaling=1
    p.add_argument("--lora_dropout", type=float, default=0)  # Small dropout

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
    # FIXED: Changed to right padding for sequence classification
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
    return compute_metrics

# ---------- Model init (with optional QLoRA) ----------
def load_base_model(args, tok, num_labels: int):
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
            args.model_name,
            num_labels=num_labels,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=low_dtype,
            pad_token_id=tok.pad_token_id,  # Set during load
            ignore_mismatched_sizes=True,  # Allow classifier head mismatch
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            torch_dtype=low_dtype,
            pad_token_id=tok.pad_token_id,  # Set during load
            ignore_mismatched_sizes=True,  # Allow classifier head mismatch
        )

    # CRITICAL FIX: Properly initialize the classifier head with small weights
    # The default initialization from transformers can be too large
    if hasattr(base, 'score'):
        # Qwen uses 'score' for classification head
        # Use very small initialization to start near uniform distribution
        torch.nn.init.normal_(base.score.weight, mean=0.0, std=0.01)
        if base.score.bias is not None:
            torch.nn.init.zeros_(base.score.bias)
        print(f"Re-initialized score layer: mean={base.score.weight.data.mean().item():.6f}, std={base.score.weight.data.std().item():.6f}")
    elif hasattr(base, 'classifier'):
        torch.nn.init.normal_(base.classifier.weight, mean=0.0, std=0.01)
        if base.classifier.bias is not None:
            torch.nn.init.zeros_(base.classifier.bias)
        print(f"Re-initialized classifier layer: mean={base.classifier.weight.data.mean().item():.6f}, std={base.classifier.weight.data.std().item():.6f}")
    
    base.resize_token_embeddings(len(tok))
    base.config.use_cache = False
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
        modules_to_save=["score"]  # or ["classifier"] depending on model
    )
    model = get_peft_model(base, lora_cfg)
    # Show trainable params for sanity
    model.print_trainable_parameters()
    return model

# ---------- Main ----------
def main():
    args = parse_args()
    SCRATCH_PREFIX = "/pscratch/sd/l/lsx/lora"
    # SCRATCH_PREFIX = "./"
    # ensure output_dir always lives under this directory
    if not args.output_dir.startswith(SCRATCH_PREFIX):
        args.output_dir = os.path.join(SCRATCH_PREFIX, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    tok = build_tokenizer(args.model_name)
    ds = load_dataset("nyu-mll/glue", args.task_name)
    num_labels = get_num_labels(args.task_name)

    tok_fn, remove_cols = tokenizers_for_task(args.task_name, tok, args.max_len)
    ds_tok = ds.map(tok_fn, batched=False)

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
    else:
        eval_dataset = ds_tok["validation"]
    train_dataset = ds_tok["train"]

    collator = DataCollatorWithPadding(
        tokenizer=tok, pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # Model
    base = load_base_model(args, tok, num_labels)
    model = wrap_with_lora(base, args)

    # CRITICAL: Re-initialize classifier AFTER LoRA wrapping
    # PEFT's modules_to_save creates a copy at: base_model.model.score.modules_to_save.default
    print("\n=== Re-initializing Classifier After PEFT Wrapping ===")
    
    # Find and reinitialize the trainable classifier
    classifier_reinitialized = False
    for name, param in model.named_parameters():
        # Look for the trainable score layer (in modules_to_save)
        if 'modules_to_save' in name and 'score' in name and 'weight' in name:
            print(f"Found trainable classifier: {name}")
            print(f"  Before: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            
            # Re-initialize with small std
            with torch.no_grad():
                param.data = torch.randn_like(param.data) * 0.01
            
            print(f"  After:  mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            classifier_reinitialized = True
        
        # Also zero out bias if it exists
        if 'modules_to_save' in name and 'score' in name and 'bias' in name:
            print(f"Found trainable classifier bias: {name}")
            with torch.no_grad():
                param.data.zero_()
            classifier_reinitialized = True
    
    if not classifier_reinitialized:
        print("WARNING: Could not find trainable classifier to reinitialize!")
        print("Attempting fallback method...")
        # Fallback: try to access directly
        try:
            score_layer = model.base_model.model.score.modules_to_save.default
            torch.nn.init.normal_(score_layer.weight, mean=0.0, std=0.01)
            if score_layer.bias is not None:
                torch.nn.init.zeros_(score_layer.bias)
            print(f"✓ Reinitialized via direct access: std={score_layer.weight.data.std().item():.6f}")
        except Exception as e:
            print(f"ERROR: Could not reinitialize classifier: {e}")

    model.print_trainable_parameters()

    # Inspect the classifier head's grad status and stats
    print("\n=== Classifier Head Inspection ===")
    for n, p in model.named_parameters():
        if "score" in n or "classifier" in n:
            print(f"{n}: requires_grad={p.requires_grad}, "
                  f"shape={p.shape}, mean={p.data.float().mean().item():.4f}, "
                  f"std={p.data.float().std().item():.4f}")
    
    # Sanity check: forward pass with dummy batch
    print("\n=== Sanity Check Forward Pass ===")
    dummy_input = {
        'input_ids': torch.randint(0, len(tok), (2, 128)).to(model.device),
        'attention_mask': torch.ones(2, 128).to(model.device),
        'labels': torch.tensor([0, 1]).to(model.device) if num_labels == 2 else torch.tensor([0, 1]).to(model.device)
    }
    with torch.no_grad():
        outputs = model(**dummy_input)
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Logits sample: {outputs.logits[0]}")
        print(f"Loss: {outputs.loss.item():.4f}")
        print(f"Expected loss for random init: ~{np.log(num_labels):.4f}")
        if outputs.loss.item() > 10:
            print("WARNING: Initial loss is very high! Check classifier initialization.")
    
    print(f"\n=== Trainable Parameters ===")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
    
    
    # Precision
    use_bf16_hw = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    fp16 = args.fp16   # default to fp16 on older GPUs
    bf16 = args.bf16 or (use_bf16_hw and not args.fp16)
    print(f"fp16 is {fp16}")
    print(f"bf16 is {bf16}")
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,  # ADDED: Gradient clipping
        warmup_ratio=args.warmup_ratio,  # ADDED: Warmup
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",  # ADDED: Save checkpoints
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,  # ADDED: Load best model
        metric_for_best_model="accuracy",  # ADDED: Metric for best model
        greater_is_better=True,
        ddp_find_unused_parameters=False,
        fp16=fp16,
        bf16=bf16,
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