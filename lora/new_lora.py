#!/usr/bin/env python3
# Train a LoRA adapter for Qwen2.5-0.5B on GLUE (SST-2 or MNLI)

'''
python3 lora_finetune_qwen_glue.py \
  --task_name mnli \
  --output_dir qwen25_mnli_lora_adapter \
  --lora_r 16 \
  --learning_rate 3e-4 \
  --num_epochs 5
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
    p.add_argument("--max_len", type=int, default=256)  # Reduced for efficiency
    p.add_argument("--batch_size", type=int, default=16, help="Per-device train/eval batch size.")  # Reduced to avoid OOM
    p.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")  # Compensate for smaller batch
    p.add_argument("--num_epochs", type=float, default=5.0)  # More epochs
    p.add_argument("--learning_rate", type=float, default=3e-4)  # Better default
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)  # More warmup
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=16)  # Higher rank
    p.add_argument("--lora_alpha", type=float, default=32)  # 2x rank
    p.add_argument("--lora_dropout", type=float, default=0.05)  # Small dropout

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

def tokenize_dataset(task: str, tok, ds, max_len: int):
    """Tokenize with batched=True for efficiency"""
    if task == "sst2":
        def tok_fn(examples):
            return tok(examples["sentence"], truncation=True, max_length=max_len, padding=False)
        remove_cols = ["sentence", "idx"]
    else:  # mnli
        def tok_fn(examples):
            return tok(
                examples["premise"], 
                examples["hypothesis"], 
                truncation=True, 
                max_length=max_len,
                padding=False
            )
        remove_cols = ["premise", "hypothesis", "idx"]

    # Use batched=True for better performance
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=remove_cols)
    
    # Rename label column if exists
    if "label" in ds_tok["train"].column_names:
        ds_tok = ds_tok.rename_column("label", "labels")
    
    return ds_tok

# ---------- Metrics ----------
def make_compute_metrics(task: str):
    accuracy_metric = evaluate.load("accuracy")
    
    if task == "mnli":
        # For MNLI, also track matched/mismatched separately if needed
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return accuracy_metric.compute(predictions=predictions, references=labels)
    else:
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return accuracy_metric.compute(predictions=predictions, references=labels)
    
    return compute_metrics

# ---------- Model init (with optional QLoRA) ----------
def load_base_model(args, tok, num_labels: int):
    use_qlora = str2bool(args.use_qlora) if isinstance(args.use_qlora, str) else args.use_qlora
    
    # FIX: Don't force float32, use appropriate dtype
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
            torch_dtype=compute_dtype,
            pad_token_id=tok.pad_token_id,
            ignore_mismatched_sizes=True,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        # Use bf16 or fp16 depending on hardware
        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
            else torch.float16
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            torch_dtype=dtype,
            pad_token_id=tok.pad_token_id,
            ignore_mismatched_sizes=True,
        )

    # FIX: Use std=0.02 which is standard for transformers, not 0.01
    # For better initial loss, we want logits close to 0 (uniform distribution)
    if hasattr(base, 'score'):
        # Very small initialization to start near uniform distribution
        torch.nn.init.normal_(base.score.weight, mean=0.0, std=0.02)
        if base.score.bias is not None:
            torch.nn.init.zeros_(base.score.bias)
        
        # Verify initialization
        with torch.no_grad():
            weight_std = base.score.weight.std().item()
            weight_mean = base.score.weight.mean().item()
        logging.info(f"Initialized score layer: mean={weight_mean:.6f}, std={weight_std:.6f}")
        
        # Critical: Check that initialization will give reasonable initial loss
        # For a well-initialized classifier, logits should be close to 0
        # This gives loss ≈ log(num_labels)
        
    elif hasattr(base, 'classifier'):
        torch.nn.init.normal_(base.classifier.weight, mean=0.0, std=0.02)
        if base.classifier.bias is not None:
            torch.nn.init.zeros_(base.classifier.bias)
        
        with torch.no_grad():
            weight_std = base.classifier.weight.std().item()
            weight_mean = base.classifier.weight.mean().item()
        logging.info(f"Initialized classifier layer: mean={weight_mean:.6f}, std={weight_std:.6f}")
    
    base.resize_token_embeddings(len(tok))
    base.config.use_cache = False
    base.config.pad_token_id = tok.pad_token_id
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.pad_token_id = tok.pad_token_id

    return base

def wrap_with_lora(base, args, num_labels):
    """Wrap model with LoRA, ensuring classifier is trainable"""
    
    # More comprehensive target modules for better coverage
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ]
    
    # Determine the correct classifier module name
    classifier_name = None
    if hasattr(base, 'score'):
        classifier_name = "score"
    elif hasattr(base, 'classifier'):
        classifier_name = "classifier"
    
    if classifier_name is None:
        raise ValueError("Could not find classifier head (score or classifier)")
    
    logging.info(f"Using classifier name: {classifier_name}")
    
    # Store classifier state before PEFT wrapping to verify it doesn't get corrupted
    if hasattr(base, 'score'):
        pre_peft_weight = base.score.weight.data.clone()
        pre_peft_std = pre_peft_weight.std().item()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
        modules_to_save=[classifier_name],  # This makes classifier trainable
    )
    
    model = get_peft_model(base, lora_cfg)
    
    # Verify classifier wasn't corrupted by PEFT wrapping
    if hasattr(base, 'score'):
        # Access the wrapped classifier
        if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'score'):
            post_peft_std = model.base_model.model.score.original_module.weight.data.std().item()
            if abs(post_peft_std - pre_peft_std) > 0.001:
                logging.warning(f"⚠️  Classifier std changed during PEFT wrapping: {pre_peft_std:.6f} → {post_peft_std:.6f}")
    
    model.print_trainable_parameters()
    
    return model

# ---------- Main ----------
def main():
    args = parse_args()
    SCRATCH_PREFIX = "./"
    if not args.output_dir.startswith(SCRATCH_PREFIX):
        args.output_dir = os.path.join(SCRATCH_PREFIX, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    tok = build_tokenizer(args.model_name)
    ds = load_dataset("nyu-mll/glue", args.task_name)
    num_labels = get_num_labels(args.task_name)
    
    logging.info(f"Task: {args.task_name}, Num labels: {num_labels}")
    logging.info(f"Train samples: {len(ds['train'])}")

    # FIX: Use batched tokenization
    ds_tok = tokenize_dataset(args.task_name, tok, ds, args.max_len)

    # Splits
    if args.task_name == "mnli":
        eval_split = "validation_matched"
        eval_dataset = ds_tok[eval_split]
    else:
        eval_dataset = ds_tok["validation"]
    train_dataset = ds_tok["train"]
    
    logging.info(f"Eval samples: {len(eval_dataset)}")

    collator = DataCollatorWithPadding(
        tokenizer=tok, 
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # Model
    base = load_base_model(args, tok, num_labels)
    model = wrap_with_lora(base, args, num_labels)
    
    # CRITICAL: Reinitialize classifier AFTER PEFT wrapping
    # PEFT's modules_to_save can sometimes corrupt the initialization
    logging.info("\n=== Post-PEFT Classifier Reinitialization ===")
    try:
        # Find the actual classifier layer after PEFT wrapping
        classifier_layer = None
        classifier_found = False
        
        # Try multiple paths to find the classifier
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            base_model = model.base_model.model
            if hasattr(base_model, 'score'):
                # Check if it's wrapped by modules_to_save
                if hasattr(base_model.score, 'modules_to_save'):
                    if hasattr(base_model.score.modules_to_save, 'default'):
                        classifier_layer = base_model.score.modules_to_save.default
                        classifier_found = True
                        logging.info("Found classifier at: base_model.score.modules_to_save.default")
                elif hasattr(base_model.score, 'original_module'):
                    classifier_layer = base_model.score.original_module
                    classifier_found = True
                    logging.info("Found classifier at: base_model.score.original_module")
                else:
                    classifier_layer = base_model.score
                    classifier_found = True
                    logging.info("Found classifier at: base_model.score")
        
        if not classifier_found:
            # Try direct access
            for name, module in model.named_modules():
                if 'score' in name and isinstance(module, torch.nn.Linear):
                    classifier_layer = module
                    classifier_found = True
                    logging.info(f"Found classifier at: {name}")
                    break
        
        if classifier_layer is not None:
            pre_std = classifier_layer.weight.data.std().item()
            
            # CRITICAL: Make sure it's trainable
            if not classifier_layer.weight.requires_grad:
                logging.warning("⚠️  Classifier is FROZEN! Enabling requires_grad...")
                classifier_layer.weight.requires_grad = True
                if classifier_layer.bias is not None:
                    classifier_layer.bias.requires_grad = True
            
            # Reinitialize with VERY small weights for near-uniform initial predictions
            # The logits should be close to 0 to give loss ≈ log(num_labels)
            with torch.no_grad():
                torch.nn.init.normal_(classifier_layer.weight, mean=0.0, std=0.01)  # Even smaller!
                if classifier_layer.bias is not None:
                    torch.nn.init.zeros_(classifier_layer.bias)
            
            post_std = classifier_layer.weight.data.std().item()
            logging.info(f"Classifier reinitialized: {pre_std:.6f} → {post_std:.6f}")
            logging.info(f"Classifier requires_grad: {classifier_layer.weight.requires_grad}")
            
            if not classifier_layer.weight.requires_grad:
                raise RuntimeError("❌ FATAL: Classifier is still frozen after manual enable!")
        else:
            raise ValueError("❌ Could not find classifier layer!")
    except Exception as e:
        logging.error(f"Classifier setup failed: {e}")
        raise
    
    # Sanity check: forward pass with proper labels for MNLI
    logging.info("\n=== Sanity Check Forward Pass ===")
    dummy_input = {
        'input_ids': torch.randint(0, len(tok), (4, 64)).to(model.device),
        'attention_mask': torch.ones(4, 64).to(model.device),
        'labels': torch.tensor([0, 1, 2, 0] if num_labels == 3 else [0, 1, 0, 1]).to(model.device)
    }
    with torch.no_grad():
        outputs = model(**dummy_input)
        expected_loss = np.log(num_labels)
        logging.info(f"Initial loss: {outputs.loss.item():.4f} (expected ~{expected_loss:.4f})")
        logging.info(f"Logits sample: {outputs.logits[0].cpu().tolist()}")
        logit_std = outputs.logits.std().item()
        logging.info(f"Logit std: {logit_std:.4f}")
        if abs(outputs.loss.item() - expected_loss) > 0.5:
            logging.warning(f"⚠️  Initial loss is far from expected! Logit std={logit_std:.4f}")
            if logit_std > 1.0:
                logging.warning("⚠️  Logits have high variance - classifier initialization may be too large!")
    
    # Verify trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
    
    # Precision - CRITICAL: Must use mixed precision with gradient checkpointing
    use_bf16_hw = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    # Force mixed precision if neither is set (required for gradient checkpointing to work properly)
    if not args.fp16 and not args.bf16:
        if use_bf16_hw:
            bf16 = True
            fp16 = False
            logging.info("Auto-enabling BF16 (required for gradient checkpointing)")
        else:
            fp16 = True
            bf16 = False
            logging.info("Auto-enabling FP16 (required for gradient checkpointing)")
    else:
        fp16 = args.fp16
        bf16 = args.bf16 or (use_bf16_hw and not args.fp16)
    logging.info(f"Training precision - FP16: {fp16}, BF16: {bf16}")
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # CRITICAL: Required for proper gradients
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        ddp_find_unused_parameters=False,
        fp16=fp16,
        bf16=bf16,
        report_to="none",  # Disable wandb/tensorboard if not needed
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

    # Train
    logging.info("\n=== Starting Training ===")
    
    # CRITICAL: Verify classifier is trainable
    logging.info("\n=== Trainable Parameters Verification ===")
    classifier_trainable = False
    lora_trainable = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'score' in name or 'classifier' in name:
                classifier_trainable = True
                logging.info(f"✓ Classifier trainable: {name}")
            elif 'lora' in name:
                lora_trainable = True
    
    if not classifier_trainable:
        raise RuntimeError("❌ FATAL: Classifier is not trainable! Training will fail.")
    if not lora_trainable:
        logging.warning("⚠️  No LoRA parameters trainable!")
    
    logging.info(f"✓ Classifier trainable: {classifier_trainable}")
    logging.info(f"✓ LoRA trainable: {lora_trainable}")
    
    trainer.train()

    # Final evaluation
    logging.info("\n=== Final Evaluation ===")
    eval_results = trainer.evaluate()
    logging.info(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Save ONLY the LoRA adapter
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    logging.info(f"\nLoRA adapter saved to: {args.output_dir}")

if __name__ == "__main__":
    main()