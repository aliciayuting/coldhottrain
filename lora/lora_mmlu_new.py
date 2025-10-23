#!/usr/bin/env python3
"""
LoRA finetune Qwen on MMLU dataset - FIXED VERSION
Key fixes:
1. Uses chat template (fixes accuracy/loss paradox)
2. Correct answer format (no space)
3. Better hyperparameters (2 epochs, r=16, lr=2e-4)
4. Early stopping to prevent overfitting

Usage:
    python lora_mmlu_FIXED.py --model Qwen/Qwen2.5-0.5B-Instruct --batch_size 4 --bf16
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import transformers
import numpy as np
from sklearn.metrics import accuracy_score

# Set random seed
transformers.set_seed(42)

import logging 
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format="[%(levelname)s] %(message)s"
)

def parse_args():
    p = argparse.ArgumentParser(description="LoRA finetune on MMLU - FIXED")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", type=str, default="cais/mmlu")
    p.add_argument("--output_dir", type=str, default="/pscratch/sd/l/lsx/lora/qwen05b_fixed")
    p.add_argument("--max_length", type=int, default=1024)  # FIXED: Increased for chat template
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)  # FIXED: Increased
    p.add_argument("--num_epochs", type=float, default=2.0)  # FIXED: Reduced from 10 to 2
    p.add_argument("--learning_rate", type=float, default=2e-4)  # FIXED: Increased from 1e-4
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=16)  # FIXED: Increased from 8 to 16
    p.add_argument("--lora_alpha", type=float, default=32)  # FIXED: Increased from 8 to 32
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=200)  # FIXED: More frequent logging
    p.add_argument("--eval_steps", type=int, default=200)  # FIXED: More frequent eval
    p.add_argument("--bf16", action="store_true", help="Use BF16")
    p.add_argument("--fp16", action="store_true", help="Use FP16")
    p.add_argument("--early_stopping_patience", type=int, default=3, 
                   help="Stop if no improvement for N evaluations")
    p.add_argument("--max_train_samples", type=int, default=None, 
                   help="Limit training samples for debugging")
    p.add_argument("--max_eval_samples", type=int, default=None, 
                   help="Limit eval samples for debugging")

    return p.parse_args()

def main():
    args = parse_args()
    
    # Setup output directory
    SCRATCH_PREFIX = "/pscratch/sd/l/lsx/lora"
    if not args.output_dir.startswith(SCRATCH_PREFIX):
        args.output_dir = os.path.join(SCRATCH_PREFIX, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    print(f"\n{'='*70}")
    print("FIXED VERSION - Key improvements:")
    print("✅ Uses chat template (fixes accuracy/loss paradox)")
    print("✅ Correct answer format (no space)")
    print(f"✅ Reduced epochs: {args.num_epochs} (was 10)")
    print(f"✅ Higher learning rate: {args.learning_rate} (was 1e-4)")
    print(f"✅ Better LoRA config: r={args.lora_r}, alpha={args.lora_alpha} (was r=8, alpha=8)")
    print(f"✅ Early stopping enabled (patience={args.early_stopping_patience})")
    print(f"{'='*70}\n")
    
    # ========== Load Tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded: pad_token='{tokenizer.pad_token}'")
    
    # ========== Load Model ==========
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        # device_map="auto"
    )

    model.gradient_checkpointing_enable()  
    model.config.use_cache = False  
    
    # ========== Configure LoRA ==========
    print("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    
    # Verify trainable parameters are reasonable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"\n{'='*70}")
    print("Parameter Check:")
    print(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"  Total: {total_params:,}")
    
    if trainable_pct > 5:
        print(f"  ⚠️  WARNING: {trainable_pct:.1f}% trainable is high for LoRA!")
        print(f"  Expected: <2% for r={args.lora_r}")
        print(f"  This might cause overfitting.")
    else:
        print(f"  ✅ Trainable % looks good for LoRA")
    print(f"{'='*70}\n")

    # ========== Load MMLU Dataset ==========
    print("Loading MMLU dataset...")
    dataset = load_dataset(args.dataset, "all")
    train_dataset = dataset["auxiliary_train"]
    eval_dataset = dataset["test"]
    
    # Optionally limit samples for debugging
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
        print(f"Limited training to {len(train_dataset)} samples")
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
        print(f"Limited eval to {len(eval_dataset)} samples")
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # ========== Format MMLU Examples with Chat Template ==========
    def format_mmlu_example(example):
        """
        CRITICAL FIX: Uses chat template instead of plain text
        This fixes the accuracy/loss paradox!
        """
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        
        # Format choices
        choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # FIXED: Use chat template (this is the key fix!)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers multiple choice questions. Respond with only the letter of the correct answer (A, B, C, or D)."
            },
            {
                "role": "user",
                "content": f"{question}\n\n{choice_text}"
            }
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Adds the assistant prefix
        )
        
        # FIXED: No space before letter!
        answer_letter = chr(65 + answer_idx)  # "A", "B", "C", or "D"
        
        full_text = prompt + answer_letter
        
        return {
            "prompt": prompt,
            "answer": answer_letter,
            "full_text": full_text
        }

    print("\nFormatting datasets with chat template...")
    train_dataset = train_dataset.map(format_mmlu_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_mmlu_example, remove_columns=eval_dataset.column_names)
    
    # Show example
    print("\n" + "="*70)
    print("SAMPLE FORMATTED EXAMPLE:")
    print("="*70)
    print(f"Prompt:\n{train_dataset[0]['prompt']}")
    print(f"\nAnswer: '{train_dataset[0]['answer']}'")
    print(f"\nFull text:\n{train_dataset[0]['full_text']}")
    print("="*70 + "\n")
    
    # ========== Tokenize ==========
    def tokenize_function(examples):
        """Tokenize examples, masking prompt tokens in labels"""
        # Tokenize full text
        full_encodings = tokenizer(
            examples["full_text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Tokenize just prompt to find where answer starts
        prompt_encodings = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels
        labels = []
        for i, (input_ids, attention_mask) in enumerate(zip(full_encodings["input_ids"], 
                                                             full_encodings["attention_mask"])):
            label = input_ids.copy()
            prompt_len = len(prompt_encodings["input_ids"][i])
            
            # Mask prompt tokens (set to -100 so they're ignored in loss)
            for j in range(min(prompt_len, len(label))):
                label[j] = -100
            
            # Mask padding tokens
            for j in range(len(label)):
                if attention_mask[j] == 0:
                    label[j] = -100
            
            labels.append(label)
        
        return {
            "input_ids": full_encodings["input_ids"],
            "attention_mask": full_encodings["attention_mask"],
            "labels": labels
        }
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "answer", "full_text"]
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "answer", "full_text"]
    )
    
    # Show tokenized example
    print("\n" + "="*70)
    print("SAMPLE TOKENIZED EXAMPLE:")
    print("="*70)
    example = train_dataset[0]
    non_pad_ids = [id for id in example["input_ids"] if id != tokenizer.pad_token_id]
    non_ignore_labels = [(i, l) for i, l in enumerate(example["labels"]) if l != -100]
    print(f"Input length (non-padded): {len(non_pad_ids)} tokens")
    print(f"Label tokens (non -100): {len(non_ignore_labels)} tokens")
    print(f"Decoded input: {tokenizer.decode(non_pad_ids)}")
    print(f"Decoded labels: {tokenizer.decode([l for _, l in non_ignore_labels])}")
    print("="*70 + "\n")
    
    # ========== Setup Choice Tokens ==========
    # FIXED: No space before letters!
    CHOICES = ["A", "B", "C", "D"]
    CHOICE_TOKEN_IDS = []
    
    print("Setting up choice tokens...")
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            print(f"⚠️  WARNING: Choice '{choice}' tokenizes to {len(ids)} tokens: {ids}")
            print(f"   Using first token: {ids[0]}")
            CHOICE_TOKEN_IDS.append(ids[0])
        else:
            CHOICE_TOKEN_IDS.append(ids[0])
    
    CHOICE_TOKEN_IDS = torch.tensor(CHOICE_TOKEN_IDS, dtype=torch.long)
    print(f"Choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")
    print(f"Decoded: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}\n")
    
    # ========== Metrics Helpers ==========
    def _first_answer_pos(labels: torch.Tensor) -> torch.Tensor:
        """Find position of first non-masked label token"""
        not_ign = (labels != -100)
        first_pos = not_ign.float().argmax(dim=1)
        has_any = not_ign.any(dim=1)
        first_pos = torch.where(has_any, first_pos, torch.full_like(first_pos, -1))
        return first_pos
    
    def preprocess_logits_for_metrics(logits, labels):
        """Extract logits at answer position for the 4 choice tokens"""
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        
        logits = logits.float()
        labels = labels.to(logits.device)
        
        B, T, V = logits.shape
        
        # Find first answer token position
        ans_pos = _first_answer_pos(labels)
        
        # Handle cases with no answer token
        invalid = (ans_pos < 0)
        if invalid.any():
            logging.warning(f"{invalid.sum().item()}/{B} examples have no answer tokens")
            ans_pos = torch.where(invalid, torch.full_like(ans_pos, T - 1), ans_pos)
        
        # Extract logits at answer position
        row_idx = torch.arange(B, device=logits.device)
        logits_at_ans = logits[row_idx, ans_pos, :]
        
        # Get logits for the 4 choice tokens
        choice_ids = CHOICE_TOKEN_IDS.to(logits.device)
        choice_logits = logits_at_ans[:, choice_ids]
        
        return choice_logits
    
    def compute_metrics(eval_pred):
        """Compute accuracy from choice logits"""
        choice_logits = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # Predicted choice (0-3)
        pred_idx = choice_logits.argmax(axis=1)
        
        # Find gold answer
        labels_tensor = torch.tensor(labels)
        ans_pos = _first_answer_pos(labels_tensor)
        has_answer = (ans_pos >= 0)
        
        # Extract gold token IDs
        row_idx = torch.arange(labels_tensor.size(0))
        gold_tokens = torch.full((labels_tensor.size(0),), -1, dtype=torch.long)
        gold_tokens[has_answer] = labels_tensor[row_idx[has_answer], ans_pos[has_answer]]
        
        # Map gold tokens to choice index
        choice_ids = CHOICE_TOKEN_IDS
        eq_matrix = (gold_tokens[:, None] == choice_ids[None, :])
        gold_idx = eq_matrix.long().argmax(dim=1).numpy()
        gold_valid = eq_matrix.any(dim=1).numpy()
        
        # Compute accuracy
        if gold_valid.any():
            acc = accuracy_score(gold_idx[gold_valid], pred_idx[gold_valid])
        else:
            acc = 0.0
        
        # Show examples
        print(f"\n{'='*70}")
        print(f"Evaluation Accuracy: {acc:.4f}")
        print(f"{'='*70}")
        for i in range(min(5, len(pred_idx))):
            if gold_valid[i]:
                pred_letter = chr(65 + pred_idx[i])
                gold_letter = chr(65 + gold_idx[i])
                status = "✓" if pred_idx[i] == gold_idx[i] else "✗"
                print(f"  Example {i}: pred={pred_letter}, gold={gold_letter} {status}")
        print(f"{'='*70}\n")
        
        return {"accuracy": acc}
    
    # ========== Training Arguments ==========
    use_bf16 = args.bf16 or (torch.cuda.is_available() and 
                             torch.cuda.get_device_capability()[0] >= 8 and 
                             not args.fp16)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Saving and evaluation
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,  # Keep more checkpoints
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        
        # Best model selection - FIXED: Use eval_loss instead of accuracy
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # FIXED: Watch loss, not accuracy
        greater_is_better=False,  # Lower loss is better
        
        # Mixed precision
        fp16=args.fp16,
        bf16=use_bf16,
        
        # Optimization
        optim="adamw_torch",
        gradient_checkpointing=True,
        
        # Other
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Mixed precision: bf16={use_bf16}, fp16={args.fp16}")
    print(f"  Early stopping: patience={args.early_stopping_patience}\n")
    
    # ========== Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.01
            )
        ]
    )
    
    # Log GPU memory if available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved\n")
    
    # ========== Train ==========
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print("Watch for:")
    print("  ✅ eval_loss DECREASING (not increasing!)")
    print("  ✅ accuracy INCREASING")
    print("  ✅ Both metrics moving correctly together")
    print("  ❌ If eval_loss increases → early stopping will trigger")
    print("="*70 + "\n")
    
    trainer.train()
    
    # ========== Final Evaluation ==========
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    eval_results = trainer.evaluate()
    
    print(f"\nFinal Results:")
    print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Runtime: {eval_results['eval_runtime']:.1f}s")
    
    # Check if results are good
    if eval_results['eval_accuracy'] < 0.30:
        print(f"\n⚠️  WARNING: Accuracy is still low ({eval_results['eval_accuracy']:.1%})")
        print("   Expected: >35% for 0.5B model")
        print("   Possible issues:")
        print("   - Model too small for task")
        print("   - Need more epochs")
        print("   - Try larger model (1.5B or 7B)")
    elif eval_results['eval_accuracy'] < 0.40:
        print(f"\n✓ Accuracy is decent ({eval_results['eval_accuracy']:.1%})")
        print("  This is reasonable for 0.5B model on MMLU")
    else:
        print(f"\n✅ Accuracy is good ({eval_results['eval_accuracy']:.1%})")
        print("  This is excellent for 0.5B model!")
    
    # ========== Save ==========
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✅ Training complete!")
    print(f"✅ LoRA adapter saved to: {args.output_dir}")
    print(f"✅ Final accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"\nLoad with:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  from peft import PeftModel")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{args.model}')")
    print(f"  model = PeftModel.from_pretrained(base, '{args.output_dir}')")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()