#!/usr/bin/env python3
"""
LoRA finetune Vicuna (or any causal LM) on MMLU dataset
Usage:
    python lora_vicuna_mmlu.py --model luffycodes/vicuna-mmlu-val-only-correct-mcq-7b-ep2 --batch_size 8
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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import numpy as np
from sklearn.metrics import accuracy_score
from probe2 import *

# Set random seed
transformers.set_seed(42)

import logging 
logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

def parse_args():
    p = argparse.ArgumentParser(description="LoRA finetune on MMLU")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", type=str, default="cais/mmlu")
    p.add_argument("--output_dir", type=str, default="/pscratch/sd/l/lsx/lora/qwen05b")
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_qlora", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--bf16", action="store_true", help="Use BF16")
    p.add_argument("--fp16", action="store_true", help="Use FP16")

    return p.parse_args()

def main():
    args = parse_args()
    
    # Setup output directory
    # SCRATCH_PREFIX = "/pscratch/sd/l/lsx/lora"
    # SCRATCH_PREFIX = "./scratch_lora"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    
    # ========== Load Tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # ========== Load Model ==========
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # ========== Configure LoRA ==========
    print("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"]  # make lm_head trainable
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("\n=== LM Head Status ===")
    # Check base model specifically
    print("\nChecking base model for lm_head:")
    if hasattr(model, 'base_model'):
        for name, param in model.base_model.named_parameters():
            if 'lm_head' in name:
                print(f"{name}: requires_grad={param.requires_grad}")

    # ========== Load MMLU Dataset ==========
    print("Loading MMLU dataset...")
    dataset = load_dataset(args.dataset, "all")
    train_dataset = dataset["auxiliary_train"]
    eval_dataset = dataset["test"]
    
    # Optionally truncate for debugging
    # train_dataset = train_dataset.select(range(100))
    # eval_dataset = eval_dataset.select(range(50))
    
    # ========== Format MMLU Examples ==========
    def format_mmlu_example(example):
        """Convert MMLU example to instruction-following format"""
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        
        # Format choices
        choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # Create instruction-following format
        prompt = f"""Answer the following multiple choice question.

    Question: {question}

    Choices:
    {choice_text}

    Answer:"""
        
        # Get the correct answer letter
        answer = chr(65 + answer_idx)
        answer = f" {answer}"
        full_text = prompt + answer
        
        return {"prompt": prompt, 
                "label": answer,
                "full_text": full_text}

    # ========== Tokenize ==========
    def tokenize_function(examples):
        enc = tokenizer(
            examples["full_text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            add_special_tokens=False,
        )
        enc_prompt = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )
        prompt_lens = [len(x) for x in enc_prompt["input_ids"]]
        
        labels = [seq.copy() for seq in enc["input_ids"]]
        
        # Mask padding
        for i, mask in enumerate(enc["attention_mask"]):
            for j, m in enumerate(mask):
                if m == 0:
                    labels[i][j] = -100
        
        # Mask prompt, keep only answer tokens
        for i, plen in enumerate(prompt_lens):
            upto = min(plen, len(labels[i]))
            for j in range(upto):
                if labels[i][j] != -100:
                    labels[i][j] = -100
        
        enc["labels"] = labels
        return enc

    def show_dataset_example(dataset, num_examples=1):
        for i in range(len(dataset)):
            if i >= num_examples:
                break
            print(f"--- Example {i}: ---")
            print(f"### Prompt:\n{dataset[i]['prompt']}")
            print(f"### Label:\n{dataset[i]['label']}")
            print()
        
    print("Formatting datasets...")
    train_dataset = train_dataset.map(format_mmlu_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_mmlu_example, remove_columns=eval_dataset.column_names)
    # print("Sample formatted training examples:")
    # show_dataset_example(train_dataset, num_examples=1)
    

    def show_tokenized_dataset_examples(dataset, num_examples=2):
        for i in range(len(dataset)):
            if i >= num_examples:
                break
            example = dataset[i]
            print(f"--- Example {i}: ---")
            print(f"example keys: {list(example.keys())}")
            print(f"### Input IDs length: {len(example['input_ids'])}")
            # print(f"### Input ids: {example['input_ids']}")
            non_padded_input_idxs = [idx for idx, id in enumerate(example['input_ids']) if id != tokenizer.pad_token_id]
            non_padded_labels_idxs = [idx for idx, label in enumerate(example['labels']) if label != -100]
            non_masked_attention_idxs = [idx for idx, mask in enumerate(example['attention_mask']) if mask != 0]
            print(f"### Non-padded Input IDs (len: {len(non_padded_input_idxs)})")
            print(f"### Non-padded Labels (len: {len(non_padded_labels_idxs)})")
            print(f"### Non-masked Attention (len: {len(non_masked_attention_idxs)})")
            print(f"### Input text (non-padded):\n{tokenizer.decode([id for id in example['input_ids'] if id != tokenizer.pad_token_id])}")
            non_ignore_labels = [l for l in example['labels'] if l != -100]
            print(f"### Labels (non -100): {non_ignore_labels}")
            print(f"### Decoded: {tokenizer.decode(non_ignore_labels)}")
            print()

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "label"]
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "label"]
    )

    # print("Sample tokenized training examples:")
    # show_tokenized_dataset_examples(eval_dataset, num_examples=1)

    
    # ========== Setup Choice Tokens ==========
    CHOICES = [" A", " B", " C", " D"]
    CHOICE_TOKEN_IDS_LIST = []
    for s in CHOICES:
        ids = tokenizer.encode(s, add_special_tokens=False)
        assert(len(ids) == 1)
        CHOICE_TOKEN_IDS_LIST.append(ids)
    
    # Check if single or multi-token
    SINGLE_TOKEN = all(len(ids) == 1 for ids in CHOICE_TOKEN_IDS_LIST)
    
    if SINGLE_TOKEN:
        CHOICE_TOKEN_IDS = torch.tensor([ids[0] for ids in CHOICE_TOKEN_IDS_LIST], dtype=torch.long)
        print(f"Choice tokens are SINGLE tokens: {CHOICE_TOKEN_IDS.tolist()}")
    else:
        print(f"Choice tokens are MULTI-TOKEN: {CHOICE_TOKEN_IDS_LIST}")
        # For multi-token (like Vicuna), use second token (the letter)
        if all(len(ids) >= 2 for ids in CHOICE_TOKEN_IDS_LIST):
            CHOICE_TOKEN_IDS = torch.tensor([ids[1] for ids in CHOICE_TOKEN_IDS_LIST], dtype=torch.long)
            print(f"Using second token (letter): {CHOICE_TOKEN_IDS.tolist()}")
            print(f"Decoded: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}")
        else:
            raise ValueError("Inconsistent token structure for choices")
    
    # ========== Metrics Helpers ==========
    def _first_answer_pos(labels: torch.Tensor) -> torch.Tensor:
        not_ign = (labels != -100)
        first_pos = not_ign.float().argmax(dim=1)
        has_any = not_ign.any(dim=1)
        first_pos = torch.where(has_any, first_pos, torch.full_like(first_pos, -1))
        return first_pos
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float()
        labels = labels.to(logits.device)
        
        B, T, V = logits.shape
        ans_pos = _first_answer_pos(labels)
        
        bad = (ans_pos < 0)
        if bad.any():
            print("!!! Warning: some examples have no answer token; using last non-pad position instead.")
            ans_pos = torch.where(bad, torch.full_like(ans_pos, T - 1), ans_pos)
        
        row_idx = torch.arange(B, device=logits.device)
        logits_at_ans = logits[row_idx, ans_pos, :]
        
        choice_ids = CHOICE_TOKEN_IDS.to(logits.device)
        four_logits = logits_at_ans.index_select(dim=1, index=choice_ids)
        return four_logits
    
    def compute_metrics(eval_pred):
        four_logits = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        pred_idx = np.asarray(four_logits).argmax(axis=1)
        
        labels = torch.tensor(label_ids)
        ans_pos = _first_answer_pos(labels)
        has_any = (ans_pos >= 0)
        
        row_idx = torch.arange(labels.size(0))
        gold_token_ids = torch.full((labels.size(0),), -1, dtype=torch.long)
        gold_token_ids[has_any] = labels[row_idx[has_any], ans_pos[has_any]]
        
        choice_ids = CHOICE_TOKEN_IDS
        eq_matrix = (gold_token_ids[:, None] == choice_ids[None, :])
        gold_idx = eq_matrix.long().argmax(dim=1).numpy()
        gold_valid = eq_matrix.any(dim=1).numpy()
        
        if gold_valid.any():
            acc = accuracy_score(gold_idx[gold_valid], pred_idx[gold_valid])
        else:
            acc = 0.0        
        print(f"\nBatch Accuracy: {acc:.4f}")
        for i in range(min(5, len(pred_idx))):
            print(f"  Example {i}: pred={chr(65+pred_idx[i])}, gold={chr(65+gold_idx[i])}")
        
        return {"accuracy": acc}
    
    # ========== Training Arguments ==========
    use_bf16_hw = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    fp16 = args.fp16
    bf16 = args.bf16 or (use_bf16_hw and not args.fp16)
    
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
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=fp16,
        bf16=bf16,
        optim="adamw_torch",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        resume_from_checkpoint=False,
        logging_strategy="steps",
    )
    
    # ========== Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
    
    # ========== Train ==========
    print("\nStarting training...")
    trainer.train()
    
    # ========== Save ==========
    print("\nSaving LoRA adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\nLoRA adapter saved to: {args.output_dir}")
    print("Load with:")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{args.model}')")
    print(f"  model = PeftModel.from_pretrained(base, '{args.output_dir}')")

if __name__ == "__main__":
    main()