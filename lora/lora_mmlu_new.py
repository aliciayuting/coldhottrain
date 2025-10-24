#!/usr/bin/env python3
"""
LoRA finetune Qwen on MMLU dataset
Usage:
    python lora_mmlu.py --model Qwen/Qwen2.5-0.5B-Instruct --batch_size 4
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
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)  # More frequent eval
    p.add_argument("--use_qlora", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--bf16", action="store_true", help="Use BF16")
    p.add_argument("--fp16", action="store_true", help="Use FP16")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing factor (0.1 recommended for calibration)")
    p.add_argument("--debug_samples", type=int, default=5,
                   help="Number of samples to print detailed debug info for")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Setup output directory
    SCRATCH_PREFIX = "/pscratch/sd/l/lsx/lora"
    # SCRATCH_PREFIX = "./"
    # ensure output_dir always lives under this directory
    if not args.output_dir.startswith(SCRATCH_PREFIX):
        args.output_dir = os.path.join(SCRATCH_PREFIX, args.output_dir)
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
    # This model doesn't finetune lm_head
    # print("\n=== LM Head Status ===")
    # # Check base model specifically
    # print("\nChecking base model for lm_head:")
    # if hasattr(model, 'base_model'):
    #     for name, param in model.base_model.named_parameters():
    #         if 'lm_head' in name:
    #             print(f"{name}: requires_grad={param.requires_grad}")

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
        # not_ign = (labels != -100)
        # first_pos = not_ign.float().argmax(dim=1)
        # has_any = not_ign.any(dim=1)
        # first_pos = torch.where(has_any, first_pos, torch.full_like(first_pos, -1))
        # return first_pos

        not_ign = (labels != -100)           # boolean mask
        flipped = torch.flip(not_ign, dims=[1])
        last_pos_from_end = flipped.float().argmax(dim=1)
        last_pos = (not_ign.size(1) - 1) - last_pos_from_end

        has_any = not_ign.any(dim=1)
        last_pos = torch.where(has_any, last_pos, torch.full_like(last_pos, -1))
        return last_pos
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float()
        labels = labels.to(logits.device)
        
        B, T, V = logits.shape
        ans_pos = _first_answer_pos(labels)
        
        bad = (ans_pos < 0)
        if bad.any():
            # print("!!! Warning: some examples have no answer token; using last non-pad position instead.")
            ans_pos = torch.where(bad, torch.full_like(ans_pos, T - 1), ans_pos)
            num_bad = bad.sum().item()
            # print(f"\n!!! WARNING: {num_bad}/{B} examples have no answer token")
            
            # DEBUG: Print details of first bad example
            bad_idx = torch.where(bad)[0][0].item()
            # print(f"First bad example (index {bad_idx}):")
            # print(f"  Labels shape: {labels[bad_idx].shape}")
            # print(f"  Labels: {labels[bad_idx].tolist()[:50]}...")  # First 50
            # print(f"  Non-ignore count: {(labels[bad_idx] != -100).sum().item()}")
            # print(f"  Unique values: {torch.unique(labels[bad_idx]).tolist()}")
            ans_pos = torch.where(bad, torch.full_like(ans_pos, T - 1), ans_pos)
        row_idx = torch.arange(B, device=logits.device)
        logits_at_ans = logits[row_idx, ans_pos, :]
        
        choice_ids = CHOICE_TOKEN_IDS.to(logits.device)
        four_logits = logits_at_ans.index_select(dim=1, index=choice_ids)
        return four_logits
    
    def compute_metrics(eval_pred):
        """
        Modified to extract and display debugging information
        """
        extended_preds = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        # Split the extended predictions
        # Shape: (N, 4 + top_k + top_k)
        four_logits = extended_preds[:, :4]
        topk_logits = extended_preds[:, 4:14]  # next 10
        topk_indices = extended_preds[:, 14:24].astype(int)  # next 10
        
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
        
        # Calculate probabilities
        probs = torch.softmax(torch.tensor(four_logits), dim=1).numpy()
        
        # Probability of predicted answer
        pred_probs = probs[np.arange(len(pred_idx)), pred_idx]
        
        # ========== NEW: ABCD-only Cross Entropy Loss ==========
        # This is the loss if we only considered A,B,C,D tokens
        if gold_valid.any():
            # Get probabilities of correct answers (renormalized over ABCD only)
            correct_probs = probs[gold_valid, gold_idx[gold_valid]]
            
            # Cross-entropy loss (ABCD-only)
            abcd_only_loss = -np.log(correct_probs + 1e-10).mean()
            
            calibration = correct_probs.mean()
            simulated_loss = abcd_only_loss  # Keep backward compatibility
        else:
            calibration = 0.0
            abcd_only_loss = 10.0
            simulated_loss = 10.0
        
        pred_counts = np.bincount(pred_idx, minlength=4)
        max_pred_pct = pred_counts.max() / len(pred_idx)
        
        # ========== NEW: DETAILED DEBUGGING ==========
        print(f"\n{'='*80}")
        print(f"EVALUATION DEBUGGING - Showing first {args.debug_samples} examples:")
        print(f"{'='*80}")
        
        for i in range(min(args.debug_samples, len(four_logits))):
            print(f"\n--- Example {i} ---")
            
            # ABCD probabilities
            abcd_probs = probs[i]
            print(f"ABCD Probabilities:")
            for j, letter in enumerate(['A', 'B', 'C', 'D']):
                token_id = CHOICE_TOKEN_IDS[j].item()
                print(f"  {letter} (token {token_id}): {abcd_probs[j]:.4f}")
            
            # Predicted answer among ABCD
            pred_letter = ['A', 'B', 'C', 'D'][pred_idx[i]]
            print(f"Predicted (ABCD only): {pred_letter}")
            
            # Gold answer
            if gold_valid[i]:
                gold_letter = ['A', 'B', 'C', 'D'][gold_idx[i]]
                gold_token = gold_token_ids[i].item()
                gold_prob_abcd = probs[i, gold_idx[i]]
                example_abcd_loss = -np.log(gold_prob_abcd + 1e-10)
                print(f"Gold answer: {gold_letter} (token {gold_token})")
                print(f"Gold probability (ABCD-only): {gold_prob_abcd:.4f}")
                print(f"ABCD-only CE Loss (this example): {example_abcd_loss:.4f}")
                print(f"Correct: {pred_idx[i] == gold_idx[i]}")
            else:
                print(f"Gold answer: INVALID")
            
            # Top 10 tokens overall
            print(f"\nTop 10 tokens by probability (FULL VOCABULARY):")
            top_probs_full = torch.softmax(torch.tensor(topk_logits[i]), dim=0).numpy()
            for j in range(10):
                token_id = topk_indices[i, j]
                token_str = tokenizer.decode([token_id])
                prob = top_probs_full[j]
                
                # Check if this is one of ABCD
                is_choice = ""
                for k, choice_id in enumerate(CHOICE_TOKEN_IDS):
                    if token_id == choice_id.item():
                        is_choice = f" ← {['A','B','C','D'][k]}"
                        break
                
                print(f"  #{j+1}: token {token_id:6d} = '{token_str:10s}' prob={prob:.4f}{is_choice}")
            
            # Check if argmax is outside ABCD
            true_argmax_idx = topk_indices[i, 0]
            true_argmax_token = tokenizer.decode([true_argmax_idx])
            
            is_abcd = any(true_argmax_idx == cid.item() for cid in CHOICE_TOKEN_IDS)
            if not is_abcd:
                print(f"\n⚠️  TRUE ARGMAX IS NOT IN ABCD!")
                print(f"   True argmax: token {true_argmax_idx} = '{true_argmax_token}'")
                print(f"   This explains why loss increases while ABCD-accuracy might stay high!")
        
        print(f"\n{'='*80}")
        print(f"Overall Evaluation Metrics:")
        print(f"  Accuracy (ABCD only): {acc:.4f}")
        print(f"  ABCD-only Cross Entropy Loss: {abcd_only_loss:.4f}")
        print(f"  Calibration (correct answer prob): {calibration:.4f}")
        print(f"  Prediction confidence (mean): {pred_probs.mean():.4f}")
        
        if max_pred_pct > 0.4:
            print(f"  🚨 MODEL COLLAPSE: {max_pred_pct*100:.1f}% predictions are one answer!")
        elif max_pred_pct > 0.35:
            print(f"  ⚠️  Prediction imbalance: {max_pred_pct*100:.1f}%")
        else:
            print(f"  ✅ Predictions balanced: max={max_pred_pct*100:.1f}%")
        
        if calibration < 0.3 and acc > 0.3:
            print(f"  ⚠️  Poor calibration: model overconfident in wrong answers")
        elif calibration > 0.5:
            print(f"  ✅ Good calibration")
        
        print(f"  Prediction distribution: A={pred_counts[0]}, B={pred_counts[1]}, "
              f"C={pred_counts[2]}, D={pred_counts[3]}")
        
        # Check how many examples have argmax outside ABCD
        num_argmax_outside_abcd = 0
        for i in range(len(topk_indices)):
            true_argmax = topk_indices[i, 0]
            is_abcd = any(true_argmax == cid.item() for cid in CHOICE_TOKEN_IDS)
            if not is_abcd:
                num_argmax_outside_abcd += 1
        
        pct_outside = 100.0 * num_argmax_outside_abcd / len(topk_indices)
        print(f"\n  📊 Argmax outside ABCD: {num_argmax_outside_abcd}/{len(topk_indices)} ({pct_outside:.1f}%)")
        if pct_outside > 10:
            print(f"     ⚠️  Model is frequently predicting tokens outside A,B,C,D!")
            print(f"     This causes high loss even if ABCD-accuracy looks okay.")
        
        # ========== NEW: Estimate full vocabulary loss for comparison ==========
        # For examples where we have gold answers, estimate what the loss would be
        # if we considered the full vocabulary
        if gold_valid.any():
            full_vocab_loss_estimates = []
            for i in range(len(topk_indices)):
                if not gold_valid[i]:
                    continue
                    
                gold_token = gold_token_ids[i].item()
                
                # Check if gold token is in top-K
                top_k_tokens = topk_indices[i]
                if gold_token in top_k_tokens:
                    # Find position
                    pos = np.where(top_k_tokens == gold_token)[0][0]
                    # Get probability (need to softmax the top-k logits)
                    top_k_probs = torch.softmax(torch.tensor(topk_logits[i]), dim=0).numpy()
                    gold_prob_full = top_k_probs[pos]
                else:
                    # Gold token not in top-10, assume very low probability
                    gold_prob_full = 1e-6
                
                full_vocab_loss_estimates.append(-np.log(gold_prob_full + 1e-10))
            
            if full_vocab_loss_estimates:
                estimated_full_vocab_loss = np.mean(full_vocab_loss_estimates)
                print(f"\n  📉 Loss Comparison:")
                print(f"     ABCD-only loss: {abcd_only_loss:.4f}")
                print(f"     Estimated full-vocab loss: {estimated_full_vocab_loss:.4f}")
                print(f"     Difference: {estimated_full_vocab_loss - abcd_only_loss:.4f}")
                
                if estimated_full_vocab_loss > abcd_only_loss + 0.5:
                    print(f"     ⚠️  Full-vocab loss is MUCH higher!")
                    print(f"     This indicates the model assigns low probability to correct tokens")
                    print(f"     in the full vocabulary, even though ABCD-only metrics look okay.")
        
        
        print(f"{'='*80}\n")
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
        save_steps=args.eval_steps,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
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
        label_smoothing_factor=args.label_smoothing
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