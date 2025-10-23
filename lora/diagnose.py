#!/usr/bin/env python3
"""
Standalone diagnostic script - evaluates model on dataset WITHOUT training
Quickly shows prediction patterns, data issues, and model behavior

Usage:
    python diagnose_only.py --model Qwen/Qwen2.5-0.5B-Instruct
    python diagnose_only.py --model /path/to/your/lora/adapter --use_lora
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser(description="Diagnose model on MMLU without training")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--use_lora", action="store_true", help="Load as LoRA adapter")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Base model if using LoRA")
    p.add_argument("--dataset", type=str, default="cais/mmlu")
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_examples", type=int, default=1000,
                   help="Number of examples to evaluate (0 = all)")
    p.add_argument("--split", type=str, default="test", choices=["test", "auxiliary_train"])
    return p.parse_args()

def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC EVALUATION (NO TRAINING)")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    if args.num_examples > 0:
        print(f"Evaluating: {args.num_examples} examples")
    else:
        print(f"Evaluating: ALL examples")
    print(f"{'='*70}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model if args.use_lora else args.model,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    if args.use_lora:
        print(f"  Base: {args.base_model}")
        print(f"  LoRA adapter: {args.model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    model.eval()
    print("Model loaded successfully\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, "all")
    eval_dataset = dataset[args.split]
    
    if args.num_examples > 0:
        eval_dataset = eval_dataset.select(range(min(args.num_examples, len(eval_dataset))))
    
    print(f"Dataset loaded: {len(eval_dataset)} examples\n")
    
    # Format examples
    def format_mmlu_example(example):
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        
        choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = f"""Answer the following multiple choice question.

Question: {question}

Choices:
{choice_text}

Answer:"""
        
        answer = chr(65 + answer_idx)
        answer = f" {answer}"
        full_text = prompt + answer
        
        return {
            "prompt": prompt,
            "label": answer,
            "full_text": full_text,
            "answer_idx": answer_idx
        }
    
    print("Formatting examples...")
    eval_dataset = eval_dataset.map(format_mmlu_example)
    
    # Setup choice tokens
    CHOICES = [" A", " B", " C", " D"]
    CHOICE_TOKEN_IDS_LIST = []
    for s in CHOICES:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            print(f"⚠️  WARNING: Choice '{s}' tokenizes to {len(ids)} tokens: {ids}")
            CHOICE_TOKEN_IDS_LIST.append(ids[0] if ids else 0)
        else:
            CHOICE_TOKEN_IDS_LIST.append(ids[0])
    
    CHOICE_TOKEN_IDS = torch.tensor(CHOICE_TOKEN_IDS_LIST, dtype=torch.long)
    print(f"Choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")
    print(f"Decoded: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}\n")
    
    # Evaluate
    print("="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    predictions = []
    gold_labels = []
    all_probs = []
    invalid_count = 0
    
    device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(eval_dataset), args.batch_size), desc="Evaluating"):
        batch = eval_dataset[i:i+args.batch_size]
        
        # Tokenize batch
        prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
        
        inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab]
        
        # Get logits at the last token position (after "Answer:")
        # Find last non-pad position for each example
        batch_size = logits.shape[0]
        last_pos = (inputs.attention_mask.sum(dim=1) - 1).long()
        
        # Extract logits at last position
        row_idx = torch.arange(batch_size, device=device)
        logits_at_last = logits[row_idx, last_pos, :]
        
        # Get logits for the 4 choice tokens
        choice_logits = logits_at_last[:, CHOICE_TOKEN_IDS.to(device)]
        
        # Get probabilities
        probs = torch.softmax(choice_logits, dim=1).cpu().numpy()
        
        # Get predictions
        pred_idx = choice_logits.argmax(dim=1).cpu().numpy()
        
        # Get gold labels
        gold_idx = batch["answer_idx"] if isinstance(batch["answer_idx"], list) else [batch["answer_idx"]]
        
        predictions.extend(pred_idx.tolist())
        gold_labels.extend(gold_idx)
        all_probs.extend(probs.tolist())
    
    predictions = np.array(predictions)
    gold_labels = np.array(gold_labels)
    all_probs = np.array(all_probs)
    
    # ============================================
    # DIAGNOSTIC OUTPUT
    # ============================================
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RESULTS")
    print(f"{'='*70}")
    
    # 1. Basic accuracy
    correct = (predictions == gold_labels)
    accuracy = correct.mean()
    
    print(f"\n1. OVERALL ACCURACY: {accuracy:.4f} ({correct.sum()}/{len(correct)} correct)")
    if accuracy < 0.26:
        print(f"   ⚠️  Close to random (25%) - model not learning or not trained")
    elif accuracy < 0.3:
        print(f"   📊 Slightly above random - minimal learning")
    elif accuracy < 0.4:
        print(f"   ✅ Decent - model is learning")
    else:
        print(f"   ✅ Good - model performing well")
    
    # 2. Prediction distribution
    pred_counts = np.bincount(predictions, minlength=4)
    print(f"\n2. PREDICTION DISTRIBUTION:")
    for i, letter in enumerate(['A', 'B', 'C', 'D']):
        pct = pred_counts[i] / len(predictions) * 100
        bar = '█' * int(pct / 2)
        print(f"   {letter}: {pred_counts[i]:5d} ({pct:5.1f}%) {bar}", end="")
        
        if pct > 40:
            print(" 🚨 MODEL COLLAPSE!", end="")
        elif pct > 35:
            print(" ⚠️  Imbalanced", end="")
        print()
    
    max_pred_pct = pred_counts.max() / len(predictions)
    if max_pred_pct > 0.4:
        print(f"\n   🚨 SEVERE MODEL COLLAPSE: {max_pred_pct*100:.1f}% predictions are one answer")
        print(f"   This means the model always (or almost always) predicts the same choice!")
    elif max_pred_pct > 0.35:
        print(f"\n   ⚠️  PREDICTION BIAS: {max_pred_pct*100:.1f}% predictions favor one answer")
    else:
        print(f"\n   ✅ Predictions are reasonably balanced")
    
    # 3. Gold label distribution
    gold_counts = np.bincount(gold_labels, minlength=4)
    print(f"\n3. GOLD LABEL DISTRIBUTION (Ground Truth):")
    for i, letter in enumerate(['A', 'B', 'C', 'D']):
        pct = gold_counts[i] / len(gold_labels) * 100
        bar = '█' * int(pct / 2)
        print(f"   {letter}: {gold_counts[i]:5d} ({pct:5.1f}%) {bar}")
    
    max_gold_pct = gold_counts.max() / len(gold_labels)
    if max_gold_pct > 0.35:
        print(f"\n   ⚠️  Dataset imbalance: {max_gold_pct*100:.1f}% of answers are one choice")
        print(f"   This can bias the model toward predicting that choice more often")
    
    # 4. Probability analysis
    pred_probs = all_probs[np.arange(len(predictions)), predictions]
    correct_probs = all_probs[np.arange(len(gold_labels)), gold_labels]
    
    print(f"\n4. PROBABILITY ANALYSIS:")
    print(f"   Predicted answer confidence:")
    print(f"     Mean: {pred_probs.mean():.4f}")
    print(f"     Median: {np.median(pred_probs):.4f}")
    print(f"     Min: {pred_probs.min():.4f}, Max: {pred_probs.max():.4f}")
    
    print(f"\n   Correct answer probability:")
    print(f"     Mean: {correct_probs.mean():.4f}")
    print(f"     Median: {np.median(correct_probs):.4f}")
    print(f"     Min: {correct_probs.min():.4f}, Max: {correct_probs.max():.4f}")
    
    # Simulated loss
    simulated_loss = -np.log(correct_probs + 1e-10).mean()
    print(f"\n   Simulated loss: {simulated_loss:.4f}")
    
    if pred_probs.mean() > 0.9:
        print(f"\n   🚨 OVERCONFIDENCE: Model is too confident (avg {pred_probs.mean():.2f})")
        print(f"   This can lead to high loss when wrong")
    elif pred_probs.mean() > 0.7:
        print(f"\n   ⚠️  High confidence: {pred_probs.mean():.2f}")
    
    if correct_probs.mean() < 0.3:
        print(f"\n   🚨 LOW CONFIDENCE ON CORRECT ANSWERS: {correct_probs.mean():.4f}")
        print(f"   This explains why loss might be high even with decent accuracy!")
        print(f"   The model gets the right answer but assigns it low probability")
    
    # 5. Per-answer-choice accuracy
    print(f"\n5. ACCURACY BY GOLD LABEL:")
    for choice_idx in range(4):
        mask = gold_labels == choice_idx
        if mask.sum() > 0:
            choice_acc = (predictions[mask] == choice_idx).mean()
            choice_letter = chr(65 + choice_idx)
            bar = '█' * int(choice_acc * 50)
            print(f"   When answer is {choice_letter}: {choice_acc*100:5.1f}% correct ({mask.sum():4d} examples) {bar}")
    
    # 6. Confusion matrix (top errors)
    print(f"\n6. MOST COMMON ERRORS:")
    errors = []
    for i in range(len(predictions)):
        if predictions[i] != gold_labels[i]:
            pred_letter = chr(65 + predictions[i])
            gold_letter = chr(65 + gold_labels[i])
            errors.append((pred_letter, gold_letter))
    
    if errors:
        error_counts = Counter(errors)
        print(f"   Total errors: {len(errors)}/{len(predictions)} ({len(errors)/len(predictions)*100:.1f}%)")
        print(f"\n   Top confusion patterns:")
        for (pred_letter, gold_letter), count in error_counts.most_common(10):
            pct = count / len(errors) * 100
            print(f"     Predicted {pred_letter} when answer was {gold_letter}: {count:4d} times ({pct:4.1f}% of errors)")
    
    # 7. Sample predictions
    print(f"\n7. SAMPLE PREDICTIONS:")
    sample_indices = np.random.choice(len(predictions), size=min(10, len(predictions)), replace=False)
    for idx in sample_indices:
        pred_letter = chr(65 + predictions[idx])
        gold_letter = chr(65 + gold_labels[idx])
        prob = all_probs[idx, predictions[idx]]
        correct_prob = all_probs[idx, gold_labels[idx]]
        
        status = "✓" if predictions[idx] == gold_labels[idx] else "✗"
        print(f"   Example {idx}: pred={pred_letter} (p={prob:.3f}), gold={gold_letter} (p={correct_prob:.3f}) {status}")
    
    # 8. Summary and recommendations
    print(f"\n{'='*70}")
    print(f"SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")
    
    issues_found = []
    
    if max_pred_pct > 0.4:
        issues_found.append("MODEL COLLAPSE")
        print(f"🚨 MODEL COLLAPSE DETECTED")
        print(f"   The model predicts one answer {max_pred_pct*100:.1f}% of the time")
        print(f"   → Reduce learning rate (try 5e-5 instead of 1e-4)")
        print(f"   → Reduce LoRA rank (try r=64 instead of r=256)")
        print(f"   → Use fewer target modules (only q_proj, v_proj)")
        print(f"   → Add label smoothing (--label_smoothing 0.1)")
    
    if correct_probs.mean() < 0.3:
        issues_found.append("LOW CONFIDENCE")
        print(f"🚨 LOW CONFIDENCE ON CORRECT ANSWERS")
        print(f"   Model assigns avg probability of {correct_probs.mean():.3f} to correct answers")
        print(f"   This causes high loss even with decent accuracy")
        print(f"   → Add label smoothing to prevent overconfidence elsewhere")
        print(f"   → Train longer or with higher learning rate")
    
    if max_gold_pct > 0.35:
        issues_found.append("DATASET IMBALANCE")
        print(f"⚠️  DATASET IMBALANCE DETECTED")
        print(f"   {max_gold_pct*100:.1f}% of answers are one choice")
        print(f"   → Consider using class weights")
        print(f"   → Use label smoothing")
    
    if accuracy < 0.3:
        issues_found.append("LOW ACCURACY")
        print(f"⚠️  LOW ACCURACY")
        print(f"   Model barely better than random guessing")
        print(f"   → Check if model was actually trained")
        print(f"   → Increase learning rate")
        print(f"   → Train for more epochs")
    
    if not issues_found:
        print(f"✅ NO MAJOR ISSUES DETECTED")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Predictions balanced: {max_pred_pct*100:.1f}% max")
        print(f"   Correct answer confidence: {correct_probs.mean():.4f}")
        if accuracy < 0.4:
            print(f"\n   To improve accuracy further:")
            print(f"   → Train longer (more epochs)")
            print(f"   → Increase LoRA rank (if currently low)")
            print(f"   → Tune learning rate")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()