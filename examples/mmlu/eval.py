import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Configuration
MODEL_DIR = "/pscratch/sd/l/lsx/shouxu_runs/Qwen_Qwen2.5-0.5B-Instruct-cais_mmlu"
DATASET = "cais/mmlu"
MAX_LENGTH = 512
BATCH_SIZE = 1

# Load tokenizer and model from your trained checkpoint
print("Loading tokenizer and model from checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    trust_remote_code=True,
)

# Load MMLU dataset
print("Loading MMLU test dataset...")
dataset = load_dataset(DATASET, "all")

# Format MMLU data into instruction format (same as training)
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
    completion = f" {answer}"
    
    # Combine for training
    full_text = prompt + completion
    
    return {"text": full_text}

def tokenize_function(examples):
    """Tokenize the text data"""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# Process test dataset
print("Processing test dataset...")
test_dataset = dataset["test"].map(format_mmlu_example, remove_columns=dataset["test"].column_names)
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# You can also evaluate on validation set
val_dataset = dataset["validation"].map(format_mmlu_example, remove_columns=dataset["validation"].column_names)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# CRITICAL: Preprocess logits to only keep the predicted token indices
def preprocess_logits_for_metrics(logits, labels):
    """
    Reduce logits to just the argmax predictions to save memory.
    This is called for each batch before accumulating predictions.
    """
    # logits shape: (batch_size, seq_length, vocab_size)
    # We only need the predicted token index, not the full distribution
    pred_ids = torch.argmax(logits, dim=-1)  # (batch_size, seq_length)
    return pred_ids

def compute_metrics(eval_pred):
    """Compute accuracy for MMLU evaluation"""
    predictions, labels = eval_pred
    # predictions are already argmax'd token IDs from preprocess_logits_for_metrics
    
    # Create mask for non-padding/non-masked positions
    mask = labels != -100
    
    # Only compare at positions where we have labels
    correct = 0
    total = 0
    
    for pred_seq, label_seq, mask_seq in zip(predictions, labels, mask):
        # Get the last non-masked position (where the answer letter should be)
        valid_positions = np.where(mask_seq)[0]
        if len(valid_positions) > 0:
            # Check the last valid position (the answer)
            last_pos = valid_positions[-1]
            if pred_seq[last_pos] == label_seq[last_pos]:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}

# Create TrainingArguments for evaluation
eval_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_eval_batch_size=BATCH_SIZE,
    bf16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
)

# Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    args=eval_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# Evaluate on validation set
print("\n" + "="*50)
print("Evaluating on validation set...")
print("="*50)
val_results = trainer.evaluate(val_dataset)
print(f"\nValidation Results:")
print(f"  Loss: {val_results['eval_loss']:.4f}")
print(f"  Accuracy: {val_results['eval_accuracy']:.4f}")

# # Evaluate on test set
# print("\n" + "="*50)
# print("Evaluating on test set...")
# print("="*50)
# test_results = trainer.evaluate(test_dataset)
# print(f"\nTest Results:")
# print(f"  Loss: {test_results['eval_loss']:.4f}")
# print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")

print("\nEvaluation complete!")