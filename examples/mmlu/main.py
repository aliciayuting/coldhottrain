import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers

import numpy as np


# Set random seed for reproducibility
transformers.set_seed(42)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DATASET = "cais/mmlu"
OUTPUT_DIR = f"/pscratch/sd/l/lsx/shouxu_runs/{MODEL_NAME.replace('/', '_')}-{DATASET.replace('/', '_')}"
MAX_LENGTH = 512
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 50
WARMUP_STEPS = 100

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    trust_remote_code=True,
)

# Configure LoRA for efficient finetuning
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# Load MMLU dataset
print("Loading MMLU dataset...")
dataset = load_dataset(DATASET, "all")

# Format MMLU data into instruction format
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

# Process datasets
print("Processing datasets...")
train_dataset = dataset["auxiliary_train"].map(format_mmlu_example, remove_columns=dataset["auxiliary_train"].column_names)
# train_dataset = dataset["test"].map(format_mmlu_example, remove_columns=dataset["test"].column_names)
val_dataset = dataset["validation"].map(format_mmlu_example, remove_columns=dataset["validation"].column_names)
# test_dataset = dataset["test"].map(format_mmlu_example, remove_columns=dataset["test"].column_names)

# Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# for i, example in enumerate(train_dataset):
#     if i < 1:
#         print(f"Example {i}:")
#         print(f"  Input IDs length: {len(example['input_ids'])}")
#         print(f"  Labels length: {len(example['labels'])}")
#         non_ignore_labels = [l for l in example['labels'] if l != -100]
#         print(f"  Labels (non -100): {non_ignore_labels}")
#         print(f"  Decoded: {tokenizer.decode(non_ignore_labels)}")
#         print()
#         break


# for i, example in enumerate(val_dataset):
#     if i < 1:
#         print(f"Example {i}:")
#         print(f"  Input IDs length: {len(example['input_ids'])}")
#         print(f"  Labels length: {len(example['labels'])}")
#         non_ignore_labels = [l for l in example['labels'] if l != -100]
#         print(f"  Labels (non -100): {non_ignore_labels}")
#         print(f"  Decoded: {tokenizer.decode(non_ignore_labels)}")
#         print()
#         break

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
    print()
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
            # print(f"Predicted: {tokenizer.decode(pred_seq[last_pos]).strip()}, Actual: {tokenizer.decode(label_seq[last_pos]).strip()}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}

# Training arguments with data parallelism
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=1,
    # eval_strategy="epoch",
    eval_strategy="steps",
    eval_steps=1000,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Data parallelism settings
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    resume_from_checkpoint=False,
    # Reporting
    # report_to="tensorboard",
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    # max_steps=5,
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=200,
    
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# Train
print("Starting training...")
trainer.train()

# Save final model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")

# # Optional: Evaluate on test set
# print("\nEvaluating on test set...")
# test_results = trainer.evaluate(test_dataset)
# print(f"Test results: {test_results}")