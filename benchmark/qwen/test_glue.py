import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import evaluate
import sys

DATASET = "sst2"
NUM_LABELS = 2
EPOCH_LENGTH = 527
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")


main_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "jamal_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca-neurons-80p-1e-randommask-20250927-002901")
checkpoint_dir = os.path.join(main_dir, "ckpt")
gradient_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(main_dir, "grad_dump/step004500")
masks_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(main_dir, "neuron_masks_0.pt")
epoch_num = int(sys.argv[4]) if len(sys.argv) > 4 else 30
MODEL = os.path.join(checkpoint_dir, f"checkpoint-{EPOCH_LENGTH*epoch_num}")

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# If tokenizer has no pad token (common for causal LMs), set it:
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    
    
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    num_labels=NUM_LABELS
    # device_map="auto")
)
model.config.pad_token_id = tok.pad_token_id

ds = load_dataset("nyu-mll/glue",DATASET)


# Preprocess into prompt–response format
def tokenize_function(examples):
    return tok(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_ds = ds.map(tokenize_function, batched=False)

tokenized_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

print(tokenized_ds)
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tok)

accuracy_metric = evaluate.load("accuracy")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model.to(device)
model.eval()

eval_loader = DataLoader(
    tokenized_ds['validation'],
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=device.type == "cuda",
)
all_predictions = []
all_labels = []
loss_sum = 0.0
num_batches = 0

with torch.inference_mode():
    for batch in eval_loader:
        # Move batch to device
        print("Batch keys:", batch.keys())
        print("Batch structure:", {k: v.shape if hasattr(v, 'shape') else type(v) for k, v in batch.items()})
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # Note: "label" not "labels"
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get predictions
        logits = outputs.logits  # Shape: (batch_size, num_labels)
        predictions = torch.argmax(logits, dim=-1)
        
        # Collect predictions and labels
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Accumulate loss
        loss_sum += outputs.loss.item()
        num_batches += 1
        
        print(f"Processed {num_batches * 8} samples", end="\r")

# Compute final metrics
accuracy = accuracy_metric.compute(
    predictions=all_predictions,
    references=all_labels
)

avg_loss = loss_sum / num_batches

print()
print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy['accuracy']:.4f}")