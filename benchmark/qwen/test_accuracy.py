import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

DATASET = "tatsu-lab/alpaca"
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
EPOCH_LENGTH = 407
VALIDATION_FRACTION = 0.1     # Hold out 10% for validation

main_dir = os.path.join(SCRATCH, "jamal_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca-neurons-50p-1e-20250917-163518")
checkpoint_dir = os.path.join(main_dir, "ckpt")
gradient_dir = os.path.join(main_dir, "grad_dump/step004500")
masks_path = os.path.join(main_dir, "neuron_masks.pt")

MODEL = os.path.join(checkpoint_dir, f"checkpoint-{EPOCH_LENGTH*27}")

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# If tokenizer has no pad token (common for causal LMs), set it:
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    # device_map="auto")
)

ds = load_dataset(DATASET)



# Preprocess into prompt–response format
def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    response = example["output"]

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return tok(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_ds = ds.map(format_example, batched=False)


split_seed = 42

tokenized_ds = tokenized_ds["train"].train_test_split(
    test_size=VALIDATION_FRACTION,
    seed=split_seed,
)
tokenized_ds["validation"] = tokenized_ds.pop("test")

train_dataset = tokenized_ds["train"]
eval_dataset = tokenized_ds["validation"]
eval_dataset = eval_dataset.remove_columns(['instruction', 'input', 'output'])

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
model.eval()

eval_loader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collator,
    pin_memory=device.type == "cpu",
)

loss_sum = 0.0
token_count = 0

with torch.inference_mode():
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid_tokens = (batch["labels"] != -100).sum().item()
        if valid_tokens == 0:
            continue
        outputs = model(**batch)
        loss_sum += outputs.loss.item() * valid_tokens
        token_count += valid_tokens

eval_loss = loss_sum / max(token_count, 1)
print(f"Eval loss: {eval_loss:.4f}")
