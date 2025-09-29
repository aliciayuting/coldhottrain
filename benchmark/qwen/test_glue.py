import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
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


tokenized_ds: DatasetDict = ds.map(format_example, batched=False) # type: ignore


split_seed = 42

tokenized_ds = tokenized_ds["train"].train_test_split(
    test_size=VALIDATION_FRACTION,
    seed=split_seed,
)
tokenized_ds["validation"] = tokenized_ds.pop("test")

train_dataset = tokenized_ds["train"]
eval_dataset = tokenized_ds["validation"]
print(eval_dataset)
eval_dataset = eval_dataset.remove_columns(['instruction', 'input', 'output', 'text'])

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model.to(device)
model.eval()

eval_loader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collator,
    pin_memory=device.type == "cuda",
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
        print(f"Processed {token_count} tokens", end="\r")

eval_loss = loss_sum / max(token_count, 1)
print()
print(f"Eval loss: {eval_loss:.4f}")
