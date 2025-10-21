import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import transformers
import numpy as np
from sklearn.metrics import accuracy_score

# ---------- Repro ----------
transformers.set_seed(42)

# ---------- Config ----------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "cais/mmlu"
OUTPUT_DIR = f"/pscratch/sd/l/lsx/shouxu_runs/{MODEL_NAME.replace('/', '_')}-{DATASET.replace('/', '_')}"
MAX_LENGTH = 768
BATCH_SIZE = 16

# ---------- Tokenizer ----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token

# ---------- Model (eval mode, no grad) ----------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
torch.set_grad_enabled(False)

# ---------- Load MMLU (eval split only) ----------
print("Loading MMLU dataset...")
dataset = load_dataset(DATASET, "all")
eval_dataset = dataset["test"]

# ---------- Format MMLU as prompt + gold letter ----------
def format_mmlu_example(example):
    question = example["question"]
    choices = example["choices"]
    answer_idx = example["answer"]
    choice_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    prompt = (
        "Answer the following multiple choice question.\n\n"
        f"Question: {question}\n\nChoices:\n{choice_text}\n\nAnswer:"
    )
    answer = f" {chr(65 + answer_idx)}"
    full_text = prompt + answer
    return {"prompt": prompt, "answer": answer, "full_text": full_text}

print("Formatting eval dataset...")
eval_dataset = eval_dataset.map(format_mmlu_example, remove_columns=eval_dataset.column_names)

# ---------- Tokenize ----------
def tokenize_function(examples):
    enc = tokenizer(
        examples["full_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        add_special_tokens=False,
    )
    enc_prompt = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    prompt_lens = [len(x) for x in enc_prompt["input_ids"]]

    labels = [seq.copy() for seq in enc["input_ids"]]
    for i, mask in enumerate(enc["attention_mask"]):
        for j, m in enumerate(mask):
            if m == 0:
                labels[i][j] = -100
    for i, plen in enumerate(prompt_lens):
        upto = min(plen, len(labels[i]))
        for j in range(upto):
            if labels[i][j] != -100:
                labels[i][j] = -100
    enc["labels"] = labels
    return enc

print("Tokenizing eval dataset...")
eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["prompt", "answer"],
)

# ---------- 4-choice setup ----------
CHOICES = [" A", " B", " C", " D"]
CHOICE_TOKEN_IDS = []
for s in CHOICES:
    ids = tokenizer.encode(s, add_special_tokens=False)
    assert len(ids) == 1, f"Choice token '{s}' did not encode to a single token: {ids}"
    CHOICE_TOKEN_IDS.append(ids[0])
CHOICE_TOKEN_IDS = torch.tensor(CHOICE_TOKEN_IDS, dtype=torch.long)
print(f"Choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")

# ---------- Helpers for metrics ----------
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
        ans_pos = torch.where(bad, torch.full_like(ans_pos, T - 1), ans_pos)

    row_idx = torch.arange(B, device=logits.device)
    logits_at_ans = logits[row_idx, ans_pos, :]

    choice_ids = CHOICE_TOKEN_IDS.to(logits.device)
    four_logits = logits_at_ans.index_select(dim=1, index=choice_ids)
    return four_logits

def compute_metrics(eval_pred):
    four_logits = eval_pred.predictions  # (N, 4) - logits for A, B, C, D
    label_ids = eval_pred.label_ids      # (N, T) - tokenized labels

    # Get predicted choice index (0=A, 1=B, 2=C, 3=D)
    pred_idx = np.asarray(four_logits).argmax(axis=1)  # (N,)

    # Get gold choice index from labels
    labels = torch.tensor(label_ids)
    ans_pos = _first_answer_pos(labels)
    has_any = (ans_pos >= 0)

    row_idx = torch.arange(labels.size(0))
    gold_token_ids = torch.full((labels.size(0),), -1, dtype=torch.long)
    gold_token_ids[has_any] = labels[row_idx[has_any], ans_pos[has_any]]

    choice_ids = CHOICE_TOKEN_IDS
    eq_matrix = (gold_token_ids[:, None] == choice_ids[None, :])
    gold_idx = eq_matrix.long().argmax(dim=1).numpy()  # (N,)
    gold_valid = eq_matrix.any(dim=1).numpy()

    acc = accuracy_score(gold_idx[gold_valid], pred_idx[gold_valid]) if gold_valid.any() else 0.0

    # ========== CHECK ALL EXAMPLES FOR INVALID PREDICTIONS/LABELS ==========
    print("\n" + "="*80)
    print("CHECKING ALL PREDICTIONS AND LABELS")
    print("="*80)
    print(f"Expected choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")
    print(f"Expected tokens: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}")
    print("="*80)
    
    invalid_cases = []
    
    for i in range(len(pred_idx)):
        # Convert index to letter
        pred_letter = chr(65 + pred_idx[i])      # 0→A, 1→B, 2→C, 3→D
        gold_letter = chr(65 + gold_idx[i])      # 0→A, 1→B, 2→C, 3→D
        
        # Check if valid
        pred_valid = pred_letter in ['A', 'B', 'C', 'D']
        gold_valid_check = gold_valid[i]  # Check if actual token matches expected tokens
        
        # Store if label is invalid (prediction should always be valid since we argmax over 4 choices)
        if not gold_valid_check:
            # Get the actual token ID from labels
            actual_token_id = gold_token_ids[i].item()
            actual_token_decoded = tokenizer.decode([actual_token_id]) if actual_token_id >= 0 else "N/A"
            
            invalid_cases.append({
                'index': i,
                'pred_letter': pred_letter,
                'gold_letter': gold_letter,
                'actual_token_id': actual_token_id,
                'actual_token_decoded': actual_token_decoded,
                'gold_valid': gold_valid_check
            })
    
    # Print results
    if invalid_cases:
        print(f"\n⚠️  FOUND {len(invalid_cases)} INVALID LABEL CASES:\n")
        print("These labels have token IDs that don't match [' A', ' B', ' C', ' D']\n")
        for case in invalid_cases[:20]:  # Show first 20
            print(f"Example {case['index']:5d}: "
                  f"Actual token ID={case['actual_token_id']:6d} "
                  f"Decoded='{case['actual_token_decoded']}'  |  "
                  f"Expected one of {CHOICE_TOKEN_IDS.tolist()}  |  "
                  f"Prediction={case['pred_letter']}")
        if len(invalid_cases) > 20:
            print(f"\n... and {len(invalid_cases) - 20} more invalid cases")
    else:
        print("\n✓ ALL LABELS ARE VALID (match expected token IDs for A, B, C, D)")
    
    print("\n" + "="*80)
    print(f"Total examples checked: {len(pred_idx)}")
    print(f"Invalid labels: {len(invalid_cases)}")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("="*80 + "\n")

    return {"accuracy": acc}

# ---------- Training Arguments ----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=False,
    bf16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=False,
    logging_strategy="no",
)

# ---------- Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# ---------- Run evaluation ----------
print("\nEvaluating on MMLU (test split)...")
metrics = trainer.evaluate()
print("\nFinal metrics:", metrics)