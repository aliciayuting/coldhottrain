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
MODEL_NAME = "luffycodes/vicuna-mmlu-val-only-correct-mcq-7b-ep2"  # 7B model
DATASET = "cais/mmlu"
OUTPUT_DIR = f"/pscratch/sd/l/lsx/shouxu_runs/{MODEL_NAME.replace('/', '_')}-{DATASET.replace('/', '_')}"
MAX_LENGTH = 768
BATCH_SIZE = 4  # Reduced for 7B model (safer with larger model)

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
CHOICE_TOKEN_IDS_LIST = []  # List of token sequences
for s in CHOICES:
    ids = tokenizer.encode(s, add_special_tokens=False)
    CHOICE_TOKEN_IDS_LIST.append(ids)

# Check if single token or multi-token
SINGLE_TOKEN = all(len(ids) == 1 for ids in CHOICE_TOKEN_IDS_LIST)

if SINGLE_TOKEN:
    CHOICE_TOKEN_IDS = torch.tensor([ids[0] for ids in CHOICE_TOKEN_IDS_LIST], dtype=torch.long)
    print(f"Choice tokens are SINGLE tokens")
    print(f"Choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")
else:
    print(f"Choice tokens are MULTI-TOKEN")
    print(f"Choice token sequences: {CHOICE_TOKEN_IDS_LIST}")
    # For multi-token where first token is space, use the SECOND token (the actual letter)
    if all(len(ids) >= 2 for ids in CHOICE_TOKEN_IDS_LIST):
        CHOICE_TOKEN_IDS = torch.tensor([ids[1] for ids in CHOICE_TOKEN_IDS_LIST], dtype=torch.long)
        print(f"Using second token (actual letter): {CHOICE_TOKEN_IDS.tolist()}")
        print(f"Decoded: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}")
    else:
        raise ValueError("Multi-token choices don't have consistent 2-token structure")
    print("⚠️  Note: This assumes answer format is always ' X' (space + letter)")

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
    print(f"Expected choice tokens: {CHOICES}")
    if SINGLE_TOKEN:
        print(f"Expected token IDs: {CHOICE_TOKEN_IDS.tolist()}")
        print(f"Decoded: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}")
    else:
        print(f"Expected token sequences: {CHOICE_TOKEN_IDS_LIST}")
        print(f"Using letter tokens: {CHOICE_TOKEN_IDS.tolist()}")
        print(f"Decoded letters: {[tokenizer.decode([tid]) for tid in CHOICE_TOKEN_IDS.tolist()]}")
    print("="*80)
    
    invalid_cases = []
    format_issues = []
    
    for i in range(len(pred_idx)):
        # Convert index to letter
        pred_letter = chr(65 + pred_idx[i])
        gold_letter = chr(65 + gold_idx[i])
        
        # Check if label token is valid
        gold_valid_check = gold_valid[i]
        
        # For invalid cases, decode the actual label tokens from the dataset
        if not gold_valid_check:
            # Find the answer position in labels
            label_tensor = torch.tensor(label_ids[i])
            ans_positions = torch.where(label_tensor != -100)[0]
            
            if len(ans_positions) > 0:
                # Get all non-masked label tokens
                actual_token_ids = [label_ids[i][pos] for pos in ans_positions.tolist()]
                actual_decoded = tokenizer.decode(actual_token_ids)
                
                invalid_cases.append({
                    'index': i,
                    'pred_letter': pred_letter,
                    'gold_letter': gold_letter,
                    'actual_token_ids': actual_token_ids,
                    'actual_decoded': actual_decoded,
                    'gold_valid': gold_valid_check
                })
            else:
                invalid_cases.append({
                    'index': i,
                    'pred_letter': pred_letter,
                    'gold_letter': gold_letter,
                    'actual_token_ids': [],
                    'actual_decoded': 'NO_TOKENS',
                    'gold_valid': gold_valid_check
                })
        else:
            # Even for valid cases, verify format in first few examples
            if i < 10:
                label_tensor = torch.tensor(label_ids[i])
                ans_positions = torch.where(label_tensor != -100)[0]
                if len(ans_positions) > 0:
                    actual_token_ids = [label_ids[i][pos] for pos in ans_positions.tolist()]
                    actual_decoded = tokenizer.decode(actual_token_ids)
                    expected_decoded = f" {gold_letter}"
                    
                    if actual_decoded != expected_decoded:
                        format_issues.append({
                            'index': i,
                            'expected': expected_decoded,
                            'actual': actual_decoded,
                            'token_ids': actual_token_ids
                        })
    
    # Print format check for first 10
    if format_issues:
        print(f"\n⚠️  FORMAT ISSUES in first 10 examples:\n")
        for issue in format_issues:
            print(f"Example {issue['index']}: Expected='{issue['expected']}' "
                  f"Got='{issue['actual']}' Token IDs={issue['token_ids']}")
        print()
    else:
        print(f"\n✓ First 10 examples have correct format (space + letter)\n")
    
    # Print invalid cases
    if invalid_cases:
        print(f"⚠️  FOUND {len(invalid_cases)} INVALID LABEL CASES:\n")
        print("These labels don't match expected choice tokens\n")
        for case in invalid_cases[:20]:
            print(f"Example {case['index']:5d}: "
                  f"Expected one of {CHOICES}  |  "
                  f"Got='{case['actual_decoded']}' "
                  f"Token IDs={case['actual_token_ids']}  |  "
                  f"Prediction={case['pred_letter']}")
        if len(invalid_cases) > 20:
            print(f"\n... and {len(invalid_cases) - 20} more invalid cases")
    else:
        print("✓ ALL LABELS ARE VALID (match expected tokens)")
    
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