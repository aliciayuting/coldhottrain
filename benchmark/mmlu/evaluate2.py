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
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "cais/mmlu"
OUTPUT_DIR = f"/pscratch/sd/l/lsx/shouxu_runs/{MODEL_NAME.replace('/', '_')}-{DATASET.replace('/', '_')}"
OUTPUT_PRED_DEBUG_DIR = "./mmlu_eval_debug"
MAX_LENGTH = 768
BATCH_SIZE = 4

# ---------- Tokenizer ----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)
# Some Qwen variants need this to avoid pad warnings
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
# (Optional) truncate for quick debugging
# eval_dataset = eval_dataset.select(range(2))
original_questions = []

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
    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": full_text,
        "original_question": question,              # NEW: Store question
        "original_choices": choices,                # NEW: Store choices
        "correct_answer_idx": answer_idx,           # NEW: Store answer index
        "correct_answer_letter": chr(65 + answer_idx)  # NEW: Store answer letter
    }


print("Formatting eval dataset...")
eval_dataset = eval_dataset.map(format_mmlu_example, remove_columns=eval_dataset.column_names)

# ---------- Tokenize: mask everything except the single gold letter ----------
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
    # mask out padding
    for i, mask in enumerate(enc["attention_mask"]):
        for j, m in enumerate(mask):
            if m == 0:
                labels[i][j] = -100
    # mask out prompt positions, keep only answer token(s)
    for i, plen in enumerate(prompt_lens):
        upto = min(plen, len(labels[i]))
        for j in range(upto):
            if labels[i][j] != -100:
                labels[i][j] = -100
    enc["labels"] = labels

    enc["original_question"] = examples["original_question"]      # NEW: Preserve
    enc["original_choices"] = examples["original_choices"]        # NEW: Preserve
    enc["correct_answer_idx"] = examples["correct_answer_idx"]    # NEW: Preserve
    enc["correct_answer_letter"] = examples["correct_answer_letter"]  # NEW: Preserve
    
    return enc

print("Tokenizing eval dataset...")
eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["prompt", "answer"],
)

# ---------- Inspect a couple examples (optional) ----------
def show_tokenized_dataset_examples(dataset, num_examples=2):
    for i in range(min(num_examples, len(dataset))):
        ex = dataset[i]
        non_ignore_labels = [l for l in ex["labels"] if l != -100]
        print(f"--- Example {i} ---")
        print("Decoded input (no pads):")
        print(tokenizer.decode([t for t in ex["input_ids"] if t != tokenizer.pad_token_id]))
        print("Gold label tokens (no -100):", non_ignore_labels)
        print("Gold decoded:", tokenizer.decode(non_ignore_labels))
        print()
# show_tokenized_dataset_examples(eval_dataset)

# ---------- 4-choice setup ----------
CHOICES = [" A", " B", " C", " D"]
CHOICE_TOKEN_IDS = []
for s in CHOICES:
    ids = tokenizer.encode(s, add_special_tokens=False)
    assert len(ids) == 1, f"Choice token '{s}' did not encode to a single token: {ids}"
    CHOICE_TOKEN_IDS.append(ids[0])
CHOICE_TOKEN_IDS = torch.tensor(CHOICE_TOKEN_IDS, dtype=torch.long)
print(f"Choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")

all_predictions = []

# ---------- Helpers for metrics ----------
def _first_answer_pos(labels: torch.Tensor) -> torch.Tensor:
    """
    labels: (B, T) with -100 except the single answer token.
    returns: LongTensor (B,) of first non -100 position; -1 if none.
    """
    not_ign = (labels != -100)
    first_pos = not_ign.float().argmax(dim=1)
    has_any = not_ign.any(dim=1)
    first_pos = torch.where(has_any, first_pos, torch.full_like(first_pos, -1))
    return first_pos

def preprocess_logits_for_metrics(logits, labels):
    """
    Input:
        logits: (B, T, V) or tuple(logits, ...)
        labels: (B, T)
    Return:
        (B, 4) logits for tokens [" A"," B"," C"," D"] at the answer position.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.float()
    labels = labels.to(logits.device)

    B, T, V = logits.shape
    ans_pos = _first_answer_pos(labels)  # (B,)

    bad = (ans_pos < 0)
    if bad.any():
        ans_pos = torch.where(bad, torch.full_like(ans_pos, T - 1), ans_pos)

    row_idx = torch.arange(B, device=logits.device)
    logits_at_ans = logits[row_idx, ans_pos, :]  # (B, V)

    choice_ids = CHOICE_TOKEN_IDS.to(logits.device)  # (4,)
    four_logits = logits_at_ans.index_select(dim=1, index=choice_ids)  # (B, 4)
    return four_logits

def compute_metrics(eval_pred):
    """
    eval_pred.predictions: (N, 4)
    eval_pred.label_ids:   (N, T)
    """
    four_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    pred_idx = np.asarray(four_logits).argmax(axis=1)  # (N,)

    labels = torch.tensor(label_ids)
    ans_pos = _first_answer_pos(labels)  # (N,)
    has_any = (ans_pos >= 0)
    assert has_any.any(), "No valid answers found in labels!"

    row_idx = torch.arange(labels.size(0))
    gold_token_ids = torch.full((labels.size(0),), -1, dtype=torch.long)
    gold_token_ids[has_any] = labels[row_idx[has_any], ans_pos[has_any]]

    choice_ids = CHOICE_TOKEN_IDS
    eq_matrix = (gold_token_ids[:, None] == choice_ids[None, :])  # (N, 4)
    gold_idx = eq_matrix.long().argmax(dim=1).numpy()
    gold_valid = eq_matrix.any(dim=1).numpy()

    acc = accuracy_score(gold_idx[gold_valid], pred_idx[gold_valid]) if gold_valid.any() else 0.0
    print("accuracy:", acc)
    # Store all predictions with details
    global all_predictions
    all_predictions = []
    print("\n" + "="*80)
    print("DETAILED PREDICTION RESULTS")
    print("="*80 + "\n")
    for i in range(min(5, len(pred_idx))):
        pred_letter = chr(65 + pred_idx[i])
        gold_letter = chr(65 + gold_idx[i])
        is_correct = pred_idx[i] == gold_idx[i] if gold_valid[i] else False
        
        print(f"Example {i}: Predicted: {pred_letter} | Correct: {gold_letter} | {'✓' if is_correct else '✗'}")
    

    return {"accuracy": acc}


class MMOLUTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_predictions = []
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        # Call parent evaluation
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Now collect all predictions with original questions
        print("\n" + "="*80)
        print("DETAILED PREDICTION RESULTS")
        print("="*80 + "\n")
        
        pred_idx = output.predictions.argmax(axis=1)
        

        self.all_predictions = []
        for i, example in enumerate(self.eval_dataset):
            pred_letter = chr(65 + pred_idx[i])
            gold_letter = example["correct_answer_letter"]  # Access from dataset
            gold_idx = example["correct_answer_idx"]        # Access from dataset
            is_correct = pred_idx[i] == gold_idx
            
            result = {
                "index": i,
                "question": example["original_question"],
                "choices": example["original_choices"],
                "predicted_answer": pred_letter,
                "correct_answer": gold_letter,
                "is_correct": bool(is_correct),
                "logits": output.predictions[i].tolist()
            }
            self.all_predictions.append(result)
            
            # Print first 10 and all incorrect
            if i < 10 or not is_correct:
                print(f"Example {i}:")
                print(f"Question: {example['original_question']}")
                print("Choices:")
                for j, choice in enumerate(example["original_choices"]):
                    marker = "→" if j == pred_idx[i] else " "
                    correct_marker = "✓" if j == gold_idx else " "
                    print(f"  {marker} {correct_marker} {chr(65+j)}. {choice}")
                print(f"Predicted: {pred_letter} | Correct: {gold_letter} | {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                print("-" * 80 + "\n")
        
        # Summary
        correct_count = sum(1 for p in self.all_predictions if p["is_correct"])
        total_count = len(self.all_predictions)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total questions: {total_count}")
        print(f"Correct: {correct_count}")
        print(f"Incorrect: {total_count - correct_count}")
        print(f"Accuracy: {output.metrics[f'{metric_key_prefix}_accuracy']:.4f} ({output.metrics[f'{metric_key_prefix}_accuracy']*100:.2f}%)")
        print("="*80 + "\n")
        
        return output

# ---------- Data collator ----------
data_collator = default_data_collator

# ---------- Eval-only TrainingArguments ----------
# Keep only what evaluation needs; no training knobs, no checkpointing/saving.
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

# ---------- Trainer (eval only) ----------
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
# )

trainer = MMOLUTrainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# ---------- Run evaluation ----------
print("\nEvaluating on MMLU (test split)...")
metrics = trainer.evaluate()
print("\nFinal metrics:", metrics)

# ---------- Save predictions to file ----------
output_file = os.path.join(OUTPUT_PRED_DEBUG_DIR, "predictions.json")
os.makedirs(OUTPUT_PRED_DEBUG_DIR, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump({
        "model": MODEL_NAME,
        "dataset": DATASET,
        "accuracy": metrics["eval_accuracy"],
        "total_examples": len(trainer.all_predictions),
        "predictions": trainer.all_predictions
    }, f, indent=2)
print(f"\nPredictions saved to: {output_file}")