#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import random
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from category import subcategories, categories, subject_to_categories

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # or "Qwen/Qwen2.5-0.5B"
MMLU_CONFIG = "all"                         # 57 subjects
K_FEW_SHOT = 5                              # number of exemplars per subject
MAX_LEN = 768                               # prompt max length (prompt only)
SEED = 1234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Helpers
# ---------------------------
def set_pad_if_missing(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def build_fewshot_bank(val_split):
    """
    Returns dict: subject -> list of K_FEW_SHOT few-shot exemplars (question, choices, answer)
    Uses the first K examples per subject (deterministic).
    """
    bank = {}
    by_subj = defaultdict(list)
    for r in val_split:
        by_subj[r["subject"]].append(r)
    for s, rows in by_subj.items():
        bank[s] = rows[:K_FEW_SHOT]
    return bank

def render_example(r):
    """Render a single MMLU example (without the final Answer:)."""
    A, B, C, D = r["choices"]
    return (
        f"Q: {r['question']}\n"
        f"A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
    )

def render_answer(a):
    """Render the answer letter."""
    # Map 0->A, 1->B, 2->C, 3->D
    return "ABCD"[a]

def build_prompt(subject, fewshot_bank, test_row):
    """
    Builds the full prompt ending in 'Answer: ' (note the trailing space).
    """
    header = f"Subject: {subject}\n\n"
    shots = []
    for r in fewshot_bank.get(subject, []):
        shots.append(render_example(r) + f"Answer: {render_answer(r['answer'])}\n\n")
    test_part = render_example(test_row) + "Answer: "
    return header + "".join(shots) + test_part

def prepare_label_ids(tok):
    """
    Returns:
        label_texts: [" A"," B"," C"," D"]
        label_ids_list: List[List[int]] tokenized labels
        single_token: bool (True if all labels are exactly one token)
        first_token_ids: List[int] (first token of each label)
    """
    label_texts = [" A", " B", " C", " D"]   # leading space matters
    label_ids_list = [tok(t, add_special_tokens=False).input_ids for t in label_texts]
    single_token = all(len(ids) == 1 for ids in label_ids_list)
    first_token_ids = [ids[0] for ids in label_ids_list]
    return label_texts, label_ids_list, single_token, first_token_ids

@torch.no_grad()
def probs_over_4_labels_single_token(model, input_ids, attention_mask, first_token_ids):
    """
    Single forward pass → logits for next token → gather 4 logits → softmax over the 4.
    input_ids: [1, T]
    attention_mask: [1, T]
    returns: torch.Tensor shape [4] with probs summing to 1.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    next_logits = out.logits[0, -1]                            # [V]
    label_logits = next_logits[first_token_ids]                # [4]
    probs = label_logits.softmax(dim=-1)                       # normalized over the 4 labels
    return probs

@torch.no_grad()
def probs_over_4_labels_span(model, input_ids, attention_mask, label_ids_list):
    """
    Safe version for multi-token labels:
    1) Run the prompt once to get past_key_values.
    2) For each label, step through its tokens, summing log-probs.
    3) Softmax over the 4 summed log-probs to get a 4-way distribution.
    """
    # Step 1: run the prompt once
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    base_past = getattr(out, "past_key_values", None)

    logps = []
    for lid in label_ids_list:
        lid_t = torch.tensor([lid], device=input_ids.device)   # [1, k]
        past = base_past
        running_attn_len = input_ids.shape[1]                  # track for models that need mask length
        logp = 0.0

        # Feed label tokens one by one
        for t in range(lid_t.size(1)):
            step_ids = lid_t[:, t:t+1]                         # [1, 1]
            if past is None:
                # Fallback: concatenate prompt + prefix (rarely needed)
                concat = torch.cat([input_ids, lid_t[:, :t+1]], dim=1)
                out_step = model(input_ids=concat, use_cache=True)
                next_logprobs = out_step.logits[:, -1, :].log_softmax(-1)
            else:
                out_step = model(input_ids=step_ids, past_key_values=past, use_cache=True)
                next_logprobs = out_step.logits[:, -1, :].log_softmax(-1)

            tgt = step_ids[0, 0].item()
            logp += float(next_logprobs[0, tgt].item())
            past = getattr(out_step, "past_key_values", None)
            running_attn_len += 1

        logps.append(logp)

    # Convert log-scores to a probability distribution over the 4 labels
    logps_t = torch.tensor(logps, device=input_ids.device)     # [4]
    probs = torch.softmax(logps_t, dim=-1)                     # [4]
    return probs

def gold_to_index(gold_letter):
    """Map 'A'/'B'/'C'/'D' to 0..3."""
    return "ABCD".index(gold_letter)

# ---------------------------
# Main
# ---------------------------
def main():
    print(f"Loading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok = set_pad_if_missing(tok)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else None)
    model.to(DEVICE).eval()

    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", MMLU_CONFIG)
    val, test = ds["validation"], ds["test"]

    print("Building few-shot bank...")
    fewshot_bank = build_fewshot_bank(val)

    # Prepare label tokens
    label_texts, label_ids_list, single_token, first_token_ids = prepare_label_ids(tok)
    if single_token:
        print("Label tokens are single-token. Using fast single-step scoring.")
        print(f"label_texts: {label_texts} label_ids_list: {label_ids_list} first_token_ids: {first_token_ids}")
    else:
        print("Detected multi-token labels. Using span scoring.")


    # Group test rows by subject for macro accuracy
    subjects = sorted(set(test["subject"]))
    by_subject = defaultdict(list)
    for r in test:
        by_subject[r["subject"]].append(r)

    per_subject_acc = {}
    total_correct = 0
    total_count = 0


    debug_thres = 20
    debug_count = 0

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)


    subj_to_cats = subject_to_categories()

    print(subj_to_cats)
    for subj in subjects:
        cat = subj_to_cats.get(subj, ["Unknown"])[0]
        if cat not in cat_correct:
            cat_correct[cat] = 0
            cat_total[cat] = 0

        rows = by_subject[subj]
        correct = 0
        for r in tqdm(rows, desc=f"Evaluating {subj:>20} {cat}", leave=False):
            # print(f"Row: {r}")
            # print(f"")
            prompt = build_prompt(subj, fewshot_bank, r)
            # print(f"Prompt: {prompt}")
            enc = tok(prompt, truncation=True, max_length=MAX_LEN, add_special_tokens=False, return_tensors="pt")
            input_ids = enc.input_ids.to(DEVICE)
            attn_mask = enc.attention_mask.to(DEVICE)

            if single_token:
                probs = probs_over_4_labels_single_token(model, input_ids, attn_mask, first_token_ids)
            else:
                probs = probs_over_4_labels_span(model, input_ids, attn_mask, label_ids_list)

            pred_idx = int(probs.argmax().item())
            # gold_idx = gold_to_index(r["answer"])
            gold_idx = int(r["answer"])
            correct += int(pred_idx == gold_idx)
            cat_correct[cat] += int(pred_idx == gold_idx)
            cat_total[cat] += 1

            # print(f"predicted: {label_texts[pred_idx]} (idx {pred_idx}), label: {label_texts[gold_idx]} (idx {gold_idx}), per-subject-acc:{correct}/{len(rows)}, cat-acc: {cat_correct[cat]}/{cat_total[cat]}")

        acc = correct / max(1, len(rows))
        per_subject_acc[subj] = acc
        total_correct += correct
        total_count += len(rows)

    macro = sum(per_subject_acc[s] for s in subjects) / len(subjects)
    micro = total_correct / max(1, total_count)

    # Report
    n_show = 10
    print("\nPer-subject accuracy (first {} subjects alphabetically):".format(n_show))
    for s in subjects[:n_show]:
        print(f"  {s:30s}  {per_subject_acc[s]*100:6.2f}%")

    # per 
    print(f"\nMacro accuracy over {len(subjects)} subjects: {macro*100:.2f}%")
    print(f"Micro accuracy over {total_count} questions:  {micro*100:.2f}%")


    for cat in cat_total.keys():
        correct = cat_correct[cat]
        total = cat_total[cat]
        acc = correct / max(1, total)
        print(f"Category: {cat:30s}  Correct: {correct}  Total: {total}  Accuracy: {acc*100:6.2f}% ({correct}/{total})")

if __name__ == "__main__": 
    main()
