#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from torch.utils.data import default_collate

import numpy as np
# ---------------------------
# Config
# ---------------------------
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "cais/mmlu"        # a.k.a. MMLU
DATASET_CONFIG = "all"          # bundle of 57 subjects
USE_LORA = False                 # set False for full finetune
SEED = 43
MAX_TOKENS = 768                # prompt+answer max length
TRAIN_SPLIT = "validation"      # demo: train on val, eval on test
EVAL_SPLIT = "test"
OUTPUT_DIR = "./output/out_qwen25_7b_mmlu_lora"
MAX_TRAIN_SAMPLES = 32
MAX_EVAL_SAMPLES = 32

# LoRA hyperparams (good starting point)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Trainer hyperparams (tune to your GPU)
BATCH_TRAIN = 1
BATCH_EVAL = 1
GRAD_ACCUM = 1
LR = 2e-4 if USE_LORA else 1e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.03
MAX_STEPS = 5


# ---------------------------
# Utils
# ---------------------------
def render_mc_question(q: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D", "E", "F"]
    lines = [q.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{letters[i]}. {c}")
    lines.append("")
    lines.append("Choose only one letter (A, B, C, or D).")
    return "\n".join(lines)


def make_chat_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    # Qwen Instruct: use chat template so the formatting matches its pretraining
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer multiple-choice questions "
                "by outputting exactly one letter: A, B, C, or D."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def to_letter(answer, num_choices=4):
    """Normalize MMLU answer to 'A'.. based on index or string."""
    # Integer index (common in cais/mmlu)
    if isinstance(answer, (int, float)):
        idx = int(answer)
        if not (0 <= idx < num_choices):
            raise ValueError(f"answer index {idx} out of range 0..{num_choices-1}")
        return LETTERS[idx]

    # String like "A" / "a" / "A." / "2" etc.
    if isinstance(answer, str):
        s = answer.strip()
        if s.isdigit():
            idx = int(s)
            if not (0 <= idx < num_choices):
                raise ValueError(f"answer index {idx} out of range 0..{num_choices-1}")
            return LETTERS[idx]
        # take first alpha char and normalize
        for ch in s:
            if ch.isalpha():
                return ch.upper()
    raise TypeError(f"Unsupported answer type: {type(answer)}: {answer}")

def build_input_and_labels(tokenizer: AutoTokenizer, prompt: str, answer_letter: str):
    # We train the model to output just the letter (no extra text).
    target = answer_letter.strip()
    full_text = prompt + target

    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_TOKENS,
        add_special_tokens=False,
    )
    # Mask the prompt tokens; only learn on the answer letter
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + tok["input_ids"][len(prompt_ids):]
    tok["labels"] = labels[:MAX_TOKENS]
    tok["input_ids"] = tok["input_ids"][:MAX_TOKENS]
    tok["attention_mask"] = tok["attention_mask"][:MAX_TOKENS]
    return tok


@dataclass
class PadCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        keys = ["input_ids", "attention_mask", "labels"]
        # pad manually so labels get padded with -100
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attn, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attn.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def compute_letter_token_accuracy(eval_preds):
    """
    We masked all prompt tokens, so accuracy reduces to the
    last non -100 position(s). For our setup the answer is a single token.
    """
    logits, labels = eval_preds
    preds = logits.argmax(-1)
    mask = labels != -100
    # keep only positions where labels are not -100
    correct = (preds[mask] == labels[mask]).sum()
    total = mask.sum()
    acc = (correct.astype(np.float32) / total.astype(np.float32)).item() if total > 0 else 0.0
    return {"letter_acc": acc}


# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(SEED)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Dataset
    ds: DatasetDict = load_dataset(DATASET_ID, DATASET_CONFIG)


    def limit(ds_split, n, seed=SEED):
        if n is None: 
            return ds_split
        n = min(n, len(ds_split))
        return ds_split.shuffle(seed=seed).select(range(n))

    train_raw = limit(ds[TRAIN_SPLIT], MAX_TRAIN_SAMPLES)
    eval_raw  = limit(ds[EVAL_SPLIT],  MAX_EVAL_SAMPLES)


    # MMLU fields: question(str), choices(list[str]), answer(str like "A"/"B"/"C"/"D"), subject(str)
    # Convert to instruction-following samples
    def _map_fn(ex):
        q_text = render_mc_question(ex["question"], ex["choices"])
        prompt = make_chat_prompt(tok, q_text)
        # ex["answer"] may be 0..3; convert to 'A'.. first
        ans_letter = to_letter(ex["answer"], num_choices=len(ex["choices"]))
        return build_input_and_labels(tok, prompt, ans_letter)

    ds_proc = DatasetDict({
        "train": train_raw.map(_map_fn, remove_columns=train_raw.column_names),
        "eval": eval_raw.map(_map_fn, remove_columns=eval_raw.column_names),
    })

    print(f"Train samples: {len(ds_proc['train'])}, Eval samples: {len(ds_proc['eval'])}")


    # Model (QLoRA by default)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if USE_LORA else None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        # device_map="auto",
        # quantization_config=quant_cfg,
    )

    if USE_LORA:
        model = prepare_model_for_kbit_training(model)
        lconf = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters()
    else:
        # full finetune: make sure you have enough GPU memory
        for p in model.parameters():
            p.requires_grad_(True)

    # Trainer
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="no",
        # save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to=[],
        max_steps=MAX_STEPS,
        overwrite_output_dir=True,
        # resume_from_checkpoint="no",
        ddp_backend="nccl",               # default on Linux + NVIDIA
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["eval"],
        tokenizer=tok,
        data_collator=PadCollator(tok),
        compute_metrics=compute_letter_token_accuracy,

    )

    trainer.train()

    # Save (merging LoRA into a single HF model if desired)
    if USE_LORA:
        # Adapter-only:
        adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
        model.save_pretrained(adapter_dir)
        print(f"Saved LoRA adapter to: {adapter_dir}")

        # Optional: merge weights for easy deployment
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))
            tok.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))
            print("Saved merged model.")
        except Exception as e:
            print(f"Skip merge (ok for training-only use). Reason: {e}")
    else:
        model.save_pretrained(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        print(f"Saved full model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
