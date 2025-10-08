#!/usr/bin/env python3
# Evaluate a Qwen2.5-0.5B + LoRA adapter on GLUE MNLI validation split

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import PeftModel, prepare_model_for_kbit_training  # prepare_* not used unless QLoRA


MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "mnli"
NUM_LABELS = 3
MAX_LEN = 256
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LoRA-tuned Qwen2.5-0.5B on GLUE MNLI")
    p.add_argument("--adapter_dir", type=str, default="qwen25_mnli_lora_adapter",
                   help="Path to saved LoRA adapter directory from training.")
    p.add_argument("--validation_split", type=str, default="validation_matched",
                   choices=["validation_matched", "validation_mismatched"],
                   help="Which MNLI validation split to evaluate.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device eval batch size.")
    p.add_argument("--use_qlora", type=str, default="false",
                   help="If 'true', load base in 4-bit for evaluation (requires bitsandbytes).")
    p.add_argument("--save_preds", type=str, default="",
                   help="Optional path to save predictions as a .csv (premise,hypothesis,label,pred).")
    return p.parse_args()


def str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "t", "yes", "y"}


def main():
    args = parse_args()
    use_qlora = str2bool(args.use_qlora)

    # 1) Load data & tokenizer
    ds_all = load_dataset("glue", DATASET)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def preprocess(ex):
        return tok(ex["premise"], ex["hypothesis"], truncation=True, max_length=MAX_LEN)

    # Keep the original for optional CSV export
    eval_raw = ds_all[args.validation_split]

    eval_ds = eval_raw.map(
        preprocess,
        batched=True,
        remove_columns=[c for c in eval_raw.column_names if c not in ("label")] + ["premise", "hypothesis"],
    )
    # Keep only the features Trainer expects
    keep_cols = ("input_ids", "attention_mask", "label")
    drop_cols = [c for c in eval_ds.column_names if c not in keep_cols]
    if drop_cols:
        eval_ds = eval_ds.remove_columns(drop_cols)

    # 2) Load base model + attach LoRA adapter
    cfg = AutoConfig.from_pretrained(MODEL, num_labels=NUM_LABELS, problem_type="single_label_classification")

    if use_qlora:
        from transformers import BitsAndBytesConfig
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
            else torch.float16
        )
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL, config=cfg, quantization_config=bnb_cfg, device_map="auto"
        )
        # Not strictly necessary for pure evaluation, but harmless:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    else:
        base = AutoModelForSequenceClassification.from_pretrained(MODEL, config=cfg)

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    # Dtype/bfloat16 selection
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    # 3) Metrics & collator
    metric = evaluate.load("glue", DATASET)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(axis=-1)
        return metric.compute(predictions=preds, references=p.label_ids)

    data_collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    # 4) Use Trainer purely for evaluation
    eval_args = TrainingArguments(
        output_dir="eval_tmp",
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=4,
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results: Dict[str, Any] = trainer.evaluate()
    print(f"\n=== Evaluation on {args.validation_split} ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # 5) (Optional) Save predictions to CSV
    if args.save_preds:
        import pandas as pd
        import numpy as np
        preds = trainer.predict(eval_ds).predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        y_pred = np.argmax(preds, axis=-1)

        # Pull back text columns from the raw split for readability
        df = eval_raw.to_pandas()[["premise", "hypothesis", "label"]].copy()
        df["pred"] = y_pred
        df.to_csv(args.save_preds, index=False)
        print(f"\nSaved predictions to: {args.save_preds}")

    # 6) (Optional) Also evaluate the other split if available
    other = "validation_mismatched" if args.validation_split == "validation_matched" else "validation_matched"
    if other in ds_all:
        # Map & filter like above
        other_raw = ds_all[other]
        other_ds = other_raw.map(
            preprocess,
            batched=True,
            remove_columns=[c for c in other_raw.column_names if c not in ("label")] + ["premise", "hypothesis"],
        )
        other_ds = other_ds.remove_columns([c for c in other_ds.column_names if c not in keep_cols])

        other_res = trainer.evaluate(other_ds)
        print(f"\n=== Evaluation on {other} ===")
        for k, v in other_res.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
