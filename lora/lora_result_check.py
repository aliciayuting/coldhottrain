#!/usr/bin/env python3
# Evaluate a Qwen2.5-0.5B + LoRA adapter on GLUE MNLI validation split


'''
python3 lora_result_check.py \
    --adapter_dir qwen25_sst2_lora_adapter 
'''

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import argparse
from typing import Dict, Any
import numpy as np

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

from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training  # prepare_* not used unless QLoRA
from lora_finetune_qwen_glue import tokenizers_for_task, make_compute_metrics,build_tokenizer, get_num_labels,str2bool
 
MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LoRA-tuned Qwen2.5-0.5B on GLUE MNLI")
    p.add_argument("--adapter_dir", type=str, default="qwen25_sst2_lora_adapter",
                   help="Path to saved LoRA adapter directory from training.")
    p.add_argument("--task_name", type=str, default="sst2", choices=["sst2", "mnli"],
                   help="GLUE task to fine-tune on.")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                   help="Base model to fine-tune.")
    p.add_argument("--validation_split", type=str, default="validation_matched",
                   choices=["validation_matched", "validation_mismatched"],
                   help="Which MNLI validation split to evaluate.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device eval batch size.")
    p.add_argument("--use_qlora", type=str, default="false",
                   help="If 'true', load base in 4-bit for evaluation (requires bitsandbytes).")
    p.add_argument("--save_preds", type=str, default="",
                   help="Optional path to save predictions as a .csv (premise,hypothesis,label,pred).")
    return p.parse_args()


# def str2bool(s: str) -> bool:
#     return s.lower() in {"1", "true", "t", "yes", "y"}



def main():
    args = parse_args()
    use_qlora = str2bool(args.use_qlora)

    # 1) Load data & tokenizer
    # tok = build_tokenizer(args.model_name)
    peft_cfg = PeftConfig.from_pretrained(args.adapter_dir)
    base_name = peft_cfg.base_model_name_or_path or args.model_name

    # 1) Load tokenizer: prefer adapter_dir (you saved it there), else the base
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("nyu-mll/glue", args.task_name)
    num_labels = get_num_labels(args.task_name)

    tok_fn, remove_cols = tokenizers_for_task(args.task_name, tok, args.max_len)
    ds_tok = ds.map(tok_fn, batched=True)
    if "label" in ds_tok["train"].column_names:
        ds_tok = ds_tok.rename_column("label", "labels")
    # Keep label + model inputs only (HF Trainer handles "label")
    cols_to_remove = [c for c in remove_cols if c in ds_tok["train"].column_names]
    if cols_to_remove:
        ds_tok = ds_tok.remove_columns(cols_to_remove)

    # Splits
    if args.task_name == "mnli":
        eval_split = "validation_matched"
        eval_ds = ds_tok[eval_split]
        # (You could also evaluate mismatched separately if desired)
    else:
        eval_ds = ds_tok["validation"]

    # 2) Load base model + attach LoRA adapter
    # cfg = AutoConfig.from_pretrained(MODEL, num_labels=num_labels, problem_type="single_label_classification")
    cfg = AutoConfig.from_pretrained(base_name, num_labels=num_labels, problem_type="single_label_classification")
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
            base_name, config=cfg, quantization_config=bnb_cfg, device_map="auto"
        )
        # Not strictly necessary for pure evaluation, but harmless:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    else:
        base = AutoModelForSequenceClassification.from_pretrained(base_name, config=cfg)
    base.resize_token_embeddings(len(tok))
    base.config.pad_token_id = tok.pad_token_id
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    # Dtype/bfloat16 selection
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    # 3) Metrics & collator
    compute_metrics = make_compute_metrics(args.task_name)

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
    print(f"\n=== Evaluation on {args.task_name} ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # --- NEW: dump model parameter dimensions to file ---
    param_info_path = os.path.join(args.adapter_dir, "adapter_model_param_information.txt")
    with open(param_info_path, "w") as f:
        f.write(f"Parameter dimensions for model loaded from adapter: {args.adapter_dir}\n\n")
        for name, param in model.named_parameters():
            shape_str = "x".join(map(str, param.shape))
            f.write(f"{name:80s} {shape_str}\n")
    print(f"[INFO] Parameter shape info written to {param_info_path}")

    # # 5) (Optional) Save predictions to CSV
    # if args.save_preds:
    #     import pandas as pd
    #     import numpy as np
    #     preds = trainer.predict(eval_ds).predictions
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     y_pred = np.argmax(preds, axis=-1)

    #     # Pull back text columns from the raw split for readability
    #     df = eval_raw.to_pandas()[["premise", "hypothesis", "label"]].copy()
    #     df["pred"] = y_pred
    #     df.to_csv(args.save_preds, index=False)
    #     print(f"\nSaved predictions to: {args.save_preds}")

    # # 6) (Optional) Also evaluate the other split if available
    # other = "validation_mismatched" if args.validation_split == "validation_matched" else "validation_matched"
    # if other in ds_all:
    #     # Map & filter like above
    #     other_raw = ds_all[other]
    #     other_ds = other_raw.map(
    #         preprocess,
    #         batched=True,
    #         remove_columns=[c for c in other_raw.column_names if c not in ("label")] + ["premise", "hypothesis"],
    #     )
    #     other_ds = other_ds.remove_columns([c for c in other_ds.column_names if c not in keep_cols])

    #     other_res = trainer.evaluate(other_ds)
    #     print(f"\n=== Evaluation on {other} ===")
    #     for k, v in other_res.items():
    #         print(f"{k}: {v}")


if __name__ == "__main__":
    main()
