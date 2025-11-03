#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForSequenceClassification

# ---- CONFIG ----
RUN_DIR = "/share/desa/nfs02/yy354/cold/runs/glue/mnli/runs/roberta-base-skipRatio0.92-changeIters100-seed0-lr2e-5"

CKPT_A = os.path.join(RUN_DIR, "checkpoint-300")
CKPT_B = os.path.join(RUN_DIR, "checkpoint-400")

# if your run is MNLI with 3 labels
NUM_LABELS = 3


def load_model(ckpt_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path,
        num_labels=NUM_LABELS,
        local_files_only=True,
    )
    return model


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor):
    """
    Compare two tensors of the same shape.
    Returns:
        elem_changed, elem_total, row_changed, row_total
    """
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch for {name}: {t1.shape} vs {t2.shape}")

    # element-wise diff
    diff = (t1 != t2)
    elem_changed = diff.sum().item()
    elem_total = t1.numel()

    # row-level diff (only for 2D)
    row_changed = 0
    row_total = 0
    if t1.ndim == 2:
        row_total = t1.shape[0]
        row_changed = (diff.any(dim=1)).sum().item()
        # col_total = t1.shape[1]
        # col_changed = (diff.any(dim=0)).sum().item()
    # print(f"--- {name}: shape={t1.shape} ")
    return elem_changed, elem_total, row_changed, row_total,  t1.shape

    # col_changed = 0
    # col_total = 0
    # if t1.ndim == 2:
        # col_total = t1.shape[1]
        # col_changed = (diff.any(dim=0)).sum().item()
    # return elem_changed, elem_total, col_changed, col_total, t1.shape


def main():
    print(f"Loading model A from: {CKPT_A}")
    model_a = load_model(CKPT_A)
    print(f"Loading model B from: {CKPT_B}")
    model_b = load_model(CKPT_B)

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    total_elem_changed = 0
    total_elem = 0
    total_row_changed = 0
    total_rows = 0

    print("\n===== PER PARAMETER DIFF =====")
    for name, p_a in params_a.items():
        if name not in params_b:
            print(f"[warn] {name} not found in model B, skipping")
            continue

        p_b = params_b[name]

        # move to cpu to be safe
        t1 = p_a.detach().cpu()
        t2 = p_b.detach().cpu()

        elem_changed, elem_total, row_changed, row_total, shapes = compare_tensors(name, t1, t2)

        # accumulate
        total_elem_changed += elem_changed
        total_elem += elem_total
        total_row_changed += row_changed
        total_rows += row_total

        # only print interesting ones
        if elem_changed > 0:
            elem_ratio = elem_changed / elem_total
            if row_total > 0:
                row_ratio = row_changed / row_total
                print(f"{name}: elems {elem_changed}/{elem_total} ({elem_ratio:.6f}), "
                      f"rows {row_changed}/{row_total} ({row_ratio:.6f}), shapes {shapes}")
            else:
                print(f"{name}: elems {elem_changed}/{elem_total} ({elem_ratio:.6f}), shapes {shapes}")

    print("\n===== SUMMARY =====")
    print(f"TOTAL elem changed: {total_elem_changed}/{total_elem} "
          f"({(total_elem_changed/total_elem):.8f})")
    if total_rows > 0:
        print(f"TOTAL rows changed: {total_row_changed}/{total_rows} "
              f"({(total_row_changed/total_rows):.8f})")


if __name__ == "__main__":
    main()