# mnli_to_alpaca.py
import json
from datasets import load_dataset

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

INSTRUCTION = (
    "Decide if the hypothesis is entailed by the premise. "
    "Answer with exactly one of: entailment, neutral, contradiction."
)

def rec(premise: str, hypothesis: str, label_id: int):
    return {
        "instruction": INSTRUCTION,
        "input": f"Premise: {premise}\nHypothesis: {hypothesis}",
        "output": LABEL_MAP[int(label_id)],
        # Optional fields LLaMA-Factory can ingest if present:
        "system": "",
        "history": [],
    }

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ds = load_dataset("nyu-mll/glue", "mnli")
    # Train
    train_rows = [
        rec(x["premise"], x["hypothesis"], x["label"])
        for x in ds["train"]
        if x["label"] != -1
    ]
    # Validation: use "validation_matched" split
    dev_rows = [
        rec(x["premise"], x["hypothesis"], x["label"])
        for x in ds["validation_matched"]
        if x["label"] != -1
    ]

    ouptut_dir = "/mnt/coldhot/shouxu_runs/datasets/mnli_alpaca"

    dump(f"{ouptut_dir}/mnli_alpaca_train.jsonl", train_rows)
    dump(f"{ouptut_dir}/mnli_alpaca_dev_matched.jsonl", dev_rows)

    print("Wrote mnli_alpaca_train.jsonl and mnli_alpaca_dev_matched.jsonl")
