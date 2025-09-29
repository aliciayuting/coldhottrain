import os
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import importlib.metadata

# ---------------- Config ----------------
REPO_ID  = "Qwen/Qwen2.5-0.5B"
REVISION = None   # pin commit hash if you want reproducibility

SNAPSHOT = Path(
    "~/.cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/"
)

print("Transformers version:", transformers.__version__)
print("huggingface_hub version:", importlib.metadata.version("huggingface_hub"))

# ---------------- Helpers ----------------
def has_model_files(root: Path) -> bool:
    if not root.exists():
        return False
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    cfg_files = ["config.json"]
    tok_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json", "merges.txt",
        "spiece.model", "sentencepiece.bpe.model",
    ]
    def any_exists(names): return any((root / n).exists() for n in names)
    return any_exists(weight_files) and any_exists(cfg_files) and any_exists(tok_files)

# ---------------- Ensure local snapshot ----------------
if has_model_files(SNAPSHOT):
    print(f"[OK] Using cached snapshot at: {SNAPSHOT}")
else:
    print(f"[MISS] Snapshot incomplete at: {SNAPSHOT}")
    print(f"[DL ] Downloading {REPO_ID} ...")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    dl_path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(SNAPSHOT),
        local_dir_use_symlinks=False,
    )
    print(f"[OK ] Downloaded to: {dl_path}")

# ---------------- Load offline ----------------
cfg = AutoConfig.from_pretrained(str(SNAPSHOT), local_files_only=True)
print("model_type:", getattr(cfg, "model_type", None))

tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    str(SNAPSHOT),
    local_files_only=True,
    torch_dtype=torch.float32,
    device_map=None,
)
print("Loaded model:", type(model).__name__)





for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.size())  
