import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn import Linear
from torch import Tensor
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import importlib.metadata


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling



from datasets import load_dataset



from module import *

def safe_destroy():
    if dist.is_available() and dist.is_initialized():
        try:
            # Optional but helpful to flush in-flight NCCL ops
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# ---------------- Config ----------------
REPO_ID  = "Qwen/Qwen2.5-0.5B"
REVISION = None   # pin commit hash if you want reproducibility

SNAPSHOT = Path(
    # "~/.cache/huggingface/transformers/"
    "/pscratch/sd/l/lsx/.cache/huggingface/transformers/"
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
    # torch_dtype=torch.float32,
    torch_dtype=torch.bfloat16,
    device_map=None,
)
print("Loaded model:", type(model).__name__)





# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())  



def get_decoder_layers(m: nn.Module):
    """
    Return the list-like container of decoder blocks for common HF decoder-only models.
    Works for Qwen2ForCausalLM (m.model.layers).
    """
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
        return m.transformer.layers
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError("Could not locate decoder layers (tried model.layers / transformer.layers / layers).")

# ---- your previously defined helpers (from my last message) ----
def make_hot_idx(out_features: int, frac: float | None = None, idx: torch.Tensor | None = None, device=None):
    if idx is not None:
        hot = torch.as_tensor(idx, dtype=torch.long, device=device)
        assert hot.ndim == 1 and hot.numel() > 0
        assert hot.min().item() >= 0 and hot.max().item() < out_features
        return hot
    assert frac is not None and 0.0 <= frac <= 1.0
    k = max(0, min(out_features, int(round(frac * out_features))))
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    perm = torch.randperm(out_features, device=device)
    return perm[:k].sort().values

def replace_linear_with_colwise(mod: nn.Module, hot_idx: torch.Tensor) -> LinearColWise:
    assert isinstance(mod, nn.Linear)
    hot_idx = hot_idx.to(mod.weight.device)
    wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped

# ---- apply to layer 23 of Qwen2.5 -------------------------------------------
layers = get_decoder_layers(model)   # <-- the fix
layer_idx = 23
layer = layers[layer_idx]

policy_by_name = {
    "self_attn.q_proj": 0.30,
    "self_attn.k_proj": 0.30,
    "self_attn.v_proj": 0.30,
    "self_attn.o_proj": 0.30,
    "mlp.up_proj":      0.30,
    "mlp.down_proj":    0.30,
    # "mlp.gate_proj":  0.30,  # include if you want to split the gate too
}

mapping = {
    "self_attn.q_proj": layer.self_attn.q_proj,
    "self_attn.k_proj": layer.self_attn.k_proj,
    "self_attn.v_proj": layer.self_attn.v_proj,
    "self_attn.o_proj": layer.self_attn.o_proj,
    "mlp.up_proj":      layer.mlp.up_proj,
    "mlp.down_proj":    layer.mlp.down_proj,
    # "mlp.gate_proj": layer.mlp.gate_proj,
}

for name, linear in mapping.items():
    assert isinstance(linear, nn.Linear), f"{name} expected nn.Linear, got {type(linear)}"
    print(f"Processing {name}: {tuple(linear.weight.shape)}")
    out_features = linear.out_features
    hot_idx = make_hot_idx(out_features, frac=policy_by_name[name], device=linear.weight.device)
    wrapped = replace_linear_with_colwise(linear, hot_idx)
    if name.startswith("self_attn."):
        setattr(layer.self_attn, name.split(".", 1)[1], wrapped)
    else:
        setattr(layer.mlp,       name.split(".", 1)[1], wrapped)

# quick sanity check
print(type(layer.self_attn.q_proj), layer.self_attn.q_proj.W_hot.shape, layer.self_attn.q_proj.W_cold.shape)
print(type(layer.mlp.up_proj),      layer.mlp.up_proj.W_hot.shape,      layer.mlp.up_proj.W_cold.shape)




''' Training loop'''

DATASET = "tatsu-lab/alpaca"

# Load Alpaca-52K dataset
ds = load_dataset(DATASET)

# Preprocess into prompt–response format
def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    response = example["output"]

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return tok(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_ds = ds.map(format_example, batched=False)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# output_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}"

# Training arguments
args = TrainingArguments(
    # output_dir=f"{output_dir}/ckpt",
    per_device_train_batch_size=4,
    # gradient_accumulation_steps=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    # fp16=True,
    bf16=True,
    logging_steps=10,
    # save_steps=100,
    # save_total_limit=2,
    ddp_find_unused_parameters=False,
    # max_steps = 16,
)



# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    data_collator=collator,
)

num_gpus = torch.cuda.device_count()
num_samples = len(tokenized_ds["train"])
global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, num_gpus)
iters_per_epoch = (num_samples + global_batch - 1) // global_batch
print(f"#GPUs: {num_gpus}  Global batch: {global_batch}  Iters/epoch: {iters_per_epoch}")


# # dump_out_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace("/", "_")}-{DATASET.replace('/', '_')}-grad_dump"
# dump_out_dir = f"{output_dir}/grad_dump"

# dump_cb = PerModuleGradDumper(
#     out_dir=dump_out_dir,
#     model=model,
#     capture_steps=100,
#     include_bias=True,
#     also_embeddings=False,  # set True if you also want embeddings/lm_head
# )
# trainer.add_callback(dump_cb)

# probe_cb = Probe()
# trainer.add_callback(probe_cb)



# Start training
trainer.train()


safe_destroy()
