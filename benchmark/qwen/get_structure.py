from transformers import AutoConfig, AutoModelForCausalLM
import torch, math
from collections import defaultdict

import os, json, transformers, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer



SNAPSHOT = "/pscratch/sd/l/lsx/.cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/"

print("Transformers version:", transformers.__version__)
print("Snapshot has:", os.listdir(SNAPSHOT))

# 1) Load config directly from the folder (offline)
cfg = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True)
print("model_type:", getattr(cfg, "model_type", None))

# 2) Load tokenizer from the same folder (offline)
tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, use_fast=True)

# 3) Load model from the same folder (offline)
model = AutoModelForCausalLM.from_pretrained(
    SNAPSHOT,
    local_files_only=True,
    torch_dtype=torch.float32,  # change to bfloat16 later if you want
    device_map=None
)
print("Loaded model:", type(model).__name__)


model.eval()

# quick totals
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Loaded {MODEL_ID}")
print(f"Params (total): {total_params/1e6:.2f}M, trainable: {trainable_params/1e6:.2f}M\n")



def get(attr, default=None):
    return getattr(cfg, attr, default)

# Try common llama/Qwen-style names
n_layers = get("num_hidden_layers", get("n_layer"))
d_model  = get("hidden_size",      get("n_embd"))
n_heads  = get("num_attention_heads", get("n_head"))
n_kv     = get("num_key_value_heads", get("n_kv_heads", n_heads))
d_ff     = get("intermediate_size", None)
max_pos  = get("max_position_embeddings", None)
rope_th  = get("rope_theta", None)
vocab    = get("vocab_size", None)
act_fn   = get("hidden_act", "silu")  # Qwen2.x uses SwiGLU (silu+gate) in MLP
norm_eps = get("rms_norm_eps", get("layer_norm_eps", None))

print("=== Qwen2.5-0.5B Config (key fields) ===")
print(f"layers={n_layers}, d_model={d_model}, heads={n_heads}, kv_heads={n_kv}")
print(f"intermediate_size={d_ff}, activation={act_fn}")
print(f"vocab_size={vocab}, max_position_embeddings={max_pos}, rope_theta={rope_th}")
print(f"rms_norm_eps={norm_eps}")
print()







'''Compact module tree with param counts (depth-limited)'''
def human(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)

def module_param_count(m):
    return sum(p.numel() for p in m.parameters())

def print_tree(m, name="", depth=0, max_depth=2):
    indent = "  " * depth
    pcount = module_param_count(m)
    cls = m.__class__.__name__
    print(f"{indent}{name or '[root]'} :: {cls} :: {human(pcount)} params")
    if depth >= max_depth:
        return
    for child_name, child in m.named_children():
        print_tree(child, child_name, depth+1, max_depth)

print("=== Module Tree (depth≤2) ===")
print_tree(model, "[root]", 0, 5)
print()





'''Per-layer structural summary (attention & MLP shapes)'''
# This tries to be resilient to differing attribute names across model impls.
layers = None
for candidate in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
    try:
        layers = eval(f"model.{candidate}")
        break
    except Exception:
        pass
if layers is None:
    raise RuntimeError("Could not locate layers list on this model.")

def first_shape_or_none(*tensors):
    for t in tensors:
        if t is not None:
            try:
                return tuple(t.weight.shape)
            except Exception:
                pass
    return None

# Try to detect typical submodules
def layer_summary(i, blk):
    # Norms
    in_norm  = getattr(blk, "input_layernorm",  getattr(blk, "pre_attention_layernorm", getattr(blk, "ln1", None)))
    post_norm= getattr(blk, "post_attention_layernorm", getattr(blk, "ln2", None))
    
    attn     = getattr(blk, "self_attn", getattr(blk, "attention", None))
    mlp      = getattr(blk, "mlp", getattr(blk, "feed_forward", None))
    
    # Attn projections
    q_proj = getattr(attn, "q_proj", None) if attn else None
    k_proj = getattr(attn, "k_proj", None) if attn else None
    v_proj = getattr(attn, "v_proj", None) if attn else None
    o_proj = getattr(attn, "o_proj", None) if attn else None
    
    # Some impls fuse qkv into one weight
    qkv = getattr(attn, "qkv_proj", None) or getattr(attn, "qkv", None)
    
    # MLP projections (llama/qwen style: gate/up/down)
    gate = getattr(mlp, "gate_proj", None) if mlp else None
    up   = getattr(mlp, "up_proj",   None) if mlp else None
    down = getattr(mlp, "down_proj", None) if mlp else None
    
    # Shapes
    q_shape   = first_shape_or_none(q_proj)
    k_shape   = first_shape_or_none(k_proj)
    v_shape   = first_shape_or_none(v_proj)
    o_shape   = first_shape_or_none(o_proj)
    qkv_shape = first_shape_or_none(qkv)
    gate_shape= first_shape_or_none(gate)
    up_shape  = first_shape_or_none(up)
    down_shape= first_shape_or_none(down)
    
    # Heads/kv heads (fall back to config if per-layer attr missing)
    n_head  = getattr(attn, "num_heads",  n_heads) if attn else n_heads
    n_kv_h  = getattr(attn, "num_key_value_heads", n_kv) if attn else n_kv
    
    # Param counts for big chunks
    def pcount(mod): return sum(p.numel() for p in mod.parameters()) if mod else 0
    attn_params = pcount(attn) if attn else 0
    mlp_params  = pcount(mlp)  if mlp  else 0
    norms_params= (pcount(in_norm) if in_norm else 0) + (pcount(post_norm) if post_norm else 0)
    
    return {
        "layer": i,
        "heads": int(n_head) if n_head is not None else None,
        "kv_heads": int(n_kv_h) if n_kv_h is not None else None,
        "q_shape": q_shape, "k_shape": k_shape, "v_shape": v_shape, "o_shape": o_shape,
        "qkv_fused_shape": qkv_shape,
        "gate_shape": gate_shape, "up_shape": up_shape, "down_shape": down_shape,
        "params:attn": attn_params, "params:mlp": mlp_params, "params:norms": norms_params
    }

per_layer = [layer_summary(i, blk) for i, blk in enumerate(layers)]

# Pretty print a few layers (first 3 and last 1)
def show_layer(d):
    print(
        f"L{d['layer']:02d}  heads={d['heads']}, kv_heads={d['kv_heads']}  "
        f"Q{d['q_shape']} K{d['k_shape']} V{d['v_shape']} O{d['o_shape']}  "
        f"QKVfused={d['qkv_fused_shape']}  "
        f"G{d['gate_shape']} U{d['up_shape']} D{d['down_shape']}  "
        f"| params attn={human(d['params:attn'])}, mlp={human(d['params:mlp'])}, norms={human(d['params:norms'])}"
    )

print("=== Per-layer summary (sample) ===")
for d in per_layer[:3]: show_layer(d)
if len(per_layer) > 3:
    print("...")
    show_layer(per_layer[-1])
print()

# (Optional) save full table
import pandas as pd
df = pd.DataFrame(per_layer)
df
