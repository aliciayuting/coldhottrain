# qwen2_5_0_5b_viz.py
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoConfig
from graphviz import Digraph

MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUT_DIR = Path("qwen2_5_0_5b_viz")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def load_config(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return cfg

def safe_get(d, key, default=None):
    return getattr(d, key, default) if hasattr(d, key) else d.get(key, default) if isinstance(d, dict) else default

def short_num(n):
    # pretty print ints (e.g., 4980736 -> "4.98M")
    if n is None: return "?"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"

def config_summary(cfg):
    # Try to read common transformer fields defensively
    d_model = safe_get(cfg, "hidden_size", safe_get(cfg, "n_embd"))
    n_layer = safe_get(cfg, "num_hidden_layers", safe_get(cfg, "n_layer"))
    n_head  = safe_get(cfg, "num_attention_heads", safe_get(cfg, "n_head"))
    kv_head = safe_get(cfg, "num_key_value_heads", safe_get(cfg, "n_kv_head"))
    vocab   = safe_get(cfg, "vocab_size")
    ctx     = safe_get(cfg, "max_position_embeddings", safe_get(cfg, "max_sequence_length"))
    act_fn  = safe_get(cfg, "hidden_act")
    norm    = "RMSNorm" if "rms" in str(cfg).lower() else "LayerNorm?"

    lines = [
        f"Model:       {cfg.__class__.__name__} ({MODEL_ID})",
        f"Type:        Decoder-only Transformer (causal LM)",
        f"Layers:      {n_layer}",
        f"D_model:     {d_model}",
        f"Attn heads:  {n_head}" + (f" (KV heads: {kv_head})" if kv_head else ""),
        f"Vocab:       {vocab}",
        f"Context:     {ctx}",
        f"MLP act:     {act_fn}",
        f"Norm:        {norm}",
    ]
    return "\n".join(lines)

def _lbl(s: str) -> str:
    """Sanitize labels for Graphviz (avoid record grammar & unicode arrows)."""
    return (
        s.replace("{", r"\{").replace("}", r"\}")
         .replace("→", "->")
         .replace("|", r"\|")
         .replace("<", r"\<").replace(">", r"\>")
    )

def build_high_level_graph(cfg, outfile: Path):
    # Pull a few key fields safely
    def g(obj, *keys, default=None):
        for k in keys:
            if hasattr(obj, k): return getattr(obj, k)
        return default

    n_layer = g(cfg, "num_hidden_layers", "n_layer", default=24)
    d_model = g(cfg, "hidden_size", "n_embd", default=1024)
    n_head  = g(cfg, "num_attention_heads", "n_head", default=16)
    kv_head = g(cfg, "num_key_value_heads", "n_kv_head", default=None)
    act_fn  = g(cfg, "hidden_act", default="silu/gelu")
    norm    = "RMSNorm" if "rms" in str(cfg).lower() else "LayerNorm?"

    dot = Digraph("Qwen2_5_0_5B_HighLevel", format="svg", engine="dot")
    dot.attr(rankdir="LR", fontsize="10", labelloc="t",
             label=_lbl(f"{MODEL_ID} — High-Level Architecture"))

    # Styles: use box, not record (record grammar is picky about {}, |, etc.)
    node_style  = dict(shape="box", fontsize="10", style="rounded,filled", fillcolor="#F6F8FB")
    layer_style = dict(shape="box", fontsize="9",  style="rounded",       color="#555555")

    # I/O
    dot.node("inp", _lbl("Input tokens\n(int64) -> IDs"), **node_style)
    dot.node("tok", _lbl(f"Token Embedding\n(vocab -> {d_model})"), **node_style)

    # Visual container note (folder shape can stay; it doesn’t parse label as record)
    dot.node("stack", _lbl(f"Transformer Block × {n_layer}"), shape="folder", fontsize="10")

    # Final norm/head
    dot.node("norm_f", _lbl(f"{norm}"), **node_style)
    dot.node("lmh",    _lbl(f"LM Head\n({d_model} -> vocab)"), **node_style)

    dot.edges([("inp", "tok"), ("tok", "stack"), ("stack", "norm_f"), ("norm_f", "lmh")])

    # Representative blocks: first, middle, last
    samples = sorted(set([0, max(1, n_layer // 2), n_layer - 1]))
    for i in samples:
        with dot.subgraph(name=f"cluster_block_{i}") as c:
            c.attr(label=_lbl(f"Block {i}"), color="#BBBBBB")
            c.node(f"b{i}_pre",  _lbl(f"Pre-Norm\n({norm})"), **layer_style)
            c.node(f"b{i}_attn", _lbl(f"Self-Attention\nheads={n_head}" + (f"/kv={kv_head}" if kv_head else "")), **layer_style)
            c.node(f"b{i}_res1", _lbl("Residual Add"), **layer_style)
            c.node(f"b{i}_mlp",  _lbl(f"MLP\n(act={act_fn})"), **layer_style)
            c.node(f"b{i}_res2", _lbl("Residual Add"), **layer_style)
            c.edges([
                (f"b{i}_pre",  f"b{i}_attn"),
                (f"b{i}_attn", f"b{i}_res1"),
                (f"b{i}_res1", f"b{i}_mlp"),
                (f"b{i}_mlp",  f"b{i}_res2"),
            ])

    # Connect stack through sample blocks (schematic)
    dot.edge("tok", "b0_pre", lhead="cluster_block_0")
    prev = samples[0]
    for nxt in samples[1:]:
        dot.edge(f"b{prev}_res2", f"b{nxt}_pre", ltail=f"cluster_block_{prev}", lhead=f"cluster_block_{nxt}")
        prev = nxt
    dot.edge(f"b{prev}_res2", "norm_f", ltail=f"cluster_block_{prev}")

    out = outfile.with_suffix(".svg")
    dot.render(outfile, cleanup=True)
    return out

def main():
    print("Loading config…")
    cfg = load_config(MODEL_ID)
    print("\n=== Config Summary ===")
    print(config_summary(cfg))

    print("\nDrawing high-level architecture…")
    svg_path = build_high_level_graph(cfg, OUT_DIR / "qwen2_5_0_5b_architecture")
    print(f"✅ Wrote diagram: {svg_path}")

    # Optional: quick parameter count without loading weights
    total_params = safe_get(cfg, "num_parameters", None)
    if total_params is not None:
        print(f"Parameters (reported): {short_num(total_params)}")
    else:
        print("Parameter count not reported in config (skip).")

def print_module_tree():
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
    def walk(module, prefix=""):
        print(prefix + module.__class__.__name__)
        for name, child in module.named_children():
            walk(child, prefix + f"{name}.")
    walk(model)

# Uncomment to run:
# print_module_tree()


if __name__ == "__main__":
    #main()
    print_module_tree()
