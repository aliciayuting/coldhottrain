from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification

# import torch.nn
import torch
import torch.nn as nn
from module import EmbeddingColWise, LinearColWise


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
    #TODO: remove this later
    if k == out_features:
        k = out_features - 1
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    perm = torch.randperm(out_features, device=device)
    return perm[:k].sort().values

def replace_linear_with_colwise(mod: nn.Module, hot_idx: torch.Tensor, mode: str = "1linear_efficient") -> LinearColWise:
    assert isinstance(mod, nn.Linear)
    hot_idx = hot_idx.to(mod.weight.device)
    wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx, mode=mode)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped

def replace_embedding_with_colwise(mod: nn.Module, hot_idx: torch.Tensor) -> EmbeddingColWise:
    assert isinstance(mod, nn.Embedding)
    hot_idx = hot_idx.to(mod.weight.device)
    wrapped = EmbeddingColWise.from_embedding(mod, hot_idx=hot_idx)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped

# ---- apply to layer 23 of Qwen2.5 -------------------------------------------
policy_by_name = {
    "self_attn.q_proj": 0.30,
    "self_attn.k_proj": 0.30,
    "self_attn.v_proj": 0.30,
    "self_attn.o_proj": 0.30,
    "mlp.up_proj":      0.30,
    "mlp.down_proj":    0.30,
    # "mlp.gate_proj":  0.30,  # include if you want to split the gate too
}
