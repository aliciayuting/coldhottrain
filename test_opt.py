from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MODEL = "facebook/opt-13b"
MODEL = "Qwen/Qwen2.5-0.5B"

# Load tokenizer & model
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,   # works well on A100
    device_map="auto"             # sharded across visible GPUs if needed
)

# Simple prompt (feel free to swap for a GSM8K-style problem)
# prompt = "Solve: If a book costs $12 and a pen costs $3, how much for 5 books and 4 pens?\nAnswer:"
prompt = "### Instruction:\nIf a book costs $12 and a pen costs $3, how much for 5 books and 4 pens?\n\n### Response:"


inputs = tok(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tok.decode(out[0], skip_special_tokens=True))