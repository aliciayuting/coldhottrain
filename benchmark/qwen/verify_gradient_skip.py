import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
main_dir = os.path.join(SCRATCH, "jamal_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca-neurons-50p-1e-20250916-162317")
checkpoint_dir = os.path.join(main_dir, "ckpt")
masks_path = os.path.join(main_dir, "neuron_masks.pt")

#TODO: open .pt file named neuron_masks.pt
print(f"Loading masks from {masks_path}")
masks = torch.load(masks_path)
print(f"Loaded masks keys: {masks.keys()}")

torch.set_printoptions(profile="full")

# Print Qwen/Qwen2.5-0.5B model layer names (derived from mask keys for speed)
MODEL = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
print(model.named_parameters())

#example = masks['model.layers.11.mlp.up_proj.weight']
#print(f"Example mask: {example}")
for k,example in masks.items(): 
    print(f"Layer: {k}")
    print(f"Example mask shape: {example.shape}")
    print("proportion of masked:", sum(example) / example.numel())
