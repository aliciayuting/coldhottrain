import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
EPOCH_LENGTH = 407
main_dir = os.path.join(SCRATCH, "jamal_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca-neurons-50p-1e-20250916-162317")
checkpoint_dir = os.path.join(main_dir, "ckpt")
masks_path = os.path.join(main_dir, "neuron_masks.pt")

#TODO: open .pt file named neuron_masks.pt
print(f"Loading masks from {masks_path}")
masks = torch.load(masks_path)
print(f"Loaded masks keys: {masks.keys()}")

torch.set_printoptions(profile="full")

# Print Qwen/Qwen2.5-0.5B model layer names (derived from mask keys for speed)
# MODEL = "Qwen/Qwen2.5-0.5B"
# tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     torch_dtype=torch.bfloat16,
#     device_map="cpu",
# )

# for param in model.named_parameters():
#     print(param[0])

#example = masks['model.layers.11.mlp.up_proj.weight']
#print(f"Example mask: {example}")
total_masked = 0
total_elements = 0
for k,example in masks.items(): 
    print(f"Layer: {k}")
    print(f"Example mask shape: {example.shape}")
    print("proportion of masked:", sum(example) / example.numel())
    total_masked += sum(example)
    total_elements += example.numel()

print(f"average masked: {total_masked / total_elements}")



#load model from checkpoint
compare_1 = EPOCH_LENGTH*0
compare_2 = EPOCH_LENGTH*1
model1 = AutoModelForCausalLM.from_pretrained(
    os.path.join(checkpoint_dir, f"checkpoint-{compare_1}"),
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

model2 = AutoModelForCausalLM.from_pretrained(
    os.path.join(checkpoint_dir, f"checkpoint-{compare_2}"),
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)


# Compare weights of model1 and model2
for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    if name1 != name2:
        print(f"Layer names do not match: {name1} != {name2}")
        continue
    if param1.shape != param2.shape:
        print(f"Layer shapes do not match for {name1}: {param1.shape} != {param2.shape}")
        continue
    for row in range(len(param1)):
        if masks[name1][row] and not torch.equal(param1[row], param2[row]):
            print(f"Layer {name1} has differing weights at row {row}")
