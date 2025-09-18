import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
EPOCH_LENGTH = 407
main_dir = os.path.join(SCRATCH, "jamal_runs/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca-neurons-50p-1e-20250917-163518")
checkpoint_dir = os.path.join(main_dir, "ckpt")
gradient_dir = os.path.join(main_dir, "grad_dump/step004500")
masks_path = os.path.join(main_dir, "neuron_masks.pt")

#TODO: open .pt file named neuron_masks.pt
print(f"Loading masks from {masks_path}")
masks = torch.load(masks_path)
print(f"Loaded masks keys: {masks.keys()}")

torch.set_printoptions(profile="full")

def convert_name(s: str) -> str:
    parts = s.split(".")
    # Extract layer number, zero-padded to 2 digits
    layer_num = int(parts[2])
    layer_str = f"L{layer_num:02d}"
    # Join remaining parts with underscores
    rest = "_".join(parts[3:])
    return f"{layer_str}_{rest}.npy"

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

print("total masked:", total_masked)
print("total elements:", total_elements)
print(f"average masked: {total_masked / total_elements}")

for k,example in masks.items(): 
    converted_name = convert_name(k)
    print(f"Converted {k} to {converted_name}")
    # fetch numpy file with name converted_name (already includes .npy) in
    # gradient_dir and check rows where example == True are all zeros
    grad_path = os.path.join(gradient_dir, converted_name)
    if not os.path.exists(grad_path):
        print(f"Gradient file not found: {grad_path}")
        continue

    # Fast load without copying entire array into memory
    arr = np.load(grad_path, mmap_mode="r")

    # Convert mask to numpy boolean and index rows
    rows_mask = example.to(torch.bool).cpu().numpy()
    if rows_mask.ndim != 1:
        rows_mask = rows_mask.reshape(-1)

    # If mask length mismatches rows, skip minimal checking to avoid slow ops
    if arr.ndim == 0 or arr.shape[0] != rows_mask.shape[0]:
        print(f"Shape mismatch for {converted_name}: grad rows {arr.shape} vs mask {rows_mask.shape}")
        continue

    if not np.any(rows_mask):
        print("No masked rows to verify (all False).")
        continue

    # Check that all elements in masked rows are exactly zero
    masked_block = arr[rows_mask]
    nonzero_count = np.count_nonzero(masked_block)
    if nonzero_count == 0:
        print("OK: masked rows are all zeros.")
    else:
        print(f"FAIL: {nonzero_count} non-zero elements in masked rows.")
    


#load model from checkpoint
#compare_1 = EPOCH_LENGTH*0
#compare_1 = "Qwen/Qwen2.5-0.5B"
# compare_1 = os.path.join(checkpoint_dir, f"checkpoint-{EPOCH_LENGTH*27}")
# compare_2 = os.path.join(checkpoint_dir, f"checkpoint-{EPOCH_LENGTH*28}")

# model1 = AutoModelForCausalLM.from_pretrained(
#     compare_1,
#     torch_dtype=torch.bfloat16,
#     device_map="cpu",
# )

# model2 = AutoModelForCausalLM.from_pretrained(
#     compare_2,
#     torch_dtype=torch.bfloat16,
#     device_map="cpu",
# )


# # Compare weights of model1 and model2
# for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
#     if name1 != name2:
#         print(f"Layer names do not match: {name1} != {name2}")
#         continue
#     if param1.shape != param2.shape:
#         print(f"Layer shapes do not match for {name1}: {param1.shape} != {param2.shape}")
#         continue
#     if name1 not in masks:
#         print(f"Layer {name1} not found in masks")
#         continue
#     for row in range(len(param1)):
#         if masks[name1][row]==True and not torch.equal(param1[row], param2[row]):
#             # go through each element in the row and print the differing elements
#             for col in range(len(param1[row])):
#                 if param1[row][col] != param2[row][col]:
#                     print(f"Layer {name1} has differing weights at row {row}, col {col}: {param1[row][col]} != {param2[row][col]}")


