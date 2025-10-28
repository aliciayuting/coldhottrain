import torch
import torch.nn as nn
import torch.optim as optim
from module import *
from helper import *


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Toy dataset ---
torch.manual_seed(0)
N, in_dim, hidden_dim, out_dim = 128, 10, 8192, 1
X = torch.randn(N, in_dim, device=device)
true_w = torch.randn(in_dim, out_dim, device=device)
y = X @ true_w + 0.1 * torch.randn(N, out_dim, device=device)

# --- Model, loss, optimizer ---
model = MLP(in_dim, hidden_dim, out_dim).to(device)
# model = MLP_Frozen(in_dim, hidden_dim, out_dim).to(device)

# generate a random mask to freeze half of the neurons in the hidden layer
# trainable_indices = generate_train_indices(in_features=hidden_dim, out_features=out_dim, frac=0.5)
# print("Trainable indices:", trainable_indices)
# model = MLP_Frozen(in_dim, hidden_dim, out_dim, trainable_indices=trainable_indices).to(device)


# print("Parameters:")
# for name, param in model.fc2.named_parameters():
#     print(f"  {name:10s} {tuple(param.shape)} #elements: {param.numel()} {param.numel()*param.element_size()}bytes ")

# print("\nBuffers:")
# for name, buf in model.fc2.named_buffers():
#     print(f"  {name:10s} {tuple(buf.shape)} #elements: {buf.numel()} {buf.numel()*buf.element_size()}bytes ")

# model_size = get_size(model.fc2)
# print(f"\nTotal size of fc2 layer: {model_size} bytes")


# print out the original weights of fc2 layer
# print("Original weights of fc2 layer:")
# print(model.fc2.W_frozen)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


print("Buffers:")
for name, buf in model.named_buffers():
    print(f"  {name:10s} {tuple(buf.shape)} {buf.numel()} {buf.numel()*buf.element_size()} B ")

print("Parameters:")
for name, param in model.named_parameters():
    print(f"  {name:10s} {tuple(param.shape)} {param.numel()} {param.numel()*param.element_size()} B ")

    state = optimizer.state.get(param)
    if state is not None:
        for state_name, value in state.items():
            if torch.is_tensor(value):
                size_bytes = value.numel() * value.element_size()
                print(f"    Optimizer state '{state_name}': shape={tuple(value.shape)} "
                      f"numel={value.numel()} size={size_bytes} bytes")
            else:
                print(f"    Optimizer state '{state_name}': {value}")
    else:
        print("    No optimizer state for this parameter.")



# Map tensor IDs to names for readability

# print("Optimizer parameters (full model):")
# param_to_name = {id(p): n for n, p in model.named_parameters()}
# for i, group in enumerate(optimizer.param_groups):
#     print(f"Param group {i}:")
#     for p in group['params']:
#         name = param_to_name.get(id(p), "Unnamed")
#         size_bytes = p.numel() * p.element_size()
#         print(f"  {name:20s} shape={tuple(p.shape)} "
#               f"numel={p.numel()} size={size_bytes} bytes")

# exit()


param_size = 0
buff_size = 0
for name, param in model.named_parameters():
    print(f"  {name:10s} {tuple(param.shape)} {param.numel()} {param.numel()*param.element_size()} B ")
    param_size += param.numel() * param.element_size()
for name, buf in model.named_buffers():
    print(f"  {name:10s} {tuple(buf.shape)} {buf.numel()} {buf.numel()*buf.element_size()} B ")
    buff_size += buf.numel() * param.element_size()
total_size = param_size + buff_size

# calculated
param_size_calculated = 0
fc
print(f"Total size of model: {total_size} bytes")







# --- Training loop ---
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d}: loss = {loss.item():.4f}")

    # print("Frozen weights of fc2 layer:")
    # print(model.fc2.W_frozen)

    # print("Trainable weights of fc2 layer:")
    # for idx, val in zip(model.fc2.idx, model.fc2.vals):
    #     print(f"Index {idx.tolist()}: Value {val.item()}")


    for name, param in model.named_parameters():
        print(f"  {name:10s} {tuple(param.shape)} {param.numel()} {param.numel()*param.element_size()} B ")

        state = optimizer.state.get(param)
        if state is not None:
            for state_name, value in state.items():
                if torch.is_tensor(value):
                    size_bytes = value.numel() * value.element_size()
                    print(f"    Optimizer state '{state_name}': shape={tuple(value.shape)} "
                        f"numel={value.numel()} size={size_bytes} bytes")
                else:
                    print(f"    Optimizer state '{state_name}': {value}")
        else:
            print("    No optimizer state for this parameter.")

    print_gpu_memoory_usage()

    print("-" * 30)
