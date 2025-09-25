import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
# from torch.profiler import emit_nvtx

from module import *
from helper import *

def generate_regression_data(num_samples, in_dim, out_dim):
    x = torch.randn(num_samples, in_dim)
    y = torch.randn(num_samples, out_dim)   # continuous outputs
    return x, y

def generate_classification_data(num_samples, in_dim, num_classes):
    x = torch.randn(num_samples, in_dim)
    y = torch.randint(0, num_classes, (num_samples,))  # integer labels
    return x, y


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


warmup_steps = 5
benchmark_steps = 10


# --- Toy dataset ---
torch.manual_seed(0)
batch_size = 32
N, in_dim, hidden_dim, out_dim = batch_size * (warmup_steps + benchmark_steps), 2, 2, 2
num_epochs = 1

x, y = generate_regression_data(N, in_dim, out_dim)

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- MLP ---
model = nn.Linear(in_dim, out_dim, bias=False)


hot_ratio = 0.2

# skip element wise
# trainable_indices = generate_train_indices(in_features=in_dim, out_features=out_dim, frac=hot_ratio)
# model=LinearElementwise(in_dim, out_dim, trainable_indices, bias=False)

# # skip column wise
# trainable_indices = random_unique_columns(ncol=out_dim, m=int(out_dim*hot_ratio))
# model = LinearColWise.from_linear(model, hot_idx=trainable_indices)
# model = model.to(device)


# trainable_indices_list = []
# trainable_indices_list.append(generate_train_indices(in_features=in_dim, out_features=hidden_dim, frac=hot_ratio))
# trainable_indices_list.append(generate_train_indices(in_features=hidden_dim, out_features=out_dim, frac=hot_ratio))
# model = MLP_Frozen(in_dim, hidden_dim, out_dim, trainable_indices_list)



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model = model.to(device)


forward_list = []
backward_list = []
optimizer_list = []
num_steps = 0

for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # with torch.autograd.profiler.record_function("forward"):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("forward")
        t0 = time.time()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        torch.cuda.synchronize()
        t1 = time.time()
        torch.cuda.nvtx.range_pop()
        forward_list.append(t1 - t0)
        
        torch.cuda.nvtx.range_push("BACKWARD")
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        torch.cuda.nvtx.range_pop()
        backward_list.append(t1 - t0)

        # with torch.autograd.profiler.record_function("optimizer"):
        torch.cuda.nvtx.range_push("optimizer")
        t0 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        torch.cuda.nvtx.range_pop()
        optimizer_list.append(t1 - t0)

        total_loss += loss.item()

        num_steps += 1

        if num_steps >= warmup_steps + benchmark_steps:
            break


    print(f"Epoch {epoch+1}, Loss = {total_loss/len(dataloader):.4f}")

print(f"Avg forward time: {np.mean(forward_list[warmup_steps:])*1000000:.2f} us")
print(f"{np.asarray(forward_list)*1000000}")
print(f"Avg backward time: {np.mean(backward_list[warmup_steps:])*1000000:.2f} us")
print(f"{np.asarray(backward_list)*1000000}")
print(f"Avg optimizer step time: {np.mean(optimizer_list[warmup_steps:])*1000000:.2f} us")
print(f"{np.asarray(optimizer_list)*1000000}")



# nsys profile --trace=cuda,osrt,nvtx --cuda-memory-usage=true --show-output=true -o full python benchmark.py
# nsys profile --stats=true --show-output=true -f true --output ./profile/full python benchmark.py

