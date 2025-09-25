import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read argument k")
    parser.add_argument("--k", type=int, help="an integer argument k")
    parser.add_argument("--hot-ratio", type=float, help="a float argument hot_ratio", default=0.2)
    parser.add_argument("--skip", action="store_true", help="set this flag to skip something")


    args = parser.parse_args()
    dim = args.k
    skip = args.skip
    hot_ratio = args.hot_ratio
    print(f"k = {dim}, hot_ratio={hot_ratio} skip = {skip}")


    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    warmup_steps = 5
    benchmark_steps = 10


    # --- Toy dataset ---
    torch.manual_seed(0)
    batch_size = 32
    N, in_dim, hidden_dim, out_dim = batch_size * (warmup_steps + benchmark_steps), dim, dim, dim
    num_epochs = 1

    x, y = generate_regression_data(N, in_dim, out_dim)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- MLP ---
    model = nn.Linear(in_dim, out_dim, bias=False)

    # skip element wise
    # trainable_indices = generate_train_indices(in_features=in_dim, out_features=out_dim, frac=hot_ratio)
    # model=LinearElementwise(in_dim, out_dim, trainable_indices, bias=False)

    # skip column wise
    if skip:
        trainable_indices = random_unique_columns(ncol=out_dim, m=int(out_dim*hot_ratio))
        print(f"#Trainable={trainable_indices.numel()} Total: {in_dim * out_dim}")
        model = LinearColWise.from_linear(model, hot_idx=trainable_indices)
        model = model.to(device)


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
            # print("*"*10 + f"step: {num_steps}" + "*"*10)
            
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

            # print_params(model)

            

            if num_steps >= warmup_steps + benchmark_steps:
                print_gpu_memoory_usage()
                allocated, cached = print_gpu_memoory_usage()
                break


        print(f"Epoch {epoch+1}, Loss = {total_loss/len(dataloader):.4f}")

    print(f"Avg forward time: {np.mean(forward_list[warmup_steps:])*1000000:.2f} us")
    # print(f"{np.asarray(forward_list)*1000000}")
    print(f"Avg backward time: {np.mean(backward_list[warmup_steps:])*1000000:.2f} us")
    # print(f"{np.asarray(backward_list)*1000000}")
    print(f"Avg optimizer step time: {np.mean(optimizer_list[warmup_steps:])*1000000:.2f} us")
    # print(f"{np.asarray(optimizer_list)*1000000}")


    # wirte the hidden_size, allocated, cached, forward_time, backward_time, optimizer_time to a csv file
    with open("./performance/benchmark_results.csv", "a") as f:
        f.write(f"{in_dim},{out_dim},{hidden_dim},{skip},{hot_ratio},{allocated/1024**2:.2f},{cached/1024**2:.2f},{np.mean(forward_list[warmup_steps:])*1000000:.2f},{np.mean(backward_list[warmup_steps:])*1000000:.2f},{np.mean(optimizer_list[warmup_steps:])*1000000:.2f}\n")



# nsys profile --trace=cuda,osrt,nvtx --cuda-memory-usage=true --show-output=true -o full python benchmark.py
# nsys profile --stats=true --show-output=true -f true --output ./profile/full python benchmark.py

