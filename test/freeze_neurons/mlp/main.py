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
# model = MLP(in_dim, hidden_dim, out_dim).to(device)

# generate a random mask to freeze half of the neurons in the hidden layer
trainable_indices = generate_train_indices(in_features=hidden_dim, out_features=out_dim, frac=0.5)
print("Trainable indices:", trainable_indices)
model = MLP_Frozen(in_dim, hidden_dim, out_dim, trainable_indices=trainable_indices).to(device)

# print out the original weights of fc2 layer
# print("Original weights of fc2 layer:")
# print(model.fc2.W_frozen)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

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

    print_gpu_memoory_usage()

    print("-" * 30)
