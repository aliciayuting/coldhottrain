import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
class MLP_Frozen(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, trainable_indices):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = LinearElementwise(hidden_dim, out_dim, trainable_indices)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
def generate_train_indices(in_features, out_features, frac=0.5, seed=None):
    """
    Randomly select a fraction of weight positions (row, col) as trainable.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # All possible (row, col) pairs
    rows = torch.arange(out_features).repeat_interleave(in_features)
    cols = torch.arange(in_features).repeat(out_features)
    all_indices = torch.stack([rows, cols], dim=1)  # shape [out_features*in_features, 2]
    
    # Pick a subset (half by default)
    nnz = int(all_indices.size(0) * frac)
    perm = torch.randperm(all_indices.size(0))[:nnz]
    train_indices = all_indices[perm]
    
    return train_indices

class LinearElementwise(nn.Module):
    """
    Train only specific (row, col) positions.
    - indices: LongTensor [nnz, 2] with pairs (row, col) for trainable entries
    - vals: nn.Parameter [nnz] are the only trainable weights
    The frozen dense matrix (for all other positions) is kept as a buffer W_frozen.
    """
    def __init__(self, in_features, out_features, train_indices, weight_init=None, bias=True, bias_init=None, W_frozen_init=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        idx = torch.as_tensor(train_indices, dtype=torch.long)  # shape [nnz, 2]
        assert idx.ndim == 2 and idx.size(1) == 2
        self.register_buffer("idx", idx)                        # [nnz,2]
        self.register_buffer("row", idx[:,0])
        self.register_buffer("col", idx[:,1])

        # Initialize dense W to split frozen vs trainable
        if weight_init is None:
            bound = (1.0 / in_features) ** 0.5
            W_full = torch.empty(out_features, in_features).uniform_(-bound, bound)
        else:
            W_full = weight_init.clone().detach()

        # Trainable values as a flat vector
        self.vals = nn.Parameter(W_full[self.row, self.col])

        # Frozen dense for all other positions
        if W_frozen_init is None:
            W_frozen = W_full.clone()
            W_frozen[self.row, self.col] = 0  # remove trainable entries
        else:
            W_frozen = W_frozen_init.clone().detach()
        self.register_buffer("W_frozen", W_frozen)

        # Bias
        self.has_bias = bias
        if bias:
            if bias_init is None:
                bound = (1.0 / in_features) ** 0.5
                b = torch.empty(out_features).uniform_(-bound, bound)
            else:
                b = bias_init.clone().detach()
            self.b = nn.Parameter(b)   # or buffer if you want it frozen
        else:
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        y = x.new_zeros(B, self.out_features)

        if self.W_frozen is not None:
            with torch.no_grad():
                y = y + F.linear(x, self.W_frozen, bias=None)  # no bias here

        if self.vals.numel() > 0:
            contrib = x.index_select(1, self.col) * self.vals.view(1, -1)
            y.index_add_(1, self.row, contrib)

        if self.has_bias and self.b is not None:
            y = y + self.b  # bias gets gradients

        # y = torch.relu(y)
        return y