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
    def __init__(self, in_dim, hidden_dim, out_dim, trainable_indices_list):
        super().__init__()
        # self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = LinearElementwise(in_dim, hidden_dim, trainable_indices_list[0])
        self.fc2 = LinearElementwise(hidden_dim, out_dim, trainable_indices_list[1])

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
        # self.register_buffer("idx", idx)                        # [nnz,2]
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


        self.stream_frozen = torch.cuda.Stream(device='cuda')
        self.stream_sparse = torch.cuda.Stream(device='cuda')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "streams only help on CUDA tensors"
        B = x.size(0)

        # Allocate separate outputs so two streams never race on the same buffer
        y_frozen = x.new_zeros(B, self.out_features)
        y_sparse = x.new_zeros(B, self.out_features)

        # 1) Frozen matmul on stream_frozen (no grads)
        s1 = self.stream_frozen
        with torch.cuda.stream(s1), torch.no_grad():
            # F.linear launches a cuBLAS GEMM; stays on s1
            # (Bias is handled later, so bias=None here.)
            torch._C._nn.linear(x, self.W_frozen, None, out=y_frozen) \
                if hasattr(torch._C._nn, "linear") else \
                y_frozen.add_(F.linear(x, self.W_frozen, bias=None))

        # 2) Sparse contribution on stream_sparse (needs grads wrt self.vals)
        s2 = self.stream_sparse
        with torch.cuda.stream(s2):
            # x[:, col] * vals -> scatter-add into y_sparse on dim=1 by row
            x_sel = x.index_select(1, self.col)            # [B, nnz]
            contrib = x_sel * self.vals.view(1, -1)        # [B, nnz]
            y_sparse.index_add_(1, self.row, contrib)      # accumulate

        # 3) Join: wait for both streams, then sum on the current stream
        cur = torch.cuda.current_stream()
        cur.wait_stream(s1)
        cur.wait_stream(s2)

        # (Optional but nice-to-have) Tie storage lifetimes to the consumer stream
        y_frozen.record_stream(cur)
        y_sparse.record_stream(cur)

        y = y_frozen.add_(y_sparse)

        if self.has_bias and self.b is not None:
            y = y + self.b

        # y = torch.relu(y)
        return y
    
# def get_size(module):
#     total = 0
#     print("Module size breakdown:")

#     # parameters
#     print("Parameters:")
#     for name, param in module.named_parameters():
#         total += param.numel() * param.element_size()
#         print(f"  {name:10s} {tuple(param.shape)} {param.numel()} {param.numel()*param.element_size()} B ")

#     # buffers
#     print("Buffers:")
#     for name, buf in module.named_buffers():
#         total += buf.numel() * buf.element_size()
#         print(f"  {name:10s} {tuple(buf.shape)} {buf.numel()} {buf.numel()*buf.element_size()} B ")

#     print(f"Total size: {total} bytes")
#     return total


def random_unique_columns(ncol: int, m: int, device=None):
    if m > ncol:
        raise ValueError(f"m ({m}) must be <= ncol ({ncol})")
    return torch.randperm(ncol, device=device, dtype=torch.long)[:m]

class LinearColWise(nn.Module):
    """
    Freeze whole output columns of a Linear(in_features -> out_features).
    - hot_idx: 1D LongTensor of output indices that remain trainable (“hot”).
               The complement is “cold” (frozen).
    - If bias=True: bias is split as well (hot bias is Parameter, cold bias is buffer).
    """
    def __init__(self, in_features: int, out_features: int,
                 hot_idx: torch.Tensor, bias: bool = False,
                 init_weight: torch.Tensor | None = None,
                 init_bias: torch.Tensor | None = None):
        super().__init__()
        assert hot_idx.ndim == 1
        self.in_features = in_features
        self.out_features = out_features

        # Normalize and derive cold indices
        hot_idx = hot_idx.to(torch.long).unique(sorted=True)
        all_idx = torch.arange(out_features, dtype=torch.long, device=hot_idx.device)
        cold_idx = torch.tensor(sorted(set(all_idx.tolist()) - set(hot_idx.tolist())),
                                dtype=torch.long, device=hot_idx.device)

        self.register_buffer("hot_idx", hot_idx, persistent=True)
        self.register_buffer("cold_idx", cold_idx, persistent=True)

        # Shapes for F.linear: weight is [out_features, in_features]
        if init_weight is None:
            # Kaiming uniform like nn.Linear default
            fan_in = in_features
            bound = 1.0 / fan_in**0.5
            full_W = (torch.empty(out_features, in_features).uniform_(-bound, bound))
        else:
            assert init_weight.shape == (out_features, in_features)
            full_W = init_weight.detach()

        # Split weights
        W_hot_init  = full_W[hot_idx]   # [hot_dim,  in_features]
        W_cold_init = full_W[cold_idx]  # [cold_dim, in_features]

        # Trainable hot, frozen cold
        self.W_hot = nn.Parameter(W_hot_init)
        self.register_buffer("W_cold", W_cold_init, persistent=True)

        # Bias handling (optional)
        self.has_bias = bias
        if bias:
            if init_bias is None:
                bound = 1.0 / in_features**0.5
                full_b = torch.empty(out_features).uniform_(-bound, bound)
            else:
                assert init_bias.shape == (out_features,)
                full_b = init_bias.detach()

            b_hot_init  = full_b[hot_idx]
            b_cold_init = full_b[cold_idx]
            self.b_hot  = nn.Parameter(b_hot_init)
            self.register_buffer("b_cold", b_cold_init, persistent=True)
        else:
            self.register_buffer("b_cold", None, persistent=False)
            self.b_hot = None

    @torch.no_grad()
    def set_cold_from_full(self, full_weight: torch.Tensor, full_bias: torch.Tensor | None = None):
        """Optional utility to refresh the cold part from a full matrix."""
        self.W_cold.copy_(full_weight[self.cold_idx])
        if self.has_bias and full_bias is not None:
            self.b_cold.copy_(full_bias[self.cold_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        y = x.new_empty(B, self.out_features)

        # Compute cold output with no grad
        with torch.no_grad():
            out_cold = F.linear(x, self.W_cold, self.b_cold if self.has_bias else None)  # [B, cold_dim]

        # Compute hot output with grad
        out_hot = F.linear(x, self.W_hot, self.b_hot if self.has_bias else None)         # [B, hot_dim]

        # Stitch back to original column order
        y.index_copy_(1, self.cold_idx, out_cold)
        y.index_copy_(1, self.hot_idx,  out_hot)
        return y

    @staticmethod
    def from_linear(base: nn.Linear, hot_idx: torch.Tensor) -> "LinearColWise":
        """Convenience: wrap an existing nn.Linear (weight [out,in], bias [out] or None)."""
        mod = LinearColWise(
            in_features=base.in_features,
            out_features=base.out_features,
            hot_idx=hot_idx,
            bias=base.bias is not None,
            init_weight=base.weight.data.clone(),
            init_bias=None if base.bias is None else base.bias.data.clone(),
        )
        return mod
