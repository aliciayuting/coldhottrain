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




class EfficientFullLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_cold, W_hot, b_cold, b_hot, reorder_idx):
        # Save only the original components (not W_cat or W_full!)
        ctx.save_for_backward(x, W_cold, W_hot, reorder_idx)
        ctx.b_cold = b_cold  # Store as context (might be None)
        ctx.b_hot = b_hot
        
        # Create W_full (not saved for backward)
        W_cat = torch.cat([W_cold, W_hot], dim=0)
        W_full = W_cat.index_select(0, reorder_idx)
        
        # Create b_full if needed (not saved)
        b_full = None
        if b_cold is not None:
            b_cat = torch.cat([b_cold, b_hot], dim=0)
            b_full = b_cat.index_select(0, reorder_idx)
        
        # Compute output
        return F.linear(x, W_full, b_full)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, W_cold, W_hot, reorder_idx = ctx.saved_tensors
        b_cold = ctx.b_cold
        b_hot = ctx.b_hot
        
        # Recreate W_full for input gradient computation
        W_cat = torch.cat([W_cold, W_hot], dim=0)
        W_full = W_cat.index_select(0, reorder_idx)
        
        # Compute input gradient
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = F.linear(grad_output, W_full.t())
        
        # Compute weight gradients
        grad_W_cold = grad_W_hot = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            # Flatten x for matmul: [B, T, in] -> [B*T, in]
            x_flat = x.reshape(-1, x.size(-1))
            grad_output_flat = grad_output.reshape(-1, grad_output.size(-1))
            
            # grad w.r.t W_full: [out, in]
            grad_W_full = grad_output_flat.t().mm(x_flat)
            
            # Inverse permutation to get grad_W_cat
            inverse_idx = torch.empty_like(reorder_idx)
            inverse_idx[reorder_idx] = torch.arange(len(reorder_idx), device=reorder_idx.device)
            grad_W_cat = grad_W_full.index_select(0, inverse_idx)
            
            # Split back to cold/hot
            cold_dim = W_cold.size(0)
            grad_W_cold = grad_W_cat[:cold_dim] if ctx.needs_input_grad[1] else None
            grad_W_hot = grad_W_cat[cold_dim:] if ctx.needs_input_grad[2] else None
        
        # Compute bias gradients
        grad_b_cold = grad_b_hot = None
        if b_cold is not None:
            if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
                # Sum over all dims except last (features)
                grad_b_full = grad_output.sum(dim=list(range(grad_output.ndim - 1)))
                
                # Inverse permute
                inverse_idx = torch.empty_like(reorder_idx)
                inverse_idx[reorder_idx] = torch.arange(len(reorder_idx), device=reorder_idx.device)
                grad_b_cat = grad_b_full.index_select(0, inverse_idx)
                
                # Split
                cold_dim = b_cold.size(0)
                grad_b_cold = grad_b_cat[:cold_dim] if ctx.needs_input_grad[3] else None
                grad_b_hot = grad_b_cat[cold_dim:] if ctx.needs_input_grad[4] else None
        
        return grad_x, grad_W_cold, grad_W_hot, grad_b_cold, grad_b_hot, None

class Efficient2Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_cold, W_hot, b_cold, b_hot, cold_idx, hot_idx, out_features):
        # Save inputs for backward (but NOT the intermediate activations!)
        ctx.save_for_backward(x, W_cold, W_hot, cold_idx, hot_idx)
        ctx.b_cold = b_cold
        ctx.b_hot = b_hot
        ctx.out_features = out_features
        
        # Compute outputs (not saved)
        out_cold = F.linear(x, W_cold, b_cold)  # [B, T, cold_dim]
        out_hot = F.linear(x, W_hot, b_hot)     # [B, T, hot_dim]
        
        # Assemble output
        y = x.new_empty(*x.shape[:-1], out_features)
        y.index_copy_(-1, cold_idx, out_cold)
        y.index_copy_(-1, hot_idx, out_hot)
        
        # Only return y, intermediates are freed!
        return y
    
    @staticmethod  
    def backward(ctx, grad_output):
        x, W_cold, W_hot, cold_idx, hot_idx = ctx.saved_tensors
        b_cold = ctx.b_cold
        b_hot = ctx.b_hot
        
        # Extract gradients for cold and hot paths directly from grad_output
        grad_out_cold = grad_output.index_select(-1, cold_idx)  # [B, T, cold_dim]
        grad_out_hot = grad_output.index_select(-1, hot_idx)    # [B, T, hot_dim]
        
        # Input gradient
        grad_x = None
        if ctx.needs_input_grad[0]:
            # Compute grad_x by applying transposed weights
            grad_x_cold = F.linear(grad_out_cold, W_cold.t())
            grad_x_hot = F.linear(grad_out_hot, W_hot.t())
            grad_x = grad_x_cold + grad_x_hot
        
        # Weight gradients
        grad_W_cold = grad_W_hot = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            # Flatten for matmul
            x_flat = x.reshape(-1, x.size(-1))  # [B*T, in]
            grad_out_cold_flat = grad_out_cold.reshape(-1, grad_out_cold.size(-1))  # [B*T, cold_dim]
            grad_out_hot_flat = grad_out_hot.reshape(-1, grad_out_hot.size(-1))     # [B*T, hot_dim]
            
            if ctx.needs_input_grad[1]:
                grad_W_cold = grad_out_cold_flat.t().mm(x_flat)  # [cold_dim, in]
            if ctx.needs_input_grad[2]:
                grad_W_hot = grad_out_hot_flat.t().mm(x_flat)    # [hot_dim, in]
        
        # Bias gradients  
        grad_b_cold = grad_b_hot = None
        if b_cold is not None:
            if ctx.needs_input_grad[3]:
                grad_b_cold = grad_out_cold.sum(dim=list(range(grad_out_cold.ndim - 1)))
            if ctx.needs_input_grad[4]:
                grad_b_hot = grad_out_hot.sum(dim=list(range(grad_out_hot.ndim - 1)))
        
        return grad_x, grad_W_cold, grad_W_hot, grad_b_cold, grad_b_hot, None, None, None


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
                 init_bias: torch.Tensor | None = None,
                 #mode: str = "1linear_efficient"):
                 mode: str = "2linear_efficient"):
        super().__init__()
        assert hot_idx.ndim == 1
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        # Normalize and derive cold indices
        hot_idx = hot_idx.to(torch.long).unique(sorted=True)
        all_idx = torch.arange(out_features, dtype=torch.long, device=hot_idx.device)
        cold_idx = torch.tensor(sorted(set(all_idx.tolist()) - set(hot_idx.tolist())),
                                dtype=torch.long, device=hot_idx.device)

        self.register_buffer("hot_idx", hot_idx, persistent=True)
        self.register_buffer("cold_idx", cold_idx, persistent=True)
        
        #TODO: delete if unneccessary
        reorder_idx = torch.empty(out_features, dtype=torch.long, device=hot_idx.device)
        reorder_idx[cold_idx] = torch.arange(len(cold_idx), device=hot_idx.device)
        reorder_idx[hot_idx]  = torch.arange(len(hot_idx),  device=hot_idx.device) + len(cold_idx)
        self.register_buffer("reorder_idx", reorder_idx, persistent=True)

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

        #TODO: purpose of these?
        self.s0 = torch.cuda.current_stream("cuda")
        self.s_cold = torch.cuda.Stream(device="cuda")
        self.s_hot  = torch.cuda.Stream(device="cuda")

    @torch.no_grad()
    def set_cold_from_full(self, full_weight: torch.Tensor, full_bias: torch.Tensor | None = None):
        """Optional utility to refresh the cold part from a full matrix."""
        self.W_cold.copy_(full_weight[self.cold_idx])
        if self.has_bias and full_bias is not None:
            self.b_cold.copy_(full_bias[self.cold_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "1linear":
            return self.forward_1linear(x)
        elif self.mode == "1linear_efficient":
            return self.forward_1linear_efficient(x)
        elif self.mode == "2linear":
            return self.forward_2linear(x)
        elif self.mode == "2linear_efficient":
            return self.forward_2linear_efficient(x)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        
    def forward_1linear(self, x: torch.Tensor) -> torch.Tensor:
        # Assemble full weight/bias in *concatenated* order then permute to full layout.
        # This preserves grads to W_hot only; W_cold stays a buffer (no optimizer state).
        W_cat = torch.cat([self.W_cold, self.W_hot], dim=0)            # [cold+hot, in]
        W_full = W_cat.index_select(0, self.reorder_idx)               # [out, in]

        if self.has_bias:
            b_cat = torch.cat([self.b_cold, self.b_hot], dim=0)        # [cold+hot]
            b_full = b_cat.index_select(0, self.reorder_idx)           # [out]
        else:
            b_full = None

        return F.linear(x, W_full, b_full)
    
    def forward_1linear_efficient(self, x: torch.Tensor) -> torch.Tensor:
        # Memory-efficient version using custom autograd
        return EfficientFullLinear.apply(
            x, self.W_cold, self.W_hot, 
            self.b_cold, self.b_hot,
            self.reorder_idx
        )
    
    def forward_2linear(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_features]
        B, T, _ = x.shape
        device = x.device

        # Allocate output with 3D shape
        y = x.new_empty(B, T, self.out_features)

        # Compute cold and hot paths (F.linear works on the last dim)
        #TODO: see if turning this on/off for out_cold has an effect on memory. We want the gradients to flow through though right?
        #with torch.no_grad():
        out_cold = F.linear(x, self.W_cold, self.b_cold if self.has_bias else None)   # [B, T, cold_dim]
        out_hot = F.linear(x, self.W_hot, self.b_hot if self.has_bias else None)          # [B, T, hot_dim]

        # Scatter into the last (feature) dimension
        # NOTE: use dim=2 because features are at the last axis
        y.index_copy_(2, self.cold_idx, out_cold)
        y.index_copy_(2, self.hot_idx,  out_hot)


        # # ---- cold half ----
        # out_cold = F.linear(x, self.W_cold, self.b_cold if self.has_bias else None)
        # y.index_copy_(2, self.cold_idx, out_cold)
        # del out_cold                      # let it go out of scope ASAP

        # # ---- hot half ----
        # out_hot = F.linear(x, self.W_hot, self.b_hot if self.has_bias else None)
        # y.index_copy_(2, self.hot_idx, out_hot)
        # del out_hot

        return y
    
    def forward_2linear_efficient(self, x: torch.Tensor) -> torch.Tensor:
        # Memory-efficient 2linear using custom autograd
        return Efficient2Linear.apply(
            x, self.W_cold, self.W_hot,
            self.b_cold, self.b_hot,
            self.cold_idx, self.hot_idx,
            self.out_features
        )

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

class EmbeddingColWise(nn.Module):
    """
    Split/freeze rows of an Embedding(num_embeddings, embedding_dim).
    - hot_idx: token IDs (rows) that remain trainable; the rest are frozen.
    - padding_idx: preserved like nn.Embedding (output zeros; no grads flow).
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 hot_idx: torch.Tensor,
                 padding_idx: int | None = None,
                 init_weight: torch.Tensor | None = None,
                 max_norm: float | None = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 mode: str = "2linear"):
        super().__init__()
        assert hot_idx.ndim == 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Normalize and derive cold indices
        hot_idx = hot_idx.to(torch.long).unique(sorted=True)
        all_idx = torch.arange(num_embeddings, dtype=torch.long, device=hot_idx.device)
        cold_idx = torch.tensor(sorted(set(all_idx.tolist()) - set(hot_idx.tolist())),
                                dtype=torch.long, device=hot_idx.device)

        self.register_buffer("hot_idx",  hot_idx,  persistent=True)
        self.register_buffer("cold_idx", cold_idx, persistent=True)

        # Initialize from full table or random
        if init_weight is None:
            bound = 1.0 / embedding_dim**0.5
            full_W = torch.empty(num_embeddings, embedding_dim, device=hot_idx.device).uniform_(-bound, bound)
            if padding_idx is not None:
                full_W[padding_idx].zero_()
        else:
            assert init_weight.shape == (num_embeddings, embedding_dim)
            full_W = init_weight.detach()

        # Trainable hot, frozen cold
        self.W_hot  = nn.Parameter(full_W[hot_idx])
        self.register_buffer("W_cold", full_W[cold_idx], persistent=True)

        # Precompute ID→position remaps (in compact hot/cold tables)
        # -1 means "not present" in that table
        hot_pos  = torch.full((num_embeddings,), -1, dtype=torch.long, device=hot_idx.device)
        cold_pos = torch.full((num_embeddings,), -1, dtype=torch.long, device=hot_idx.device)
        hot_pos[hot_idx]   = torch.arange(hot_idx.numel(),  device=hot_idx.device)
        cold_pos[cold_idx] = torch.arange(cold_idx.numel(), device=hot_idx.device)

        self.register_buffer("hot_pos",  hot_pos,  persistent=True)
        self.register_buffer("cold_pos", cold_pos, persistent=True)
        self.register_buffer("is_hot_row", hot_pos.ne(-1), persistent=True)

    @torch.no_grad()
    def set_cold_from_full(self, full_weight: torch.Tensor):
        """Optional: refresh frozen rows from a full table."""
        self.W_cold.copy_(full_weight[self.cold_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] token IDs in [0, num_embeddings)
        # Map IDs to compact positions
        pos_hot  = self.hot_pos[x]        # [-1 or 0..|hot|-1]
        pos_cold = self.cold_pos[x]       # [-1 or 0..|cold|-1]
        mask_hot = self.is_hot_row[x]     # True where token is hot

        # Do the two compact lookups; clamp -1→0 (dummy row) then mask-select later
        #emb_hot  = F.embedding(pos_hot.clamp_min(0),  self.W_hot)   # [B, T, D]
        #emb_cold = F.embedding(pos_cold.clamp_min(0), self.W_cold)  # [B, T, D]
        emb_hot = F.embedding(
            pos_hot.clamp_min(0), self.W_hot,
            padding_idx=None,
            max_norm=self.max_norm, norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )
        emb_cold = F.embedding(
            pos_cold.clamp_min(0), self.W_cold,
            padding_idx=None,
            max_norm=self.max_norm, norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )
        y = torch.where(mask_hot.unsqueeze(-1), emb_hot, emb_cold)

        # Preserve padding semantics
        if self.padding_idx is not None:
            pad_mask = x.eq(self.padding_idx)
            if pad_mask.any():
                y = y.masked_fill(pad_mask.unsqueeze(-1), 0)

        return y

    @staticmethod
    def from_embedding(base: nn.Embedding, hot_idx: torch.Tensor) -> "EmbeddingColWise":
        mod =  EmbeddingColWise(
            num_embeddings=base.num_embeddings,
            embedding_dim=base.embedding_dim,
            hot_idx=hot_idx,
            padding_idx=base.padding_idx,
            init_weight=base.weight.detach().clone(),
            max_norm=getattr(base, "max_norm", None),
            norm_type=getattr(base, "norm_type", 2.0),
            scale_grad_by_freq=getattr(base, "scale_grad_by_freq", False),
            sparse=getattr(base, "sparse", False),
        )
        return mod


def print_params(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        print(f"  {name:10s} {param.data} ")

    print("Buffers:")
    for name, buf in model.named_buffers():
        print(f"  {name:10s} {buf}  ")
