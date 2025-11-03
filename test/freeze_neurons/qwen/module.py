from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 

logger = logging.getLogger(__name__)
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
    @staticmethod
    def from_linear(
        base: nn.Linear,
        train_weight_indices: torch.Tensor,
        train_bias_indices: Optional[torch.Tensor] = None,
    ) -> "LinearElementwise":
        # preserve dtype/device and avoid .data
        w0 = base.weight.detach()
        if base.bias is not None:
            b0 = base.bias.detach()
        else:
            b0 = torch.zeros(base.out_features, device=w0.device, dtype=w0.dtype)

        return LinearElementwise(
            in_features=base.in_features,
            out_features=base.out_features,
            train_indices=train_weight_indices,
            #train_bias_indices=train_bias_indices,
            weight_init=w0,
            bias_init=b0,
        )
    
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
        with torch.no_grad():
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
            with torch.no_grad():
                inverse_idx = torch.empty_like(reorder_idx)
                inverse_idx[reorder_idx] = torch.arange(len(reorder_idx), device=reorder_idx.device)

            grad_W_cat = grad_W_full.index_select(0, inverse_idx).detach()
            
            # Split back to cold/hot
            cold_dim = W_cold.size(0)

            #this is the thing that saves memory. allows grad_W_cat to be deleted, and only the needed hot grads are saved.
            #with no clone, the grad_W_cat is a view of grad_W_full, means the whole matrix is needed for grad_x computation instead of just the 20% of hot ones.
            grad_W_cold = grad_W_cat[:cold_dim].clone() if ctx.needs_input_grad[1] else None
            grad_W_hot = grad_W_cat[cold_dim:].clone() if ctx.needs_input_grad[2] else None
            
            # Explicitly delete intermediates
            del grad_W_full, grad_W_cat, inverse_idx
        
        # Compute bias gradients
        grad_b_cold = grad_b_hot = None
        if b_cold is not None:
            if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
                # Sum over all dims except last (features)
                grad_b_full = grad_output.sum(dim=list(range(grad_output.ndim - 1)))
                
                # Inverse permute
                with torch.no_grad():
                    inverse_idx = torch.empty_like(reorder_idx)
                    inverse_idx[reorder_idx] = torch.arange(len(reorder_idx), device=reorder_idx.device)

                grad_b_cat = grad_b_full.index_select(0, inverse_idx).detach()
                
                # Split
                cold_dim = b_cold.size(0)
                grad_b_cold = grad_b_cat[:cold_dim].clone() if ctx.needs_input_grad[3] else None
                grad_b_hot = grad_b_cat[cold_dim:].clone() if ctx.needs_input_grad[4] else None
                del grad_b_full, grad_b_cat, inverse_idx
        del W_cat, W_full
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
                 mode: str = "1linear_efficient"):
                 #mode: str = "2linear_efficient"):
                 #mode: str = "1linear"):
                 #mode: str = "2linear"):
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

        #keep optimizer on cpu
        self._offloaded_opt_state = {
            "weight": {},   # for W_* rows (shape [in_features] per row)
            "bias": {}      # for b_* rows (shape [] or [1] per row effectively 1D)
        }

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
        #TODO: see if turning this on/off for out_cold has an effect on memory. We want the gradients to flow through though right? #with torch.no_grad():
        out_cold = F.linear(x, self.W_cold, self.b_cold if self.has_bias else None)   # [B, T, cold_dim]
        out_hot = F.linear(x, self.W_hot, self.b_hot if self.has_bias else None)          # [B, T, hot_dim]

        # Scatter into the last (feature) dimension
        # NOTE: use dim=2 because features are at the last axis
        y.index_copy_(2, self.cold_idx, out_cold)
        y.index_copy_(2, self.hot_idx,  out_hot)


        # with torch.cuda.stream(self.s_cold):
        #     out_cold = F.linear(x, self.W_cold, self.b_cold if self.has_bias else None)  # enqueued on s_cold

        # with torch.cuda.stream(self.s_hot):
        #     out_hot = F.linear(x, self.W_hot, self.b_hot if self.has_bias else None)     # enqueued on s_hot

        # # Make the default stream wait for both results before using them
        # self.s0.wait_stream(self.s_cold)
        # self.s0.wait_stream(self.s_hot)
        # y = x.new_empty(B, T, self.out_features)
        # y.index_copy_(2, self.cold_idx, out_cold)
        # y.index_copy_(2, self.hot_idx,  out_hot)


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


    @torch.no_grad()
    def switch_hot(self,
                   new_hot_idx: torch.Tensor,
                   optimizer: torch.optim.Optimizer | None = None,
                   keep_state: bool = True):
        """
        Change which output rows are trainable (“hot”) without ever creating a full [out,in] Parameter.
        If `optimizer` is provided, its state is patched so only hot rows hold state.
        
        Must be called between optimizer steps (no outstanding graph using the old params).
        """
        device = self.W_cold.device
        dtype  = self.W_cold.dtype
        new_hot_idx = new_hot_idx.to(device=device, dtype=torch.long).unique(sorted=True)

        # Fast no-op if identical
        if torch.equal(new_hot_idx, self.hot_idx):
            pass
            #return

        # --- derive new cold indices and new reorder ---
        all_idx = torch.arange(self.out_features, device=device, dtype=torch.long)
        new_cold_idx = torch.tensor(
            sorted(set(all_idx.tolist()) - set(new_hot_idx.tolist())),
            device=device, dtype=torch.long
        )

        new_reorder = torch.empty(self.out_features, dtype=torch.long, device=device)
        new_reorder[new_cold_idx] = torch.arange(len(new_cold_idx), device=device)
        new_reorder[new_hot_idx]  = torch.arange(len(new_hot_idx),  device=device) + len(new_cold_idx)

        # --- build mapping: output-index -> row-in-current W_hot/W_cold ---
        # (-1) means “doesn’t live here”
        hot_row_of_out = torch.full((self.out_features,), -1, device=device, dtype=torch.long)
        hot_row_of_out[self.hot_idx] = torch.arange(self.W_hot.size(0), device=device)
        cold_row_of_out = torch.full((self.out_features,), -1, device=device, dtype=torch.long)
        cold_row_of_out[self.cold_idx] = torch.arange(self.W_cold.size(0), device=device)

      
        # --- helper: gather rows in the exact order of out_idx, pulling from hot/cold as needed ---
        def gather_weight(out_idx: torch.Tensor):
            from_hot_mask = torch.isin(out_idx, self.hot_idx)
            rows = out_idx.numel()
            W = self.W_cold.new_empty(rows, self.in_features)
            # positions in the *destination* where source is old-hot / old-cold
            pos_from_hot  = torch.nonzero(from_hot_mask, as_tuple=False).squeeze(-1)
            pos_from_cold = torch.nonzero(~from_hot_mask, as_tuple=False).squeeze(-1)
            if pos_from_hot.numel():
                src_rows_hot = hot_row_of_out.index_select(0, out_idx.index_select(0, pos_from_hot))
                W.index_copy_(0, pos_from_hot, self.W_hot.index_select(0, src_rows_hot))
            if pos_from_cold.numel():
                src_rows_cold = cold_row_of_out.index_select(0, out_idx.index_select(0, pos_from_cold))
                W.index_copy_(0, pos_from_cold, self.W_cold.index_select(0, src_rows_cold))
            return W, from_hot_mask

        def gather_bias(out_idx: torch.Tensor, from_hot_mask: torch.Tensor):
            if not self.has_bias:
                return None
            b = torch.empty(out_idx.numel(), device=device, dtype=dtype)
            pos_from_hot  = torch.nonzero(from_hot_mask, as_tuple=False).squeeze(-1)
            pos_from_cold = torch.nonzero(~from_hot_mask, as_tuple=False).squeeze(-1)
            if pos_from_hot.numel():
                src_rows_hot = hot_row_of_out.index_select(0, out_idx.index_select(0, pos_from_hot))
                b.index_copy_(0, pos_from_hot, self.b_hot.index_select(0, src_rows_hot))
            if pos_from_cold.numel():
                src_rows_cold = cold_row_of_out.index_select(0, out_idx.index_select(0, pos_from_cold))
                b.index_copy_(0, pos_from_cold, self.b_cold.index_select(0, src_rows_cold))
            return b

        # --- build new tensors in *exact* new_hot_idx/new_cold_idx order ---
        new_W_hot_tensor, mask_new_hot_from_old_hot = gather_weight(new_hot_idx)
        new_W_cold_tensor, mask_new_cold_from_old_hot = gather_weight(new_cold_idx)

        new_b_hot_tensor  = gather_bias(new_hot_idx,  mask_new_hot_from_old_hot) if self.has_bias else None
        new_b_cold_tensor = gather_bias(new_cold_idx, mask_new_cold_from_old_hot) if self.has_bias else None

        # --- for optimizer state remap: where (dest positions) came from old-hot, and which old rows ---
        new_rows_from_old = torch.nonzero(mask_new_hot_from_old_hot, as_tuple=False).squeeze(-1)
        if new_rows_from_old.numel():
            old_rows_for_those = hot_row_of_out.index_select(0, new_hot_idx.index_select(0, new_rows_from_old))
        else:
            old_rows_for_those = new_rows_from_old  # empty

        # --- swap tensors into module (new Parameter for hot parts) ---
        #old new param swap
        # old_W_hot_param = self.W_hot
        # self.W_hot  = nn.Parameter(new_W_hot_tensor, requires_grad=True)
        # self.W_cold = new_W_cold_tensor  # buffer

        # if self.has_bias:
        #     old_b_hot_param = self.b_hot
        #     self.b_hot  = nn.Parameter(new_b_hot_tensor, requires_grad=True) if new_b_hot_tensor is not None else nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=True)
        #     self.b_cold = new_b_cold_tensor if new_b_cold_tensor is not None else torch.empty(0, device=device, dtype=dtype)
        # else:
        #     old_b_hot_param = None

        # # --- update index buffers ---
        # self.hot_idx     = new_hot_idx
        # self.cold_idx    = new_cold_idx
        # self.reorder_idx = new_reorder

        #new in place swap
        # weights
        self.W_hot.detach().copy_(new_W_hot_tensor)   # SAME Parameter object
        self.W_cold.copy_(new_W_cold_tensor)          # SAME buffer object

        # bias
        if self.has_bias:
            self.b_hot.detach().copy_(new_b_hot_tensor)
            self.b_cold.copy_(new_b_cold_tensor)

        # indices (also avoid rebinding buffer attributes)
        self.hot_idx.detach().copy_(new_hot_idx)
        self.cold_idx.detach().copy_(new_cold_idx)
        self.reorder_idx.detach().copy_(new_reorder)


        # --- patch optimizer state (optional) ---
        # if optimizer is not None:
        #     self._swap_param_in_optimizer(
        #         optimizer, old_W_hot_param, self.W_hot,
        #         keep_state=keep_state,
        #         new_rows_from_old=new_rows_from_old,
        #         old_rows_for_those=old_rows_for_those
        #     )
        #     if self.has_bias and old_b_hot_param is not None:
        #         # same row correspondence for bias
        #         self._swap_param_in_optimizer(
        #             optimizer, old_b_hot_param, self.b_hot,
        #             keep_state=keep_state,
        #             new_rows_from_old=new_rows_from_old,
        #             old_rows_for_those=old_rows_for_those
        #         )
        if optimizer is not None and keep_state:
            def _remap_state_inplace(param: torch.nn.Parameter,
                                    old_rows: torch.Tensor,
                                    new_rows: torch.Tensor):
                st = optimizer.state.get(param, None)
                if not isinstance(st, dict):
                    return
                for k, v in list(st.items()):
                    if not torch.is_tensor(v):
                        continue
                    # We only remap row-wise tensors that match the param's shape.
                    # (Adam/AdamW: 'exp_avg', 'exp_avg_sq'; SGD: 'momentum_buffer'.)
                    if v.shape != param.shape:
                        continue
                    # Make indices live on the same device as the state tensor
                    old_rows_d = old_rows.to(device=v.device, dtype=torch.long)
                    new_rows_d = new_rows.to(device=v.device, dtype=torch.long)
                    # Snapshot the rows that persist, then zero and scatter back
                    if old_rows_d.numel():
                        src_rows = v.index_select(0, old_rows_d).clone()
                    else:
                        src_rows = None
                    v.zero_()
                    if src_rows is not None and new_rows_d.numel():
                        v.index_copy_(0, new_rows_d, src_rows)

            _remap_state_inplace(self.W_hot,  old_rows_for_those, new_rows_from_old)
            if self.has_bias:
                _remap_state_inplace(self.b_hot, old_rows_for_those, new_rows_from_old)

    @staticmethod
    def _find_param_in_optimizer(optimizer, target):
        for g in optimizer.param_groups:
            for i, p in enumerate(g['params']):
                if p is target:           # identity check avoids __eq__
                    return g, i
        return None, None

    @staticmethod
    def _swap_param_in_optimizer(optimizer: torch.optim.Optimizer,
                                 old_param: nn.Parameter,
                                 new_param: nn.Parameter,
                                 *,
                                 keep_state: bool,
                                 new_rows_from_old: torch.Tensor,
                                 old_rows_for_those: torch.Tensor):
        """
        Replace `old_param` by `new_param` in the optimizer param groups.
        If keep_state=True, slice row-wise state tensors so rows that stayed hot keep their state.
        Newly hot rows get zero-initialized state. Cold rows' state is dropped.
        """
        # Locate the param group containing old_param
        group_found, idx_in_group = LinearColWise._find_param_in_optimizer(optimizer, old_param)
        if group_found is None:
            logger.warning("Old param not found in optimizer; cannot swap state.")
        # Old param not managed by this optimizer; nothing to do
            return
        #logger.info(f"Swapping param in optimizer group with {len(group_found['params'])} params. idx={idx_in_group}")
        # Replace param in group
        group_found['params'][idx_in_group] = new_param

        # Move / rebuild state
        old_state = optimizer.state.pop(old_param, None)
        if not keep_state or old_state is None or len(old_state) == 0:
            logger.info("No old state to keep; initializing new param state as empty.")
            optimizer.state[new_param] = {}  # lazy init by optimizer on first step
            return

        # Build a fresh state dict with same keys but row-sliced tensors
        new_state = {}
        for k, v in old_state.items():
            #logger.info(f"  key '{k}': type {type(v)}, shape {v.shape if torch.is_tensor(v) else 'N/A'}, old_param shape {old_param.shape}")
            if torch.is_tensor(v) and v.shape == old_param.shape:
                # v is a row-wise state tensor (e.g., exp_avg, exp_avg_sq, momentum_buffer)
                # Initialize zeros for the whole new_param, then copy the subset rows that remained hot.
                tgt = new_param.new_zeros(new_param.shape)
                if new_rows_from_old.numel() > 0:
                    #logger.info(f"    remapping {new_rows_from_old.numel()} rows from old state")
                    src_rows = v.index_select(0, old_rows_for_those)
                    tgt.index_copy_(0, new_rows_from_old, src_rows)
                else:
                    logger.info("    no rows remapped from old state")
                new_state[k] = tgt
            else:
                # scalars (e.g., step) or tensors not matching param shape: keep as is
                new_state[k] = v

        optimizer.state[new_param] = new_state
    def _assert_optimizer_has(self, opt):
        # 1) The exact object the model uses must be in the optimizer
        ids = {id(p) for g in opt.param_groups for p in g['params']}
        assert id(self.W_hot) in ids, "Optimizer not pointing to current W_hot"

        # 2) State tensors match current shape
        st = opt.state.get(self.W_hot, {})
        for k, v in st.items():
            if torch.is_tensor(v) and 'step' not in k:   # skip scalars
                assert v.shape == self.W_hot.shape, f"State[{k}] shape mismatch"

        # 3) After backward, grads exist and have magnitude
        # (call this after loss.backward())
        if self.W_hot.grad is not None:
            assert torch.isfinite(self.W_hot.grad).all()

    @staticmethod
    def from_linear(base: nn.Linear, hot_idx: torch.Tensor, mode: str = "1linear_efficient") -> "LinearColWise":
        """Convenience: wrap an existing nn.Linear (weight [out,in], bias [out] or None)."""
        mod = LinearColWise(
            in_features=base.in_features,
            out_features=base.out_features,
            hot_idx=hot_idx,
            bias=base.bias is not None,
            init_weight=base.weight.data.clone(),
            init_bias=None if base.bias is None else base.bias.data.clone(),
            mode=mode
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