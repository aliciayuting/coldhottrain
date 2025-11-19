import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class LinearElementwise(nn.Module):
    """
    Linear layer where only specific weights *and* specific bias entries are trainable.

    Args
    ----
    in_features, out_features : ints
    train_weight_indices      : LongTensor/list with shape [nnz_w, 2] (row, col) trainable weights
    train_bias_indices        : LongTensor/list with shape [nnz_b] (row)   trainable bias entries (optional)
    weight_init               : optional full [out_features, in_features] initial weight (used to init both parts)
    W_frozen_init             : optional full frozen [out_features, in_features] (we still zero trainable slots)
    bias_init                 : optional full [out_features] initial bias (used to init both parts)
    b_frozen_init             : optional frozen [out_features] bias (we still zero trainable slots)
    metadata                  : dict, additional metadata for the layer
    Semantics
    ---------
    Effective weight = W_frozen (zeros at trainable slots) + sparse(vals at train_weight_indices)
    Effective bias   = b_frozen (zeros at trainable slots) + sparse(bias_vals at train_bias_indices)
    """


    def __init__(self, in_features, out_features,
                 train_weight_indices,
                 train_bias_indices: Optional[torch.Tensor] = None,
                 weight_init=None, W_frozen_init=None,
                 bias_init=None, b_frozen_init=None,
                 metadata: Dict = {}):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        metadata = metadata if metadata is not None else {}
        self.metadata = dict(metadata)
        self._metadata_buffer_attr = {}
        for key, value in self.metadata.items():
            if torch.is_tensor(value):
                tensor = value.detach()
                tensor.requires_grad_(False)
                safe_key = str(key).replace(".", "_")
                base_name = f"_metadata_{safe_key}"
                buf_name = base_name
                suffix = 0
                while hasattr(self, buf_name):
                    suffix += 1
                    buf_name = f"{base_name}_{suffix}"
                self.register_buffer(buf_name, tensor)
                self._metadata_buffer_attr[key] = buf_name

        # ----- Trainable WEIGHT indices -----
        w_idx = torch.as_tensor(train_weight_indices, dtype=torch.long)
        if w_idx.ndim != 2 or w_idx.size(1) != 2:
            raise ValueError("train_weight_indices must be [nnz, 2] (row, col)")
        w_row, w_col = w_idx[:, 0], w_idx[:, 1]
        if torch.any(w_row < 0) or torch.any(w_row >= self.out_features):
            raise ValueError("weight row indices out of range")
        if torch.any(w_col < 0) or torch.any(w_col >= self.in_features):
            raise ValueError("weight col indices out of range")

        # Enforce uniqueness of (row, col) pairs to avoid silent summation bugs.
        lin = w_row * self.in_features + w_col
        if torch.unique(lin).numel() != lin.numel():
            raise ValueError("train_weight_indices contains duplicate (row, col) entries")

        self.register_buffer("row", w_row)
        self.register_buffer("col", w_col)

        # ----- Initialize full/base weights used to seed both parts -----
        if weight_init is None:
            bound = (1.0 / self.in_features) ** 0.5
            W_full = torch.empty(self.out_features, self.in_features, dtype=torch.get_default_dtype())
            W_full.uniform_(-bound, bound)
        else:
            W_full = torch.as_tensor(weight_init).clone().detach()
            if W_full.shape != (self.out_features, self.in_features):
                raise ValueError("weight_init has wrong shape")

        # Trainable weight values (exactly those slots)
        self.vals = nn.Parameter(W_full[self.row, self.col].clone())

        # Frozen weight base (zero out trainable slots to avoid double-counting)
        if W_frozen_init is None:
            W_frozen = W_full.clone()
        else:
            W_frozen = torch.as_tensor(W_frozen_init, dtype=W_full.dtype).clone().detach()
            if W_frozen.shape != (self.out_features, self.in_features):
                raise ValueError("W_frozen_init has wrong shape")
        W_frozen[self.row, self.col] = 0
        self.register_buffer("W_frozen", W_frozen)

        # ----- Trainable BIAS indices (optional) -----
        if train_bias_indices is not None:
            b_idx = torch.as_tensor(train_bias_indices, dtype=torch.long)
            if b_idx.ndim != 1:
                raise ValueError("train_bias_indices must be 1D [nnz_b]")
            if torch.any(b_idx < 0) or torch.any(b_idx >= self.out_features):
                raise ValueError("bias indices out of range")
            if torch.unique(b_idx).numel() != b_idx.numel():
                raise ValueError("train_bias_indices contains duplicate rows")
            self.register_buffer("bias_idx", b_idx)
        else:
            self.bias_idx = None  # keep attribute for checks

        # Determine bias seeds
        if bias_init is not None:
            b_full_for_vals = torch.as_tensor(bias_init, dtype=W_full.dtype).clone().detach()
            if b_full_for_vals.numel() != self.out_features:
                raise ValueError("bias_init must be length out_features")
        else:
            bound = (1.0 / self.in_features) ** 0.5
            b_full_for_vals = torch.empty(self.out_features, dtype=W_full.dtype).uniform_(-bound, bound)

        # Trainable bias values (if any)
        if self.bias_idx is not None:
            self.bias_vals = nn.Parameter(b_full_for_vals[self.bias_idx].clone())
        else:
            self.bias_vals = None

        # Frozen bias base (zeros by default unless provided); zero-out trainable slots
        if b_frozen_init is not None:
            b_frozen = torch.as_tensor(b_frozen_init, dtype=W_full.dtype).clone().detach()
            if b_frozen.numel() != self.out_features:
                raise ValueError("b_frozen_init must be length out_features")
        elif bias_init is not None:
            b_frozen = b_full_for_vals.clone()
        else:
            b_frozen = torch.zeros(self.out_features, dtype=W_full.dtype)
        if self.bias_idx is not None:
            b_frozen[self.bias_idx] = 0
        self.register_buffer("b_frozen", b_frozen)

        # Convenience: do we add any bias?
        self._use_bias = (self.bias_idx is not None) or (self.b_frozen.abs().sum().item() != 0.0)

        # ----- Sanity: initial effective params equal to intended seeds -----
        with torch.no_grad():
            # Weight: W_eff0 = W_frozen + scatter(vals)
            W_eff0 = self.W_frozen.clone()
            if self.vals.numel() > 0:
                W_eff0[self.row, self.col] = self.vals.detach()
            # Expected: if W_frozen_init given, use it except trainable slots come from weight_init;
            # otherwise, exactly W_full.
            if W_frozen_init is None:
                expect_W0 = W_full
            else:
                expect_W0 = torch.as_tensor(W_frozen_init, dtype=W_full.dtype).clone().detach()
                expect_W0[self.row, self.col] = W_full[self.row, self.col]
            if not torch.allclose(W_eff0, expect_W0):
                raise AssertionError("Internal init error: effective initial weight != expected seed")

            # Bias: b_eff0 = b_frozen (+ bias_vals at bias_idx)
            if self._use_bias:
                b_eff0 = self.b_frozen.clone()
                if self.bias_vals is not None and self.bias_vals.numel() > 0:
                    b_eff0[self.bias_idx] = self.bias_vals.detach()
                # Expected bias seed: start from b_frozen_init if provided else zeros/ bias_init
                if b_frozen_init is None:
                    expect_b0 = b_frozen.clone()  # already zeroed at trainable slots
                else:
                    expect_b0 = torch.as_tensor(b_frozen_init, dtype=W_full.dtype).clone().detach()
                    if self.bias_idx is not None:
                        expect_b0[self.bias_idx] = 0
                if self.bias_idx is not None:
                    # trainable slots take bias_init values
                    expect_b0[self.bias_idx] = b_full_for_vals[self.bias_idx]
                if not torch.allclose(b_eff0, expect_b0):
                    raise AssertionError("Internal init error: effective initial bias != expected seed")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features must be in the last dim, works for [B, T, C] etc.
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x[..., {x.shape[-1]}] != in_features={self.in_features}")

        x_flat = x.reshape(-1, self.in_features)

        # Build effective weight = frozen + sparse delta as a dense tensor (correctness > perf)
        W_eff = self.W_frozen
        if self.vals.numel() > 0:
            # create a fresh delta and index-write trainable values
            delta = torch.zeros_like(self.W_frozen)
            delta[self.row, self.col] = self.vals
            W_eff = W_eff + delta

        # Bias (if any): start from frozen, then write trainable slots
        bias = None
        if self._use_bias:
            b = self.b_frozen
            if self.bias_vals is not None and self.bias_vals.numel() > 0:
                b = b.clone()
                b[self.bias_idx] = self.bias_vals
            bias = b

        # Cast weights/bias to match input dtype if needed (simple and safe)
        W_eff = W_eff.to(dtype=x_flat.dtype)
        if bias is not None:
            bias = bias.to(dtype=x_flat.dtype)

        y_flat = F.linear(x_flat, W_eff, bias=bias)
        return y_flat.view(*x.shape[:-1], self.out_features)

    @torch.no_grad()
    def hotswap(
        self,
        new_weight_indices: torch.Tensor,
        new_bias_indices: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        keep_state: int = 0,
    ) -> None:
        """
        Replace the set of trainable weight/bias slots while preserving optimizer state when possible.

        `new_weight_indices` must contain the same number of (row, col) pairs as the current module.
        If bias entries are trainable, `new_bias_indices` must be provided with the same length as the
        current bias index tensor.
        """
        device = self.W_frozen.device
        new_w_idx = torch.as_tensor(new_weight_indices, dtype=torch.long, device=device)
        if new_w_idx.ndim != 2 or new_w_idx.size(1) != 2:
            raise ValueError("new_weight_indices must be [nnz, 2] (row, col)")

        old_row = self.row.clone()
        old_col = self.col.clone()
        old_lin = old_row * self.in_features + old_col

        new_row, new_col = new_w_idx[:, 0], new_w_idx[:, 1]
        if new_row.numel() != old_row.numel():
            raise ValueError("hotswap currently requires the same number of trainable weights")
        if torch.any(new_row < 0) or torch.any(new_row >= self.out_features):
            raise ValueError("new weight row indices out of range")
        if torch.any(new_col < 0) or torch.any(new_col >= self.in_features):
            raise ValueError("new weight col indices out of range")
        new_lin = new_row * self.in_features + new_col
        if torch.unique(new_lin).numel() != new_lin.numel():
            raise ValueError("new_weight_indices contains duplicate (row, col) entries")

        had_trainable_bias = self.bias_vals is not None
        if new_bias_indices is None:
            if had_trainable_bias:
                raise ValueError("new_bias_indices must be provided because bias is trainable")
            new_bias_idx = None
        else:
            if not had_trainable_bias:
                raise ValueError("module has no trainable bias to hotswap")
            new_bias_idx = torch.as_tensor(new_bias_indices, dtype=torch.long, device=device)
            if new_bias_idx.ndim != 1:
                raise ValueError("new_bias_indices must be 1D [nnz_b]")
            if new_bias_idx.numel() != self.bias_idx.numel():
                raise ValueError("hotswap currently requires the same number of trainable bias entries")
            if torch.any(new_bias_idx < 0) or torch.any(new_bias_idx >= self.out_features):
                raise ValueError("new bias indices out of range")
            if torch.unique(new_bias_idx).numel() != new_bias_idx.numel():
                raise ValueError("new_bias_indices contains duplicate entries")

        weight_state_old_pos = weight_state_new_pos = None
        if optimizer is not None and self.vals.numel() > 0:
            old_map = {int(idx): pos for pos, idx in enumerate(old_lin.detach().cpu().tolist())}
            new_list = new_lin.detach().cpu().tolist()
            keep_new_pos = []
            keep_old_pos = []
            for new_pos, idx in enumerate(new_list):
                old_pos = old_map.get(int(idx), None)
                if old_pos is not None:
                    keep_new_pos.append(new_pos)
                    keep_old_pos.append(old_pos)
            weight_state_new_pos = torch.tensor(keep_new_pos, dtype=torch.long)
            weight_state_old_pos = torch.tensor(keep_old_pos, dtype=torch.long)

        bias_state_old_pos = bias_state_new_pos = None
        if had_trainable_bias and optimizer is not None and keep_state:
            old_bias = self.bias_idx.detach().cpu().tolist()
            old_bias_map = {int(idx): pos for pos, idx in enumerate(old_bias)}
            new_bias_list = new_bias_idx.detach().cpu().tolist()
            keep_new_bias = []
            keep_old_bias = []
            for new_pos, idx in enumerate(new_bias_list):
                old_pos = old_bias_map.get(int(idx), None)
                if old_pos is not None:
                    keep_new_bias.append(new_pos)
                    keep_old_bias.append(old_pos)
            bias_state_new_pos = torch.tensor(keep_new_bias, dtype=torch.long)
            bias_state_old_pos = torch.tensor(keep_old_bias, dtype=torch.long)

        full_weight = self.W_frozen.clone()
        if self.vals.numel() > 0:
            full_weight[old_row, old_col] = self.vals.detach()

        new_vals = full_weight[new_row, new_col].clone()
        new_W_frozen = full_weight
        new_W_frozen[new_row, new_col] = 0

        if self.vals.numel() > 0:
            self.vals.detach().copy_(new_vals)
        self.W_frozen.copy_(new_W_frozen)
        self.row.detach().copy_(new_row)
        self.col.detach().copy_(new_col)

        if had_trainable_bias:
            full_bias = self.b_frozen.clone()
            full_bias[self.bias_idx] = self.bias_vals.detach()
            new_bias_vals = full_bias[new_bias_idx].clone()
            new_b_frozen = full_bias
            new_b_frozen[new_bias_idx] = 0

            self.bias_vals.detach().copy_(new_bias_vals)
            self.bias_idx.detach().copy_(new_bias_idx)
            self.b_frozen.copy_(new_b_frozen)
        else:
            full_bias = self.b_frozen

        self._use_bias = (self.bias_vals is not None) or (self.b_frozen.abs().sum().item() != 0.0)

        if optimizer is not None:
            self._update_optimizer_state(
                optimizer,
                self.vals if self.vals.numel() > 0 else None,
                weight_state_old_pos,
                weight_state_new_pos,
                keep_state,
            )
            if had_trainable_bias:
                self._update_optimizer_state(
                    optimizer,
                    self.bias_vals,
                    bias_state_old_pos,
                    bias_state_new_pos,
                    keep_state,
                )

    @staticmethod
    def _update_optimizer_state(
        optimizer: torch.optim.Optimizer | None,
        param: Optional[nn.Parameter],
        old_pos: Optional[torch.Tensor],
        new_pos: Optional[torch.Tensor],
        keep_state: bool,
    ) -> None:
        if optimizer is None or param is None:
            return
        if not keep_state:
            optimizer.state.pop(param, None)
            return

        state = optimizer.state.get(param, None)
        if not isinstance(state, dict):
            return
        if old_pos is None or new_pos is None or old_pos.numel() == 0 or new_pos.numel() == 0:
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape == param.shape:
                    value.zero_()
            return

        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if value.shape != param.shape:
                continue
            old_idx = old_pos.to(value.device)
            new_idx = new_pos.to(value.device)
            src = value.index_select(0, old_idx).clone()
            value.zero_()
            value.index_copy_(0, new_idx, src)

    @staticmethod
    def from_linear(
        base: nn.Linear,
        train_weight_indices: torch.Tensor,
        train_bias_indices: Optional[torch.Tensor] = None,
        metadata: Dict = {},
    ) -> "LinearElementwise":
        # Preserve dtype/device via seeds, avoid .data
        w0 = base.weight.detach()
        if base.bias is not None:
            b0 = base.bias.detach()
        else:
            b0 = torch.zeros(base.out_features, device=w0.device, dtype=w0.dtype)

        return LinearElementwise(
            in_features=base.in_features,
            out_features=base.out_features,
            train_weight_indices=train_weight_indices,
            train_bias_indices=train_bias_indices,
            weight_init=w0,
            bias_init=b0,
            metadata=metadata,
        )
