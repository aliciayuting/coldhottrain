import torch
from torch import nn
from torch.optim import AdamW
from typing import Dict, Optional, Iterable, Union, List
import logging
ParamGroups = Union[Iterable[nn.Parameter], List[Dict]]  # what HF passes


logger = logging.getLogger(__name__)

class MaskedAdamW(AdamW):
    def __init__(
        self,
        params: ParamGroups,
        *,
        mask_dict: Optional[Dict[str, torch.Tensor]] = None,
        named_parameters: Optional[Dict[str, nn.Parameter]] = None,
        freeze_state: str = "zero",  # "none", "decay", "full", or "zero"
        **kwargs,
    ):
        """
        - params: can be HF-style param groups (list of dicts) or plain iterable of params.
        - mask_dict: maps PARAMETER NAMES -> 1D bool mask (len == param.shape[0]).
        - named_parameters: dict(model.named_parameters()) so we can map ids->names
                            when `params` are grouped (HF Trainer).
        - freeze_state (affects masked rows only):
            "none"  = keep weights frozen; let moments update (no grad zeroing; no state restore)
            "decay" = keep weights frozen; zero masked grads so moments just decay
            "full"  = keep weights frozen; restore moments to pre-step values
            "zero"  = set weights to 0 after step; zero the corresponding moments
        """
        super().__init__(params, **kwargs)

        # Build id(param) -> name using provided named_parameters (HF case)
        self._param_to_name: Dict[int, str] = {}
        if named_parameters is not None:
            for n, p in named_parameters.items():
                self._param_to_name[id(p)] = n

        # Validate + store masks (by name)
        self._mask_dict: Dict[str, torch.Tensor] = {}
        self._warned_missing: set[str] = set()
        self.freeze_state = freeze_state
        if mask_dict:
            self.set_mask_dict(mask_dict, strict=(named_parameters is not None))

    @torch.no_grad()
    def set_mask_dict(self, mask_dict: Dict[str, torch.Tensor], strict: bool = False):
        logging.debug(f"Setting mask_dict with {len(mask_dict)} entries.")
        """Set/update masks. Tensors must be 1D bool; len == param.shape[0]."""
        # quick look-up from our current params
        name_to_param: Dict[str, nn.Parameter] = {}
        for group in self.param_groups:
            for p in group["params"]:
                n = self._param_to_name.get(id(p))
                if n is not None:
                    name_to_param[n] = p

        normed = {}
        for name, mask in mask_dict.items():
            p = name_to_param.get(name)
            if p is None:
                if strict:
                    raise KeyError(f"Mask provided for '{name}', but no such parameter is in optimizer param groups.")
                normed[name] = mask
                continue
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if p.ndim == 0:
                raise ValueError(f"Parameter '{name}' is scalar; row-wise masking is undefined.")
            if mask.ndim != 1 or mask.numel() != p.shape[0]:
                raise ValueError(
                    f"Mask for '{name}' must be 1D bool of length {p.shape[0]}, got shape {tuple(mask.shape)}."
                )
            normed[name] = mask.to(device=p.device, non_blocking=True)
        self._mask_dict = normed

    @torch.no_grad()
    def step(self, closure=None):
        # Prepare caches for masked rows (params, and optionally state)
        pre_rows: Dict[int, torch.Tensor] = {}
        pre_state_avg: Dict[int, torch.Tensor] = {}
        pre_state_var: Dict[int, torch.Tensor] = {}
        pre_state_amsmax: Dict[int, torch.Tensor] = {}

        # Optionally zero grads on masked rows so moments don't pick them up
        zero_grads = (self.freeze_state in ("decay", "full", "zero"))

        for group in self.param_groups:
            for p in group["params"]:
                name = self._param_to_name.get(id(p))
                if name is None:
                    continue
                mask = self._mask_dict.get(name)
                if mask is None or not mask.any():
                    continue
                m = mask if mask.device == p.device else mask.to(p.device)

                # Cache param rows (to undo AdamW update incl. weight decay).
                # For "zero" we don't need this cache, but keeping it is harmless.
                if self.freeze_state != "zero":
                    pre_rows[id(p)] = p.data[m].clone()

                # Optionally freeze state completely: cache moments to restore later
                if self.freeze_state == "full":
                    st = self.state[p]
                    exp_avg = st.get("exp_avg", None)
                    exp_avg_sq = st.get("exp_avg_sq", None)
                    ams_max = st.get("max_exp_avg_sq", None)  # AMSGrad
                    if exp_avg is not None:    pre_state_avg[id(p)] = exp_avg[m].clone()
                    if exp_avg_sq is not None: pre_state_var[id(p)] = exp_avg_sq[m].clone()
                    if ams_max is not None:    pre_state_amsmax[id(p)] = ams_max[m].clone()

                # Optionally zero masked grads so moments don't get fresh signal
                if zero_grads and p.grad is not None:
                    g = p.grad
                    if g.ndim == 0:
                        raise ValueError(f"Scalar param '{name}' cannot be row-masked.")
                    if g.is_sparse:
                        raise ValueError(f"Sparse grads not supported for masked row-wise AdamW on '{name}'.")
                    g[m] = 0  # in-place

        loss = super().step(closure=closure)

        # Restore/modify params (and possibly state) for masked rows
        for group in self.param_groups:
            for p in group["params"]:
                name = self._param_to_name.get(id(p))
                if name is None:
                    continue
                mask = self._mask_dict.get(name)
                if mask is None or not mask.any():
                    continue
                m = mask if mask.device == p.device else mask.to(p.device)

                # Default behavior across modes (except "zero"): keep weights unchanged
                cached = pre_rows.get(id(p))
                if cached is not None:
                    p.data[m] = cached

                # Full restore of optimizer moments
                if self.freeze_state == "full":
                    st = self.state[p]
                    exp_avg = st.get("exp_avg", None)
                    exp_avg_sq = st.get("exp_avg_sq", None)
                    ams_max = st.get("max_exp_avg_sq", None)
                    if exp_avg is not None and id(p) in pre_state_avg:
                        exp_avg[m] = pre_state_avg[id(p)]
                    if exp_avg_sq is not None and id(p) in pre_state_var:
                        exp_avg_sq[m] = pre_state_var[id(p)]
                    if ams_max is not None and id(p) in pre_state_amsmax:
                        ams_max[m] = pre_state_amsmax[id(p)]

                # ZERO mode: set weights and moments to 0 for masked rows
                if self.freeze_state == "zero":
                    st = self.state[p]
                    exp_avg = st.get("exp_avg", None)
                    exp_avg_sq = st.get("exp_avg_sq", None)
                    ams_max = st.get("max_exp_avg_sq", None)  # AMSGrad
                    if exp_avg is not None:
                        exp_avg[m].zero_()
                    if exp_avg_sq is not None:
                        exp_avg_sq[m].zero_()
                    if ams_max is not None:
                        ams_max[m].zero_()
                    p.data[m].zero_()

        return loss
