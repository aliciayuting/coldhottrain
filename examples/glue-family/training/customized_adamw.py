# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor

from torch.optim.adam import Adam


from torch.optim.optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    DeviceDtypeDict,
    Optimizer,
    ParamsT,
)

class PlainAdamW(Adam):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )

        self.all_optimizer_states = None
        self.all_optimizer_states_name_mapping = None
        self.colwise = False
    

    def set_states(self, all_optimizer_states, all_optimizer_states_name_mapping):
        self.all_optimizer_states = all_optimizer_states
        self.all_optimizer_states_name_mapping = all_optimizer_states_name_mapping
        self.colwise = True

    # Preserve decoupled_weight_decay from AdamW for backwards compatibility. The following
    # guarantees that decoupled_weight_decay will always be True for loading any state into
    # AdamW
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group["decoupled_weight_decay"] = True

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        colwise: bool = False,
        all_optimizer_states = None,
        all_optimizer_states_name_mapping = None,
    ):
        # print(f"INIT GROUP GOT CALLED colwise={colwise}")
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    if not colwise:
                        state["step"] = (
                            torch.zeros(
                                (),
                                dtype=_get_scalar_dtype(is_fused=group["fused"]),
                                device=p.device,
                            )
                            if group["capturable"] or group["fused"]
                            else torch.tensor(0.0, dtype=_get_scalar_dtype())
                        )
                    else:
                        state["step"] = torch.zeros((p.shape[0],), dtype=_get_scalar_dtype())
                        # print(f"name: {all_optimizer_states_name_mapping[id(p)]} step shape: {state['step'].shape}")


                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                colwise=self.colwise,
                all_optimizer_states=self.all_optimizer_states,
                all_optimizer_states_name_mapping=self.all_optimizer_states_name_mapping,
            )

            _single_tensor_adam(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=group["amsgrad"],
                    has_complex=has_complex,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                    decoupled_weight_decay=True,
                    colwise=self.colwise,
                    all_optimizer_states=self.all_optimizer_states,
                    all_optimizer_states_name_mapping=self.all_optimizer_states_name_mapping,
                )
            
        return loss



def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
    colwise: bool = False,
    all_optimizer_states = None,
    all_optimizer_states_name_mapping = None,

):
    assert grad_scale is None and found_inf is None

    if colwise:
        assert all_optimizer_states is not None and all_optimizer_states_name_mapping is not None, "all_optimizer_states and all_optimizer_states_name_mapping must be provided for colwise AdamW"

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
    else:
        lr = _to_scalar(lr)
        # TODO: Support nonzero-dim Tensor betas, see #147921
    

    # We only shuffle around the beta when it is a Tensor, otherwise, we prefer
    # treating it as a scalar.
    # Note: ensure type declaration is under conditional check for isinstance
    # or else torchscript will get cranky about the DeviceDict type.
    if isinstance(beta1, Tensor):
        assert False, "Tensor beta1 not supported in customized AdamW"
        beta1_dict: Optional[DeviceDtypeDict] = {(beta1.device, beta1.dtype): beta1}
    else:
        beta1_dict = None

    

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), (
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
            )

        # update step
        if not colwise:
            step_t += 1
        else:
            assert step_t.dim() == 1 and step_t.shape[0] == param.shape[0], f"{all_optimizer_states_name_mapping[id(param)]} step_t must be 1-dimensional for colwise, but get shape: {step_t.shape} dim: {step_t.dim()}, param.shape: {param.shape}"
            step_t.add_(1) # per-element step for colwise

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform stepweight decay
                param.mul_(1 - lr * weight_decay)
            else:
                # Nested if is necessary to bypass jitscript rules
                if differentiable and isinstance(weight_decay, Tensor):
                    if weight_decay.requires_grad:
                        grad = grad.addcmul_(param.clone(), weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        device = param.device

        if beta1_dict is not None:
            dtype = param.dtype  # type: ignore[union-attr]

            # cast to workaround https://github.com/pytorch/pytorch/issues/140601
            key = (device, dtype)
            if key not in beta1_dict:
                beta1_dict[key] = beta1.to(  # type: ignore[union-attr]
                    device=device, dtype=dtype, non_blocking=True
                )

            device_beta1: Union[float, Tensor] = beta1_dict[key]
        else:
            device_beta1 = beta1

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - device_beta1)

        # Nested if is necessary to bypass jitscript rules
        if differentiable and isinstance(beta2, Tensor):
            if beta2.requires_grad:
                # Using lerp to only use 2 operations bc addcmul's value cannot be a tensor
                # Showing equivalence of differentiable path and nondifferentiable path
                # expavg * b2 + grad^2 * (1-b2)
                #           add expavg * (1-b2) - expavg * (1-b2) = 0
                # expavg * b2 + expavg * (1-b2) - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg + (grad^2 - expavg) * (1-b2)
                # expavg.lerp(grad^2, 1-beta2)
                exp_avg_sq.lerp_(torch.square(grad), weight=1 - beta2)
            else:
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=cast(float, 1 - beta2)
                )
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # type: ignore[arg-type]

        if capturable or differentiable:
            assert False, "capturable or differentiable not supported in customized AdamW"
            step = step_t

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta1, Tensor):
                if beta1.requires_grad:
                    bias_correction1 = 1 - beta1 ** step.clone()
                else:
                    bias_correction1 = 1 - beta1**step
            else:
                bias_correction1 = 1 - beta1**step

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta2, Tensor):
                if beta2.requires_grad:
                    bias_correction2 = 1 - beta2 ** step.clone()
                else:
                    bias_correction2 = 1 - beta2**step
            else:
                bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            if differentiable:
                param.addcdiv_(exp_avg.clone(), denom)
            else:
                param.addcdiv_(exp_avg, denom)
        else:
            if not colwise:
                step = _get_value(step_t)
            else:
                step = step_t.to(param.device)  # ensure step is on param device
                # if all_optimizer_states_name_mapping[id(param)] == "roberta.encoder.layer.0.attention.self.query.W_hot":
                #     print(f"param_name: {all_optimizer_states_name_mapping[id(param)]} step: {step}")

            if not colwise:

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2**0.5

                if amsgrad:
                    assert False, "amsgrad not supported in customized AdamW"
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)  # type: ignore[arg-type]
            else:
                rows = param.size(0)
                # step = step.to(device=param.device, dtype=param.dtype)  # [R]

                shape_row = (rows,) + (1,) * (param.ndim - 1)           # [R] for 1D, [R,1] for 2D, [R,1,1] for 3D, ...
                bc1 = (1.0 - torch.pow(beta1, step)).view(shape_row)
                bc2 = (1.0 - torch.pow(beta2, step)).view(shape_row)

                denom = exp_avg_sq.sqrt().div_(bc2.sqrt()).add_(eps)    # same shape as param
                step_size = lr / bc1                                    # [R,1,...]
                delta = (exp_avg / denom) * step_size                   # broadcasted
                param.add_(-delta)

            # if all_optimizer_states is not None and all_optimizer_states_name_mapping is not None:
            #     param_name = all_optimizer_states_name_mapping[id(param)]
            #     if param_name == "roberta.encoder.layer.0.attention.self.query.W_hot" or param_name == "roberta.encoder.layer.0.attention.self.query.b_hot":
            #         print(f"param_name: {param_name} step: {step}")
            #         print(f"denorm ({denom.shape}): {denom}")
            #         print(f"step_size: {step_size}")

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])