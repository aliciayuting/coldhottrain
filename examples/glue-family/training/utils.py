import inspect
import torch

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def optimizer_info(model, optim):
    param_to_name = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_to_name[id(param)] = name

    print(f"Optimizer type: {type(optim)}")
    for i, group in enumerate(optim.param_groups):
        print(f"--- Param Group {i} ---")
        for key, value in group.items():
            if key != "params":
                print(f"\t{key}: {value}")
        for param in group['params']:
            state = optim.state[param]
            param_name = param_to_name[id(param)]
            print(f"\t--param name: {param_name} lens(state): {len(state)}--")
            for skey, sval in state.items():
                if torch.is_tensor(sval):
                    print(f"\t\tstate key: {skey} shape: {sval.shape}")
                else:
                    print(f"\t\t state key: {skey} value: {sval}")