import torch

def print_gpu_memoory_usage():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    total_mem = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    cached = torch.cuda.memory_reserved(0)
    
    print(f"Total: {total_mem/1024**2:.2f} MB")
    print(f"Allocated: {allocated/1024**2:.2f} MB")
    print(f"Cached: {cached/1024**2:.2f} MB")

    return allocated, cached

def calculate_linear_layer_size(in_features, out_features, bias=True):
    total_params = in_features * out_features
    if bias:
        total_params += out_features
    return total_params * 4  # assuming float32 (4 bytes per parameter)

def calculate_adam_optimizer_state_size(in_features, out_features, bias=True):
    npara = in_features * out_features
    if bias:
        npara += out_features
    return (2 * npara + 1) * 4  # m and v, assuming float32 (4 bytes each)

def calculate_mlp_size(in_features, out_features, bias=True):
    model_size = (in_features * out_features + (out_features if bias else 0)) * 4
    optimizer_size = (2 * (in_features * out_features + (out_features if bias else 0)) + 1) * 4

def calculate_frozen_mlp_size(in_features, out_features, frozen_ratio, bias=True):
    # total_params = in_features * out_features
    # frozen_params = int(total_params * frozen_ratio)
    # trainable_params = total_params - frozen_params
    # if bias:
    #     trainable_params += out_features  # assuming bias is trainable

    frozen_matrix_size = in_features * out_features # full dense matrix for frozen weights
    hot_param_size = in_features * out_features * (1-frozen_ratio)  # values for trainable weights
    hot_idx_size = in_features * out_features * (1-frozen_ratio) * 2  # row and col indices for trainable weights

    hot_params_size = trainable_params * 4  # values for trainable weights

    model_size = (frozen_params + trainable_params) * 4  # assuming float32 (4 bytes each)
    optimizer_size = (2 * trainable_params + 1) * 4  # m and v for trainable params, assuming float32 (4 bytes each)

    return model_size + optimizer_size