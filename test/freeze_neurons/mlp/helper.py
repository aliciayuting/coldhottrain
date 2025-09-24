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
