import os
import torch
import torch.distributed as dist
import time
import sys

def diagnose():
    """Minimal script to diagnose where the hang occurs"""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    
    print(f"[{rank}] Step 1: Environment check", flush=True)
    print(f"[{rank}] LOCAL_RANK={local_rank}, RANK={rank}", flush=True)
    print(f"[{rank}] CUDA available: {torch.cuda.is_available()}", flush=True)
    
    if torch.cuda.is_available():
        print(f"[{rank}] Step 2: Setting CUDA device to {local_rank}", flush=True)
        torch.cuda.set_device(local_rank)
        
        # Test basic CUDA operation
        print(f"[{rank}] Step 3: Testing basic CUDA operation", flush=True)
        test_tensor = torch.tensor([1.0], device=f"cuda:{local_rank}")
        print(f"[{rank}] CUDA tensor created successfully: {test_tensor}", flush=True)
        torch.cuda.synchronize()
        print(f"[{rank}] CUDA synchronize successful", flush=True)
    
    print(f"[{rank}] Step 4: Initializing process group", flush=True)
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    try:
        dist.init_process_group(backend=backend, init_method="env://")
        print(f"[{rank}] Process group initialized successfully", flush=True)
    except Exception as e:
        print(f"[{rank}] ERROR: Failed to initialize process group: {e}", flush=True)
        sys.exit(1)
    
    # Test 1: Simple barrier
    print(f"[{rank}] Step 5: Testing barrier", flush=True)
    try:
        dist.barrier()
        print(f"[{rank}] Barrier successful", flush=True)
    except Exception as e:
        print(f"[{rank}] ERROR: Barrier failed: {e}", flush=True)
    
    # Test 2: Small tensor all_reduce
    print(f"[{rank}] Step 6: Testing all_reduce with small tensor", flush=True)
    
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([float(rank)], device=device)
    
    print(f"[{rank}] Tensor before all_reduce: {x.item()}", flush=True)
    
    try:
        # Add a small delay to ensure all processes are ready
        time.sleep(0.1 * rank)
        
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(f"[{rank}] all_reduce successful. Result: {x.item()}", flush=True)
        
    except Exception as e:
        print(f"[{rank}] ERROR: all_reduce failed: {e}", flush=True)
    
    print(f"[{rank}] Step 7: Final barrier", flush=True)
    dist.barrier()
    
    print(f"[{rank}] Step 8: Cleanup", flush=True)
    dist.destroy_process_group()
    
    print(f"[{rank}] DIAGNOSIS COMPLETE - All tests passed!", flush=True)

if __name__ == "__main__":
    diagnose()