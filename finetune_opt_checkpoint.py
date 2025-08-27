# ==== 0) Config ====
groups = 3                  # e.g., {GSM8K, ASDiv, SVAMP}
ema_alpha = 0.1             # EMA smoothing for |grad|
HIT_T = 1e-5                # "hot" threshold for hit counts
log_every = 50              # all-reduce + in-memory log update
snap_every = 200            # write a snapshot to disk
outdir = "runs/opt13b_hotcold"
os.makedirs(outdir, exist_ok=True)

# ==== 1) Discover layers/neurons (OPT-13B) & unfreeze what you train ====
import re, torch.nn as nn
fc1_modules = []   # (layer_idx, name, module)
num_layers = 0
neurons = None

for name, mod in model.named_modules():
    # OPT decoder layers: model.decoder.layers.{L}.*
    m = re.match(r"model\.decoder\.layers\.(\d+)\.fc1$", name)
    if isinstance(mod, nn.Linear) and m:
        lid = int(m.group(1))
        fc1_modules.append((lid, name, mod))
        num_layers = max(num_layers, lid+1)
        neurons = neurons or mod.out_features
        assert neurons == mod.out_features, "varying FFN size not supported here"

# ==== 2) Allocate per-group accumulators on device ====
device = next(model.parameters()).device
G = [torch.zeros((num_layers, neurons), dtype=torch.float32, device=device) for _ in range(groups)]
H = [torch.zeros((num_layers, neurons), dtype=torch.int32,   device=device) for _ in range(groups)]

# current group for this microbatch (set in the train loop)
_curr_group = torch.tensor(0, device=device)
def set_group(g: int): _curr_group.fill_(int(g))
def get_group() -> int: return int(_curr_group.item())

# ==== 3) Register gradient hooks on fc1 weights ====
def make_grad_hook(layer_idx: int):
    def _hook(grad: torch.Tensor):
        # grad: [out_features, in_features] for fc1.weight
        with torch.no_grad():
            g = get_group()
            row_l1 = grad.abs().mean(dim=1).to(torch.float32)       # [neurons]
            # EMA update
            G[g][layer_idx].mul_(1-ema_alpha).add_(ema_alpha*row_l1)
            # hit count
            H[g][layer_idx] += (row_l1 > HIT_T).to(torch.int32)
    return _hook

for lid, name, mod in fc1_modules:
    mod.weight.register_hook(make_grad_hook(lid))

# ==== 4) (Optional) Activation probe (Deja-Vu-ish) ====
# Captures ReLU outputs after fc1 before fc2.
act_ema_alpha = 0.1
A = [torch.zeros((num_layers, neurons), dtype=torch.float32, device=device) for _ in range(groups)]

def make_forward_hook(layer_idx: int):
    # For OPT-13B, the forward of FFN is: y = fc2( relu(fc1(x)) )
    def _fwd(module, inp, out):
        # This hook is on fc1; out = fc1(x)  shape [B,T,neurons] or [*,neurons]
        with torch.no_grad():
            g = get_group()
            # apply ReLU just like the FFN does
            y = torch.relu(out)
            # reduce over batch/time -> mean per neuron
            if y.dim() == 3:   # [B, T, H]
                row = y.abs().mean(dim=(0,1)).to(torch.float32)      # [H]
            else:               # [*, H]
                row = y.abs().mean(dim=0).to(torch.float32)          # [H]
            A[g][layer_idx].mul_(1-act_ema_alpha).add_(act_ema_alpha*row)
    return _fwd

for lid, name, mod in fc1_modules:
    mod.register_forward_hook(make_forward_hook(lid))

# ==== 5) Distributed aggregation helper (FSDP/ZeRO) ====
import torch.distributed as dist
def reduce_stats():
    if not dist.is_available() or not dist.is_initialized():
        return
    for g in range(groups):
        dist.all_reduce(G[g], op=dist.ReduceOp.SUM)
        dist.all_reduce(H[g], op=dist.ReduceOp.SUM)
        dist.all_reduce(A[g], op=dist.ReduceOp.SUM)

# ==== 6) Snapshot writer ====
import numpy as np
def save_snapshot(step:int):
    # move small copies to CPU for writing; use np.save for speed or torch.save
    snap = {
        "step": step,
        "layers": num_layers,
        "neurons": neurons,
        "groups": groups,
        "G": [t.detach().cpu().numpy() for t in G],
        "H": [t.detach().cpu().numpy() for t in H],
        "A": [t.detach().cpu().numpy() for t in A],   # optional
    }
    # compressed .npz keeps files small; one file per snapshot
    np.savez_compressed(f"{outdir}/hotcold_step{step:07d}.npz", **snap)

# ==== 7) Train loop (insert into yours) ====
global_step = 0
for epoch in range(num_epochs):
    for batch in train_loader:
        # set group for this microbatch (adapt if your batch has mixed groups)
        set_group(int(batch["group"][0]))

        # move tensors and run forward/backward as usual
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / grad_accum

        loss.backward()

        if (global_step+1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)

        # periodic reduce + snapshot
        if (global_step+1) % log_every == 0:
            reduce_stats()

        if (global_step+1) % snap_every == 0:
            # save only on rank 0
            if (not dist.is_available()) or dist.get_rank() == 0:
                save_snapshot(global_step+1)

        global_step += 1