from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from gradient_callback import *
from probe import *
import hashlib

MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "tatsu-lab/alpaca"
RUN_NAME = "neurons"
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
ZERO_BOTTOM_K_PERCENT = 0.1   # Zero bottom 10% of gradients
ZERO_MODE = "neurons"         # Options: "weights" or "neurons"
FREEZE_AFTER_EPOCHS = 1       # Choose bottom-k once after this many epochs


def safe_destroy():
    if dist.is_available() and dist.is_initialized():
        try:
            # Optional but helpful to flush in-flight NCCL ops
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# Load tokenizer & model
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# If tokenizer has no pad token (common for causal LMs), set it:
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    # device_map="auto")
)


# Load Alpaca-52K dataset
ds = load_dataset(DATASET)

# Preprocess into prompt–response format
def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    response = example["output"]

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return tok(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_ds = ds.map(format_example, batched=False)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# output_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}"
output_dir = f"{SCRATCH}/jamal_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}"

weight_out_dir = f"{output_dir}/weight_dump"
# Training arguments
args = TrainingArguments(
    output_dir=f"{output_dir}/ckpt",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    # gradient_accumulation_steps=1,
    num_train_epochs=30,
    learning_rate=2e-5,
    # fp16=True,
    bf16=True,
    logging_steps=100,
    save_strategy="epoch",
    #save_steps=100,
    # save_total_limit=2,
    ddp_find_unused_parameters=False,
    # max_steps = 16,
)


#Q: would it be more efficient to calc the gradient mask on rank 0 and broadcast it to all ranks?
class CustomTrainer(Trainer):
    def __init__(self, zero_bottom_k_percent=0.1, zero_mode="weights", freeze_after_epochs=3, **kwargs):
        """
        zero_bottom_k_percent: float, percentage of bottom gradients to zero (0.1 = 10%)
        zero_mode: str, either "weights" (individual weights) or "neurons" (full neurons)
        freeze_after_epochs: int, epoch count after which bottom-k is selected once
        and then fixed for the remainder of training.
        """
        super().__init__(**kwargs)
        self.zero_bottom_k_percent = float(zero_bottom_k_percent)
        self.zero_mode = str(zero_mode)
        self.freeze_after_epochs = int(freeze_after_epochs)

        # Fixed masks computed once after freeze_after_epochs
        self._fixed_weight_masks = None   # dict[name -> bool Tensor same shape as param]
        self._fixed_neuron_masks = None   # dict[name -> bool Tensor of shape [out_features]]
        self._fixed_masks_ready = False

    def training_step(self, model, inputs, num_items_in_batch):
        """Override training_step to intercept gradients"""
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.zero_bottom_k_percent <= 0:
            return loss

        # Determine epoch progress as float; None early in training
        cur_epoch = float(self.state.epoch) if self.state.epoch is not None else 0.0

        # Only start zeroing after we have selected a fixed mask post N epochs
        if cur_epoch >= self.freeze_after_epochs:
            if not self._fixed_masks_ready:
                #make sure we are not in the middle of a gradient accumulation step
                if getattr(self.model, "require_backward_grad_sync", True):
                    # Compute fixed masks using the CURRENT gradients once
                    self._compute_fixed_masks_from_current_grads()
                    if self.zero_mode == "neurons":
                        self._ddp_assert_neuron_masks_identical()
                    self._fixed_masks_ready = True
                # Fall-through to also apply them on this same step
            self._apply_fixed_masks()

        # Before freeze_after_epochs, we do not zero gradients (no dynamic selection)
        return loss

    def _compute_fixed_masks_from_current_grads(self):
        """Compute and cache fixed masks from gradients of the current step."""
        if self.zero_mode == "weights":
            self._fixed_weight_masks = self._build_weight_masks_from_grads()
            total_elems = sum(m.numel() for m in self._fixed_weight_masks.values()) if self._fixed_weight_masks else 0
            zeroed = sum(int(m.sum().item()) for m in self._fixed_weight_masks.values()) if self._fixed_weight_masks else 0
            print(f"[CustomTrainer] Computed fixed weight masks: zeroing {zeroed}/{total_elems} elements (~{(zeroed/max(1,total_elems))*100:.2f}%).")
        elif self.zero_mode == "neurons":
            self._fixed_neuron_masks = self._build_neuron_masks_from_grads()
            total_rows = sum(m.numel() for m in self._fixed_neuron_masks.values()) if self._fixed_neuron_masks else 0
            zeroed = sum(int(m.sum().item()) for m in self._fixed_neuron_masks.values()) if self._fixed_neuron_masks else 0
            print(f"[CustomTrainer] Computed fixed neuron masks: zeroing {zeroed}/{total_rows} rows (~{(zeroed/max(1,total_rows))*100:.2f}%).")
        else:
            raise ValueError(f"Unknown zero_mode: {self.zero_mode}")

    #Q: do we want to mask out bias grads too?
    @torch.no_grad()
    def _apply_fixed_masks(self):
        """Apply previously computed fixed masks to current gradients."""
        if self.zero_mode == "weights":
            if not self._fixed_weight_masks:
                return
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                mask = self._fixed_weight_masks.get(name)
                if mask is not None:
                    p.grad[mask] = 0.0
        elif self.zero_mode == "neurons":
            if not self._fixed_neuron_masks:
                return
            for name, p in self.model.named_parameters():
                if p.grad is None or p.grad.ndim < 2:
                    continue
                row_mask = self._fixed_neuron_masks.get(name)
                if row_mask is not None:
                    # Zero out entire rows where row_mask is True
                    p.grad[row_mask] = 0.0
        else:
            raise ValueError(f"Unknown zero_mode: {self.zero_mode}")

    @torch.no_grad()
    def _build_weight_masks_from_grads(self):
        """Build boolean masks per parameter for bottom-K% gradient magnitudes."""
        # Collect all gradient magnitudes flattened
        mags = []
        per_param_shapes = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().abs()
            mags.append(g.flatten())
            per_param_shapes[name] = g.shape
        if not mags:
            return {}
        all_mags = torch.cat(mags)
        k = int(all_mags.numel() * self.zero_bottom_k_percent)
        if k <= 0:
            return {}
        # Threshold for bottom-K values
        thresh = torch.topk(all_mags, k, largest=False).values[-1]

        # Build mask per param (True where we want to zero)
        masks = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            mask = (p.grad.abs() <= thresh)
            # Ensure mask is same dtype/device as grad (bool on same device)
            masks[name] = mask.to(dtype=torch.bool, device=p.grad.device)
        return masks

    @torch.no_grad()
    def _build_neuron_masks_from_grads(self):
        """Build boolean row masks per parameter for bottom-K% neuron gradient L2 norms."""
        # Compute per-row L2 norms over all 2D+ params
        scores = []
        param_refs = []  # (name, rows)
        for name, p in self.model.named_parameters():
            if p.grad is None or p.grad.ndim < 2:
                continue
            if p.grad.ndim > 2:
                print(f"[CustomTrainer] param {name} has ndim={p.grad.ndim}")
            g = p.grad.detach()
            rows = g.shape[0]
            flat = g.view(rows, -1)
            l2 = (flat.pow(2).sum(dim=1)).float()  # [rows]
            scores.append(l2)
            param_refs.append((name, rows))

        if not scores:
            return {}

        all_scores = torch.cat(scores)
        total_rows = all_scores.numel()
        k = int(total_rows * self.zero_bottom_k_percent)
        if k <= 0:
            return {}

        # Bottom-K indices across all rows of all relevant params
        bottom_idx = torch.topk(all_scores, k, largest=False).indices

        # Build boolean row masks per param
        masks = {}
        # Map global index back to (param, local_row)
        offset = 0
        for (name, rows), l2 in zip(param_refs, scores):
            #find out which rows of this param are in bottom k
            this_slice = bottom_idx[(bottom_idx >= offset) & (bottom_idx < offset + rows)] - offset
            if this_slice.numel() > 0:
                row_mask = torch.zeros(rows, dtype=torch.bool, device=l2.device)
                row_mask[this_slice] = True
                masks[name] = row_mask
            offset += rows
        return masks
    @torch.no_grad()
    def _ddp_assert_neuron_masks_identical(self):
        """Check that self._fixed_neuron_masks are identical across ranks.
        Call only on a synchronized step (require_backward_grad_sync == True)."""
        if not (dist.is_available() and dist.is_initialized()):
            return  # single GPU or non-DDP

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Canonical param list: all 2D+ params in named order
        names = []
        shapes = []
        for name, p in self.model.named_parameters():
            if p.ndim >= 2:
                names.append(name)
                shapes.append(p.shape)

        # Build a vector of int64 checksums, one per param in 'names'
        checks = []
        for (name, shape) in zip(names, shapes):
            rows = shape[0]
            mask = self._fixed_neuron_masks.get(name, None)
            if mask is None:
                # Treat as all-zeros (no rows selected)
                mask_cpu = torch.zeros(rows, dtype=torch.bool).cpu()
            else:
                # Ensure correct length and on CPU
                assert mask.ndim == 1 and mask.numel() == rows, f"Row mask shape mismatch for {name}"
                mask_cpu = mask.detach().to("cpu", dtype=torch.bool)

            # Hash the raw bytes for exact equality (name+shape+mask contents)
            h = hashlib.sha256()
            h.update(name.encode("utf-8"))
            h.update(str(tuple(shape)).encode("utf-8"))
            h.update(mask_cpu.numpy().tobytes())
            # take first 8 bytes as unsigned 64-bit int
            h64 = int.from_bytes(h.digest()[:8], byteorder="big", signed=True)
            checks.append(h64)

        local_vec = torch.tensor(checks, dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")

        # Broadcast rank-0's vector as the reference
        ref_vec = local_vec.clone()
        dist.broadcast(ref_vec, src=0)

        # Compare to reference and report any mismatches
        mism = (local_vec != ref_vec)
        num_mism = int(mism.sum().item())

        # Aggregate across all ranks to see if any mismatches anywhere
        total_mism = torch.tensor([num_mism], dtype=torch.int32, device=local_vec.device)
        dist.all_reduce(total_mism, op=dist.ReduceOp.SUM)

        if total_mism.item() != 0:
            # Print which params differ on this rank (keep it concise)
            bad_idxs = mism.nonzero(as_tuple=False).flatten().tolist()
            bad_names = [names[i] for i in bad_idxs[:10]]  # truncate to first 10 for readability
            print(f"[MaskCheck][rank {rank}] {num_mism} param masks differ from rank 0. e.g., {bad_names}")
            # Optional: hard assert to fail fast
            # raise RuntimeError(f"DDP mask mismatch on rank {rank}")
        elif rank == 0:
            print("[MaskCheck] All ranks have identical neuron masks ✔")

# Configuration - adjust these values

# Trainer
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    data_collator=collator,
    zero_bottom_k_percent=ZERO_BOTTOM_K_PERCENT,
    zero_mode=ZERO_MODE,
    freeze_after_epochs=FREEZE_AFTER_EPOCHS,
)

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_ds["train"],
#     data_collator=collator,
# )

num_gpus = torch.cuda.device_count()
num_samples = len(tokenized_ds["train"])
global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, num_gpus)
iters_per_epoch = (num_samples + global_batch - 1) // global_batch
print(f"#GPUs: {num_gpus}  Global batch: {global_batch}  Iters/epoch: {iters_per_epoch}")


# dump_out_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace("/", "_")}-{DATASET.replace('/', '_')}-grad_dump"
dump_out_dir = f"{output_dir}/grad_dump"

dump_cb = PerModuleGradDumper(
    out_dir=dump_out_dir,
    model=model,
    capture_steps=100,
    include_bias=True,
    also_embeddings=True,  # set True if you also want embeddings/lm_head
    weight_out_dir=weight_out_dir,
)
trainer.add_callback(dump_cb)

# probe_cb = Probe()
# trainer.add_callback(probe_cb)



# Start training
trainer.train()


safe_destroy()
