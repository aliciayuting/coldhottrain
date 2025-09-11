from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from gradient_callback import *
from probe import *

MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "tatsu-lab/alpaca"


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
output_dir = f"/pscratch/sd/l/lsx/yyt_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}"

weight_out_dir = f"{output_dir}/weight_dump"
# Training arguments
args = TrainingArguments(
    output_dir=f"{output_dir}/ckpt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    # gradient_accumulation_steps=1,
    num_train_epochs=40,
    learning_rate=2e-5,
    # fp16=True,
    bf16=True,
    logging_steps=100,
    save_steps=100,
    # save_total_limit=2,
    ddp_find_unused_parameters=False,
    # max_steps = 16,
)



class CustomTrainer(Trainer):
    def __init__(self, zero_bottom_k_percent=0.1, zero_mode="weights", **kwargs):
        """
        zero_bottom_k_percent: float, percentage of bottom gradients to zero (0.1 = 10%)
        zero_mode: str, either "weights" (individual weights) or "neurons" (full neurons)
        """
        super().__init__(**kwargs)
        self.zero_bottom_k_percent = zero_bottom_k_percent
        self.zero_mode = zero_mode
        
    def training_step(self, model, inputs):
        """Override training_step to intercept gradients"""
        loss = super().training_step(model, inputs)
        
        # Intercept gradients after backward pass but before optimizer step
        if self.zero_bottom_k_percent > 0:
            self._zero_bottom_k_gradients()
            
        return loss
    
    def _zero_bottom_k_gradients(self):
        """Zero out bottom K% of gradients by magnitude"""
        if self.zero_mode == "weights":
            self._zero_bottom_k_weights()
        elif self.zero_mode == "neurons":
            self._zero_bottom_k_neurons()
    
    def _zero_bottom_k_weights(self):
        """Zero individual weights with bottom K% gradient magnitudes"""
        # Collect all gradient magnitudes
        all_grad_mags = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().flatten()
                all_grad_mags.append(grad_mag)
        
        if not all_grad_mags:
            return
            
        # Concatenate all gradients and find bottom K% threshold
        all_grads = torch.cat(all_grad_mags)
        k = int(len(all_grads) * self.zero_bottom_k_percent)
        if k == 0:
            return
            
        # Use topk with largest=False to get bottom K values
        threshold = torch.topk(all_grads, k, largest=False).values[-1]
        
        # Zero gradients below or equal to threshold
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                mask = param.grad.abs() <= threshold
                param.grad[mask] = 0.0
    
    def _zero_bottom_k_neurons(self):
        """Zero full neurons with bottom K% L2 norm squared gradient magnitudes"""
        neuron_scores = []
        neuron_info = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and len(param.grad.shape) >= 2:
                # For weight matrices, treat each output neuron (first dimension)
                for neuron_idx in range(param.grad.shape[0]):
                    neuron_grad = param.grad[neuron_idx]
                    l2_norm_squared = (neuron_grad ** 2).sum().item()
                    neuron_scores.append(l2_norm_squared)
                    neuron_info.append((name, param, neuron_idx))
        
        if not neuron_scores:
            return
            
        # Find bottom K% neurons by average gradient magnitude
        k = int(len(neuron_scores) * self.zero_bottom_k_percent)
        if k == 0:
            return
            
        bottom_k_indices = torch.topk(torch.tensor(neuron_scores), k, largest=False).indices
        
        # Zero gradients for bottom K% neurons
        for idx in bottom_k_indices:
            name, param, neuron_idx = neuron_info[idx]
            param.grad[neuron_idx] = 0.0

# Configuration - adjust these values
ZERO_BOTTOM_K_PERCENT = 0.1  # Zero bottom 10% of gradients
ZERO_MODE = "weights"        # Options: "weights" or "neurons"

# Trainer
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    data_collator=collator,
    zero_bottom_k_percent=ZERO_BOTTOM_K_PERCENT,
    zero_mode=ZERO_MODE,
)

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
    also_embeddings=False,  # set True if you also want embeddings/lm_head
    weight_out_dir=weight_out_dir,
)
#trainer.add_callback(dump_cb)

# probe_cb = Probe()
# trainer.add_callback(probe_cb)



# Start training
trainer.train()


safe_destroy()
