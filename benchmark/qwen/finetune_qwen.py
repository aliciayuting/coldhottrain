from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from gradient_callback import *
from probe import *
import hashlib
import time
import torch.distributed as dist
from skip_gradient_callback import SkipGradientCallback
import logging

logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'DEBUG').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "tatsu-lab/alpaca"
RUN_NAME = "neurons-0p"
_RUN_TS = time.strftime("%Y%m%d-%H%M%S")
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
ZERO_BOTTOM_K_PERCENT = 0.5   # Zero bottom 50% of gradients
ZERO_MODE = "neurons"         # Options: "weights" or "neurons"
FREEZE_AFTER_EPOCHS = 1       # Choose bottom-k once after this many epochs
VALIDATION_FRACTION = 0.1     # Hold out 10% for validation


MODE="random"
RANDOM_HOT_K_PERCENT = 1.0
CHANGE_RANDOM_EVERY_ITERS = 100

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


split_seed = 42

tokenized_ds = tokenized_ds["train"].train_test_split(
    test_size=VALIDATION_FRACTION,
    seed=split_seed,
)
tokenized_ds["validation"] = tokenized_ds.pop("test")

train_dataset = tokenized_ds["train"]
eval_dataset = tokenized_ds["validation"]
print(eval_dataset)
print(train_dataset)
# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# output_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}"
output_dir = f"{SCRATCH}/jamal_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}-{_RUN_TS}"

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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)



num_gpus = torch.cuda.device_count()
num_samples = len(train_dataset)
global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, num_gpus)
iters_per_epoch = (num_samples + global_batch - 1) // global_batch
print(f"#GPUs: {num_gpus}  Global batch: {global_batch}  Iters/epoch: {iters_per_epoch}")


# dump_out_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace("/", "_")}-{DATASET.replace('/', '_')}-grad_dump"
dump_out_dir = f"{output_dir}/grad_dump"




skipgradient_cb = SkipGradientCallback(
    model=model,
    zero_bottom_k_percent=ZERO_BOTTOM_K_PERCENT,
    zero_mode=ZERO_MODE,
    epoch_start_track=FREEZE_AFTER_EPOCHS-1,   # start tracking gradient norms after this many epochs
    epoch_compute_masks=FREEZE_AFTER_EPOCHS,  # compute & fix masks at this epoch
    use_cold_every_iters=20,
    output_dir=output_dir,
    mode=MODE,
    random_hot_k_percent=RANDOM_HOT_K_PERCENT,
    change_random_every_iters=CHANGE_RANDOM_EVERY_ITERS,
)
trainer.add_callback(skipgradient_cb)

dump_cb = PerModuleGradDumper(
    out_dir=dump_out_dir,
    model=model,
    capture_steps=100,
    include_bias=True,
    also_embeddings=True,  # set True if you also want embeddings/lm_head
    weight_out_dir=weight_out_dir,
)

# trainer.add_callback(dump_cb)

# probe_cb = Probe()
# trainer.add_callback(probe_cb)



# Start training
trainer.train()



safe_destroy()
