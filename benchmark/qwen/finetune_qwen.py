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



# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    data_collator=collator,
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
trainer.add_callback(dump_cb)

# probe_cb = Probe()
# trainer.add_callback(probe_cb)



# Start training
trainer.train()


safe_destroy()
