from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
import torch
from gradient_callback import *
from probe import *
import hashlib
import time
import torch.distributed as dist
from skip_gradient_callback import SkipGradientCallback
import logging

logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

MODEL = "Qwen/Qwen2.5-0.5B"


#DATASET = "sst2"
#VALIDATION_SET = "validation"
#NUM_LABELS = 2
#EVAL_LOSS_STEPS=100

DATASET = "mnli"
VALIDATION_SET = "validation_matched"
#VALIDATION_SET = "validation_mismatched"
NUM_LABELS = 3
EVAL_LOSS_STEPS=500
NUM_EPOCHS=3

RUN_NAME = "base"
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
    
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    num_labels=NUM_LABELS
    # device_map="auto")
)
model.config.pad_token_id = tok.pad_token_id

ds = load_dataset("nyu-mll/glue", DATASET)

# Preprocess into prompt–response format
def tokenize_function_sst(examples):
    return tok(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
#WILL NOT WORK WITH BATCHED!!!!!
def tokenize_function_mnli(example):
    return tok(
        f"Premise: {example['premise']}; Hypothesis: {example['hypothesis']}",
        padding="max_length",
        truncation=True,
        max_length=512
    )

if DATASET == "sst2":
    tokenized_ds = ds.map(tokenize_function_sst, batched=False)
elif DATASET == "mnli":
    tokenized_ds = ds.map(tokenize_function_mnli, batched=False)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")


print(tokenized_ds)
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tok)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return accuracy  # Returns {"accuracy": 0.923}

# output_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}"
output_dir = f"{SCRATCH}/jamal_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}-{_RUN_TS}"

weight_out_dir = f"{output_dir}/weight_dump"
# Training arguments
args = TrainingArguments(
    output_dir=f"{output_dir}/ckpt",
    logging_dir=f"{output_dir}/logs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    # gradient_accumulation_steps=1,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-5,
    # fp16=True,
    bf16=True,
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=EVAL_LOSS_STEPS,
    weight_decay=0.01,
    #save_steps=100,
    # save_total_limit=2,
    ddp_find_unused_parameters=False,
    # max_steps = 16,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds[VALIDATION_SET],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



num_gpus = torch.cuda.device_count()
num_samples = len(tokenized_ds["train"])
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
