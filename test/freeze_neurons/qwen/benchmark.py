from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
import torch
# from custom_adam import MaskedAdamW
from gradient_callback import *
# from probe import *
import hashlib
import time
import torch.distributed as dist
# from skip_gradient_callback import SkipGradientCallback
import logging
import torch.nn as nn
import os
import numpy as np
from iteration_time_callback import IterationTimeCallback

from torch.cuda import nvtx
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from contextlib import contextmanager
from collections import defaultdict

from module import *
from helper import *

logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

MODEL = "Qwen/Qwen2.5-0.5B"
# MODEL = "Qwen/Qwen2.5-14B"


DATASET = "sst2"
VALIDATION_SET = "validation"
NUM_LABELS = 2
EVAL_LOSS_STEPS=5

# DATASET = "mnli"
# VALIDATION_SET = "validation_matched"
# #VALIDATION_SET = "validation_mismatched"
# NUM_LABELS = 3
# EVAL_LOSS_STEPS=500

NUM_EPOCHS=3

RUN_NAME = "random-20p"
_RUN_TS = time.strftime("%Y%m%d-%H%M%S")
SCRATCH = os.getenv("SCRATCH", "/pscratch/sd/l/lsx")
ZERO_BOTTOM_K_PERCENT = 0.5   # Zero bottom 50% of gradients
ZERO_MODE = "neurons"         # Options: "weights" or "neurons"
FREEZE_AFTER_EPOCHS = 1       # Choose bottom-k once after this many epochs
VALIDATION_FRACTION = 0.1     # Hold out 10% for validation


MODE="random"
RANDOM_HOT_K_PERCENT = 0.2
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


# if main program
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




@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

def register_nvtx_hooks(model: torch.nn.Module):
    """
    Forward: push at pre_hook, pop at post_hook.
    Backward: push at full_backward_pre_hook, pop at full_backward_hook.
    """
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # only leaf-ish modules to reduce noise; remove this to tag everything

        fq = name  # fully-qualified path already in named_modules()

        def fpre(_, __, _fq=fq):
            torch.cuda.nvtx.range_push(_fq)

        def fpost(_, __, ___, _fq=fq):
            torch.cuda.nvtx.range_pop()

        def bpre(_, __, _fq=fq):
            torch.cuda.nvtx.range_push(_fq + " [backward]")

        def bpost(_, __, ___, _fq=fq):
            torch.cuda.nvtx.range_pop()

        module.register_forward_pre_hook(fpre)
        module.register_forward_hook(fpost)
        # Backward hooks (PyTorch 1.8+)
        # module.register_full_backward_pre_hook(bpre)
        # module.register_full_backward_hook(bpost)


class NVTXOptCallback(TrainerCallback):

    step = 0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        torch.cuda.nvtx.range_push("optimizer_step")

    def on_optimizer_step(self, args, state, control, **kwargs):
        torch.cuda.nvtx.range_pop()


    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.nvtx.range_push(f"training_step {self.step}")
        self.step += 1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.nvtx.range_pop()

class NVTXTrainer(Trainer):

    # Forward: only tag the model's forward/compute_loss region
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        # with torch.autograd.profiler.emit_nvtx():
            with nvtx.range("forward"):
                return super().compute_loss(model, inputs, return_outputs)

if __name__ == "__main__":
    # read arguments from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-time", action="store_true", help="Benchmark time using IterationTimeCallback")
    parser.add_argument("--skip-ratio", type=float, default=0.0, help="Skip Ratio for LinearColWise")
    args_cmd = parser.parse_args()
    skip_ratio = args_cmd.skip_ratio
    benchmark_time = args_cmd.benchmark_time

    print(f"skip_ratio = {skip_ratio}, benchmark_time = {benchmark_time}")



    # output_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}/{skip_ratio}"
    
    # print(f"Output dir: {output_dir}")
    # output_dir = f"/home/sl3343/coldhottrain/shouxu_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{skip_ratio}"
    output_dir = f"/mnt/coldhot/shouxu_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    weight_out_dir = f"{output_dir}/weight_dump"
    # Training arguments
    args = TrainingArguments(
        output_dir=f"{output_dir}/ckpt",
        logging_dir=f"{output_dir}/logs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        # gradient_accumulation_steps=1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        # fp16=True,
        bf16=True,
        logging_steps=100,
        save_strategy="epoch",
        # save_strategy="no",
        # eval_strategy="steps",
        # eval_steps=EVAL_LOSS_STEPS,
        weight_decay=0.01,
        #save_steps=100,
        # save_total_limit=2,
        ddp_find_unused_parameters=False,
        max_steps = 5,
        # logging_strategy="no",
        # disable_tqdm=True,
        report_to="none"
    )



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

    select_n_layers = 1
    model.model.layers = model.model.layers[:select_n_layers]
    model.config.num_hidden_layers =  select_n_layers
    # instrument_model_for_nvtx(model)
    # instrument_model_forward_backward_nvtx(model, only_leaf=True)
    register_nvtx_hooks(model)


    # print out if rank 0
    if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for name, module in model.named_modules():
            print(name, "->", module.__class__.__name__)


        cfg = model.config
        print("hidden_size:", cfg.hidden_size)                       # 896
        print("num_attention_heads:", cfg.num_attention_heads)       # 14
        print("num_key_value_heads:", getattr(cfg, "num_key_value_heads", None))  # 2
        print("head_dim:", cfg.hidden_size // cfg.num_attention_heads)            # 64
        print("kv_proj_out:", (getattr(cfg, "num_key_value_heads", 2) * (cfg.hidden_size // cfg.num_attention_heads)))  # 128
        print("num_layers: ", model.config.num_hidden_layers)


    # if skip_ratio > 0.0:
    #     print("SKIP is set to True, skipping replacement of linear layers with LinearColWise.")

    #     layers = get_decoder_layers(model)   # <-- the fix
    #     layer_idx = 23
    #     for i, layer in enumerate(layers):
                
    #         mapping = {
    #             "self_attn.q_proj": layer.self_attn.q_proj,
    #             "self_attn.k_proj": layer.self_attn.k_proj,
    #             "self_attn.v_proj": layer.self_attn.v_proj,
    #             "self_attn.o_proj": layer.self_attn.o_proj,
    #             "mlp.up_proj":      layer.mlp.up_proj,
    #             "mlp.down_proj":    layer.mlp.down_proj,
    #             # "mlp.gate_proj": layer.mlp.gate_proj,
    #         }
            


    #         for name, linear in mapping.items():
    #             assert isinstance(linear, nn.Linear), f"{name} expected nn.Linear, got {type(linear)}"
    #             # print(f"Processing {name}: {tuple(linear.weight.shape)}")
    #             out_features = linear.out_features
    #             # hot_idx = make_hot_idx(out_features, frac=policy_by_name[name], device=linear.weight.device)
    #             hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
    #             wrapped = replace_linear_with_colwise(linear, hot_idx)
    #             if name.startswith("self_attn."):
    #                 setattr(layer.self_attn, name.split(".", 1)[1], wrapped)
    #             else:
    #                 setattr(layer.mlp,       name.split(".", 1)[1], wrapped)

    #         # # quick sanity check
    #         # print(type(layer.self_attn.q_proj), layer.self_attn.q_proj.W_hot.shape, layer.self_attn.q_proj.W_cold.shape)
    #         # print(type(layer.mlp.up_proj),      layer.mlp.up_proj.W_hot.shape,      layer.mlp.up_proj.W_cold.shape)

    # else:
    #     print("SKIP is set to False, not replacing linear layers with LinearColWise.")





    ds = load_dataset("nyu-mll/glue", DATASET)



    with args.main_process_first(desc="tokenize"):
        if DATASET == "sst2":
            tokenized_ds = ds.map(tokenize_function_sst, batched=False)
        elif DATASET == "mnli":
            tokenized_ds = ds.map(tokenize_function_mnli, batched=False)
        else:
            raise ValueError(f"Unsupported dataset: {DATASET}")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate accuracy
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        
        return accuracy  # Returns {"accuracy": 0.923}


    opt_kwargs = {
        "mask_dict": {},
        "named_parameters": dict(model.named_parameters()),
        "freeze_state": "none",  # or "decay" or "full" per your preference
    }

    time_callback = IterationTimeCallback(iter_times=None)
    # trainer = Trainer(
    trainer = NVTXTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds[VALIDATION_SET],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[NVTXOptCallback],
        # optimizer_cls_and_kwargs=(MaskedAdamW, opt_kwargs)
        # 
        
    )




    # num_gpus = torch.cuda.device_count()
    num_gpus = 1
    num_samples = len(tokenized_ds["train"])
    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, num_gpus)
    iters_per_epoch = (num_samples + global_batch - 1) // global_batch
    print(f"#GPUs: {num_gpus}  Global batch: {global_batch}  Iters/epoch: {iters_per_epoch}")


    # dump_out_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace("/", "_")}-{DATASET.replace('/', '_')}-grad_dump"
    dump_out_dir = f"{output_dir}/grad_dump"







    # Start training
    with torch.autograd.profiler.emit_nvtx():
        trainer.train()



    safe_destroy()