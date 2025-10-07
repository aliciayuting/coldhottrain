from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
import torch
# from custom_adam import MaskedAdamW
from gradient_callback import *
from probe import *
from probe2 import *
import hashlib
import time
import torch.distributed as dist
# from skip_gradient_callback import SkipGradientCallback
import logging
import torch.nn as nn
import os
import numpy as np
from iteration_time_callback import IterationTimeCallback


from module import *
from helper import *

logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

MODEL = "Qwen/Qwen2.5-0.5B"
#MODEL = "Qwen/Qwen2.5-14B"


DATASET = "sst2"
VALIDATION_SET = "validation"
NUM_LABELS = 2
EVAL_LOSS_STEPS=50

# DATASET = "mnli"
# VALIDATION_SET = "validation_matched"
# #VALIDATION_SET = "validation_mismatched"
# NUM_LABELS = 3
# EVAL_LOSS_STEPS=500

NUM_EPOCHS=1

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



    output_dir = f"/pscratch/sd/l/lsx/jamal-runs-sx/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}/{skip_ratio}"
    os.makedirs(output_dir, exist_ok=True)
    # print(f"Output dir: {output_dir}")
    # output_dir = f"/home/sl3343/coldhottrain/shouxu_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}-{_RUN_TS}"
    # output_dir = f"/mnt/coldhot/shouxu_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}-{_RUN_TS}"


    weight_out_dir = f"{output_dir}/weight_dump"
    # Training arguments
    args = TrainingArguments(
        output_dir=f"{output_dir}/ckpt",
        logging_dir=f"{output_dir}/logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        # gradient_accumulation_steps=1,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        # fp16=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # save_strategy="no",
        eval_strategy="steps",
        eval_steps=EVAL_LOSS_STEPS,
        weight_decay=0.01,
        #save_steps=100,
        # save_total_limit=2,
        ddp_find_unused_parameters=False,
        #max_steps = 10,
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


    # print out if rank 0
    if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for name, param in model.named_parameters():
            print(name, param.shape, param.numel())


    if skip_ratio > 0.0:
        print("SKIP is set to True, skipping replacement of linear layers with LinearColWise.")

        embedding: nn.Embedding = model.model.embed_tokens
        hot_idx = make_hot_idx(embedding.num_embeddings, frac=1-skip_ratio, device=embedding.weight.device)
        model.model.embed_tokens = replace_embedding_with_colwise(embedding, hot_idx)

        layers = get_decoder_layers(model)   # <-- the fix
        layer_idx = 23
        for i, layer in enumerate(layers):
            mapping = {
                "self_attn.q_proj": layer.self_attn.q_proj,
                "self_attn.k_proj": layer.self_attn.k_proj,
                "self_attn.v_proj": layer.self_attn.v_proj,
                "self_attn.o_proj": layer.self_attn.o_proj,
                "mlp.up_proj":      layer.mlp.up_proj,
                "mlp.down_proj":    layer.mlp.down_proj,
                "mlp.gate_proj": layer.mlp.gate_proj,
            }
            


            for name, linear in mapping.items():
                assert isinstance(linear, nn.Linear), f"{name} expected nn.Linear, got {type(linear)}"
                # print(f"Processing {name}: {tuple(linear.weight.shape)}")
                out_features = linear.out_features
                # hot_idx = make_hot_idx(out_features, frac=policy_by_name[name], device=linear.weight.device)
                hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
                wrapped = replace_linear_with_colwise(linear, hot_idx)
                if name.startswith("self_attn."):
                    setattr(layer.self_attn, name.split(".", 1)[1], wrapped)
                else:
                    setattr(layer.mlp,       name.split(".", 1)[1], wrapped)

            # # quick sanity check
            # print(type(layer.self_attn.q_proj), layer.self_attn.q_proj.W_hot.shape, layer.self_attn.q_proj.W_cold.shape)
            # print(type(layer.mlp.up_proj),      layer.mlp.up_proj.W_hot.shape,      layer.mlp.up_proj.W_cold.shape)

    else:
        print("SKIP is set to False, not replacing linear layers with LinearColWise.")





    ds = load_dataset("nyu-mll/glue", DATASET)



    with args.main_process_first(desc="tokenize"):
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


    opt_kwargs = {
        "mask_dict": {},
        "named_parameters": dict(model.named_parameters()),
        "freeze_state": "none",  # or "decay" or "full" per your preference
    }

    time_callback = IterationTimeCallback(iter_times=None)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds[VALIDATION_SET],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # optimizer_cls_and_kwargs=(MaskedAdamW, opt_kwargs)
        callbacks=[time_callback] if benchmark_time else [],
    )




    # num_gpus = torch.cuda.device_count()
    num_gpus = 1
    num_samples = len(tokenized_ds["train"])
    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, num_gpus)
    iters_per_epoch = (num_samples + global_batch - 1) // global_batch
    print(f"#GPUs: {num_gpus}  Global batch: {global_batch}  Iters/epoch: {iters_per_epoch}")


    # dump_out_dir = f"/pscratch/sd/l/lsx/runs/{MODEL.replace("/", "_")}-{DATASET.replace('/', '_')}-grad_dump"
    dump_out_dir = f"{output_dir}/grad_dump"




    # skipgradient_cb = SkipGradientCallback(
    #     model=model,
    #     zero_bottom_k_percent=ZERO_BOTTOM_K_PERCENT,
    #     zero_mode=ZERO_MODE,
    #     epoch_start_track=FREEZE_AFTER_EPOCHS-1,   # start tracking gradient norms after this many epochs
    #     epoch_compute_masks=FREEZE_AFTER_EPOCHS,  # compute & fix masks at this epoch
    #     use_cold_every_iters=20,
    #     output_dir=output_dir,
    #     mode=MODE,
    #     random_hot_k_percent=RANDOM_HOT_K_PERCENT,
    #     change_random_every_iters=CHANGE_RANDOM_EVERY_ITERS,
    # )
    # trainer.add_callback(skipgradient_cb)

    # dump_cb = PerModuleGradDumper(
    #     out_dir=dump_out_dir,
    #     model=model,
    #     capture_steps=100,
    #     include_bias=True,
    #     also_embeddings=True,  # set True if you also want embeddings/lm_head
    #     # weight_out_dir=weight_out_dir,
    # )

    # trainer.add_callback(dump_cb)

    probe_cb = Probe()
    ram_cb = VramBreakdownCallback()

    #trainer.add_callback(probe_cb)
    trainer.add_callback(ram_cb)
    

    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    def log_memory_stats():
        """Log current GPU memory statistics"""
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Max Allocated: {max_allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    log_memory_stats()

    # Start training
    trainer.train()
    
    log_memory_stats()

    # check if this process is rank 0 before accessing time_callback.iter_times
    if benchmark_time and (not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):

        iter_times = time_callback.iter_times
        warmup = 5
        iter_times = iter_times[warmup:]  # skip first few warmup steps
        avg_time = sum(iter_times)/len(iter_times) if len(iter_times) > 0 else float('nan')

        print("Iteration times (s):", iter_times)
        print("Average iteration time (s):", avg_time)

        # wrote to a csv file
        with open(f"./output/iteration_times/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}.csv", "a") as f:
            f.write(f"{skip_ratio},{avg_time}\n")


    # # Save final model (only on rank 0)
    # if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(f"Saving final model to {output_dir}/final_only_model")
    #     # trainer.save_model(f"{output_dir}/final_only_model")
    #     model_output_path = f"{output_dir}/final_only_model"
    #     # make dir if not exist
    #     os.makedirs(model_output_path, exist_ok=True)
    #     model.save_pretrained(model_output_path)

    safe_destroy()