from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
import torch
# from custom_adam import MaskedAdamW
from gradient_callback import *
from probe import *
from probe2 import *
import hashlib
import json
import time
import torch.distributed as dist
# from skip_gradient_callback import SkipGradientCallback
import logging
import torch.nn as nn
import os
import numpy as np
from iteration_time_callback import IterationTimeCallback
from hotswap import HotSwapCallback
from linear_elementwise import LinearElementwise
from module import *
from helper import *

logging.basicConfig(
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )

torch.manual_seed(43)

MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
#MODEL = "Qwen/Qwen2.5-1.5B"


DATASET = os.getenv("DATASET", "sst2")  # Options: "sst2", "mnli", or "gsm8k"

if DATASET == "sst2":
    VALIDATION_SET = "validation"
    NUM_LABELS = 2
    EVAL_LOSS_STEPS=10
    NUM_EPOCHS=1
elif DATASET == "mnli":
    VALIDATION_SET = "validation_matched"
    NUM_LABELS = 3
    EVAL_LOSS_STEPS=500
    NUM_EPOCHS=1
elif DATASET == "gsm8k":
    VALIDATION_SET = "test"
    NUM_LABELS = None
    EVAL_LOSS_STEPS=500
    NUM_EPOCHS=3
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))

IS_GSM8K = DATASET == "gsm8k"



RUN_NAME = "random-20p"
_RUN_TS = time.strftime("%Y%m%d-%H%M%S")
SCRATCH = os.getenv("SCRATCH", "/share/desa/nfs02/cold")
ZERO_BOTTOM_K_PERCENT = 0.5   # Zero bottom 50% of gradients
ZERO_MODE = "neurons"         # Options: "weights" or "neurons"
FREEZE_AFTER_EPOCHS = 1       # Choose bottom-k once after this many epochs
VALIDATION_FRACTION = 0.1     # Hold out 10% for validation


MODE="random"
RANDOM_HOT_K_PERCENT = 0.2
CHANGE_RANDOM_EVERY_ITERS = 100


ELEMENTWISE_LINEAR = True  # whether to use elementwise linear or not
ELEMENTWISE_SWAP_SCHEME = "preselect"  # options: "all", "neuron", "input", "preselect"

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
def tokenize_function_sst(examples):
    return tok(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


def tokenize_function_mnli(example):
    return tok(
        f"Premise: {example['premise']}; Hypothesis: {example['hypothesis']}",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


def build_gsm8k_prompt(question: str) -> str:
    """Format a GSM8K question into a generation prompt."""
    question = question.strip()
    return f"Question: {question}\nAnswer: "


HASH = "####"

def tokenize_function_gsm8k(batch):
    prompts = [build_gsm8k_prompt(q) for q in batch["question"]]  # chat or your unified template
    completions = [a.strip() for a in batch["answer"]]      # contains rationale + '#### number'

    p = tok(prompts, padding=False, truncation=False, add_special_tokens=False)
    c = tok(completions, padding=False, truncation=False, add_special_tokens=False)

    MAX_LEN = MAX_LENGTH
    input_ids, labels, attn = [], [], []
    hash_ids = tok(HASH, add_special_tokens=False)["input_ids"]

    for p_ids, c_ids in zip(p["input_ids"], c["input_ids"]):
        # find '####' in token space; if missing, fall back to last 6 tokens
        idx = -1
        for i in range(len(c_ids)-len(hash_ids)+1):
            if c_ids[i:i+len(hash_ids)] == hash_ids:
                idx = i
                break
        # keep only the ANSWER tail as supervised region (from '####' to end)
        if idx >= 0:
            supervised = c_ids[idx:] + [tok.eos_token_id]
        else:
            supervised = c_ids[-6:] + [tok.eos_token_id]  # conservative fallback

        ids = p_ids + supervised
        if len(ids) > MAX_LEN:
            # trim from start of supervised region, but keep '####' tail intact
            over = len(ids) - MAX_LEN
            supervised = supervised[min(over, max(0, len(supervised)-1)):]
            ids = p_ids + supervised

        lab = [-100]*len(p_ids) + supervised  # **mask rationale entirely**
        input_ids.append(ids)
        labels.append(lab[:len(ids)])
        attn.append([1]*len(ids))

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def tokenize_function_gsm8k_maskquestion(batch):
    prompts = [build_gsm8k_prompt(q) for q in batch["question"]]
    completions = [a.strip() for a in batch["answer"]]

    # Tokenize prompts and completions separately
    prompt_tokenized = tok(prompts, padding=False, truncation=False, add_special_tokens=True)
    completion_tokenized = tok(completions, padding=False, truncation=False, add_special_tokens=False)

    # Concatenate token IDs
    input_ids = []
    labels = []
    attention_mask = []
    
    for prompt_ids, completion_ids in zip(prompt_tokenized["input_ids"], 
                                           completion_tokenized["input_ids"]):
        
        completion_ids = completion_ids + [tok.eos_token_id]

        # Concatenate
        combined_ids = prompt_ids + completion_ids
        
        # Truncate if needed
        if len(combined_ids) > 1024:
            combined_ids = combined_ids[:1024]
        
        # Create labels: -100 for prompt, actual IDs for completion
        combined_labels = [-100] * len(prompt_ids) + completion_ids
        if len(combined_labels) > 1024:
            combined_labels = combined_labels[:1024]
        
        input_ids.append(combined_ids)
        labels.append(combined_labels)
        attention_mask.append([1] * len(combined_ids))
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")

if __name__ == "__main__":
    # read arguments from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-time", action="store_true", help="Benchmark time using IterationTimeCallback")
    parser.add_argument("--skip-ratio", type=float, default=0.0, help="Skip Ratio for LinearColWise")
    parser.add_argument("--mode", type=str, default="1linear_efficient", help="Mode for LinearColWise")
    parser.add_argument("--gradient-checkpointing", type=str2bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--per-device-train-batch-size", type=int, default=16, help="per device batch size")
    parser.add_argument("--logging-steps", type=int, default=50, help="logging steps")
    parser.add_argument("--eval-steps", type=int, default=EVAL_LOSS_STEPS, help="eval steps")
    parser.add_argument("--random-swap-iters", type=int, default=100, help="random_swap_iters for HotSwapCallback")
    parser.add_argument("--elementwise-linear", type=str2bool, default=True, help="Whether to use elementwise linear or not")
    parser.add_argument("--elementwise-swap-scheme", type=str, default="neuron", help="Elementwise swap scheme: options are 'all', 'neuron', 'input', 'preselect'")
    parser.add_argument("--preselect-file", type=str, default="", help="Preselection file for elementwise swap")
    parser.add_argument("--run-name", type=str, default="", help="Run name for logging and saving")
    parser.add_argument("--category-name", type=str, default="", help="Category name for logging and saving")
    parser.add_argument("--dump-grads", type=str2bool, default=False, help="Whether to dump gradients or not")
    args_cmd = parser.parse_args()
    skip_ratio = args_cmd.skip_ratio
    benchmark_time = args_cmd.benchmark_time
    mode = args_cmd.mode
    gradient_checkpointing = args_cmd.gradient_checkpointing
    gradient_accumulation_steps = args_cmd.gradient_accumulation_steps
    per_device_train_batch_size = args_cmd.per_device_train_batch_size
    logging_steps = args_cmd.logging_steps
    eval_steps = args_cmd.eval_steps
    random_swap_iters = args_cmd.random_swap_iters
    preselect_file = args_cmd.preselect_file
    ELEMENTWISE_LINEAR = args_cmd.elementwise_linear
    ELEMENTWISE_SWAP_SCHEME = args_cmd.elementwise_swap_scheme
    RUN_NAME = args_cmd.run_name if args_cmd.run_name else RUN_NAME
    category_name = args_cmd.category_name if args_cmd.category_name else "default"
    print(f"model= {MODEL}, skip_ratio = {skip_ratio}, benchmark_time = {benchmark_time}, mode = {mode}, gradient_checkpointing = {gradient_checkpointing}, gradient_accumulation_steps = {gradient_accumulation_steps}")

    preselect_lookup = {}
    if ELEMENTWISE_SWAP_SCHEME == "preselect":
        if not preselect_file:
            raise ValueError("ELEMENTWISE_SWAP_SCHEME='preselect' requires --preselect-file to be specified")
        try:
            with open(preselect_file, "r", encoding="utf-8") as fh:
                preselect_payload = json.load(fh)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Preselect file not found: {preselect_file}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from preselect file {preselect_file}: {exc}") from exc

        raw_layers = preselect_payload.get("layers", {})
        if not raw_layers:
            raise ValueError(f"No layer selections found in preselect file: {preselect_file}")

        aggregated = {}
        for raw_name, info in raw_layers.items():
            short_name = os.path.basename(raw_name)
            entry = aggregated.setdefault(
                short_name,
                {"weights": set(), "bias": set(), "shape": None},
            )
            shape = tuple(info.get("shape", []))
            if shape:
                if entry["shape"] is None:
                    entry["shape"] = shape
                elif entry["shape"] != shape:
                    raise ValueError(f"Conflicting shapes for {short_name}: {entry['shape']} vs {shape}")
            for pair in info.get("train_weight_indices", []):
                if len(pair) != 2:
                    raise ValueError(f"Invalid weight index {pair} for {short_name}")
                entry["weights"].add((int(pair[0]), int(pair[1])))
            for idx in info.get("train_bias_indices", []):
                entry["bias"].add(int(idx))

        for short_name, entry in aggregated.items():
            weights_sorted = sorted(entry["weights"])
            bias_sorted = sorted(entry["bias"])
            preselect_lookup[short_name] = {
                "shape": entry["shape"],
                "train_weight_indices": weights_sorted,
                "train_bias_indices": bias_sorted,
            }

        if not preselect_lookup:
            raise ValueError(f"Preselect file {preselect_file} produced no usable layer entries")

    output_dir = os.path.join(SCRATCH, f"jamal-runs-benckmarking/{category_name}/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}/{RUN_NAME}-{_RUN_TS}")
    os.makedirs(output_dir, exist_ok=True)
    # print(f"Output dir: {output_dir}")
    # output_dir = f"/home/sl3343/coldhottrain/shouxu_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}-{_RUN_TS}"
    # output_dir = f"/mnt/coldhot/shouxu_runs/{MODEL.replace('/', '_')}-{DATASET.replace('/', '_')}-{RUN_NAME}-{_RUN_TS}"


    weight_out_dir = f"{output_dir}/weight_dump"
    # Training arguments
    args = TrainingArguments(
        output_dir=f"{output_dir}/ckpt",
        logging_dir=f"{output_dir}/logs",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #gradient_accumulation_steps=1,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=2e-5,
        # fp16=True,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=500,
        # save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        weight_decay=0.00,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        #save_steps=100,
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        #max_steps = 10,
        # logging_strategy="no",
        # disable_tqdm=True,
        #report_to="none"
    )



    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    # If tokenizer has no pad token (common for causal LMs), set it:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    if IS_GSM8K:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            # device_map="auto")
        )
    else:
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
            #print(name, param.shape, param.numel())
            pass


    if skip_ratio > 0.0:
        print("SKIP is set to True, elementwise_linear =", ELEMENTWISE_LINEAR, ", ELEMENTWISE_SWAP_SCHEME =", ELEMENTWISE_SWAP_SCHEME)

        embedding: nn.Embedding = model.model.embed_tokens
        hot_idx = make_hot_idx(embedding.num_embeddings, frac=1-skip_ratio, device=embedding.weight.device)
        #model.model.embed_tokens = replace_embedding_with_colwise(embedding, hot_idx)

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
                
                mod_name, proj_name = name.split(".", 1)

                if ELEMENTWISE_LINEAR:
                    #wrapped = replace_linear_with_elementwise_random(linear, percent_hot=1-skip_ratio)
                    if ELEMENTWISE_SWAP_SCHEME == "all":
                        hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
                        wrapped = replace_linear_with_elementwise_random(linear, percent_hot=1-skip_ratio)
                    elif ELEMENTWISE_SWAP_SCHEME == "neuron":
                        hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
                        wrapped = replace_linear_with_elementwise_hotidx(linear, hot_idx)
                    elif ELEMENTWISE_SWAP_SCHEME == "input":
                        hot_idx = make_hot_idx(linear.in_features, frac=1-skip_ratio, device=linear.weight.device)
                        wrapped = replace_linear_with_elementwise_hotidx_input_features(linear, hot_idx)
                    elif ELEMENTWISE_SWAP_SCHEME == "preselect":
                        selection_key = f"L{i:02d}_{mod_name}_{proj_name}_weight.npy"
                        layer_selection = preselect_lookup.get(selection_key)
                        if layer_selection is None:
                            available = ", ".join(sorted(preselect_lookup.keys()))
                            raise KeyError(f"No preselect entry for {selection_key}. Available entries: {available}")
                        expected_shape = tuple(layer_selection.get("shape") or [])
                        if expected_shape and tuple(linear.weight.shape) != expected_shape:
                            raise ValueError(
                                f"Shape mismatch for {selection_key}: preselect {expected_shape}, "
                                f"module {tuple(linear.weight.shape)}"
                            )
                        weight_pairs = layer_selection.get("train_weight_indices", [])
                        if not weight_pairs:
                            raise ValueError(f"Preselect entry {selection_key} has no weight indices")
                        w_idx = torch.as_tensor(weight_pairs, dtype=torch.long, device=linear.weight.device)
                        bias_indices = layer_selection.get("train_bias_indices", [])
                        b_idx = (
                            torch.as_tensor(bias_indices, dtype=torch.long, device=linear.weight.device)
                            if bias_indices
                            else None
                        )
                        wrapped = replace_linear_with_elementwise_preselected(linear, w_idx, b_idx)
                    else:
                        raise ValueError(f"Unsupported ELEMENTWISE_SWAP_SCHEME: {ELEMENTWISE_SWAP_SCHEME}")
                else:
                    hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
                    wrapped = replace_linear_with_colwise(linear, hot_idx, mode=mode)
                if name.startswith("self_attn."):
                    setattr(layer.self_attn, name.split(".", 1)[1], wrapped)
                else:
                    setattr(layer.mlp,       name.split(".", 1)[1], wrapped)

            # # quick sanity check
            # print(type(layer.self_attn.q_proj), layer.self_attn.q_proj.W_hot.shape, layer.self_attn.q_proj.W_cold.shape)
            # print(type(layer.mlp.up_proj),      layer.mlp.up_proj.W_hot.shape,      layer.mlp.up_proj.W_cold.shape)

    else:
        print("SKIP is set to False, not replacing linear layers.")


    verify_shapes_across_ranks(model)
    verify_weights_across_ranks(model)   # optional but thorough



    if IS_GSM8K:
        ds = load_dataset("gsm8k", "main")
    else:
        ds = load_dataset("nyu-mll/glue", DATASET)

    with args.main_process_first(desc="tokenize"):
        if DATASET == "sst2":
            tokenized_ds = ds.map(tokenize_function_sst, batched=False)
        elif DATASET == "mnli":
            tokenized_ds = ds.map(tokenize_function_mnli, batched=False)
        elif DATASET == "gsm8k":
            tokenized_ds = ds.map(
                tokenize_function_gsm8k,
                batched=True,
                remove_columns=ds["train"].column_names,
            )
        else:
            raise ValueError(f"Unsupported dataset: {DATASET}")

    print(tokenized_ds)
    # Data collator
    if IS_GSM8K:
        #data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tok,
            model=model,
            label_pad_token_id=-100,
            padding=True
        )
        compute_metrics = None
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
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

    # wrapped_model = trainer.model_wrapped
    # r = wrapped_model.reducer
    # print("Initial DDP bucket bytes:", sum(b.buffer().numel() * b.buffer().element_size() for b in r._buckets))



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

    dump_cb = PerModuleGradDumper(
        out_dir=dump_out_dir,
        model=model,
        capture_steps=100,
        include_bias=True,
        also_embeddings=True,  # set True if you also want embeddings/lm_head
        # weight_out_dir=weight_out_dir,
    )
    if args_cmd.dump_grads:
        trainer.add_callback(dump_cb)

    probe_cb = Probe()
    ram_cb = VramBreakdownCallback()

    #trainer.add_callback(probe_cb)
    trainer.add_callback(ram_cb)
    

    hotswap_cb = HotSwapCallback(swap_iters=random_swap_iters, elementwise_scheme=ELEMENTWISE_SWAP_SCHEME)
    trainer.add_callback(hotswap_cb)

    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    def log_memory_stats():
        """Log current GPU memory statistics"""
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Max Allocated: {max_allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    log_memory_stats()

    # Start training
    start = time.time()
    trainer.train()
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
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
