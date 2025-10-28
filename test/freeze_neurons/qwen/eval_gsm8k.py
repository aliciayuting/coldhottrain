"""
evaluate_gsm8k.py

Evaluate a fine-tuned model on GSM8K test set.

Usage (single GPU):
    python evaluate_gsm8k.py --model_path ./results/checkpoint-1000 --dataset_name gsm8k

Usage (multi-GPU, DDP-style with torchrun):
    torchrun --nproc_per_node 4 evaluate_gsm8k.py --ddp --model_path ./results/checkpoint-1000 --dataset_name gsm8k
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import timedelta


# ------------------------- DDP helpers -------------------------

def init_distributed(backend: str = "nccl") -> int:
    """Initialize torch.distributed from environment (torchrun). Returns local_rank."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=18000))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def get_world_size() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

def get_rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# ------------------------- Core logic -------------------------

def extract_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from GSM8K format.
    GSM8K answers are typically in the format '#### number'
    """
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return None

def format_prompt(question: str) -> str:
    """Format the GSM8K question as a prompt."""
    return f"Question: {question}\nAnswer: "

@torch.no_grad()
def evaluate_gsm8k(
    model_path: str,
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    split: str = "test",
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    device: Optional[str] = None,
    ddp: bool = False,
    output_file: Optional[str] = None,
):
    """
    Evaluate a model on GSM8K dataset.

    ddp=True enables multi-process data-parallel inference via torchrun.
    """
    # Resolve device
    if device is None:
        if torch.cuda.is_available():
            # If DDP, we already set the CUDA device via LOCAL_RANK
            device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if ddp else "cuda")
        else:
            device = torch.device("cpu")

    if is_main_process():
        print(f"Loading model from {model_path}...")

    # Important: do NOT use device_map='auto' in DDP-style runs.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print(f"Loading dataset {dataset_name} ({dataset_config}/{split})...")

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Respect --max_samples before sharding, so the total count matches the user's intent
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Shard across ranks if DDP
    if ddp and get_world_size() > 1:
        dataset = dataset.shard(num_shards=get_world_size(), index=get_rank(), contiguous=True)

    if is_main_process():
        #total_eval_samples = (max_samples if max_samples else load_dataset(dataset_name, dataset_config, split=split)).num_rows
        print(f"Evaluating on {dataset.num_rows} samples total "
              f"({len(dataset)} per-rank chunk across {get_world_size()} ranks).")

    correct = 0
    total = 0
    results: List[Dict] = []

    # Iterate in batches
    pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Rank {get_rank()} Evaluating", disable=not is_main_process())
    for i in pbar:
        batch = dataset[i:i + batch_size]
        questions = batch["question"]
        true_answers = batch["answer"]

        prompts = [format_prompt(q) for q in questions]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the generated part (exclude prompt tokens)
        gen_only = outputs[:, inputs.input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for question, gen_text, true_ans in zip(questions, generated_texts, true_answers):
            pred_answer = extract_answer(gen_text)
            true_answer = extract_answer(true_ans)

            is_correct = False
            if pred_answer is not None and true_answer is not None:
                is_correct = abs(pred_answer - true_answer) < 1e-6
                if is_correct:
                    correct += 1
                total += 1

            results.append({
                "question": question,
                "generated": gen_text,
                "true_answer": true_answer,
                "predicted_answer": pred_answer,
                "correct": is_correct
            })

    # Reduce counters across ranks
    correct_tensor = torch.tensor([correct], device=device, dtype=torch.long)
    total_tensor = torch.tensor([total], device=device, dtype=torch.long)
    if ddp and get_world_size() > 1:
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    global_correct = int(correct_tensor.item())
    global_total = int(total_tensor.item())
    accuracy = (global_correct / global_total) if global_total > 0 else 0.0

    # Optionally gather detailed results to rank 0 (careful: can be big)
    gathered_results = None
    if output_file:
        if ddp and get_world_size() > 1:
            try:
                if is_main_process():
                    gathered_results = [None for _ in range(get_world_size())]
                    dist.gather_object(results, object_gather_list=gathered_results, dst=0)
                    # flatten chunks
                    gathered_results = [item for chunk in gathered_results if chunk is not None for item in chunk]
                else:
                    dist.gather_object(results, object_gather_list=None, dst=0)
            except Exception:
                # Fallback: write per-rank file
                rank = get_rank()
                rank_file = _ranked_filename(output_file, rank)
                with open(rank_file, "w") as f:
                    json.dump({"results": results}, f, indent=2)
        else:
            gathered_results = results

    # Print summary only on rank 0
    if is_main_process():
        print("\n" + "="*60)
        print(f"Results:")
        print(f"  Total samples: {global_total} (valid predictions)")
        print(f"  Correct: {global_correct}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("="*60)

        # Print a few examples (from whatever results we have on rank 0)
        sample_src = gathered_results if gathered_results is not None else results
        print("\nSample predictions:")
        for i, result in enumerate(sample_src[:3]):
            print(result)
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {result['question'][:100]}...")
            print(f"Generated: {result['generated'][:200]}...")
            print(f"True answer: {result['true_answer']}")
            print(f"Predicted: {result['predicted_answer']}")
            print(f"Correct: {result['correct']}")

        # Save detailed results if requested
        if output_file:
            if gathered_results is not None:
                payload = {
                    "accuracy": accuracy,
                    "correct": global_correct,
                    "total": global_total,
                    "results": gathered_results
                }
                with open(output_file, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"\nDetailed results saved to {output_file}")
            else:
                print(f"\nPer-rank result files were written next to: {output_file}")

    return {
        "accuracy": accuracy,
        "correct": global_correct,
        "total": global_total,
        "results": results if is_main_process() else None  # only return local results on non-main
    }

def _ranked_filename(path: str, rank: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext or '.json'}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name (default: gsm8k)")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset config (default: main)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-process batch size for generation (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate (default: 512)")
    parser.add_argument("--output_file", type=str, default=None, help="Save detailed results to JSON file")
    # DDP flags
    parser.add_argument("--ddp", action="store_true", help="Enable multi-GPU inference via torch.distributed + torchrun")
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="torch.distributed backend (default: nccl)")

    args = parser.parse_args()

    # Set up DDP if requested
    if args.ddp:
        local_rank = init_distributed(backend=args.ddp_backend)
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        results = evaluate_gsm8k(
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            ddp=args.ddp,
            output_file=args.output_file,
        )
    finally:
        # Cleanly tear down the process group
        if args.ddp and dist.is_available() and dist.is_initialized():
            #barrier()
            dist.destroy_process_group()
