import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers

import numpy as np
from sklearn.metrics import accuracy_score


# Set random seed for reproducibility
transformers.set_seed(42)

# Configuration
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DATASET = "cais/mmlu"
MAX_LENGTH = 768
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100


def main():
    # read model from the program arugment
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name to use")
    args = parser.parse_args()

    MODEL_NAME = args.model
    OUTPUT_DIR = f"/pscratch/sd/l/lsx/shouxu_runs/{MODEL_NAME.replace('/', '_')}-{DATASET.replace('/', '_')}"


    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA for efficient finetuning
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # model = prepare_model_for_kbit_training(model)
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    # Load MMLU dataset
    print("Loading MMLU dataset...")
    dataset = load_dataset(DATASET, "all")
    train_dataset = dataset["auxiliary_train"]
    eval_dataset = dataset["test"]
    # # truncate datasets for quick debugging (remove in real training)
    # train_dataset = train_dataset.select(range(2))
    # eval_dataset = eval_dataset.select(range(2))

    # Format MMLU data into instruction format
    def format_mmlu_example(example):
        """Convert MMLU example to instruction-following format"""
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        
        # Format choices
        choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # Create instruction-following format
        prompt = f"""Answer the following multiple choice question.

    Question: {question}

    Choices:
    {choice_text}

    Answer:"""
        
        # Get the correct answer letter
        answer = chr(65 + answer_idx)
        answer = f" {answer}"
        full_text = prompt + answer
        
        return {"prompt": prompt, 
                "label": answer,
                "full_text": full_text}


    def tokenize_function(examples):
        # Tokenize the full input (prompt + answer letter)
        enc = tokenizer(
            examples["full_text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=False,
        )

        # Tokenize prompts alone to get where the answer starts
        enc_prompt = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,              # no padding here; we want true lengths
            add_special_tokens=False,
        )
        prompt_lens = [len(x) for x in enc_prompt["input_ids"]]

        # Start with labels = input_ids
        labels = [seq.copy() for seq in enc["input_ids"]]

        # (1) Ignore padding in loss
        for i, mask in enumerate(enc["attention_mask"]):
            for j, m in enumerate(mask):
                if m == 0:
                    labels[i][j] = -100

        # (2) Ignore the prompt part in loss; keep only the answer tokens
        for i, plen in enumerate(prompt_lens):
            upto = min(plen, len(labels[i]))
            for j in range(upto):
                if labels[i][j] != -100:   # don’t touch already-padded positions
                    labels[i][j] = -100

        enc["labels"] = labels
        return enc


    def show_dataset_example(dataset, num_examples=1):
        for i in range(len(dataset)):
            if i >= num_examples:
                break
            print(f"--- Example {i}: ---")
            print(f"### Prompt:\n{dataset[i]['prompt']}")
            print(f"### Label:\n{dataset[i]['label']}")
            print()


    # Process datasets
    print("Processing datasets...")
    train_dataset = train_dataset.map(format_mmlu_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_mmlu_example, remove_columns=eval_dataset.column_names)

    print("Sample formatted training examples:")
    show_dataset_example(train_dataset, num_examples=2)


    # Tokenize
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "label"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "label"])

    def show_tokenized_dataset_examples(dataset, num_examples=2):
        for i in range(len(dataset)):
            if i >= num_examples:
                break
            example = dataset[i]
            print(f"--- Example {i}: ---")
            print(f"### Input IDs length: {len(example['input_ids'])}")
            print(f"### Input ids: {example['input_ids']}")
            non_padded_input_idxs = [idx for idx, id in enumerate(example['input_ids']) if id != tokenizer.pad_token_id]
            non_padded_labels_idxs = [idx for idx, label in enumerate(example['labels']) if label != -100]
            non_masked_attention_idxs = [idx for idx, mask in enumerate(example['attention_mask']) if mask != 0]
            print(f"### Non-padded Input IDs (len: {len(non_padded_input_idxs)})")
            print(f"### Non-padded Labels (len: {len(non_padded_labels_idxs)})")
            print(f"### Non-masked Attention (len: {len(non_masked_attention_idxs)})")
            print(f"### Input text (non-padded):\n{tokenizer.decode([id for id in example['input_ids'] if id != tokenizer.pad_token_id])}")
            non_ignore_labels = [l for l in example['labels'] if l != -100]
            print(f"### Labels (non -100): {non_ignore_labels}")
            print(f"### Decoded: {tokenizer.decode(non_ignore_labels)}")
            print()

    print("Sample tokenized training examples:")
    show_tokenized_dataset_examples(eval_dataset, num_examples=2)



    # Data collator
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )

    data_collator = default_data_collator


    # ---- Choice token IDs (computed once) ---------------------------------------
    CHOICES = [" A", " B", " C", " D"]
    CHOICE_TOKEN_IDS = []
    for s in CHOICES:
        ids = tokenizer.encode(s, add_special_tokens=False)
        assert(len(ids) == 1)
        CHOICE_TOKEN_IDS.append(ids[-1])
    CHOICE_TOKEN_IDS = torch.tensor(CHOICE_TOKEN_IDS, dtype=torch.long)
    print(f"Choice token IDs: {CHOICE_TOKEN_IDS.tolist()}")

    # ---- Helper: find first non -100 label position per example -----------------
    def _first_answer_pos(labels: torch.Tensor) -> torch.Tensor:
        """
        labels: (B, T) with -100 everywhere except the single answer token.
        returns: LongTensor (B,) positions of the first non -100; if none, -1.
        """
        not_ign = (labels != -100)  # (B, T)
        # position of first True in each row, or 0 if none
        first_pos = not_ign.float().argmax(dim=1)  # (B,)
        # But argmax returns 0 when all False; fix those to -1
        has_any = not_ign.any(dim=1)
        first_pos = torch.where(has_any, first_pos, torch.full_like(first_pos, -1))
        return first_pos

    def preprocess_logits_for_metrics(logits, labels):
        """
        Reduce memory & do the 4-way slice here.
        - logits: (B, T, V) or tuple(logits, ...) depending on model; we handle both.
        - labels: (B, T) with -100 except answer token.
        Returns: (B, 4) tensor with logits for [" A"," B"," C"," D"] at the answer position.
        """
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # (B, T, V)

        # Ensure tensor types/devices match
        logits = logits.float()
        labels = labels.to(logits.device)

        B, T, V = logits.shape
        ans_pos = _first_answer_pos(labels)  # (B,)
        # print(f"Answer positions in batch: {ans_pos.tolist()}")

        # Guard: if any example has no answer token, fall back to last non-pad position
        # (should not happen with your pipeline). We’ll mask these later anyway.
        bad = (ans_pos < 0)
        if bad.any():
            print("!!! Warning: some examples have no answer token; using last non-pad position instead.")
            # Choose a safe position (e.g., last timestep) to avoid index error
            ans_pos = torch.where(bad, torch.full_like(ans_pos, T - 1), ans_pos)

        # Gather (B, V) logits at the answer position per sample
        row_idx = torch.arange(B, device=logits.device)
        logits_at_ans = logits[row_idx, ans_pos, :]  # (B, V)
        # print(f"Logits at answer positions: {logits_at_ans.tolist()}")

        # Slice down to 4 choices
        choice_ids = CHOICE_TOKEN_IDS.to(logits.device)  # (4,)
        four_logits = logits_at_ans.index_select(dim=1, index=choice_ids)  # (B, 4)
        # print(f"Four choice logits: {four_logits.tolist()}")
        return four_logits

    def compute_metrics(eval_pred):
        """
        eval_pred.predictions: (N, 4) from preprocess_logits_for_metrics
        eval_pred.label_ids:   (N, T) original labels with -100 except gold-answer token
        Returns {"accuracy": ...}
        """
        four_logits = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Pred: argmax over the 4 choices
        pred_idx = np.asarray(four_logits).argmax(axis=1)  # (N,)
        # print(f"Predicted indices: {pred_idx.tolist()}")

        # Gold: find the answer token id for each example and map to 0..3
        labels = torch.tensor(label_ids)
        ans_pos = _first_answer_pos(labels)  # (N,)
        has_any = (ans_pos >= 0)
        assert has_any.any(), "No valid answers found in labels!"

        row_idx = torch.arange(labels.size(0))
        # For rows without any answer token (shouldn't happen), put a dummy id (-1)
        gold_token_ids = torch.full((labels.size(0),), -1, dtype=torch.long)
        gold_token_ids[has_any] = labels[row_idx[has_any], ans_pos[has_any]]
        # print(f"Gold token IDs: {gold_token_ids.tolist()}")

        # Map token id -> {0,1,2,3} using the same CHOICE_TOKEN_IDS
        choice_ids = CHOICE_TOKEN_IDS
        # Build a vectorized mapping by comparing against each choice id
        eq_matrix = (gold_token_ids[:, None] == choice_ids[None, :])  # (N, 4)
        # print(f"Gold token ID matches:\n{eq_matrix.numpy().astype(int)}")
        # If exactly one matches, argmax gives that index; if none match, row will be all False -> 0
        gold_idx = eq_matrix.long().argmax(dim=1).numpy()  # (N,)
        gold_valid = eq_matrix.any(dim=1).numpy()          # (N,)

        # Compute accuracy only on valid rows (where gold token matched one of the 4)
        if gold_valid.any():
            acc = accuracy_score(gold_idx[gold_valid], pred_idx[gold_valid])
        else:
            acc = 0.0
        print("accuracy:", acc)


        for i in range(min(5, len(pred_idx))):
            print(f"Example {i}: pred={pred_idx[i]}, gold={gold_idx[i]}, valid={gold_valid[i]}")

        return {"accuracy": acc}

    # Training arguments with data parallelism
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        # eval_strategy="epoch",
        eval_strategy="steps",
        eval_steps=500,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Data parallelism settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        resume_from_checkpoint=False,
        # Reporting
        # report_to="tensorboard",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # max_steps=1,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=100,
        
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()


    # Save final model
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")

    # # Optional: Evaluate on test set
    # print("\nEvaluating on test set...")
    # test_results = trainer.evaluate(test_dataset)
    # print(f"Test results: {test_results}")

## main
if __name__ == "__main__":
    main()