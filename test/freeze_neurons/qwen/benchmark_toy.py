import os
import contextlib
from typing import Optional, Tuple

import torch
from torch import nn
from torch.cuda import nvtx

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

class NVTXTrainer(Trainer):
    # Forward: only tag the model's forward/compute_loss region
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        with nvtx.range("forward"):
            return super().compute_loss(model, inputs, return_outputs)

    # Backward: tag exactly where Trainer performs the backward
    # def training_step(self, model: Moduleinputs: dictnum_items_in_batch: typing.Optional[torch.Tensor] = None ) -> torch.Tensor:
    #     with nvtx.range("backward"):
    #         return super().training_step(loss)

    # # Optimizer step: tag the parameter update (may be bypassed by DeepSpeed)
    # def optimizer_step(self, *args, **kwargs):
    #     with nvtx.range("optimizer.step"):
    #         return super().optimizer_step(*args, **kwargs)

    # # Optional: also tag zero_grad
    # def optimizer_zero_grad(self, *args, **kwargs):
    #     with nvtx.range("optimizer.zero_grad"):
    #         return super().optimizer_zero_grad(*args, **kwargs)




model_name = "Qwen/Qwen2.5-0.5B"
# MODEL = "Qwen/Qwen2.5-14B"

output_dir = f"/pscratch/sd/l/lsx/runs/{model_name.replace('/', '_')}-toydataset/"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# Tiny placeholder dataset (replace with yours)
# -----------------------------
class TinyText(torch.utils.data.Dataset):
    def __init__(self, tok: AutoTokenizer, n: int = 128, max_len: int = 64):
        self.tok = tok
        self.samples = ["Hello world"] * n
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        enc = self.tok(
            self.samples[i],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        ids = enc["input_ids"][0]
        return {"input_ids": ids, "labels": ids.clone()}



# logging.basicConfig(
#         level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
#         format="[%(levelname)s] %(message)s"
#     )

def main():
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)  # do NOT .cuda(); Trainer handles placement

    train_data = TinyText(tok, n=128, max_len=64)

    args = TrainingArguments(
        output_dir=f"{output_dir}/output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=0,
        report_to=[],  # keep console clean
        dataloader_pin_memory=True,
        max_steps=1,  # keep short while profiling
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
    )

    trainer = NVTXTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        tokenizer=tok,
        data_collator=default_data_collator,
    )

    # Enable autograd/operator-level NVTX names once for the whole run
    with torch.autograd.profiler.emit_nvtx():
        trainer.train()

if __name__ == "__main__":
    main()