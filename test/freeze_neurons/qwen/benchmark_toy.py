import os
import contextlib
from typing import Optional, Tuple, Union, Any
import typing
import torch
from torch import nn
from torch.cuda import nvtx
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from contextlib import contextmanager
from collections import defaultdict


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

    # # High-level forward + backward wrapper
    # def training_step(
    #         self,
    #         model: nn.Module,
    #         inputs: dict[str, Union[torch.Tensor, Any]],
    #         num_items_in_batch: Optional[torch.Tensor] = None,
    #     ) -> torch.Tensor:
    #         with torch.cuda.nvtx.range("Traing step"):
    #             return super().training_step(model, inputs, num_items_in_batch)
                



model_name = "Qwen/Qwen2.5-0.5B"
# MODEL = "Qwen/Qwen2.5-14B"

# output_dir = f"/pscratch/sd/l/lsx/runs/{model_name.replace('/', '_')}-toydataset/"
output_dir = f"/mnt/coldhot/shouxu_runs/{model_name.replace('/', '_')}-toydataset/"
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
    select_n_layers = 2
    model.model.layers = model.model.layers[:select_n_layers]
    model.config.num_hidden_layers =  select_n_layers
    # instrument_model_for_nvtx(model)
    # instrument_model_forward_backward_nvtx(model, only_leaf=True)
    register_nvtx_hooks(model)

    for name, module in model.named_modules():
        print(name, "->", module.__class__.__name__)


    cfg = model.config
    print("hidden_size:", cfg.hidden_size)                       # 896
    print("num_attention_heads:", cfg.num_attention_heads)       # 14
    print("num_key_value_heads:", getattr(cfg, "num_key_value_heads", None))  # 2
    print("head_dim:", cfg.hidden_size // cfg.num_attention_heads)            # 64
    print("kv_proj_out:", (getattr(cfg, "num_key_value_heads", 2) * (cfg.hidden_size // cfg.num_attention_heads)))  # 128
    print("num_layers: ", model.config.num_hidden_layers)

    # exit(0)


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
        max_steps=5,  # keep short while profiling
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
    )

    trainer = NVTXTrainer(
    # trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        tokenizer=tok,
        data_collator=default_data_collator,
        callbacks=[NVTXOptCallback],
    )


    # Enable autograd/operator-level NVTX names once for the whole run
    with torch.autograd.profiler.emit_nvtx():
        trainer.train()
    # trainer.train()

if __name__ == "__main__":
    main()