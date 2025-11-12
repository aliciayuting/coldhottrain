from enum import Enum
import torch
import torch.nn as nn
import sys
import os
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from adapters import AutoAdapterModel

from model.custom_module import LinearColWise
from training.helper import (
    replace_linear_with_elementwise_hotidx,
    replace_linear_with_elementwise_hotidx_input_features,
    replace_linear_with_elementwise_preselected,
    replace_linear_with_elementwise_random,
    build_elementwise_indices_from_preselected,
    extend_with_random_rows,
)

from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
from torchinfo import summary
import logging
from training.helper import get_decoder_layers
import json

logger = logging.getLogger(__name__)

class TaskType(Enum):
    TOKEN_CLASSIFICATION = (1,)
    SEQUENCE_CLASSIFICATION = (2,)
    QUESTION_ANSWERING = (3,)
    MULTIPLE_CHOICE = 4


AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}

AUTO_PEFT_TASKS = {
    TaskType.TOKEN_CLASSIFICATION: PeftTaskType.TOKEN_CLS,
    TaskType.SEQUENCE_CLASSIFICATION: PeftTaskType.SEQ_CLS,
    TaskType.QUESTION_ANSWERING: PeftTaskType.QUESTION_ANS,
    TaskType.MULTIPLE_CHOICE: PeftTaskType.SEQ_CLS,
}

def is_module_to_replace(name: str, target_modules: list[str]) -> str:
    for tm in target_modules:
        if tm in name:
            return tm
    return None

def get_model(
    args,
    task_type: TaskType,
    config: AutoConfig.from_pretrained,
    fix_bert: bool = False,
):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        # fusion_args,
        # mtl_args,
        coldneuron_args,
    ) = args


    model_class = AUTO_MODELS[task_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )


    if coldneuron_args.full_parameter_q_v_classifier:
        assert coldneuron_args.skip_ratio == 0, "Cannot use both full_parameter_q_v_classifier and ColdNeurons at the same time."
        assert not adapter_args.train_adapter, "Cannot use both full_parameter_q_v_classifier and LoRA at the same time."
        print("***** Using full parameter for q,v and classifier *****")
        model.enable_input_require_grads()
        for name, param in model.named_parameters():
            if not ("query" in name or "value" in name or "classifier" in name):
                param.requires_grad = False
    elif adapter_args.train_adapter:
        assert coldneuron_args.skip_ratio == 0, "Cannot use both LoRA and ColdNeurons at the same time."
        print("***** Using LoRA *****")
        model.enable_input_require_grads()
        peft_config = LoraConfig(
            task_type=AUTO_PEFT_TASKS[task_type],
            target_modules=["query", "value"], # TODO: make it configurable and generalizable for all models
            modules_to_save=["classifier"],
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
        model.can_return_loss = True
    elif coldneuron_args.skip_ratio > 0:
        if coldneuron_args.use_masked_skipgradient:
            print("***** Using Skip mask *****")
            model.enable_input_require_grads()
            for name, param in model.named_parameters():
                if not ("query" in name or "value" in name or "classifier" in name):
                    param.requires_grad = False
        else:
            layers = get_decoder_layers(model)
            target_modules = ["query", "value"]

            preselect_lookup = {}
            if (coldneuron_args.elementwise_swap_scheme == "preselect" or coldneuron_args.elementwise_swap_scheme == "smartswap") and coldneuron_args.elementwise_linear:
                # TODO: sync with jamal about how to modify this
                if not coldneuron_args.preselect_file:
                    raise ValueError("ELEMENTWISE_SWAP_SCHEME='preselect' requires --preselect-file to be specified")
                try:
                    with open(coldneuron_args.preselect_file, "r", encoding="utf-8") as fh:
                        preselect_payload = json.load(fh)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f"Preselect file not found: {coldneuron_args.preselect_file}") from exc
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse JSON from preselect file {coldneuron_args.preselect_file}: {exc}") from exc

                raw_layers = preselect_payload.get("layers", {})
                if not raw_layers:
                    raise ValueError(f"No layer selections found in preselect file: {coldneuron_args.preselect_file}")

                aggregated = {}
                for raw_name, info in raw_layers.items():
                    short_name = os.path.basename(raw_name)
                    if True:
                        print("short_name:", short_name)
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
                    raise ValueError(f"Preselect file {coldneuron_args.preselect_file} produced no usable layer entries")
            
            # freeze all parameters first
            for name, module in model.named_modules():
                module.requires_grad = False
                for pm, param in module.named_parameters():
                    param.requires_grad = False
            # unfreeze classifier
            for name, module in model.named_modules():
                if "classifier" in name:
                    module.requires_grad = True
                    for pm, param in module.named_parameters():
                        param.requires_grad = True


            # Iterate through each encoder layer
            model.enable_input_require_grads()
            total_buffers = sum(b.numel() for b in model.buffers())
            for layer_idx, layer in enumerate(layers):
                for name, linear in layer.named_modules():
                    # if any(tm in name for tm in target_modules):
                    proj_name = is_module_to_replace(name, target_modules)
                    # print("proj_name:", proj_name)
                    if proj_name is not None:
                        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                        attr_name = name.split('.')[-1]
                        
                        # Get parent module
                        if parent_name:
                            parent = layer
                            for part in parent_name.split('.'):
                                parent = getattr(parent, part)
                        else:
                            parent = layer
                        
                        in_features = linear.in_features
                        out_features = linear.out_features
                        preselect_proj_name = ""
                        if proj_name == "query":
                            preselect_proj_name = "q_proj"
                        elif proj_name == "value":
                            preselect_proj_name = "v_proj"
                        assert isinstance(out_features, int) and isinstance(in_features, int)
                        if not coldneuron_args.elementwise_linear: # linear colwise replacement
                            hot_idx = make_hot_idx(out_features, frac=1-coldneuron_args.skip_ratio, device=linear.weight.device)
                            wrapped = replace_linear_with_colwise(linear, hot_idx, mode="1linear")
                        else:  # linear elementwise replacement
                            # build preselect_lookup for preselect/smartswap 
                            if coldneuron_args.elementwise_swap_scheme == "all":
                                hot_idx = make_hot_idx(out_features, frac=1-coldneuron_args.skip_ratio, device=linear.weight.device)
                                wrapped = replace_linear_with_elementwise_random(linear, percent_hot=1-coldneuron_args.skip_ratio)
                            elif coldneuron_args.elementwise_swap_scheme == "neuron":
                                hot_idx = make_hot_idx(out_features, frac=1-coldneuron_args.skip_ratio, device=linear.weight.device)
                                wrapped = replace_linear_with_elementwise_hotidx(linear, hot_idx)
                            elif coldneuron_args.elementwise_swap_scheme == "input":
                                hot_idx = make_hot_idx(in_features, frac=1-coldneuron_args.skip_ratio, device=linear.weight.device)
                                wrapped = replace_linear_with_elementwise_hotidx_input_features(linear, hot_idx)
                            elif coldneuron_args.elementwise_swap_scheme == "preselect":
                                w_idx, b_idx = build_elementwise_indices_from_preselected(layer_idx, "self_attn", preselect_proj_name, linear, preselect_lookup)
                                wrapped = replace_linear_with_elementwise_preselected(linear, w_idx, b_idx)
                            elif coldneuron_args.elementwise_swap_scheme == "smartswap":
                                w_idx, b_idx = build_elementwise_indices_from_preselected(layer_idx, "self_attn", preselect_proj_name, linear, preselect_lookup)
                                #TODO: kind of cheating to do it this way because some models may not have bias, but fine for now
                                hot_neurons = b_idx
                                saved_w = w_idx
                                saved_b = b_idx
                                print(name)
                                #print(hot_neurons)
                                numels = w_idx.size(0)
                                skipped = 1-float(numels)/(linear.in_features*linear.out_features)
                                diff = skipped - coldneuron_args.skip_ratio
                                additional_n = int(diff * linear.out_features + 0.5)
                                print(diff)

                                w_idx, b_idx = extend_with_random_rows(w_idx=w_idx, b_idx=b_idx, additional_n=additional_n, in_features=linear.in_features, out_features=linear.out_features)

                                #TODO: disgustingly inefficent. fine for now though
                                wrapped = replace_linear_with_elementwise_preselected(linear, w_idx, b_idx, metadata={"hot_neurons": hot_neurons, "additional_n": additional_n, "hot_w": saved_w, "hot_b": saved_b})
                            else:
                                raise ValueError(f"Unsupported coldneuron_args.elementwise_swap_scheme: {coldneuron_args.elementwise_swap_scheme}")
                            

                        setattr(parent, attr_name, wrapped)
            after_total_buffers = sum(b.numel() for b in model.buffers())
            print(f"Total buffers after adding replacing: {after_total_buffers}, {after_total_buffers - total_buffers} added.")
        
    

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Param: {name} Numel: {param.numel()}, shape: {param.shape}, Requires grad: {param.requires_grad}")
    logger.info(summary(model, depth=5))

    return model


# ---- your previously defined helpers (from my last message) ----
def make_hot_idx(out_features: int, frac: float | None = None, idx: torch.Tensor | None = None, device=None):
    if idx is not None:
        hot = torch.as_tensor(idx, dtype=torch.long, device=device)
        assert hot.ndim == 1 and hot.numel() > 0
        assert hot.min().item() >= 0 and hot.max().item() < out_features
        return hot
    assert frac is not None and 0.0 <= frac <= 1.0
    k = max(0, min(out_features, int(round(frac * out_features))))
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    perm = torch.randperm(out_features, device=device)
    return perm[:k].sort().values


def replace_linear_with_colwise(mod: nn.Module, hot_idx: torch.Tensor, mode: str = "1linear_efficient") -> LinearColWise:
    assert isinstance(mod, nn.Linear)
    hot_idx = hot_idx.to(mod.weight.device)
    all_out = torch.arange(mod.out_features, device=hot_idx.device)
    cold_idx = all_out[~torch.isin(all_out, hot_idx)]

    #print("Non-trainable output indices:", cold_idx)

    wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx, mode=mode)
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped