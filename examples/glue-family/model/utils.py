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
)

from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
from torchinfo import summary
import logging

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

    if adapter_args.train_adapter:
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
            if not coldneuron_args.elementwise_linear:
                print("***** Using Skip customized module *****")
                model.enable_input_require_grads()
                total_buffers = sum(b.numel() for b in model.buffers())
                print(f"Total buffers before adding linearcolwise: {total_buffers}")
                replace_linear_with_colwise(model, config, coldneuron_args.skip_ratio)
                after_total_buffers = sum(b.numel() for b in model.buffers())
                print(f"Total buffers after adding linearcolwise: {after_total_buffers}, {after_total_buffers - total_buffers} added.")
            else:
                assert False, "Elementwise linear replacement not supported in main.py currently."
                # Elementwise linear replacement
                #wrapped = replace_linear_with_elementwise_random(linear, percent_hot=1-skip_ratio)
                if coldneuron_args.elementwise_swap_scheme == "all":
                    hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
                    wrapped = replace_linear_with_elementwise_random(linear, percent_hot=1-skip_ratio)
                elif coldneuron_args.elementwise_swap_scheme == "neuron":
                    hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=linear.weight.device)
                    wrapped = replace_linear_with_elementwise_hotidx(linear, hot_idx)
                elif coldneuron_args.elementwise_swap_scheme == "input":
                    hot_idx = make_hot_idx(linear.in_features, frac=1-skip_ratio, device=linear.weight.device)
                    wrapped = replace_linear_with_elementwise_hotidx_input_features(linear, hot_idx)
                elif coldneuron_args.elementwise_swap_scheme == "preselect":
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
                    raise ValueError(f"Unsupported coldneuron_args.elementwise_swap_scheme: {coldneuron_args.elementwise_swap_scheme}")

    

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


def replace_linear_with_colwise(
    model,
    config: AutoConfig.from_pretrained,
    skip_ratio: float = 0,
):
    print(f"Architecture: {config.architectures}, Type: {config.model_type}, Name: {model.__class__.__name__}")

    if model.__class__.__name__ not in ["RobertaForSequenceClassification"]:
        print(f"{model.__class__.__name__} not supported for linear module printing.")
        return
    
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    elif hasattr(model, 'roberta'):
        encoder = model.roberta.encoder
    elif hasattr(model, 'bert'):
        encoder = model.bert.encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
    else:
        raise AttributeError("Could not find encoder in model")

    for name, module in model.named_modules():
        module.requires_grad = False
        for pm, param in module.named_parameters():
            param.requires_grad = False

    for name, module in model.named_modules():
        if "classifier" in name:
            module.requires_grad = True
            for pm, param in module.named_parameters():
                param.requires_grad = True  

    # Iterate through each encoder layer
    for layer_idx, layer in enumerate(encoder.layer):
        # Iterate through all modules in this layer
        for name, module in layer.named_modules():
            # if layer_idx == 0:
            #     print(f"Layer 0 module: {name}, type: {module.__class__.__name__}")
            if name.endswith("query") or name.endswith("value"):
            # if isinstance(module, nn.Linear):
                # # Get the parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                attr_name = name.split('.')[-1]
                
                # Get parent module
                if parent_name:
                    parent = layer
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                else:
                    parent = layer
                
                out_features = module.out_features
                
                # Replace the linear module
                hot_idx = make_hot_idx(out_features, frac=1-skip_ratio, device=module.weight.device)
                hot_idx = hot_idx.to(module.weight.device)
                # wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx, mode="1linear_efficient")
                # print(f"Replaced Linear with LinearColWise: dtype: {mod.weight.dtype}")
                wrapped = LinearColWise.from_linear(module, hot_idx=hot_idx, mode="1linear")
                wrapped.to(module.weight.device, dtype=module.weight.dtype)


                setattr(parent, attr_name, wrapped)
                
                # # Track the replacement
                # full_name = f"encoder.layer.{layer_idx}.{name}"
                # print(f"Replaced: {full_name}")

                # module.requires_grad = True
                # for pm, param in module.named_parameters():
                #     param.requires_grad = True
                #     # print(f"Unfreezing param: layer {layer_idx} module {name} param {pm}")
            elif "classifier" in name:
                assert False, "classifier should be in encoder layer"



    # encoder_layers = model.roberta.encoder.layer
    # print(f"Number of encoder layers: {len(encoder_layers)}")

    # # Method 2: Iterate through each layer
    # for idx, layer in enumerate(encoder_layers):
    #     if idx > 0:
    #         break
    #     print(f"\n--- Layer {idx} ---")
    #     print(f"Self-attention: {layer.attention}")
    #     print(f"Intermediate (FFN): {layer.intermediate}")
    #     print(f"Output: {layer.output}")

    # # Method 3: Access specific components within each layer
    # for idx, layer in enumerate(encoder_layers):
    #     if idx > 0:
    #         break
    #     print(f"\nLayer {idx} components:")
    #     # Self-attention components
    #     print(f"  - Query: {layer.attention.self.query} input_features: {layer.attention.self.query.in_features} output_features: {layer.attention.self.query.out_features}")
    #     print(f"  - Key: {layer.attention.self.key} input_features: {layer.attention.self.key.in_features} output_features: {layer.attention.self.key.out_features}")
    #     print(f"  - Value: {layer.attention.self.value} input_features: {layer.attention.self.value.in_features} output_features: {layer.attention.self.value.out_features}")
    #     print(f"  - Attention output dense: {layer.attention.output.dense} input_features: {layer.attention.output.dense.in_features} output_features: {layer.attention.output.dense.out_features}")

    #     # Feed-forward network components
    #     print(f"  - Intermediate dense: {layer.intermediate.dense} input_features: {layer.intermediate.dense.in_features} output_features: {layer.intermediate.dense.out_features}")
    #     print(f"  - Output dense: {layer.output.dense} input_features: {layer.output.dense.in_features} output_features: {layer.output.dense.out_features}")

    # for name, param in model.named_parameters():
    #     print(f"Param: {name} Numel: {param.numel()}, shape: {param.shape}, Requires grad: {param.requires_grad}")
    # logger.info(summary(model, depth=5))

