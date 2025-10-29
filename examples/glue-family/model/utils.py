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

from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType

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

    # if adapter_args.train_adapter:
    if False:
        print("***** Using LoRA finetuning *****")

        # We use the AutoAdapterModel class here for better adapter support.
        model = AutoAdapterModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    else:
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

    # if coldneuron_args.use_lora:
        # assert coldneuron_args.skip_ratio == 0 and not coldneuron_args.use_masked_skipgradient, "Cannot use both LoRA and ColdNeurons at the same time."
        # print("***** Using LoRA finetuning *****")
        # peft_config = LoraConfig(
        #     task_type=AUTO_PEFT_TASKS[task_type],
        #     target_modules=["query", "value"], # TODO: make it configurable and generalizable for all models
        #     modules_to_save=["classifier"],
        #     r=coldneuron_args.lora_rank,
        #     lora_alpha=coldneuron_args.lora_scaling_factor,
        #     lora_dropout=0.05,
        #     inference_mode=False,
        # )
        # model = get_peft_model(model, peft_config)

    bert_param = 0
    # if fix_bert:
    #     if config.model_type == "bert":
    #         for param in model.bert.parameters():
    #             param.requires_grad = False
    #         for _, param in model.bert.named_parameters():
    #             bert_param += param.numel()
    #     elif config.model_type == "roberta":
    #         for param in model.roberta.parameters():
    #             param.requires_grad = False
    #         for _, param in model.roberta.named_parameters():
    #             bert_param += param.numel()
    #     elif config.model_type == "deberta":
    #         for param in model.deberta.parameters():
    #             param.requires_grad = False
    #         for _, param in model.deberta.named_parameters():
    #             bert_param += param.numel()

    # model.print_trainable_parameters() will print these info
    # all_param = 0
    # trainable_params = 0
    # for name, param in model.named_parameters():
    #     print(f"Param: {name}, Numel: {param.numel()}, Requires grad: {param.requires_grad}")
    #     all_param += param.numel()
    #     if param.requires_grad:
    #         trainable_params += param.numel()
    # total_param = all_param - bert_param
    # print("***** total param is {} trainable param is {} *****".format(total_param, trainable_params))
    # model.print_trainable_parameters()
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


def replace_linear_with_colwise(mod: nn.Module, hot_idx: torch.Tensor) -> LinearColWise:
    assert isinstance(mod, nn.Linear)
    hot_idx = hot_idx.to(mod.weight.device)
    wrapped = LinearColWise.from_linear(mod, hot_idx=hot_idx, mode="1linear")
    wrapped.to(mod.weight.device, dtype=mod.weight.dtype)
    return wrapped


def fix_linear_modules(
    model,
    config: AutoConfig.from_pretrained,
    skip_ratio: float = 0,
):

    print(f"Architecture: {config.architectures}, Type: {config.model_type}, Name: {model.__class__.__name__}")

    if model.__class__.__name__ not in ["RobertaForSequenceClassification"]:
        print(f"{model.__class__.__name__} not supported for linear module printing.")
        return
    
    # print()
    # print("=" * 50)
    # print("Linear modules in the model type: ", config.model_type)
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Linear):
    #         # linear_modules.append((name, module))
    #         print(f"{name}: {module.in_features} -> {module.out_features}")
    # print("=" * 50)
    # print()


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
    
    # Iterate through each encoder layer
    for layer_idx, layer in enumerate(encoder.layer):
        # Iterate through all modules in this layer
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Get the parent module and attribute name
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
                wrapped = replace_linear_with_colwise(module, hot_idx)
                setattr(parent, attr_name, wrapped)
                
                # Track the replacement
                full_name = f"encoder.layer.{layer_idx}.{name}"
                # print(f"Replaced: {full_name}")

    encoder_layers = model.roberta.encoder.layer
    print(f"Number of encoder layers: {len(encoder_layers)}")

    # Method 2: Iterate through each layer
    for idx, layer in enumerate(encoder_layers):
        if idx > 0:
            break
        print(f"\n--- Layer {idx} ---")
        print(f"Self-attention: {layer.attention}")
        print(f"Intermediate (FFN): {layer.intermediate}")
        print(f"Output: {layer.output}")

    # Method 3: Access specific components within each layer
    for idx, layer in enumerate(encoder_layers):
        if idx > 0:
            break
        print(f"\nLayer {idx} components:")
        # Self-attention components
        print(f"  - Query: {layer.attention.self.query} input_features: {layer.attention.self.query.in_features} output_features: {layer.attention.self.query.out_features}")
        print(f"  - Key: {layer.attention.self.key} input_features: {layer.attention.self.key.in_features} output_features: {layer.attention.self.key.out_features}")
        print(f"  - Value: {layer.attention.self.value} input_features: {layer.attention.self.value.in_features} output_features: {layer.attention.self.value.out_features}")
        print(f"  - Attention output dense: {layer.attention.output.dense} input_features: {layer.attention.output.dense.in_features} output_features: {layer.attention.output.dense.out_features}")

        # Feed-forward network components
        print(f"  - Intermediate dense: {layer.intermediate.dense} input_features: {layer.intermediate.dense.in_features} output_features: {layer.intermediate.dense.out_features}")
        print(f"  - Output dense: {layer.output.dense} input_features: {layer.output.dense.in_features} output_features: {layer.output.dense.out_features}")

