import json
import logging
import os
from collections import OrderedDict

from arguments import get_args
from model.utils import TaskType, get_model
from tasks.glue.dataset import GlueDataset
# from tasks.humset.dataset import HumsetDataset
from tasks.superglue.dataset import SuperGlueDataset
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS
from torchinfo import summary
from training.utils import get_default_args
from transformers import (
    # AdapterTrainer,
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
)
from adapters import AdapterTrainer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../benchmark/qwen/")))
from custom_adam import MaskedAdamW
from skip_gradient_callback import SkipGradientCallback
from probe2 import VramBreakdownCallback
from model.utils import fix_linear_modules
# from transformers.adapters.configuration import AdapterConfig, PfeifferConfig
from adapters.training import setup_adapter_training
import adapters
from adapters.wrappers.interfaces import get_adapter_interface
from adapters import AdapterModelInterface
from adapters import LoRAConfig


logger = logging.getLogger(__name__)


ROBERTA_INTERFACE = AdapterModelInterface(
    adapter_methods=[ "lora"],
    model_embeddings="embeddings",
    model_layers="encoder.layer",
    layer_self_attn="attention",
    layer_cross_attn=None,
    attn_q_proj="self.query",
    attn_k_proj="self.key",
    attn_v_proj="self.value",
    attn_o_proj="output.dense",
    layer_intermediate_proj="intermediate.dense",
    layer_output_proj="output.dense",
    layer_pre_self_attn=None,
    layer_pre_cross_attn=None,
    layer_pre_ffn=None,
    layer_ln_1="attention.output.LayerNorm",
    layer_ln_2="output.LayerNorm",
)


def get_trainer(args):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        # fusion_args,
        # mtl_2_args,
        coldneuron_args,
    ) = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.task_name.lower() in GLUE_DATASETS:
        dataset = GlueDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() in SUPERGLUE_DATASETS:
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)
    # elif data_args.dataset_name == "humset":
    #     dataset = HumsetDataset(tokenizer, data_args, training_args)
    logger.info(dataset.train_dataset if training_args.do_train else None, dataset.eval_dataset if training_args.do_train else None, dataset.test_dataset if training_args.do_eval else None)
    print("IS MULTIPLE CHOICE: ",dataset.multiple_choice)
    if not dataset.is_regression and not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # if data_args.dataset_name == "humset":
    #     assert False
    #     config.problem_type = "multi_label_classification"

    # # ProPETL related args
    # config.share_adapter = model_args.share_adapter
    # config.sparsity = model_args.sparsity

    if not dataset.multiple_choice:
        model = get_model(
            args=args, task_type=TaskType.SEQUENCE_CLASSIFICATION, config=config
        )
    else:
        model = get_model(args=args, task_type=TaskType.MULTIPLE_CHOICE, config=config)


    # config lora using adapters lib
    if adapter_args.train_adapter:
        assert coldneuron_args.skip_ratio == 0 and not coldneuron_args.use_masked_skipgradient, "Cannot use both LoRA and ColdNeurons at the same time."
        ''' Use adapter auto model, add head manually '''
        # if dataset.multiple_choice:
        #     model.add_multiple_choice_head(data_args.task_name, num_choices=2)
        # else:
        #     if data_args.dataset_name == "humset":
        #         multi_label = True
        #     else:
        #         multi_label = False
        #     model.add_classification_head(
        #         data_args.task_name,
        #         num_labels=dataset.num_labels,
        #         id2label={i: v for i, v in enumerate(dataset.label_list)}
        #         if not dataset.is_regression
        #         else None,
        #         layers=model_args.head_n_layers
        #         if model_args.head_n_layers
        #         else get_default_args(model.add_classification_head)["layers"],
        #         multilabel=multi_label,
        #     )

        # model.base_model.prefix_tuning = False
        # print("disabled support_prompt_tuning")
        ''' Use huggingface model directly, setup adapter training '''

        # print(f"roberta_interface none: {roberta_interface is None}")
        adapters.init(model, interface=ROBERTA_INTERFACE)
        # adapters.init(model)
        print("~~~~~~~  Initialized adapters with RoBERTa interface.  ~~~~~~~")
        
        

        # Setup adapters
        # if not data_args.omega_grid:
        if True:
            # setup_adapter_training(
            #     model,
            #     adapter_args,
            #     data_args.task_name,
            #     # for propetl
            #     adapter_config_kwargs={
            #         "sparsity": model_args.sparsity,
            #         "share_adapter": model_args.share_adapter,
            #         "r": 32,
            #         "alpha": 64,
            #     },
            # )

            # # print the lora config
            # for adapter_name in model.active_adapters:
            #     adapter_config = model.get_adapters_config(adapter_name)
            #     print(f"Adapter config for {adapter_name}: {adapter_config}")

            # adapter_config = LoRAConfig(
            #     r=32,
            #     alpha=64,
            #     dropout=0.1,
            #     # Add these if you're using them:
            #     # init_weights="bert",
            # )
            
            # # Add the adapter
            # model.add_adapter(
            #     data_args.task_name,
            #     config=adapter_config,
            #     set_active=True  # ← THIS IS KEY!
            # )
            adapter_config = LoRAConfig(
                r=32,
                alpha=64,
                dropout=0.1,
                # Add these if you're using them:
                # init_weights="bert",
            )
            
            # Add the adapter
            model.add_adapter(
                data_args.task_name,
                config=adapter_config,
                set_active=True  # ← THIS IS KEY!
            )
            
            # Train the adapter (this freezes the base model and activates adapters)
            model.train_adapter(data_args.task_name)
            
            # CRITICAL: Verify activation
            print(f"✓ Active adapters: {model.active_adapters}")
            print(f"✓ Adapter setup: {model.adapters_config.active_setup}")
            
            if model.active_adapters is None:
                raise RuntimeError("FAILED TO ACTIVATE ADAPTERS!")

    param_optimizer = list(model.named_parameters())
    logger.info("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")

    trainer_cls = (
        AdapterTrainer
        if (adapter_args.train_adapter)
        else Trainer
    )
    

    # early stopping
    if model_args.early_stopping:
        logger.info(
            "Early stopping is enabled with patience %d",
            model_args.early_stopping_patience,
        )
        early_stopping_callback = [
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience
            )
        ]
    else:
        early_stopping_callback = []

    # for name, param in model.named_parameters():
    #     if (("query" in name or "value" in name) and 'lora' in name) or "classifier" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    for name, param in model.named_parameters():
        # if param.requires_grad:
        print(f"Param: {name}, Numel: {param.numel()}, shape: {param.shape}, Requires grad: {param.requires_grad}")
    logger.info(summary(model, depth=5))

    # if coldneuron_args.skip_ratio > 0 and not coldneuron_args.use_masked_skipgradient and not adapter_args.train_adapter:
    #     print(f"***** using skipgradient with ratio {coldneuron_args.skip_ratio} *****")
    #     fix_linear_modules(model, config, coldneuron_args.skip_ratio)


    if coldneuron_args.use_masked_skipgradient:

        for name, param in model.named_parameters():
            if not ("query" in name or "value" in name or "classifier" in name):
                param.requires_grad = False

            print(f"Param: {name}, Numel: {param.numel()}, shape: {param.shape}, Requires grad: {param.requires_grad}")

        # # from torch.optim import AdamW
        # optimizer = MaskedAdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()), 
        # )
        print("***** using masked skipgradient *****")
        opt_kwargs = {
            "mask_dict": {},
            "named_parameters": dict(model.named_parameters()),
            "freeze_state": "zero", 
            "lr": training_args.learning_rate,
            "fused": True,
        }
        skipgradient_cb = SkipGradientCallback(
            model=model,
            zero_mode="neurons",
            output_dir=training_args.output_dir,
            mode="random",
            random_hot_k_percent=1-coldneuron_args.skip_ratio,
            change_random_every_iters=coldneuron_args.change_iters,
        )
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            callbacks=early_stopping_callback,
            optimizer_cls_and_kwargs=(MaskedAdamW, opt_kwargs),
        )
        trainer.create_optimizer()
        trainer.add_callback(skipgradient_cb)
    else:
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            callbacks=early_stopping_callback,
            # optimizers=(optimizer, None),
        )
        trainer.create_optimizer()

    
    vram_breakdown_callback = VramBreakdownCallback()
    trainer.add_callback(vram_breakdown_callback)

    opt = trainer.optimizer
    for i, g in enumerate(opt.param_groups):
        print(f"Group {i}:")
        for k, v in g.items():
            if k != "params":
                print(f"  {k}: {v}")

    # return trainer, model, dataset, adapter_setup
    return trainer, model, dataset, None
    # return None, None, dataset, None
