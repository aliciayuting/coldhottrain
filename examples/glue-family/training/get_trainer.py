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
# from probe2 import VramBreakdownCallback
from probe3 import VramBreakdownCallback
from model.utils import fix_linear_modules
# from transformers.adapters.configuration import AdapterConfig, PfeifferConfig
from adapters.training import setup_adapter_training
import adapters
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


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
        # print("***** Using Adapters LoRA finetuning *****")
        # ''' Use adapter auto model, add head manually '''
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

        # ''' Use huggingface model directly, setup adapter training '''
        # adapters.init(model)

        # Setup adapters
        # if not data_args.omega_grid:
        # if True:
        if False:
            setup_adapter_training(
                model,
                adapter_args,
                data_args.task_name,
                # for propetl
                adapter_config_kwargs={
                    "sparsity": model_args.sparsity,
                    "share_adapter": model_args.share_adapter,
                    # "r": 32,
                    "alpha": 64,
                },
            )

            # # print the lora config
            # for adapter_name in model.active_adapters:
            #     adapter_config = model.get_adapters_config(adapter_name)
            #     print(f"Adapter config for {adapter_name}: {adapter_config}")



    # trainer_cls = (
    #     AdapterTrainer
    #     if (adapter_args.train_adapter)
    #     else Trainer
    # )


    trainer_cls = Trainer
    

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
    


    if coldneuron_args.skip_ratio > 0 and not coldneuron_args.use_masked_skipgradient:
        print(f"***** using colwise with ratio {coldneuron_args.skip_ratio} *****")
        model.enable_input_require_grads()
        total_buffers = sum(b.numel() for b in model.buffers())
        print(f"Total buffers before adding linearcolwise: {total_buffers}")
        fix_linear_modules(model, config, coldneuron_args.skip_ratio)
        after_total_buffers = sum(b.numel() for b in model.buffers())
        print(f"Total buffers after adding linearcolwise: {after_total_buffers}, {after_total_buffers - total_buffers} added.")

        



    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Param: {name}, Numel: {param.numel()}, shape: {param.shape}, Requires grad: {param.requires_grad}")


    if coldneuron_args.use_masked_skipgradient:

        for name, param in model.named_parameters():
            if not ("query" in name or "value" in name or "classifier" in name):
                param.requires_grad = False

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
        # optimizer = AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()), 
        #     lr=training_args.learning_rate,
        #     eps=training_args.adam_epsilon,
        #     weight_decay=training_args.weight_decay,

        # )

        # lr_scheduler = trainer_cls.get_lr_scheduler(
        #     training_args.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=training_args.get_warmup_steps(),
        #     num_training_steps=trainer_cls.get_train_steps(
        #         training_args,
        #         trainer_cls.get_train_dataloader(
        #             trainer_cls(
        #                 model=model,
        #                 args=training_args,
        #                 train_dataset=dataset.train_dataset
        #                 if training_args.do_train
        #                 else None,
        #             )
        #         ),
        #     ),
        # )

        # print(f"len of filtered parameters: {len(list(filter(lambda p: p.requires_grad, model.parameters())))}")

        # print optimizer name
        # for param_group in optimizer.param_groups:
        #     print("Optimizer param group:")
        #     for k, v in param_group.items():
        #         # if k != "params":
        #         print(f"  {k}: {v}")


        if adapter_args.train_adapter:
            model.can_return_loss = True

        # optim = AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=training_args.learning_rate,
        #     eps=training_args.adam_epsilon,
        #     weight_decay=training_args.weight_decay,
        #     betas=(training_args.adam_beta1, training_args.adam_beta2),
        # )


        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optim,
        #     num_warmup_steps=0,
        #     num_training_steps=33135,
        # )


        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            callbacks=early_stopping_callback,
            # optimizers=( optim, lr_scheduler),
        )

        trainer.create_optimizer_and_scheduler(num_training_steps=33135)


    

    vram_breakdown_callback = VramBreakdownCallback()
    trainer.add_callback(vram_breakdown_callback)

    opt = trainer.optimizer
    print(f"optimizer type: {type(opt)}")
    for i, g in enumerate(opt.param_groups):
        print(f"Group {i}:")
        for k, v in g.items():
            if k != "params":
                print(f"  {k}: {v}")

    # return trainer, model, dataset, adapter_setup
    return trainer, model, dataset, None
    # return None, None, dataset, None
