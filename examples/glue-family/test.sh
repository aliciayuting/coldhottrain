RUN_NAME=full-ft-2e-5
TASK=mnli
MODEL_NAME=roberta-base
SEED=0

OUTPUT_PATH=/share/desa/nfs02/shouxu/cold/runs/$RUN_NAME/$TASK/$MODEL_NAME/$SEED


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model-name roberta-base \
    --task-name $TASK \
    --do-train True \
    --do-eval True \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 0 \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --eval_steps 1 \
    --early_stopping True \
    --early_stopping_patience 5 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --seed $SEED \
    --report_to tensorboard \
    --output_dir $OUTPUT_PATH \
    --logging_dir $OUTPUT_PATH/logs \
    --fp16 \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False


    
 