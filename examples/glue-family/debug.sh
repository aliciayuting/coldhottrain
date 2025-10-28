RUN_NAME=full-ft-2e-5
TASK=mnli
MODEL_NAME=roberta-base
SEED=0

OUTPUT_PATH=/share/desa/nfs02/cold/jamal-runs-benckmarking/$RUN_NAME/$TASK/$MODEL_NAME/debug


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
    --num_train_epochs 5 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 500 \
    --early_stopping True \
    --early_stopping_patience 5 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --seed $SEED \
    --report_to tensorboard \
    --output_dir $OUTPUT_PATH \
    --logging_dir $OUTPUT_PATH/logs \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False \
    --skip_ratio 0.1 \
    --fp16 \
    --change_iters 100 \
    --use_masked_skipgradient 
    #--use_masked_skipgradient \
    #--change_iters 0 \
    #--fp16 \
    #/share/desa/nfs02/cold/jamal-runs-benckmarking
    #--my_debug
 