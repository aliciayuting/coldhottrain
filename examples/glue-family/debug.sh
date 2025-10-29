TASK=mnli
MODEL_NAME=roberta-base
SEED=0
SKIP_RATIO=0.8
CHANGE_ITERS=1
USE_LORA=true

OUTPUT_PATH=/share/desa/nfs02/shouxu/cold/runs/glue/$TASK/

# check if use lora is True
if [ "$USE_LORA" = true ] ; then
    RUN_NAME=DEBUG-$MODEL_NAME-lora--seed$SEED
# else if skip_ratio > 0
elif (( $(echo "$SKIP_RATIO > 0" | bc -l) )); then
    RUN_NAME=DEBUG-$MODEL_NAME-skipRatio$SKIP_RATIO-changeIters$CHANGE_ITERS-seed$SEED
else
    RUN_NAME=DEBUG-$MODEL_NAME-fp-seed$SEED
fi

RUN_NAME=DEBUG

LOGGING_PATH=$OUTPUT_PATH/runs/$RUN_NAME

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
    --save_steps 100 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 100 \
    --early_stopping True \
    --early_stopping_patience 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --seed $SEED \
    --report_to tensorboard \
    --output_dir $OUTPUT_PATH \
    --logging_dir $LOGGING_PATH \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False \
    --fp16 \
    --gradient_checkpointing True \
    --train_adapter False \
    | tee ./logs/$RUN_NAME.txt
#    --max_steps 5 \
# --adapter_config lora \


    # --skip_ratio $SKIP_RATIO \
    # --my_debug True \
    # --use_masked_skipgradient \
    # --change_iters $CHANGE_ITERS \


    # --use_lora $USE_LORA \
    # --lora_rank 32 \
    # --lora_scaling_factor 64 \


    

    # --lora_attn_matrices q_proj v_proj \
    
    


    #/share/desa/nfs02/cold/jamal-runs-benckmarking
    #--my_debug
 