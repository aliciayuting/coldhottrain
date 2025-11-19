# TASK=mnli
TASK=$1
RUN_NAME_APPEND=fused-false
MODEL_NAME=roberta-base
SEED=0
# SKIP_RATIO=0.92
# CHANGE_ITERS=100
SKIP_RATIO=$2
CHANGE_ITERS=$3
LR=$4
EPOCH=$5
BATCH_SIZE=$6
KEEP_STATE=$7
USE_ELEMENTWISE=$8
SCHEME=$9
PRESELECT_FILE=${10}
USE_LORA=false

OUTPUT_BASE=/share/desa/nfs02/cold/runs/glue/$TASK

# check if use lora is True
if [ "$USE_LORA" = true ] ; then
    RUN_NAME=$MODEL_NAME-lora-seed$SEED-lr$LR
# else if skip_ratio > 0
elif (( $(echo "$SKIP_RATIO > 0" | bc -l) )); then
    RUN_NAME=$MODEL_NAME-skipRatio$SKIP_RATIO-changeIters$CHANGE_ITERS-seed$SEED-lr$LR-bs$BATCH_SIZE-keep$KEEP_STATE
    if [ "$USE_ELEMENTWISE" = true ] ; then
        RUN_NAME=$RUN_NAME-elemwise-$SCHEME-epochs$EPOCH
    else
        RUN_NAME=$RUN_NAME-colwise-use_masked_skipgradient-randomswap-epochs$EPOCH
    fi
else
    RUN_NAME=$MODEL_NAME-fp-seed$SEED-lr$LR-bs$BATCH_SIZE-qvclassifier
fi

# RUN_NAME=$MODEL_NAME-fp-seed$SEED-lr$LR-bs$BATCH_SIZE-qvclassifier
RUN_NAME=$RUN_NAME-$RUN_NAME_APPEND
OUTPUT_PATH=$OUTPUT_BASE/runs/$RUN_NAME
LOGGING_PATH=$OUTPUT_BASE/tensorboard_logs/$RUN_NAME

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python main.py \
    --model-name roberta-base \
    --task-name mnli \
    --do-train True \
    --do-eval True \
    --max_seq_length 128 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --dataloader_num_workers 0 \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 500 \
    --seed $SEED \
    --report_to tensorboard \
    --output_dir $OUTPUT_PATH \
    --logging_dir $LOGGING_PATH \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False \
    --fp16 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --gradient_checkpointing True \
    --skip_ratio $SKIP_RATIO \
    --change_iters $CHANGE_ITERS \
    --use_masked_skipgradient True \