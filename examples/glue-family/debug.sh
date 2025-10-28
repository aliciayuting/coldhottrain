TASK=mnli
MODEL_NAME=roberta-base
SEED=0
SKIP_RATIO=0.8
CHANGE_ITERS=1

# OUTPUT_PATH=/share/desa/nfs02/shouxu/cold/runs/glue/$TASK/$MODEL_NAME/$SKIP_RATIO/$SEED
OUTPUT_PATH=/share/desa/nfs02/shouxu/cold/runs/glue/$TASK/
# OUTPUT_PATH=$OUTPUT_BASE/$SKIP_RATIO/$SEED
# RUN_NAME=DEBUG-$MODEL_NAME-skipRatio$SKIP_RATIO-changeIters$CHANGE_ITERS-seed$SEED
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
    --logging_steps 1 \
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
    --logging_dir $LOGGING_PATH \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False \
    --fp16 \
    --max_steps 5


    # --use_masked_skipgradient \
    # --skip_ratio $SKIP_RATIO \
    # --change_iters $CHANGE_ITERS \
    


    #/share/desa/nfs02/cold/jamal-runs-benckmarking
    #--my_debug
 