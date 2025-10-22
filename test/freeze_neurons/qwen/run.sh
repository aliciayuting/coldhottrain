#ratio=0.00000000001
ratio=0.0
random_swap_iters=1000000
mode="1linear_efficient"
MODEL=Qwen/Qwen2.5-0.5B
DATASET="gsm8k"
gradient_checkpointing="true"
per_device_train_batch_size=8
gradient_accumulation_steps=1
logging_steps=10
eval_steps=5000

cmd="MODEL=${MODEL} DATASET=${DATASET} torchrun --standalone --nproc_per_node=4 /global/homes/l/lsx/jamal/coldhottrain-memory/test/freeze_neurons/qwen/main.py  \
--skip-ratio ${ratio} --mode ${mode} --gradient-checkpointing ${gradient_checkpointing} --gradient-accumulation-steps ${gradient_accumulation_steps} \
--per-device-train-batch-size ${per_device_train_batch_size} --random-swap-iters ${random_swap_iters} --logging-steps ${logging_steps} --eval-steps ${eval_steps}"
echo $cmd
eval $cmd