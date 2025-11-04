#ratio=0.00000000001
run_name="0.8-random-noswap"
ratio=0.8
random_swap_iters=10000000
mode="1linear"
MODEL=Qwen/Qwen2.5-0.5B
DATASET="mnli"
gradient_checkpointing="true"
per_device_train_batch_size=32
gradient_accumulation_steps=1
logging_steps=100
eval_steps=250
elementwise_linear="true"
elementwise_swap_scheme="input"
preselect_file="/share/desa/nfs02/cold/jamal-runs-benckmarking/Qwen_Qwen2.5-0.5B-mnli/0.0/preselect_grads-10p.json"
cmd="MODEL=${MODEL} DATASET=${DATASET} torchrun --standalone --nproc_per_node=1 /home/jah649/coldhottrain/test/freeze_neurons/qwen/main.py  \
--skip-ratio ${ratio} --mode ${mode} --gradient-checkpointing ${gradient_checkpointing} --gradient-accumulation-steps ${gradient_accumulation_steps} \
--per-device-train-batch-size ${per_device_train_batch_size} --random-swap-iters ${random_swap_iters} --logging-steps ${logging_steps} --eval-steps ${eval_steps} \
--elementwise-linear ${elementwise_linear} --elementwise-swap-scheme ${elementwise_swap_scheme} --preselect-file ${preselect_file} --run-name ${run_name}"
echo $cmd
eval $cmd