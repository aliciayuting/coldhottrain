#ratio=0.00000000001
category_name="debug"
#run_name="0.8-preselect"
run_name="baseline-128batch"
#run_name="0.8-colwise-100iters"
ratio=0.0
random_swap_iters=100
mode="1linear"
MODEL=Qwen/Qwen2.5-0.5B
DATASET="mnli"
max_length=128
gradient_checkpointing="true"
per_device_train_batch_size=128
gradient_accumulation_steps=1
logging_steps=100
eval_steps=250
elementwise_linear="true"
elementwise_swap_scheme="preselect"
preselect_file="/share/desa/nfs02/cold/jamal-runs-benckmarking/Qwen_Qwen2.5-0.5B-mnli/0.0/preselect_grads-10p.json"
dump_grads="false"


cmd="MODEL=${MODEL} DATASET=${DATASET} MAX_LENGTH=${max_length} torchrun --standalone --nproc_per_node=1 /home/jah649/coldhottrain/test/freeze_neurons/qwen/main.py  \
--skip-ratio ${ratio} --mode ${mode} --gradient-checkpointing ${gradient_checkpointing} --gradient-accumulation-steps ${gradient_accumulation_steps} \
--per-device-train-batch-size ${per_device_train_batch_size} --random-swap-iters ${random_swap_iters} --logging-steps ${logging_steps} --eval-steps ${eval_steps} \
--elementwise-linear ${elementwise_linear} --elementwise-swap-scheme ${elementwise_swap_scheme} --preselect-file ${preselect_file} --dump-grads ${dump_grads} \
--category-name ${category_name} --run-name ${run_name}"
echo $cmd
eval $cmd