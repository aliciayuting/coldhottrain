#ratio=0.00000000001
ratio=0.8
mode="1linear_efficient"
MODEL=Qwen/Qwen2.5-0.5B
DATASET="mnli"
gradient_checkpointing="true"
gradient_accumulation_steps=2
random_swap_iters=200

cmd="MODEL=${MODEL} DATASET=${DATASET} torchrun --standalone --nproc_per_node=4 /global/homes/l/lsx/jamal/coldhottrain-shouxu/test/freeze_neurons/qwen/main.py --skip-ratio ${ratio} --mode ${mode} --gradient-checkpointing ${gradient_checkpointing} --gradient-accumulation-steps ${gradient_accumulation_steps} --random-swap-iters ${random_swap_iters}"
echo $cmd
eval $cmd