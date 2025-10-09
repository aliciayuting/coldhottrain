#ratio=0.00000000001
ratio=0.8
mode="1linear_efficient"
MODEL=Qwen/Qwen2.5-0.5B
gradient_checkpointing="false"
gradient_accumulation_steps=2

cmd="MODEL=${MODEL} torchrun --standalone --nproc_per_node=1 main.py --skip-ratio ${ratio} --mode ${mode} --gradient-checkpointing ${gradient_checkpointing} --gradient-accumulation-steps ${gradient_accumulation_steps}"
echo $cmd
eval $cmd