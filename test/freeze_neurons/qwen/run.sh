ratio=0.2
cmd="torchrun --standalone --nproc_per_node=4 main.py --skip-ratio ${ratio} | tee output/logs/qwen2.5-0.5b-sst2-skip-${ratio}.log"
echo $cmd
eval $cmd