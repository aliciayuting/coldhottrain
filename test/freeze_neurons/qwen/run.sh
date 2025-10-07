ratio=0.8
cmd="torchrun --standalone --nproc_per_node=4 main.py --skip-ratio ${ratio}"
echo $cmd
eval $cmd