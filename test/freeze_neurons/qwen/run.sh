#ratio=0.00000000001
ratio=0.8
#ratio=0.0
cmd="torchrun --standalone --nproc_per_node=1 main.py --skip-ratio ${ratio}"
echo $cmd
eval $cmd