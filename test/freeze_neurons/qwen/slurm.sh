#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus 4
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4646

# set up for problem & define any environment variables here


module load conda
module load nccl
conda activate cold
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

ratio=0.8
mode="1linear_efficient"
MODEL=Qwen/Qwen2.5-0.5B
DATASET="mnli"
gradient_checkpointing="true"
gradient_accumulation_steps=2

cmd="MODEL=${MODEL} DATASET=${DATASET} torchrun --standalone --nproc_per_node=4 /global/homes/l/lsx/jamal/coldhottrain-shouxu/test/freeze_neurons/qwen/main.py --skip-ratio ${ratio} --mode ${mode} --gradient-checkpointing ${gradient_checkpointing} --gradient-accumulation-steps ${gradient_accumulation_steps}"
echo $cmd
eval $cmd

#torchrun --standalone --nproc_per_node=4 /global/homes/l/lsx/jamal/coldhottrain/benchmark/qwen/finetune_qwen_glue.py
#| tee /global/homes/l/lsx/jamal/coldhottrain/benchmark/qwen/logs/neuron-logs.txt
# srun -n <num_mpi_processes> -c <cpus_per_task> a.out
