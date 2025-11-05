#!/bin/bash
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32000                           # server memory requested (per node)
#SBATCH -t 8:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition       # Request partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1                  # Type/number of GPUs needed

conda activate cold
cd /home/jah649/coldhottrain/test/freeze_neurons/qwen/
./run.sh