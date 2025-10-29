#!/bin/bash
#SBATCH -J MNLI-FP                         # job name
#SBATCH -o /home/sl3343/coldhottrain/examples/glue-family/output/MNLI-FP_%j.out                  # output file (%j expands to jobID)
#SBATCH -e /home/sl3343/coldhottrain/examples/glue-family/output/MNLI-FP_%j.err                  # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                     # Request status by email
#SBATCH --mail-user=sl3343@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32000                           # server memory requested (per node)
#SBATCH -t 4:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition       # Request partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1                  # Type/number of GPUs needed

conda activate cold
cd /home/sl3343/coldhottrain/examples/glue-family/
./glue.sh