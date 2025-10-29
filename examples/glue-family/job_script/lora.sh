#!/bin/bash
#SBATCH -J MNLI-lora                         # job name
#SBATCH -o /home/sl3343/coldhottrain/examples/glue-family/output/MNLI-lora-%j.out
#SBATCH -e /home/sl3343/coldhottrain/examples/glue-family/output/MNLI-lora-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sl3343@cornell.edu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --get-user-env
#SBATCH --mem=32000
#SBATCH -t 4:00:00
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

conda activate cold
cd /home/sl3343/coldhottrain/examples/glue-family/
./glue-lora.sh