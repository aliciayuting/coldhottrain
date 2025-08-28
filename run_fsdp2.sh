#!/usr/bin/env bash
#SBATCH -J fsdp2-opt17b
#SBATCH -A <your_account>          # TODO: set your project/allocation
#SBATCH -p gpu                     # e.g., 'gpu', 'ampere', 'A100'; adjust to your cluster
#SBATCH -N 1
#SBATCH --gpus-per-node=4          # change to 1/2/8 as needed
#SBATCH --cpus-per-task=16         # dataloader / tokenization threads
#SBATCH -t 06:00:00                # walltime
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

#############################
# User config
#############################
MODEL_ID="facebook/opt-6.7b"       # or facebook/opt-13b if you have VRAM for it
OUTDIR="${SCRATCH:-$PWD}/yyt_runs/opt6b_fsdp_gsm8k"
SEED=42

EPOCHS=1
SEQ_LEN=1024
PER_DEV_BATCH=1
GRAD_ACCUM=8

LR=2e-5
WD=0.05
WARMUP_RATIO=0.03
MAX_TRAIN_SAMPLES=2000

# hot/cold logging cadence
LOG_EVERY=50
SNAP_EVERY=200

# optional: unfreeze window (we currently train all params in fsdp2)
UNFREEZE_LAST_N=12
CLIP_NORM=1.0

#############################
# Env & caches (HPC-friendly)
#############################
mkdir -p "$OUTDIR" logs
SCRATCH_BASE="${SCRATCH:-$PWD}"
mkdir -p "$SCRATCH_BASE/yyt_tmp" "$SCRATCH_BASE/.cache/pip" "$SCRATCH_BASE/.cache/huggingface"
export yyt_tmpDIR="$SCRATCH_BASE/yyt_tmp"
export PIP_CACHE_DIR="$SCRATCH_BASE/.cache/pip"
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# If you use conda/venv, activate it here:
# module load cuda/12.1           # if your cluster uses modules
# source ~/.conda/envs/cold/bin/activate
# or: source /path/to/venv/bin/activate

# (Optional but often helpful on multi-GPU nodes)
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

#############################
# Sanity checks
#############################
if [[ ! -f "finetune_fsdp2.py" ]]; then
  echo "ERROR: finetune_fsdp2.py not found in $(pwd)"; exit 1;
fi

#############################
# Run (single node, torchrun)
#############################
# NOTE: --standalone uses localhost rendezvous for 1 node.
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

echo "[*] Launching torchrun with ${GPUS_PER_NODE} GPUs on 1 node"
torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" finetune_fsdp2.py \
  --model "$MODEL_ID" \
  --outdir "$OUTDIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --seq_len "$SEQ_LEN" \
  --per_device_batch "$PER_DEV_BATCH" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --wd "$WD" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --log_every "$LOG_EVERY" \
  --snap_every "$SNAP_EVERY" \
  --unfreeze_last_n "$UNFREEZE_LAST_N" \
  --clip_norm "$CLIP_NORM"

echo "[✓] Done. Check outputs under: $OUTDIR"