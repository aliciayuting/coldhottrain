#!/usr/bin/env bash
set -euo pipefail

#############################
# Config (edit as you like) #
#############################
MODEL_ID="facebook/opt-13b"
OUTDIR="$SCRATCH/runs/opt13b_hotcold"
SEED=42
EPOCHS=1
SEQ_LEN=2048
PER_DEV_BATCH=1
GRAD_ACCUM=8
UNFREEZE_N=12
TARGETS="mlp"            # choose: mlp | mha | mlp,mha
MAX_TRAIN_SAMPLES=2000
LOG_EVERY=50
SNAP_EVERY=200


#############################
# Env & caches (HPC-friendly)
#############################
# Prefer scratch/project space to avoid $HOME quota issues
SCRATCH_BASE="${SCRATCH:-$PWD}"
# ENV_DIR="$SCRATCH_BASE/envs/opt13b_hotcold"

mkdir -p "$OUTDIR" "$SCRATCH_BASE/tmp" "$SCRATCH_BASE/.cache/pip" "$SCRATCH_BASE/.cache/huggingface"
export TMPDIR="$SCRATCH_BASE/tmp"
export PIP_CACHE_DIR="$SCRATCH_BASE/.cache/pip"
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# #############################
# # Python environment (venv) #
# #############################

# if [[ ! -d "$ENV_DIR" ]]; then
#   echo "[*] Creating venv at $ENV_DIR"
#   python$PYTHON_VER -m venv "$ENV_DIR"
# fi
# # shellcheck disable=SC1090
# source "$ENV_DIR/bin/activate"
# python -m pip install --upgrade pip setuptools wheel

# #############################
# # Install dependencies      #
# #############################
# # Python + Torch versions
# PYTHON_VER="3.10"
# # Choose one CUDA wheel line: cu121 (CUDA 12.1) or cu118 (CUDA 11.8)
# TORCH_INSTALL="pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121"
# echo "[*] Installing PyTorch"
# eval "$TORCH_INSTALL"

# echo "[*] Installing Hugging Face + utils"
# pip install "transformers>=4.40.0" "accelerate>=0.30.0" datasets tqdm numpy matplotlib

#############################
# Files check               #
#############################
if [[ ! -f "finetune_opt_checkpoint.py" ]]; then
  echo "ERROR: finetune_opt_checkpoint.py not found in current directory."
  echo "Place the training script here, or adjust paths."
  exit 1
fi

if [[ ! -f "plot_hotcold.py" ]]; then
  echo "WARNING: plot_hotcold.py not found; plotting step will be skipped."
fi

#############################
# Train                     #
#############################
echo "[*] Starting fine-tune run"
python finetune_opt_checkpoint.py \
  --model "$MODEL_ID" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --seq_len "$SEQ_LEN" \
  --per_device_batch "$PER_DEV_BATCH" \
  --grad_accum "$GRAD_ACCUM" \
  --lr 2e-5 \
  --wd 0.05 \
  --warmup_ratio 0.03 \
  --unfreeze_n "$UNFREEZE_N" \
  --targets "$TARGETS" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --log_every "$LOG_EVERY" \
  --snap_every "$SNAP_EVERY" \
  --outdir "$OUTDIR"

# #############################
# # Plot                     #
# #############################
# if [[ -f "plot_hotcold.py" ]]; then
#   echo "[*] Plotting latest snapshot"
#   python plot_hotcold.py --dir "$OUTDIR" --g0 0 --g1 1 --tau 1e-5
#   echo "[*] Plots written under: $OUTDIR"
# fi

# echo "[✓] Done."