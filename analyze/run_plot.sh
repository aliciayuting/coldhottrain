# Analyze weight

# python analyze_weight_coldhot.py \
# --ckpt_root /pscratch/sd/l/lsx/runs/opt67b_fsdp_gsm8k/ \
# --ckpt_pattern "checkpoint-epoch-*" \
# --include_final \
# --outdir /pscratch/sd/l/lsx/yyt_runs/opt67b_fsdp_gsm8k/ \
# --which both --norm l2 \
# --hot_pct 95 --cold_pct 50 --per_layer_thresholds \
# --plot

# first pass
python analyze_weight_coldhot.py \
  --ckpt_root /pscratch/sd/l/lsx/runs/opt67b_fsdp_gsm8k/ \
  --ckpt_pattern "checkpoint-epoch-*" \
  --include_final \
  --outdir /pscratch/sd/l/lsx/yyt_runs/opt67b_fsdp_gsm8k/analysis \
  --which both --norm l2 \
  --hot_pct 95 --cold_pct 50 --per_layer_thresholds \
  --plot \
  --pack_tensors


# rerun
python analyze_weight_coldhot.py \
  --ckpt_root /pscratch/sd/l/lsx/runs/opt67b_fsdp_gsm8k/\
  --ckpt_pattern "checkpoint-epoch-*" \
  --include_final \
  --outdir /pscratch/sd/l/lsx/yyt_runs/opt67b_fsdp_gsm8k/analysis \
  --which both --norm l2 \
  --hot_pct 80 --cold_pct 50 --per_layer_thresholds \
  --reuse_if_exists --pack_tensors --plot

# plot
python plot_coldhot_timelines.py \
  --analysis_dir /pscratch/sd/l/lsx/yyt_runs/opt67b_fsdp_gsm8k/analysis \
  --component mlp_fc1 \
  --layer 10 \
  --top_k 256 \
  --sort hot_freq \
  --out /path/to/run/analysis/plots/mlp_fc1_L10_timeline.png
