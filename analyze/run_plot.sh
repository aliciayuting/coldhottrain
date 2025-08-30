# Analyze weight

  python analyze_weight_hotcold.py \
    --ckpt_root /pscratch/sd/l/lsx/runs/opt67b_fsdp_gsm8k/ \
    --ckpt_pattern "checkpoint-epoch-*" \
    --include_final \
    --outdir /pscratch/sd/l/lsx/yyt_runs/opt67b_fsdp_gsm8k/ \
    --which both --norm l2 \
    --hot_pct 95 --cold_pct 50 --per_layer_thresholds \
    --plot