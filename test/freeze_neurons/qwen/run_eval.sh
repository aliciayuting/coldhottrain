# nproc_per_node = number of GPUs you want to use
torchrun --nproc_per_node 4 eval_gsm8k.py \
  --ddp \
  --model_path /pscratch/sd/l/lsx/jamal-runs-benckmarking/Qwen_Qwen2.5-0.5B-gsm8k/0.0/ckpt/checkpoint-700 \
  --dataset_name gsm8k --dataset_config main --split test \
  --max_samples 200 \
  --batch_size 8 --max_new_tokens 512 \
  --output_file ddp_results.json

