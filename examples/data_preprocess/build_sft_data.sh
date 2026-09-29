python \
    examples/data_preprocess/build_correct_sft_warmup.py \
    --model-path Path_to_Qwen3-8B \
    --source-data-path data/dapo17k/train_no_conf.parquet \
    --target-prompt-data-path data/dapo17k/train_placeholder_conf.parquet \
    --output-dir data/sft/qwen38b_dapo17k_sft_warmup \
    --num-correct 2100 \
    --num-incorrect 0 \
    --val-size 100 \
    --rollout-n 8 \
    --batch-size 256 \
    --temperature 0.7 \
    --max-tokens 4096 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code



