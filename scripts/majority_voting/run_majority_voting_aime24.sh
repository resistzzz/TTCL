#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLM_PATH=PATH_TO_Qwen3-4B
MODEL_NAME=Qwen3-4B
exec env LLM_PATH="${LLM_PATH}" MODEL_NAME="${MODEL_NAME}" DATA_NAME=AIME24 \
    TRAIN_FILES=data/aime24/aime24_placeholder_conf.parquet VAL_FILES=data/aime24/aime24_placeholder_conf.parquet \
    TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE=8 VAL_N=8 \
    bash "${SCRIPT_DIR}/run_majority_voting.sh"
