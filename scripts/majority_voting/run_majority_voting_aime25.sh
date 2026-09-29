#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

LLM_PATH=PATH_TO_Qwen3-4B
MODEL_NAME=Qwen3-4B

LR=5e-7
TRAIN_TEMP=0.7
TEST_TEMP=0.7
TEST_FREQ=10
SAVE_FREQ=30
TOTAL_EPOCHS=40
exec env LLM_PATH="${LLM_PATH}" MODEL_NAME="${MODEL_NAME}" DATA_NAME=AIME25 \
    TRAIN_FILES=data/aime25/aime25_placeholder_conf.parquet VAL_FILES=data/aime25/aime25_placeholder_conf.parquet \
    LR="${LR}" TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=1 PPO_MICRO_BATCH_SIZE=2 VAL_N=8 TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
    TRAIN_TEMP="${TRAIN_TEMP}" TEST_TEMP="${TEST_TEMP}" \
    TEST_FREQ="${TEST_FREQ}" SAVE_FREQ="${SAVE_FREQ}" \
    bash "${SCRIPT_DIR}/run_majority_voting.sh"
