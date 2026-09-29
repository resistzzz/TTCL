#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

LLM_PATH=PATH_TO_Qwen3-4B
MODEL_NAME=Qwen3-4B

TRAIN_FILES=data/math500/math500_placeholder_conf.parquet
VAL_FILES=data/math500/math500_placeholder_conf.parquet
DATA_NAME=MATH500

LR=5e-7
TRAIN_TEMP=0.7
TEST_TEMP=0.7
TEST_FREQ=10
SAVE_FREQ=30
TOTAL_EPOCHS=10
exec env LLM_PATH="${LLM_PATH}" MODEL_NAME="${MODEL_NAME}" DATA_NAME="${DATA_NAME}" \
    TRAIN_FILES="${TRAIN_FILES}" VAL_FILES="${VAL_FILES}" \
    LR="${LR}" TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=1 PPO_MICRO_BATCH_SIZE=2 VAL_N=1 TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
    TRAIN_TEMP="${TRAIN_TEMP}" TEST_TEMP="${TEST_TEMP}" \
    TEST_FREQ="${TEST_FREQ}" SAVE_FREQ="${SAVE_FREQ}" \
    bash "${SCRIPT_DIR}/run_majority_voting.sh"
