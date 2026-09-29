#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLM_PATH="${LLM_PATH:-merged_hf_model/Calibration/RLCR_AdvNoStdNorm_Qwen3-4B_DAPO14k/global_step_330}"
MODEL_NAME="${MODEL_NAME:-RLCR-Qwen3-4B-Epoch3}"

TARGET_MODE="${TARGET_MODE:-frequency}"
ENTROPY_SHRINKAGE_RHO="${ENTROPY_SHRINKAGE_RHO:-0.5}"
PRIOR_EMA_BETA="${PRIOR_EMA_BETA:-0.95}"
# online: current rollout support (legacy); epoch_ema: frozen q_0 plus epoch-lagged EMA targets.
TARGET_CACHE_MODE="${TARGET_CACHE_MODE:-epoch_ema}"
TARGET_CACHE_EMA_RATE="${TARGET_CACHE_EMA_RATE:-0.01}"
LOSS_MASK_MODE="${LOSS_MASK_MODE:-confidence}"
PSEUDO_LABEL_SOURCE="${PSEUDO_LABEL_SOURCE:-target_cache_sample}"



exec env LLM_PATH="${LLM_PATH}" MODEL_NAME="${MODEL_NAME}" DATA_NAME=CSimpleQA \
    TRAIN_FILES=data/csimpleqa/csimpleqa_answer_tag_placeholder_conf.parquet \
    VAL_FILES=data/csimpleqa/csimpleqa_answer_tag_placeholder_conf.parquet \
    TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=32 PPO_MICRO_BATCH_SIZE=16 VAL_N=1 \
    LR=5e-7 TRAIN_TEMP=0.7 TEST_TEMP=0.7 TOTAL_EPOCHS=10 TEST_FREQ=30 SAVE_FREQ=30 \
    TARGET_MODE="${TARGET_MODE}" ENTROPY_SHRINKAGE_RHO="${ENTROPY_SHRINKAGE_RHO}" PRIOR_EMA_BETA="${PRIOR_EMA_BETA}" TARGET_CACHE_MODE="${TARGET_CACHE_MODE}" TARGET_CACHE_EMA_RATE="${TARGET_CACHE_EMA_RATE}" LOSS_MASK_MODE="${LOSS_MASK_MODE}" \
    PSEUDO_LABEL_SOURCE="${PSEUDO_LABEL_SOURCE}" \
    bash "${SCRIPT_DIR}/run_majority_ttcl.sh"
