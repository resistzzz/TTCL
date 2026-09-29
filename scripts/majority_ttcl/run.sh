#!/usr/bin/env bash
# Run the listed experiments serially.  Each child training script must finish
# successfully before the next one is launched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LLM_PATH="${LLM_PATH:-path_to_Qwen3-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"

# LLM_PATH="${LLM_PATH:-path_to_Qwen3-8B}"
# MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"

# LLM_PATH="${LLM_PATH:-path_to_Llama-3.1-8B-Instruct}"
# MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"

# LLM_PATH="${LLM_PATH:-path_to_RLCR-Qwen3-8B-DAPO14K}"
# MODEL_NAME="${MODEL_NAME:-RLCR-Qwen3-8B}"

# LLM_PATH="${LLM_PATH:-path_to_RLCR-Qwen3-4B-DAPO14K}"
# MODEL_NAME="${MODEL_NAME:-RLCR-Qwen3-4B}"

# LLM_PATH="${LLM_PATH:-path_to_RLCR-Qwen3-4B-HotpotTrivia10K}"
# MODEL_NAME="${MODEL_NAME:-RLCR-Qwen3-4B-HotTrivia10K}"



ENTROPY_SHRINKAGE_RHO="${ENTROPY_SHRINKAGE_RHO:-0.5}"
PRIOR_EMA_BETA="${PRIOR_EMA_BETA:-0.95}"
# online: use the current rollout group's support immediately (legacy).
# epoch_ema: precompute frozen q_0, then update each question target only at
# epoch boundaries using TARGET_CACHE_EMA_RATE.  Set the rate to 0.0 for a
# completely frozen confidence target.
TARGET_CACHE_MODE="${TARGET_CACHE_MODE:-epoch_ema}"
TARGET_CACHE_EMA_RATE="${TARGET_CACHE_EMA_RATE:-0.01}"
case "${TARGET_CACHE_MODE}" in
    online|epoch_ema) ;;
    *) echo "Invalid TARGET_CACHE_MODE: ${TARGET_CACHE_MODE}" >&2; exit 1 ;;
esac
LOSS_MASK_MODE="${LOSS_MASK_MODE:-confidence}"
N="${N:-32}"
# Correctness pseudo labels are sampled from the frozen historical target cache.
# Use policy for the current in-group-majority baseline instead.
# target_cache_sample / target_cache_majority / policy
PSEUDO_LABEL_SOURCE="${PSEUDO_LABEL_SOURCE:-target_cache_majority}"

LOG_DIR="${LOG_DIR:-nohup_log/majority_ttcl}"
INTER_TASK_SLEEP_SECONDS="${INTER_TASK_SLEEP_SECONDS:-30}"
mkdir -p "${LOG_DIR}"

run_task() {
    local data_name="$1"
    local target_mode="$2"
    local script_name="$3"
    local entropy_suffix=""
    local pseudo_label_suffix=""
    local log_file

    if [[ "${target_mode}" == entropy_* ]]; then
        entropy_suffix="_Rho${ENTROPY_SHRINKAGE_RHO}-EmaBeta${PRIOR_EMA_BETA}"
    fi
    if [[ "${PSEUDO_LABEL_SOURCE}" == "target_cache_sample" ]]; then
        pseudo_label_suffix="_PSamp"
    elif [[ "${PSEUDO_LABEL_SOURCE}" == "target_cache_majority" ]]; then
        pseudo_label_suffix="_PMajor"
    else
        pseudo_label_suffix="_PPol"
    fi
    log_file="${LOG_DIR}/MajorityTTCL_${data_name}_${MODEL_NAME}_${target_mode}${entropy_suffix}_${TARGET_CACHE_MODE}Ema${TARGET_CACHE_EMA_RATE}${pseudo_label_suffix}_${LOSS_MASK_MODE}_K${N}.log"

    echo "[$(date '+%F %T')] Starting ${data_name} (${target_mode}); log: ${log_file}"
    env \
        LLM_PATH="${LLM_PATH}" \
        MODEL_NAME="${MODEL_NAME}" \
        TARGET_MODE="${target_mode}" \
        ENTROPY_SHRINKAGE_RHO="${ENTROPY_SHRINKAGE_RHO}" \
        PRIOR_EMA_BETA="${PRIOR_EMA_BETA}" \
        TARGET_CACHE_MODE="${TARGET_CACHE_MODE}" \
        TARGET_CACHE_EMA_RATE="${TARGET_CACHE_EMA_RATE}" \
        LOSS_MASK_MODE="${LOSS_MASK_MODE}" \
        N="${N}" \
        PSEUDO_LABEL_SOURCE="${PSEUDO_LABEL_SOURCE}" \
        bash "${SCRIPT_DIR}/${script_name}" > "${log_file}" 2>&1
    echo "[$(date '+%F %T')] Finished ${data_name} (${target_mode})"
    if (( INTER_TASK_SLEEP_SECONDS > 0 )); then
        echo "Waiting ${INTER_TASK_SLEEP_SECONDS}s before the next task so Ray/vLLM resources can be released."
        sleep "${INTER_TASK_SLEEP_SECONDS}"
    fi
}



run_task "AIME24" "frequency" "run_majority_ttcl_aime24.sh"
run_task "MATH500" "frequency" "run_majority_ttcl_math500.sh"
run_task "AIME25" "frequency" "run_majority_ttcl_aime25.sh"
run_task "AMC23" "frequency" "run_majority_ttcl_amc23.sh"
run_task "AIME26" "frequency" "run_majority_ttcl_aime26.sh"

run_task "NQ" "frequency" "run_majority_ttcl_nq.sh"
run_task "HotpotQA" "frequency" "run_majority_ttcl_hotpotqa.sh"
run_task "TriviaQA" "frequency" "run_majority_ttcl_triviaqa.sh"

run_task "SimpleQA" "frequency" "run_majority_ttcl_simpleqa.sh"
run_task "CSimpleQA" "frequency" "run_majority_ttcl_csimpleqa.sh"