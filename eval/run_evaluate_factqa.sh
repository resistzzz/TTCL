#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-PYTHON_ENV}"

# MODEL_PATHS and OUTPUT_PATHS are paired by index. Each model is evaluated
# on every parquet in DATA_LIST. ROLLOUTS_PER_QUESTION is paired with DATA_LIST
# and controls vLLM's `n` independently for each benchmark.
MODEL_PATHS=(
  PATH_TO_Qwen3-4B
  PATH_TO_Qwen3-8B
  # Add the model you want to evaluate here
)

OUTPUT_PATHS=(
  eval/eval_results/Qwen3-4B/Base_NoThink
  eval/eval_results/Qwen3-8B/Base_NoThink
  # Add corresponding dictionary here to save evaluation results
)

DATA_LIST=(
  data/simpleqa_verified/simpleqa_answer_tag_placeholder_conf.parquet
  data/csimpleqa/csimpleqa_answer_tag_placeholder_conf.parquet
  data/hallulens/hallulens_answer_tag_placeholder_conf.parquet
  data/flashrag/nq/nq_answer_tag_placeholder_conf.parquet
  data/flashrag/hotpotqa/hotpotqa_answer_tag_placeholder_conf.parquet
  data/flashrag/triviaqa/triviaqa_answer_tag_placeholder_conf.parquet
)

# One entry per DATA_LIST item. Keep one completion for the larger Fact QA
# datasets by default; increase an individual entry when needed.
ROLLOUTS_PER_QUESTION=(
  1  # simpleqa
  1  # csimpleqa
  1  # hallulens
  8  # nq
  8  # hotpotqa
  8  # triviaqa
)

if (( ${#MODEL_PATHS[@]} != ${#OUTPUT_PATHS[@]} )); then
  echo "MODEL_PATHS and OUTPUT_PATHS must have the same number of entries" >&2
  exit 1
fi
if (( ${#DATA_LIST[@]} == 0 )); then
  echo "DATA_LIST must contain at least one parquet path" >&2
  exit 1
fi
if (( ${#DATA_LIST[@]} != ${#ROLLOUTS_PER_QUESTION[@]} )); then
  echo "DATA_LIST and ROLLOUTS_PER_QUESTION must have the same number of entries" >&2
  exit 1
fi
for rollout_count in "${ROLLOUTS_PER_QUESTION[@]}"; do
  if ! [[ "${rollout_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Each ROLLOUTS_PER_QUESTION entry must be a positive integer; got ${rollout_count}" >&2
    exit 1
  fi
done

resolve_repo_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${path}"
  fi
}

BATCH_SIZE="${BATCH_SIZE:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-6144}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# serial: preserve the old single-process loop; parallel: one worker per GPU.
EVAL_MODE="${EVAL_MODE:-parallel}"  # parallel / serial
GPU_IDS="${GPU_IDS:-0,1,2,3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DTYPE="${DTYPE:-auto}"
ECE_BINS="${ECE_BINS:-10}"
CORRECTNESS_REWARD="${CORRECTNESS_REWARD:-default}" # math_verify / default
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
SEED="${SEED:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
# Qwen3 defaults to NoThink mode, matching the RLCR training prompt.
ENABLE_THINKING="${ENABLE_THINKING:-false}"

common_args=(
  --batch-size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --ece-bins "${ECE_BINS}"
  --correctness-reward "${CORRECTNESS_REWARD}"
)

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  common_args+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${SEED}" ]]; then
  common_args+=(--seed "${SEED}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  common_args+=(--trust-remote-code)
fi
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  common_args+=(--enable-thinking)
fi


EXTRA_ARGS=("$@")
TASK_MODEL_PATHS=()
TASK_DATA_PATHS=()
TASK_OUTPUT_DIRS=()
TASK_ROLLOUTS_PER_QUESTION=()

for index in "${!MODEL_PATHS[@]}"; do
  model_path="$(resolve_repo_path "${MODEL_PATHS[index]}")"
  output_root="$(resolve_repo_path "${OUTPUT_PATHS[index]}")"

  for data_index in "${!DATA_LIST[@]}"; do
    data_path_entry="${DATA_LIST[data_index]}"
    data_path="$(resolve_repo_path "${data_path_entry}")"
    dataset_name="$(basename "${data_path_entry}" .parquet)"
    output_dir="${output_root}/${dataset_name}"
    TASK_MODEL_PATHS+=("${model_path}")
    TASK_DATA_PATHS+=("${data_path}")
    TASK_OUTPUT_DIRS+=("${output_dir}")
    TASK_ROLLOUTS_PER_QUESTION+=("${ROLLOUTS_PER_QUESTION[data_index]}")
  done
done

run_task() {
  local task_index="$1"
  local gpu_id="${2:-}"
  local model_path="${TASK_MODEL_PATHS[task_index]}"
  local data_path="${TASK_DATA_PATHS[task_index]}"
  local output_dir="${TASK_OUTPUT_DIRS[task_index]}"
  local rollout_count="${TASK_ROLLOUTS_PER_QUESTION[task_index]}"
  mkdir -p "${output_dir}"

  if [[ -n "${gpu_id}" ]]; then
    echo "[GPU ${gpu_id}] evaluating $(basename "${model_path}") on $(basename "${data_path}") with n=${rollout_count}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${REPO_ROOT}/eval/evaluate_calibration.py" \
      --model-path "${model_path}" \
      --data-path "${data_path}" \
      --output-dir "${output_dir}" \
      --num-samples "${rollout_count}" \
      "${common_args[@]}" \
      "${EXTRA_ARGS[@]}" >"${output_dir}/evaluate.log" 2>&1
  else
    echo "[serial] evaluating $(basename "${model_path}") on $(basename "${data_path}") with n=${rollout_count}"
    "${PYTHON_BIN}" "${REPO_ROOT}/eval/evaluate_calibration.py" \
      --model-path "${model_path}" \
      --data-path "${data_path}" \
      --output-dir "${output_dir}" \
      --num-samples "${rollout_count}" \
      "${common_args[@]}" \
      "${EXTRA_ARGS[@]}"
  fi
}

case "${EVAL_MODE}" in
  serial)
    for task_index in "${!TASK_MODEL_PATHS[@]}"; do
      run_task "${task_index}"
    done
    ;;
  parallel)
    if (( TENSOR_PARALLEL_SIZE != 1 )); then
      echo "EVAL_MODE=parallel requires TENSOR_PARALLEL_SIZE=1; got ${TENSOR_PARALLEL_SIZE}." >&2
      exit 1
    fi
    IFS="," read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
    if (( ${#GPU_ID_ARRAY[@]} == 0 )); then
      echo "GPU_IDS must contain at least one GPU ID." >&2
      exit 1
    fi

    run_gpu_worker() {
      local gpu_id="$1"
      local worker_index="$2"
      local task_index
      for ((task_index=worker_index; task_index < ${#TASK_MODEL_PATHS[@]}; task_index += ${#GPU_ID_ARRAY[@]})); do
        run_task "${task_index}" "${gpu_id}"
      done
    }

    worker_pids=()
    for worker_index in "${!GPU_ID_ARRAY[@]}"; do
      run_gpu_worker "${GPU_ID_ARRAY[worker_index]}" "${worker_index}" &
      worker_pids+=("$!")
    done

    exit_status=0
    for worker_pid in "${worker_pids[@]}"; do
      wait "${worker_pid}" || exit_status=1
    done
    exit "${exit_status}"
    ;;
  *)
    echo "EVAL_MODE must be serial or parallel; got ${EVAL_MODE}." >&2
    exit 1
    ;;
esac
