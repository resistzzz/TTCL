#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-path/conda_env/verl061/bin/python}"
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"

# Add model and output entries at matching indices. Each model is evaluated
# on every dataset below, and each method writes to its own subdirectory.
# BASELINES accepts a comma-separated subset, e.g. BASELINES=cot or
# BASELINES=cot,self_refine. BoN remains controlled separately by RUN_BON.
MODEL_PATHS=(
  PATH_TO_MODEL_YOU_WANT_TO_EVALUATE

)
OUTPUT_PATHS=(
  PATH_TO_OUTPUT_DIR_FOR_EVALUATION_RESULTS
)
DATA_LIST=(
  data/downstream/law_em_v2/legal_val_2000.jsonl
)
BASELINES_CSV="${BASELINES:-direct,cot,self_consistency,self_refine}"
IFS=, read -r -a BASELINES <<< "${BASELINES_CSV}"
if (( ${#BASELINES[@]} == 0 )); then
  echo "BASELINES must contain at least one method." >&2
  exit 1
fi
for baseline in "${BASELINES[@]}"; do
  case "${baseline}" in
    direct|cot|self_consistency|self_refine) ;;
    bon)
      echo "BoN is selected with RUN_BON=true, not through BASELINES." >&2
      exit 1
      ;;
    *)
      echo "Unknown baseline '${baseline}'. Choose direct, cot, self_consistency, or self_refine." >&2
      exit 1
      ;;
  esac
done

# BoN is optional and uses a separately served judge model. RUN_BON=true alone
# runs only BoN; set RUN_BASELINES=true as well to run both groups of methods.
RUN_BON="${RUN_BON:-false}"
if [[ "${RUN_BON}" == "true" ]]; then
  RUN_BASELINES="${RUN_BASELINES:-false}"
else
  RUN_BASELINES="${RUN_BASELINES:-true}"
fi
BON_NUM_SAMPLES="${BON_NUM_SAMPLES:-8}"
BON_JUDGE_MODE="${BON_JUDGE_MODE:-launch}" # launch or external
BON_JUDGE_MODEL_PATH="${BON_JUDGE_MODEL_PATH:-path/models/Qwen3-32B}"
BON_JUDGE_MODEL="${BON_JUDGE_MODEL:-Qwen3-32B}"
BON_JUDGE_BASE_URL="${BON_JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}"
BON_JUDGE_GPU_IDS="${BON_JUDGE_GPU_IDS:-0,1}"
BON_JUDGE_TENSOR_PARALLEL_SIZE="${BON_JUDGE_TENSOR_PARALLEL_SIZE:-2}"
BON_JUDGE_GPU_MEMORY_UTILIZATION="${BON_JUDGE_GPU_MEMORY_UTILIZATION:-0.85}"
BON_JUDGE_MAX_MODEL_LEN="${BON_JUDGE_MAX_MODEL_LEN:-}"
BON_JUDGE_ENFORCE_EAGER="${BON_JUDGE_ENFORCE_EAGER:-true}"
BON_JUDGE_DISABLE_CUSTOM_ALL_REDUCE="${BON_JUDGE_DISABLE_CUSTOM_ALL_REDUCE:-true}"
BON_JUDGE_STARTUP_TIMEOUT="${BON_JUDGE_STARTUP_TIMEOUT:-900}"
BON_JUDGE_LOG="${BON_JUDGE_LOG:-${REPO_ROOT}/eval/baseline_results/legal_em_v2/bon_judge_server.log}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}"

EVAL_MODE="${EVAL_MODE:-parallel}" # parallel: one task per GPU; serial: one at a time
if [[ "${RUN_BON}" == "true" && "${BON_JUDGE_MODE}" == "launch" ]]; then
  GPU_IDS="${GPU_IDS:-2,3}"
else
  GPU_IDS="${GPU_IDS:-0,1,2,3}"
fi
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DTYPE="${DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
AUX_MAX_TOKENS="${AUX_MAX_TOKENS:-512}"
SC_NUM_SAMPLES="${SC_NUM_SAMPLES:-8}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-1}"
ECE_BINS="${ECE_BINS:-10}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

resolve_repo_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then printf '%s\n' "${path}"; else printf '%s/%s\n' "${REPO_ROOT}" "${path}"; fi
}

if (( ${#MODEL_PATHS[@]} != ${#OUTPUT_PATHS[@]} )); then
  echo "MODEL_PATHS and OUTPUT_PATHS must have the same length." >&2
  exit 1
fi
if (( ${#MODEL_PATHS[@]} == 0 || ${#DATA_LIST[@]} == 0 )); then
  echo "MODEL_PATHS and DATA_LIST must not be empty." >&2
  exit 1
fi

common_args=(
  --backend vllm
  --batch-size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --aux-max-tokens "${AUX_MAX_TOKENS}"
  --eval-rollouts "${EVAL_ROLLOUTS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --ece-bins "${ECE_BINS}"
  --seed "${SEED}"
  --judge-concurrency "${JUDGE_CONCURRENCY}"
)
[[ -z "${LIMIT}" ]] || common_args+=(--limit "${LIMIT}")
[[ "${TRUST_REMOTE_CODE}" != true ]] || common_args+=(--trust-remote-code)

TASK_MODEL_PATHS=()
TASK_DATA_PATHS=()
TASK_OUTPUT_DIRS=()
TASK_METHODS=()
TASK_NUM_SAMPLES=()
for model_index in "${!MODEL_PATHS[@]}"; do
  model_path="$(resolve_repo_path "${MODEL_PATHS[model_index]}")"
  output_root="$(resolve_repo_path "${OUTPUT_PATHS[model_index]}")"
  [[ -d "${model_path}" ]] || { echo "Model path does not exist: ${model_path}" >&2; exit 1; }
  for data_entry in "${DATA_LIST[@]}"; do
    data_path="$(resolve_repo_path "${data_entry}")"
    [[ -f "${data_path}" ]] || { echo "Dataset does not exist: ${data_path}" >&2; exit 1; }
    [[ "${data_path}" == *.jsonl ]] || { echo "Legal EM-v2 requires the frozen source JSONL: ${data_path}" >&2; exit 1; }
    dataset_name="$(basename "${data_entry}" .jsonl)"
    methods=()
    if [[ "${RUN_BASELINES}" == "true" ]]; then methods+=("${BASELINES[@]}"); fi
    if [[ "${RUN_BON}" == "true" ]]; then methods+=(bon); fi
    for method in "${methods[@]}"; do
      case "${method}" in
        direct|cot|self_refine) num_samples=1 ;;
        self_consistency) num_samples="${SC_NUM_SAMPLES}" ;;
        bon) num_samples="${BON_NUM_SAMPLES}" ;;
        *) echo "Unknown baseline method: ${method}" >&2; exit 1 ;;
      esac
      TASK_MODEL_PATHS+=("${model_path}")
      TASK_DATA_PATHS+=("${data_path}")
      TASK_OUTPUT_DIRS+=("${output_root}/${dataset_name}/${method}")
      TASK_METHODS+=("${method}")
      TASK_NUM_SAMPLES+=("${num_samples}")
    done
  done
done

run_task() {
  local task_index="$1" gpu_id="${2:-}"
  local output_dir="${TASK_OUTPUT_DIRS[task_index]}" method="${TASK_METHODS[task_index]}"
  local method_args=(--method "${method}")
  if [[ "${method}" == self_consistency || "${method}" == bon ]]; then
    method_args+=(--num-samples "${TASK_NUM_SAMPLES[task_index]}")
  fi
  if [[ "${method}" == bon ]]; then
    method_args+=(--bon-judge-base-url "${BON_JUDGE_BASE_URL}" --bon-judge-model "${BON_JUDGE_MODEL}")
  fi
  mkdir -p "${output_dir}"
  echo "[${gpu_id:-serial}] $(basename "${TASK_MODEL_PATHS[task_index]}") | $(basename "${TASK_DATA_PATHS[task_index]}") | ${method}"
  local command=("${PYTHON_BIN}" "${REPO_ROOT}/eval/evaluate_legal_baselines.py"
    --model-path "${TASK_MODEL_PATHS[task_index]}"
    --data-path "${TASK_DATA_PATHS[task_index]}"
    --output-dir "${output_dir}" "${common_args[@]}" "${method_args[@]}")
  if [[ -n "${gpu_id}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}" >"${output_dir}/evaluate.log" 2>&1 || {
      echo "[GPU ${gpu_id}] FAILED ${method}: ${output_dir} (see evaluate.log)" >&2; return 1;
    }
  else
    "${command[@]}"
  fi
  echo "[${gpu_id:-serial}] completed ${method}: ${output_dir}"
}

start_bon_judge() {
  [[ "${RUN_BON}" == true && "${BON_JUDGE_MODE}" == launch ]] || return 0
  [[ "${BON_JUDGE_BASE_URL}" == */v1 ]] || { echo "BON_JUDGE_BASE_URL must end in /v1" >&2; return 1; }
  local judge_path vllm_bin port health_url waited=0
  judge_path="$(resolve_repo_path "${BON_JUDGE_MODEL_PATH}")"
  [[ -d "${judge_path}" ]] || { echo "Judge model directory not found: ${judge_path}" >&2; return 1; }
  default_vllm_bin="path/conda_env/latest_vllm/bin/vllm"
  [[ -x "${default_vllm_bin}" ]] || default_vllm_bin="$(dirname -- "${PYTHON_BIN}")/vllm"
  vllm_bin="${VLLM_BIN:-${default_vllm_bin}}"
  [[ -x "${vllm_bin}" ]] || { echo "vLLM executable not found: ${vllm_bin}" >&2; return 1; }
  port="${BON_JUDGE_BASE_URL%/v1}"
  port="${port##*:}"
  health_url="${BON_JUDGE_BASE_URL%/v1}/health"
  mkdir -p "$(dirname -- "${BON_JUDGE_LOG}")"
  local serve_args=(serve "${judge_path}" --host 0.0.0.0 --port "${port}"
    --served-model-name "${BON_JUDGE_MODEL}"
    --tensor-parallel-size "${BON_JUDGE_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${BON_JUDGE_GPU_MEMORY_UTILIZATION}" --dtype "${DTYPE}" --trust-remote-code)
  [[ -z "${BON_JUDGE_MAX_MODEL_LEN}" ]] || serve_args+=(--max-model-len "${BON_JUDGE_MAX_MODEL_LEN}")
  [[ "${BON_JUDGE_ENFORCE_EAGER}" != true ]] || serve_args+=(--enforce-eager)
  [[ "${BON_JUDGE_DISABLE_CUSTOM_ALL_REDUCE}" != true ]] || serve_args+=(--disable-custom-all-reduce)
  echo "Starting BoN judge ${BON_JUDGE_MODEL} on GPUs ${BON_JUDGE_GPU_IDS}; log: ${BON_JUDGE_LOG}"
  CUDA_VISIBLE_DEVICES="${BON_JUDGE_GPU_IDS}" "${vllm_bin}" "${serve_args[@]}" >"${BON_JUDGE_LOG}" 2>&1 &
  BON_JUDGE_PID=$!
  while (( waited < BON_JUDGE_STARTUP_TIMEOUT )); do
    if ! kill -0 "${BON_JUDGE_PID}" 2>/dev/null; then
      echo "BoN judge exited before becoming ready; see ${BON_JUDGE_LOG}" >&2
      tail -n 80 "${BON_JUDGE_LOG}" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 3 "${health_url}" >/dev/null 2>&1; then
      echo "BoN judge is ready at ${BON_JUDGE_BASE_URL}"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "Timed out waiting for BoN judge; see ${BON_JUDGE_LOG}" >&2
  tail -n 80 "${BON_JUDGE_LOG}" >&2 || true
  return 1
}

stop_bon_judge() {
  if [[ -n "${BON_JUDGE_PID:-}" ]] && kill -0 "${BON_JUDGE_PID}" 2>/dev/null; then
    echo "Stopping BoN judge server (pid ${BON_JUDGE_PID})"
    kill "${BON_JUDGE_PID}" 2>/dev/null || true
    wait "${BON_JUDGE_PID}" 2>/dev/null || true
  fi
}
trap stop_bon_judge EXIT

if [[ "${RUN_BON}" != true && "${RUN_BASELINES}" != true ]]; then
  echo "Enable RUN_BASELINES and/or RUN_BON." >&2
  exit 1
fi
if [[ "${RUN_BON}" == true && "${BON_JUDGE_MODE}" != launch && "${BON_JUDGE_MODE}" != external ]]; then
  echo "BON_JUDGE_MODE must be launch or external." >&2
  exit 1
fi
if [[ "${RUN_BON}" == true && "${BON_JUDGE_MODE}" == external && "${BON_JUDGE_BASE_URL}" != */v1 ]]; then
  echo "BON_JUDGE_BASE_URL must end in /v1." >&2
  exit 1
fi

case "${EVAL_MODE}" in
  serial)
    start_bon_judge
    if [[ "${RUN_BON}" == true ]]; then
      IFS=, read -r -a serial_gpu_ids <<< "${GPU_IDS}"
      for task_index in "${!TASK_MODEL_PATHS[@]}"; do run_task "${task_index}" "${serial_gpu_ids[0]}"; done
    else
      for task_index in "${!TASK_MODEL_PATHS[@]}"; do run_task "${task_index}"; done
    fi
    ;;
  parallel)
    (( TENSOR_PARALLEL_SIZE == 1 )) || { echo "parallel mode requires TENSOR_PARALLEL_SIZE=1" >&2; exit 1; }
    IFS=, read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
    (( ${#GPU_ID_ARRAY[@]} > 0 )) || { echo "GPU_IDS must not be empty" >&2; exit 1; }
    if [[ "${RUN_BON}" == true && "${BON_JUDGE_MODE}" == launch ]]; then
      IFS=, read -r -a BON_JUDGE_GPU_ID_ARRAY <<< "${BON_JUDGE_GPU_IDS}"
      for judge_gpu in "${BON_JUDGE_GPU_ID_ARRAY[@]}"; do
        for eval_gpu in "${GPU_ID_ARRAY[@]}"; do
          [[ "${judge_gpu}" != "${eval_gpu}" ]] || { echo "Judge and evaluation GPU IDs overlap at ${judge_gpu}." >&2; exit 1; }
        done
      done
    fi
    start_bon_judge
    run_gpu_worker() {
      local gpu_id="$1" worker_index="$2" task_index status=0
      for ((task_index=worker_index; task_index<${#TASK_MODEL_PATHS[@]}; task_index+=${#GPU_ID_ARRAY[@]})); do
        run_task "${task_index}" "${gpu_id}" || status=1
      done
      return "${status}"
    }
    worker_pids=()
    for worker_index in "${!GPU_ID_ARRAY[@]}"; do
      run_gpu_worker "${GPU_ID_ARRAY[worker_index]}" "${worker_index}" & worker_pids+=("$!")
    done
    exit_status=0
    for worker_pid in "${worker_pids[@]}"; do wait "${worker_pid}" || exit_status=1; done
    exit "${exit_status}"
    ;;
  *) echo "EVAL_MODE must be serial or parallel; got ${EVAL_MODE}." >&2; exit 1 ;;
esac
