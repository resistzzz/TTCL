
#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-path/conda_env/verl061/bin/python}"
# vLLM's runtime JIT invokes helper binaries such as `ninja` via PATH. The
# script calls Python by absolute path, so expose that environment's bin dir.
PYTHON_BIN_DIR="$(dirname -- "${PYTHON_BIN}")"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

# MODEL_PATHS and OUTPUT_PATHS correspond by index. Every model is evaluated
# on every parquet in DATA_LIST. BASELINES can be overridden with a
# comma-separated list, e.g. BASELINES=cot or BASELINES=cot,self_refine.
MODEL_PATHS=(
    PATH_TO_MODEL_YOU_WANT_TO_EVALUATE
)

OUTPUT_PATHS=(
    PATH_TO_OUTPUT_DIR_FOR_EVALUATION_RESULTS
)

DATA_LIST=(
    data/simpleqa_verified/simpleqa_answer_tag_placeholder_conf.parquet
    data/csimpleqa/csimpleqa_answer_tag_placeholder_conf.parquet
    data/downstream/finance/finance_context_test_2000_with_context.parquet
)

# BoN is kept separate because it needs a second, larger judge model. Set
# RUN_BON=true to evaluate BoN using BON_JUDGE_MODEL_PATH below.
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
RUN_BON="${RUN_BON:-false}"
if [[ "${RUN_BON}" == "true" ]]; then
  RUN_BASELINES="${RUN_BASELINES:-false}"  # RUN_BON=true alone runs only BoN
else
  RUN_BASELINES="${RUN_BASELINES:-true}"
fi

# Generation / evaluation settings.
EVAL_MODE="${EVAL_MODE:-parallel}"  # parallel: one task per GPU; serial: one at a time
if [[ "${RUN_BON}" == "true" ]]; then
  GPU_IDS="${GPU_IDS:-2,3}"  # reserve GPUs 0,1 for the default 2-way judge
else
  GPU_IDS="${GPU_IDS:-0,1,2,3}"
fi
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DTYPE="${DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
AUX_MAX_TOKENS="${AUX_MAX_TOKENS:-512}"
SC_NUM_SAMPLES="${SC_NUM_SAMPLES:-8}"
BON_NUM_SAMPLES="${BON_NUM_SAMPLES:-8}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-1}"
ECE_BINS="${ECE_BINS:-10}"
CORRECTNESS_REWARD="${CORRECTNESS_REWARD:-default}"  # use math_verify for math benchmarks
ENABLE_THINKING="${ENABLE_THINKING:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"

# When RUN_BON=true, launch the judge from this local Hugging Face model path.
# Set BON_JUDGE_MODE=external to use a judge server you started yourself.
BON_JUDGE_MODEL_PATH="${BON_JUDGE_MODEL_PATH:-}"
BON_JUDGE_MODE="${BON_JUDGE_MODE:-launch}"  # launch / external
BON_JUDGE_BASE_URL="${BON_JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}"
BON_JUDGE_MODEL="${BON_JUDGE_MODEL:-Qwen/Qwen3-32B}"
BON_JUDGE_GPU_IDS="${BON_JUDGE_GPU_IDS:-0,1}"
BON_JUDGE_TENSOR_PARALLEL_SIZE="${BON_JUDGE_TENSOR_PARALLEL_SIZE:-2}"
BON_JUDGE_GPU_MEMORY_UTILIZATION="${BON_JUDGE_GPU_MEMORY_UTILIZATION:-0.85}"
BON_JUDGE_MAX_MODEL_LEN="${BON_JUDGE_MAX_MODEL_LEN:-}"
BON_JUDGE_STARTUP_TIMEOUT="${BON_JUDGE_STARTUP_TIMEOUT:-900}"
BON_JUDGE_LOG="${BON_JUDGE_LOG:-${REPO_ROOT}/eval/baseline_results/bon_judge_server.log}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}"

resolve_repo_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${path}"
  fi
}

if (( ${#MODEL_PATHS[@]} != ${#OUTPUT_PATHS[@]} )); then
  echo "MODEL_PATHS and OUTPUT_PATHS must have the same length." >&2
  exit 1
fi
if (( ${#MODEL_PATHS[@]} == 0 || ${#DATA_LIST[@]} == 0 )); then
  echo "MODEL_PATHS and DATA_LIST must not be empty." >&2
  exit 1
fi
if [[ "${RUN_BASELINES}" != "true" && "${RUN_BON}" != "true" ]]; then
  echo "Enable at least one of RUN_BASELINES or RUN_BON." >&2
  exit 1
fi
if [[ "${RUN_BON}" == "true" && "${BON_JUDGE_MODE}" == "launch" ]]; then
  if [[ "${BON_JUDGE_BASE_URL}" != */v1 ]]; then
    echo "BON_JUDGE_BASE_URL must end in /v1 when launching the local judge." >&2
    exit 1
  fi
  if [[ -z "${BON_JUDGE_MODEL_PATH}" || ! -d "$(resolve_repo_path "${BON_JUDGE_MODEL_PATH}")" ]]; then
    echo "Set BON_JUDGE_MODEL_PATH to the local Qwen3-32B Hugging Face model directory." >&2
    exit 1
  fi
fi

common_args=(
  --backend vllm
  --batch-size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --aux-max-tokens "${AUX_MAX_TOKENS}"
  --eval-rollouts "${EVAL_ROLLOUTS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --ece-bins "${ECE_BINS}"
  --correctness-reward "${CORRECTNESS_REWARD}"
  --seed "${SEED}"
  --judge-concurrency "${JUDGE_CONCURRENCY}"
)
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  common_args+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${LIMIT}" ]]; then
  common_args+=(--limit "${LIMIT}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  common_args+=(--trust-remote-code)
fi
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  common_args+=(--enable-thinking)
fi

TASK_MODEL_PATHS=()
TASK_DATA_PATHS=()
TASK_OUTPUT_DIRS=()
TASK_METHODS=()
TASK_NUM_SAMPLES=()

for model_index in "${!MODEL_PATHS[@]}"; do
  model_path="$(resolve_repo_path "${MODEL_PATHS[model_index]}")"
  output_root="$(resolve_repo_path "${OUTPUT_PATHS[model_index]}")"
  if [[ ! -d "${model_path}" ]]; then
    echo "Model path does not exist: ${model_path}" >&2
    exit 1
  fi

  for data_entry in "${DATA_LIST[@]}"; do
    data_path="$(resolve_repo_path "${data_entry}")"
    if [[ ! -f "${data_path}" ]]; then
      echo "Dataset does not exist: ${data_path}" >&2
      exit 1
    fi
    dataset_name="$(basename "${data_entry}" .parquet)"

    methods=()
    if [[ "${RUN_BASELINES}" == "true" ]]; then
      methods+=("${BASELINES[@]}")
    fi
    if [[ "${RUN_BON}" == "true" ]]; then
      methods+=(bon)
    fi

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
  local task_index="$1"
  local gpu_id="${2:-}"
  local model_path="${TASK_MODEL_PATHS[task_index]}"
  local data_path="${TASK_DATA_PATHS[task_index]}"
  local output_dir="${TASK_OUTPUT_DIRS[task_index]}"
  local method="${TASK_METHODS[task_index]}"
  local num_samples="${TASK_NUM_SAMPLES[task_index]}"
  local method_args=(--method "${method}")
  mkdir -p "${output_dir}"

  if [[ "${method}" == "self_consistency" || "${method}" == "bon" ]]; then
    method_args+=(--num-samples "${num_samples}")
  fi
  if [[ "${method}" == "bon" ]]; then
    method_args+=(
      --bon-judge-base-url "${BON_JUDGE_BASE_URL}"
      --bon-judge-model "${BON_JUDGE_MODEL}"
    )
  fi

  echo "[${gpu_id:-serial}] $(basename "${model_path}") | $(basename "${data_path}") | ${method}"
  if [[ -n "${gpu_id}" ]]; then
    if CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${REPO_ROOT}/eval/evaluate_baselines.py" \
      --model-path "${model_path}" \
      --data-path "${data_path}" \
      --output-dir "${output_dir}" \
      "${common_args[@]}" "${method_args[@]}" \
      >"${output_dir}/evaluate.log" 2>&1; then
      echo "[GPU ${gpu_id}] completed ${method}: ${output_dir}"
    else
      echo "[GPU ${gpu_id}] FAILED ${method}: ${output_dir} (see ${output_dir}/evaluate.log)" >&2
      return 1
    fi
  else
    if "${PYTHON_BIN}" "${REPO_ROOT}/eval/evaluate_baselines.py" \
      --model-path "${model_path}" \
      --data-path "${data_path}" \
      --output-dir "${output_dir}" \
      "${common_args[@]}" "${method_args[@]}"; then
      echo "[serial] completed ${method}: ${output_dir}"
    else
      echo "[serial] FAILED ${method}: ${output_dir}" >&2
      return 1
    fi
  fi
}

start_bon_judge() {
  [[ "${RUN_BON}" == "true" && "${BON_JUDGE_MODE}" == "launch" ]] || return 0
  local model_path
  model_path="$(resolve_repo_path "${BON_JUDGE_MODEL_PATH}")"
  local vllm_bin="${VLLM_BIN:-$(dirname -- "${PYTHON_BIN}")/vllm}"
  if [[ ! -x "${vllm_bin}" ]]; then
    echo "vLLM executable not found: ${vllm_bin}; set VLLM_BIN if needed." >&2
    return 1
  fi
  mkdir -p "$(dirname -- "${BON_JUDGE_LOG}")"
  local base_url="${BON_JUDGE_BASE_URL%/v1}"
  local judge_port="${base_url##*:}"
  local serve_args=(serve "${model_path}" --host 0.0.0.0 --port "${judge_port}" \
    --served-model-name "${BON_JUDGE_MODEL}" \
    --tensor-parallel-size "${BON_JUDGE_TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${BON_JUDGE_GPU_MEMORY_UTILIZATION}" --dtype "${DTYPE}" --trust-remote-code)
  if [[ -n "${BON_JUDGE_MAX_MODEL_LEN}" ]]; then
    serve_args+=(--max-model-len "${BON_JUDGE_MAX_MODEL_LEN}")
  fi
  echo "Starting BoN judge ${BON_JUDGE_MODEL} from ${model_path} on GPUs ${BON_JUDGE_GPU_IDS}; log: ${BON_JUDGE_LOG}"
  CUDA_VISIBLE_DEVICES="${BON_JUDGE_GPU_IDS}" "${vllm_bin}" "${serve_args[@]}" >"${BON_JUDGE_LOG}" 2>&1 &
  BON_JUDGE_PID=$!

  local health_url="${BON_JUDGE_BASE_URL%/v1}/health"
  local waited=0
  while (( waited < BON_JUDGE_STARTUP_TIMEOUT )); do
    if ! kill -0 "${BON_JUDGE_PID}" 2>/dev/null; then
      echo "BoN judge server exited before becoming ready. Log: ${BON_JUDGE_LOG}" >&2
      tail -n 80 "${BON_JUDGE_LOG}" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 3 "${health_url}" >/dev/null 2>&1; then
      echo "BoN judge is ready at ${BON_JUDGE_BASE_URL} (model alias: ${BON_JUDGE_MODEL})."
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "Timed out waiting for BoN judge at ${health_url}. Log: ${BON_JUDGE_LOG}" >&2
  tail -n 80 "${BON_JUDGE_LOG}" >&2 || true
  return 1
}

stop_bon_judge() {
  if [[ -n "${BON_JUDGE_PID:-}" ]] && kill -0 "${BON_JUDGE_PID}" 2>/dev/null; then
    echo "Stopping BoN judge server (pid ${BON_JUDGE_PID})."
    kill "${BON_JUDGE_PID}" 2>/dev/null || true
    wait "${BON_JUDGE_PID}" 2>/dev/null || true
  fi
}
trap stop_bon_judge EXIT

case "${EVAL_MODE}" in
  serial)
    exit_status=0
    if [[ "${RUN_BON}" == "true" ]]; then
      start_bon_judge || exit 1
    fi
    for task_index in "${!TASK_MODEL_PATHS[@]}"; do
      run_task "${task_index}" || exit_status=1
    done
    exit "${exit_status}"
    ;;
  parallel)
    if (( TENSOR_PARALLEL_SIZE != 1 )); then
      echo "EVAL_MODE=parallel requires TENSOR_PARALLEL_SIZE=1; got ${TENSOR_PARALLEL_SIZE}." >&2
      exit 1
    fi
    IFS=, read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
    if (( ${#GPU_ID_ARRAY[@]} == 0 )); then
      echo "GPU_IDS must contain at least one GPU ID." >&2
      exit 1
    fi
    if [[ "${RUN_BON}" == "true" && "${BON_JUDGE_MODE}" == "launch" ]]; then
      IFS=, read -r -a BON_JUDGE_GPU_ID_ARRAY <<< "${BON_JUDGE_GPU_IDS}"
      for judge_gpu in "${BON_JUDGE_GPU_ID_ARRAY[@]}"; do
        for eval_gpu in "${GPU_ID_ARRAY[@]}"; do
          if [[ "${judge_gpu}" == "${eval_gpu}" ]]; then
            echo "Judge GPU ${judge_gpu} overlaps with evaluation GPU ${eval_gpu}. Set disjoint BON_JUDGE_GPU_IDS and GPU_IDS." >&2
            exit 1
          fi
        done
      done
      start_bon_judge || exit 1
    fi

    run_gpu_worker() {
      local gpu_id="$1"
      local worker_index="$2"
      local task_index
      local worker_status=0
      for ((task_index=worker_index; task_index < ${#TASK_MODEL_PATHS[@]}; task_index += ${#GPU_ID_ARRAY[@]})); do
        run_task "${task_index}" "${gpu_id}" || worker_status=1
      done
      return "${worker_status}"
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
