#!/usr/bin/env bash
# Shared Lighteval/vLLM launcher. Usually invoked through one of the
# run_*.sh scripts in this directory.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_benchmark.sh <benchmark-name> <task-spec> --model MODEL_PATH [options]

Required:
  --model PATH                 Local HuggingFace model/checkpoint directory.

Optional:
  --output-dir PATH            Result directory. Default: eval/lighteval/results/<model>/<benchmark>
  --tp N                       vLLM tensor parallel size. Default: $TP_SIZE or 1.
  --gpu-memory-utilization F   vLLM GPU memory fraction. Default: $GPU_MEMORY_UTILIZATION or 0.85.
  --dtype NAME                 vLLM dtype. Default: $DTYPE or bfloat16.
  --max-samples N              Debug-only cap on examples per task.
  --no-save-details            Do not save per-example outputs.
  --                          Extra arguments passed directly to `lighteval vllm`.

Environment overrides:
  LIGHTEVAL_BIN  Lighteval executable (default: lighteval)
  VLLM_MODEL_ARGS  Extra comma-separated arguments appended to model_name=... .
  LIGHTEVAL_CACHE_ROOT  Writable directory for Lighteval sample caches.
                         Default: eval/lighteval/cache
  LIGHTEVAL_CUSTOM_TASKS  Python file with project-local Lighteval tasks.
                         Default: eval/lighteval/custom_tasks.py
  QWEN3_NO_THINK  For Qwen3 only, use /no_think in the system prompt.
                  Default: true. Set false only when intentionally evaluating
                  the thinking mode.
EOF
}

if (( $# < 3 )); then
    usage >&2
    exit 2
fi

BENCHMARK_NAME="$1"
TASK_SPEC="$2"
shift 2

MODEL_PATH=""
OUTPUT_DIR=""
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES=""
SAVE_DETAILS=true
EXTRA_ARGS=()
QWEN3_NO_THINK="${QWEN3_NO_THINK:-true}"

while (( $# > 0 )); do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tp) TP_SIZE="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --no-save-details) SAVE_DETAILS=false; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "--model is required." >&2
    usage >&2
    exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 2
fi
LCB_ALLOW_UNSANDBOXED_EXECUTION="${LCB_ALLOW_UNSANDBOXED_EXECUTION:-false}"
LCB_ALLOW_UNSANDBOXED_EXECUTION="${LCB_ALLOW_UNSANDBOXED_EXECUTION,,}"
if [[ "${BENCHMARK_NAME}" == "lcb" && "${LCB_ALLOW_UNSANDBOXED_EXECUTION}" != "true" ]]; then
    echo "Refusing to run LiveCodeBench: Lighteval executes model-generated Python on this host." >&2
    echo "Use an isolated worker/container and set LCB_ALLOW_UNSANDBOXED_EXECUTION=true to acknowledge this." >&2
    exit 2
fi
if ! [[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--tp must be a positive integer; got ${TP_SIZE}" >&2
    exit 2
fi
if [[ "${QWEN3_NO_THINK}" != "true" && "${QWEN3_NO_THINK}" != "false" ]]; then
    echo "QWEN3_NO_THINK must be true or false; got ${QWEN3_NO_THINK}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${OUTPUT_DIR}" ]]; then
    MODEL_NAME="$(basename "${MODEL_PATH%/}")"
    OUTPUT_DIR="${REPO_ROOT}/eval/lighteval/results/${MODEL_NAME}/${BENCHMARK_NAME}"
elif [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

# Lighteval 0.13.0 builds its sample-cache path as
# `cache_dir / model_name / hash`.  If model_name is absolute, pathlib drops
# cache_dir and Lighteval tries to write beside the weights.  Keep all cache
# files out of checkpoints by exposing every model through a relative symlink
# below this evaluation directory.
MODEL_REAL_PATH="$(realpath -e "${MODEL_PATH}")"
MODEL_LINK_DIR="${REPO_ROOT}/eval/lighteval/model_links"
LIGHTEVAL_CACHE_ROOT="${LIGHTEVAL_CACHE_ROOT:-${REPO_ROOT}/eval/lighteval/cache}"
if [[ "${LIGHTEVAL_CACHE_ROOT}" != /* ]]; then
    LIGHTEVAL_CACHE_ROOT="${REPO_ROOT}/${LIGHTEVAL_CACHE_ROOT}"
fi
MODEL_BASENAME="$(basename "${MODEL_REAL_PATH}")"
MODEL_LINK_HASH="$(printf '%s' "${MODEL_REAL_PATH}" | sha256sum | cut -c1-12)"
MODEL_LINK_PATH="${MODEL_LINK_DIR}/${MODEL_BASENAME}-${MODEL_LINK_HASH}"
mkdir -p "${MODEL_LINK_DIR}" "${LIGHTEVAL_CACHE_ROOT}"
if [[ -L "${MODEL_LINK_PATH}" ]]; then
    if [[ "$(readlink -f "${MODEL_LINK_PATH}")" != "${MODEL_REAL_PATH}" ]]; then
        echo "Refusing to reuse model link with an unexpected target: ${MODEL_LINK_PATH}" >&2
        exit 2
    fi
elif [[ -e "${MODEL_LINK_PATH}" ]]; then
    echo "Model link path exists but is not a symlink: ${MODEL_LINK_PATH}" >&2
    exit 2
else
    ln -s "${MODEL_REAL_PATH}" "${MODEL_LINK_PATH}"
fi
EVAL_MODEL_PATH="eval/lighteval/model_links/$(basename "${MODEL_LINK_PATH}")"

LIGHTEVAL_BIN="${LIGHTEVAL_BIN:-lighteval}"
if ! command -v "${LIGHTEVAL_BIN}" >/dev/null 2>&1; then
    echo "Cannot find '${LIGHTEVAL_BIN}'. Activate the environment that contains Lighteval, or set LIGHTEVAL_BIN." >&2
    exit 127
fi

LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-${SCRIPT_DIR}/custom_tasks.py}"
if [[ -n "${LIGHTEVAL_CUSTOM_TASKS}" && ! -f "${LIGHTEVAL_CUSTOM_TASKS}" ]]; then
    echo "Custom task file does not exist: ${LIGHTEVAL_CUSTOM_TASKS}" >&2
    exit 2
fi

MODEL_ARGS="model_name=${EVAL_MODEL_PATH},cache_dir=${LIGHTEVAL_CACHE_ROOT},dtype=${DTYPE},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},trust_remote_code=True"
if [[ -n "${VLLM_MODEL_ARGS:-}" ]]; then
    MODEL_ARGS+=",${VLLM_MODEL_ARGS}"
fi

# Lighteval's current PromptManager calls apply_chat_template() without a way
# to pass Qwen's hard `enable_thinking=False` template kwarg.  Qwen officially
# supports `/no_think` in the most recent system/user message as the equivalent
# per-request switch.  A system prompt reaches every task, including IFEval.
if [[ "${QWEN3_NO_THINK}" == "true" ]]; then
    if [[ "${MODEL_ARGS}" =~ (^|,)system_prompt= ]]; then
        echo "QWEN3_NO_THINK=true requires system_prompt=/no_think; remove system_prompt from VLLM_MODEL_ARGS or set QWEN3_NO_THINK=false." >&2
        exit 2
    fi
    MODEL_ARGS+=",system_prompt=/no_think"
fi

args=(vllm "${MODEL_ARGS}" "${TASK_SPEC}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${LIGHTEVAL_CUSTOM_TASKS}" ]]; then
    args+=(--custom-tasks "${LIGHTEVAL_CUSTOM_TASKS}")
fi
if [[ "${SAVE_DETAILS}" == true ]]; then
    args+=(--save-details)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
    args+=(--max-samples "${MAX_SAMPLES}")
fi
args+=("${EXTRA_ARGS[@]}")

echo "Benchmark: ${BENCHMARK_NAME}"
echo "Tasks:     ${TASK_SPEC}"
echo "Model:     ${MODEL_PATH}"
echo "Model link: ${EVAL_MODEL_PATH} -> ${MODEL_REAL_PATH}"
echo "Cache:     ${LIGHTEVAL_CACHE_ROOT}"
echo "Output:    ${OUTPUT_DIR}"
if [[ "${QWEN3_NO_THINK}" == "true" ]]; then
    echo "Qwen3 NoThink control: system_prompt=/no_think"
else
    echo "Qwen3 NoThink control: disabled (thinking mode requested)"
fi
cd "${REPO_ROOT}"
exec "${LIGHTEVAL_BIN}" "${args[@]}"
