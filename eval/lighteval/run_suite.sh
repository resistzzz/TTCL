#!/usr/bin/env bash
# Evaluate every configured model on every configured Lighteval benchmark.
# Edit MODEL_LIST and BENCH_LIST below, then run this file from any directory.
set -euo pipefail

# Paths may be absolute or relative to the repository root.
MODEL_LIST=(
    PATH_TO_Qwen3-4B
    PATH_TO_Qwen3-8B
    # Add the model you want to evaluate here

)

# Optional output names paired by index with MODEL_LIST. Use this to avoid
# collisions when multiple paths end with e.g. `global_step_120`. Each value
# becomes the model-level directory below OUTPUT_ROOT and may be a relative
# nested path (for example, `Qwen3-4B/Base_NoThink`). Leave this array empty
# to retain the legacy behaviour of using the final model-path component.
MODEL_OUTPUT_NAMES=(
    # Qwen3-4B/Base_NoThink
    # Qwen3-8B/Base_NoThink
)

# Supported names: mmlu_pro, ifeval, gpqa_diamond, truthfulqa, arc, lcb.
# `arc` runs ARC-Easy and ARC-Challenge together.
BENCH_LIST=(
    mmlu_pro
    ifeval
    gpqa_diamond
    truthfulqa
    arc
    # lcb  # requires LCB_ALLOW_UNSANDBOXED_EXECUTION=true; see README
)

# One independent model × benchmark job is assigned to one entry in GPU_LIST.
# For a four-GPU node and models that fit on one card, keep --tp 1 and leave
# this list as below. Change it when Slurm/CUDA_VISIBLE_DEVICES exposes a
# different set of device IDs.
GPU_LIST=(0 1 2 3)

# Number of concurrent jobs. It must not exceed the number of GPUs above.
# By default, all listed GPUs are used concurrently.
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"

# Check that a card has enough free memory for vLLM before assigning a job.
# `wait` avoids turning temporary contention from another process into a long
# list of failed benchmarks. Set GPU_BUSY_POLICY=fail for immediate failure.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
GPU_BUSY_POLICY="${GPU_BUSY_POLICY:-wait}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"

# Lighteval imports IFEval's NLTK helpers while building the task registry.
# Prepare them once before starting parallel jobs: concurrent first-time
# downloads can corrupt the shared punkt zip cache.
PREPARE_NLTK="${PREPARE_NLTK:-true}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARE_LCB="${PREPARE_LCB:-true}"

# GPQA-Diamond reserves 32768 generation tokens. Qwen3-4B supports 40960
# tokens, whereas Lighteval otherwise uses an 8192-token wrapper default.
# Set this empty to disable injection, or override it for another model family.
GPQA_MAX_MODEL_LENGTH="${GPQA_MAX_MODEL_LENGTH:-40960}"

# MMLU-Pro uses eval/lighteval/custom_tasks.py rather than Lighteval's broken
# built-in task. The fixed prompt requests concise visible reasoning and a
# final Answer line. 1024 leaves sufficient room for that line; 512 was shown
# to truncate many Qwen3 NoThink responses before their answer.
MMLU_PRO_MAX_NEW_TOKENS="${MMLU_PRO_MAX_NEW_TOKENS:-1024}"

# The built-in LiveCodeBench task reserves 32768 tokens. 4096 is a practical
# default for Python solutions; override only if a code task demonstrably needs
# more output. Its scorer executes generated code on the host, so lcb is
# deliberately opt-in via LCB_ALLOW_UNSANDBOXED_EXECUTION=true.
LCB_MAX_NEW_TOKENS="${LCB_MAX_NEW_TOKENS:-4096}"
LCB_ALLOW_UNSANDBOXED_EXECUTION="${LCB_ALLOW_UNSANDBOXED_EXECUTION:-false}"
LCB_ALLOW_UNSANDBOXED_EXECUTION="${LCB_ALLOW_UNSANDBOXED_EXECUTION,,}"

# This suite is intended for Qwen3 non-thinking evaluation. Lighteval cannot
# currently pass enable_thinking=False into Qwen's chat template, so the common
# launcher inserts Qwen's official /no_think system-prompt switch for every
# task. Do not disable this unless intentionally evaluating thinking mode.
QWEN3_NO_THINK="${QWEN3_NO_THINK:-true}"

# Task aliases and few-shot settings. Override a value before running if the
# Lighteval version in the active environment uses a different task spelling.
TASK_SPEC_MMLU_PRO="${TASK_SPEC_MMLU_PRO:-mmlu_pro_fixed|5}"
TASK_SPEC_IFEVAL="${TASK_SPEC_IFEVAL:-ifeval}"
TASK_SPEC_GPQA_DIAMOND="${TASK_SPEC_GPQA_DIAMOND:-gpqa:diamond|0}"
TASK_SPEC_TRUTHFULQA="${TASK_SPEC_TRUTHFULQA:-truthfulqa:mc|0}"
TASK_SPEC_ARC="${TASK_SPEC_ARC:-arc:easy|25,arc:challenge|25}"
TASK_SPEC_LCB="${TASK_SPEC_LCB:-lcb:codegeneration_release_v5|0}"

usage() {
    cat <<'EOF'
Usage: bash eval/lighteval/run_suite.sh [options]

Edit MODEL_LIST and BENCH_LIST in this script to select the Cartesian product
of models and benchmarks. All remaining options are forwarded to each run.

Suite option:
  --output-root PATH  Parent directory for all results.
                      Default: eval/lighteval/results
  --max-parallel N    Maximum concurrent jobs. Default: number of GPUs in GPU_LIST.

Environment options:
  PREPARE_NLTK=false  Skip the one-time NLTK cache preparation.
  PREPARE_LCB=false   Skip the one-time LCB dataset preparation.
  NLTK_CACHE_DIR=PATH Dedicated shared NLTK cache (default: eval/lighteval/nltk_data).
  PYTHON_BIN=PATH     Python interpreter used for NLTK preparation.
  GPQA_MAX_MODEL_LENGTH=N  Context length used for GPQA-Diamond (default: 40960).
  MMLU_PRO_MAX_NEW_TOKENS=N  MMLU-Pro generation cap (default: 1024).
  LCB_MAX_NEW_TOKENS=N  LiveCodeBench code-generation cap (default: 4096).
  LCB_ALLOW_UNSANDBOXED_EXECUTION=true  Required to run LCB; generated Python
                      is executed locally by Lighteval's scorer.
  QWEN3_NO_THINK=false  Evaluate Qwen3 thinking mode instead (default: true).
  GPU_MEMORY_UTILIZATION=F  Required free-memory fraction before launch (default: 0.85).
  GPU_BUSY_POLICY=wait|fail Wait for a usable GPU, or fail instead (default: wait).
  GPU_WAIT_SECONDS=N   Wait interval in seconds when GPUs are busy (default: 30).

Common examples:
  bash eval/lighteval/run_suite.sh --tp 1 --gpu-memory-utilization 0.90
  bash eval/lighteval/run_suite.sh --max-samples 5    # smoke test only

See run_benchmark.sh --help for all common options.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/eval/lighteval/results}"
NLTK_CACHE_DIR="${NLTK_CACHE_DIR:-${REPO_ROOT}/eval/lighteval/nltk_data}"
COMMON_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            COMMON_ARGS+=("$1" "$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            COMMON_ARGS+=("$1")
            shift
            ;;
    esac
done

if (( ${#MODEL_LIST[@]} == 0 )); then
    echo "MODEL_LIST is empty. Add one or more checkpoints to eval/lighteval/run_suite.sh." >&2
    exit 2
fi
if (( ${#MODEL_OUTPUT_NAMES[@]} != 0 && ${#MODEL_OUTPUT_NAMES[@]} != ${#MODEL_LIST[@]} )); then
    echo "MODEL_OUTPUT_NAMES must be empty or contain one name for every MODEL_LIST entry." >&2
    exit 2
fi
if (( ${#BENCH_LIST[@]} == 0 )); then
    echo "BENCH_LIST is empty. Add one or more supported benchmark names." >&2
    exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "GPU_LIST is empty. Add one or more GPU device IDs to run_suite.sh." >&2
    exit 2
fi
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-parallel must be a positive integer; got ${MAX_PARALLEL}" >&2
    exit 2
fi
if (( MAX_PARALLEL > ${#GPU_LIST[@]} )); then
    echo "--max-parallel (${MAX_PARALLEL}) cannot exceed GPU_LIST length (${#GPU_LIST[@]})." >&2
    exit 2
fi
if [[ -n "${GPQA_MAX_MODEL_LENGTH}" && ! "${GPQA_MAX_MODEL_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPQA_MAX_MODEL_LENGTH must be a positive integer or empty; got ${GPQA_MAX_MODEL_LENGTH}" >&2
    exit 2
fi
if [[ -n "${MMLU_PRO_MAX_NEW_TOKENS}" && ! "${MMLU_PRO_MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MMLU_PRO_MAX_NEW_TOKENS must be a positive integer or empty; got ${MMLU_PRO_MAX_NEW_TOKENS}" >&2
    exit 2
fi
if [[ -n "${LCB_MAX_NEW_TOKENS}" && ! "${LCB_MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LCB_MAX_NEW_TOKENS must be a positive integer or empty; got ${LCB_MAX_NEW_TOKENS}" >&2
    exit 2
fi
if [[ "${QWEN3_NO_THINK}" != "true" && "${QWEN3_NO_THINK}" != "false" ]]; then
    echo "QWEN3_NO_THINK must be true or false; got ${QWEN3_NO_THINK}" >&2
    exit 2
fi
if ! [[ "${GPU_MEMORY_UTILIZATION}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || [[ "${GPU_MEMORY_UTILIZATION}" == "0" ]]; then
    echo "GPU_MEMORY_UTILIZATION must be in (0, 1]; got ${GPU_MEMORY_UTILIZATION}" >&2
    exit 2
fi
if [[ "${GPU_BUSY_POLICY}" != "wait" && "${GPU_BUSY_POLICY}" != "fail" ]]; then
    echo "GPU_BUSY_POLICY must be wait or fail; got ${GPU_BUSY_POLICY}" >&2
    exit 2
fi
if ! [[ "${GPU_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_WAIT_SECONDS must be a positive integer; got ${GPU_WAIT_SECONDS}" >&2
    exit 2
fi

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

# A merged checkpoint is normally a HuggingFace directory containing
# config.json. When a user specifies a run directory with exactly one nested
# checkpoint (e.g. only global_step_60/), resolve that unambiguously here.
# This prevents a long suite from failing only after vLLM begins loading.
for model_index in "${!MODEL_LIST[@]}"; do
    candidate_path="${MODEL_LIST[${model_index}]}"
    if [[ "${candidate_path}" != /* ]]; then
        candidate_path="${REPO_ROOT}/${candidate_path}"
    fi
    if [[ ! -d "${candidate_path}" ]]; then
        echo "Model directory does not exist at MODEL_LIST index ${model_index}: ${candidate_path}" >&2
        exit 2
    fi
    if [[ ! -f "${candidate_path}/config.json" ]]; then
        mapfile -t nested_configs < <(find "${candidate_path}" -mindepth 2 -maxdepth 2 -type f -name config.json -printf '%h\n' | sort -u)
        if (( ${#nested_configs[@]} == 1 )); then
            echo "Resolved MODEL_LIST[${model_index}] to its only nested HF checkpoint: ${nested_configs[0]}"
            candidate_path="${nested_configs[0]}"
        else
            echo "MODEL_LIST[${model_index}] is not a HuggingFace model directory: ${candidate_path}" >&2
            if (( ${#nested_configs[@]} > 1 )); then
                echo "Found multiple nested checkpoints; specify one explicitly:" >&2
                printf '  %s\n' "${nested_configs[@]}" >&2
            else
                echo "Expected ${candidate_path}/config.json." >&2
            fi
            exit 2
        fi
    fi
    MODEL_LIST[${model_index}]="${candidate_path}"
done

task_spec_for() {
    case "$1" in
        mmlu_pro) printf '%s\n' "${TASK_SPEC_MMLU_PRO}" ;;
        ifeval) printf '%s\n' "${TASK_SPEC_IFEVAL}" ;;
        gpqa_diamond) printf '%s\n' "${TASK_SPEC_GPQA_DIAMOND}" ;;
        truthfulqa) printf '%s\n' "${TASK_SPEC_TRUTHFULQA}" ;;
        arc) printf '%s\n' "${TASK_SPEC_ARC}" ;;
        lcb)
            if [[ "${LCB_ALLOW_UNSANDBOXED_EXECUTION}" != "true" ]]; then
                echo "LiveCodeBench executes model-generated Python; run in an isolated worker and set LCB_ALLOW_UNSANDBOXED_EXECUTION=true." >&2
                return 2
            fi
            printf '%s\n' "${TASK_SPEC_LCB}"
            ;;
        *)
            echo "Unsupported benchmark '${1}'. Supported: mmlu_pro, ifeval, gpqa_diamond, truthfulqa, arc, lcb." >&2
            return 2
            ;;
    esac
}

# Resolve every requested benchmark before launching any background GPU job.
# In particular, an LCB acknowledgement failure must not leave already-started
# MMLU-Pro jobs orphaned when the suite exits under `set -e`.
for benchmark in "${BENCH_LIST[@]}"; do
    task_spec_for "${benchmark}" >/dev/null
done

prepare_nltk_resources() {
    if [[ "${PREPARE_NLTK}" != "true" ]]; then
        echo "Skipping NLTK cache preparation (PREPARE_NLTK=${PREPARE_NLTK})."
        return 0
    fi
    echo "Preparing dedicated NLTK cache before launching parallel evaluations: ${NLTK_CACHE_DIR}"
    NLTK_CACHE_DIR="${NLTK_CACHE_DIR}" PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/prepare_nltk.sh"
}

prepare_lcb_resources() {
    local benchmark
    for benchmark in "${BENCH_LIST[@]}"; do
        if [[ "${benchmark}" == "lcb" ]]; then
            if [[ "${PREPARE_LCB}" != "true" ]]; then
                echo "Skipping LCB dataset preparation (PREPARE_LCB=${PREPARE_LCB})."
                return 0
            fi
            echo "Preparing the LCB dataset once before launching parallel evaluations."
            PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/prepare_lcb.sh"
            return 0
        fi
    done
}

cd "${REPO_ROOT}"
prepare_nltk_resources
prepare_lcb_resources
export NLTK_DATA="${NLTK_CACHE_DIR}${NLTK_DATA:+:${NLTK_DATA}}"
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_DESCRIPTIONS=()
declare -a ACTIVE_GPUS=()
declare -a AVAILABLE_GPUS=("${GPU_LIST[@]}")

wait_for_one_active_job() {
    local completed_pid=""
    local status=0
    local index

    if wait -n -p completed_pid "${ACTIVE_PIDS[@]}"; then
        status=0
    else
        status=$?
    fi

    for index in "${!ACTIVE_PIDS[@]}"; do
        if [[ "${ACTIVE_PIDS[${index}]}" == "${completed_pid}" ]]; then
            if (( status == 0 )); then
                echo "COMPLETED: ${ACTIVE_DESCRIPTIONS[${index}]}"
            else
                echo "FAILED: ${ACTIVE_DESCRIPTIONS[${index}]}" >&2
            fi
            AVAILABLE_GPUS+=("${ACTIVE_GPUS[${index}]}")
            unset 'ACTIVE_PIDS[index]' 'ACTIVE_DESCRIPTIONS[index]' 'ACTIVE_GPUS[index]'
            ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
            ACTIVE_DESCRIPTIONS=("${ACTIVE_DESCRIPTIONS[@]}")
            ACTIVE_GPUS=("${ACTIVE_GPUS[@]}")
            return "${status}"
        fi
    done

    echo "Unable to associate completed process '${completed_pid}' with an active GPU job." >&2
    return 1
}

gpu_has_required_free_memory() {
    local requested_gpu="$1"
    local gpu_index free_memory total_memory

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is unavailable; cannot check GPU free memory." >&2
        return 2
    fi
    while IFS=',' read -r gpu_index free_memory total_memory; do
        gpu_index="${gpu_index//[[:space:]]/}"
        free_memory="${free_memory//[[:space:]]/}"
        total_memory="${total_memory//[[:space:]]/}"
        if [[ "${gpu_index}" == "${requested_gpu}" ]]; then
            awk -v free="${free_memory}" -v total="${total_memory}" -v utilization="${GPU_MEMORY_UTILIZATION}" \
                'BEGIN { exit !(free > total * utilization) }'
            return $?
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
    echo "GPU ${requested_gpu} was not returned by nvidia-smi." >&2
    return 2
}

take_ready_gpu() {
    local index gpu
    for index in "${!AVAILABLE_GPUS[@]}"; do
        gpu="${AVAILABLE_GPUS[${index}]}"
        if gpu_has_required_free_memory "${gpu}"; then
            SELECTED_GPU="${gpu}"
            unset 'AVAILABLE_GPUS[index]'
            AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]}")
            return 0
        fi
    done
    return 1
}

wait_for_ready_gpu() {
    while ! take_ready_gpu; do
        if [[ "${GPU_BUSY_POLICY}" == "fail" ]]; then
            echo "No GPU in GPU_LIST has more than ${GPU_MEMORY_UTILIZATION} of total memory free." >&2
            return 1
        fi
        echo "All currently available GPUs are busy; waiting ${GPU_WAIT_SECONDS}s for a card with >${GPU_MEMORY_UTILIZATION} free memory..." >&2
        sleep "${GPU_WAIT_SECONDS}"
    done
}

overall_status=0
for model_index in "${!MODEL_LIST[@]}"; do
    model_path="${MODEL_LIST[${model_index}]}"
    if [[ "${model_path}" != /* ]]; then
        model_path="${REPO_ROOT}/${model_path}"
    fi
    if (( ${#MODEL_OUTPUT_NAMES[@]} > 0 )); then
        model_name="${MODEL_OUTPUT_NAMES[${model_index}]}"
        if [[ -z "${model_name}" || "${model_name}" = /* || "/${model_name}/" == *"/../"* || "/${model_name}/" == *"/./"* ]]; then
            echo "Invalid MODEL_OUTPUT_NAMES entry at index ${model_index}: '${model_name}'" >&2
            echo "Use a non-empty relative path without '.' or '..' components." >&2
            exit 2
        fi
    else
        model_name="$(basename "${model_path%/}")"
    fi

    for benchmark in "${BENCH_LIST[@]}"; do
        # Unlike batch scheduling, wait only until any job completes; its GPU
        # is returned to AVAILABLE_GPUS and immediately reused below.
        while (( ${#ACTIVE_PIDS[@]} >= MAX_PARALLEL )); do
            if ! wait_for_one_active_job; then
                overall_status=1
            fi
        done

        task_spec="$(task_spec_for "${benchmark}")"
        output_dir="${OUTPUT_ROOT}/${model_name}/${benchmark}"
        if (( ${#AVAILABLE_GPUS[@]} == 0 )); then
            echo "Scheduler error: no available GPU after waiting for an active job." >&2
            exit 1
        fi
        if ! wait_for_ready_gpu; then
            exit 1
        fi
        gpu="${SELECTED_GPU}"
        job_model_args="${VLLM_MODEL_ARGS:-}"
        if [[ "${benchmark}" == "mmlu_pro" && -n "${MMLU_PRO_MAX_NEW_TOKENS}" ]]; then
            if ! [[ "${job_model_args}" =~ (^|,)generation_parameters= ]]; then
                if [[ -n "${job_model_args}" ]]; then
                    job_model_args+="," 
                fi
                job_model_args+="generation_parameters={max_new_tokens:${MMLU_PRO_MAX_NEW_TOKENS}}"
            fi
        fi
        if [[ "${benchmark}" == "gpqa_diamond" && -n "${GPQA_MAX_MODEL_LENGTH}" ]]; then
            if ! [[ "${job_model_args}" =~ (^|,)max_model_length= ]]; then
                if [[ -n "${job_model_args}" ]]; then
                    job_model_args+="," 
                fi
                job_model_args+="max_model_length=${GPQA_MAX_MODEL_LENGTH}"
            fi
        fi
        if [[ "${benchmark}" == "lcb" && -n "${LCB_MAX_NEW_TOKENS}" ]]; then
            if ! [[ "${job_model_args}" =~ (^|,)generation_parameters= ]]; then
                if [[ -n "${job_model_args}" ]]; then
                    job_model_args+="," 
                fi
                job_model_args+="generation_parameters={max_new_tokens:${LCB_MAX_NEW_TOKENS}}"
            fi
        fi

        echo
        echo "===== ${model_name} × ${benchmark} (GPU ${gpu}) ====="
        mkdir -p "${output_dir}"
        QWEN3_NO_THINK="${QWEN3_NO_THINK}" VLLM_MODEL_ARGS="${job_model_args}" CUDA_VISIBLE_DEVICES="${gpu}" bash "${SCRIPT_DIR}/run_benchmark.sh" \
            "${benchmark}" "${task_spec}" \
            --model "${model_path}" \
            --output-dir "${output_dir}" \
            "${COMMON_ARGS[@]}" \
            >"${output_dir}/launcher.log" 2>&1 &
        ACTIVE_PIDS+=("$!")
        ACTIVE_DESCRIPTIONS+=("${model_name} × ${benchmark} (GPU ${gpu}; log: ${output_dir}/launcher.log)")
        ACTIVE_GPUS+=("${gpu}")
    done
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    if ! wait_for_one_active_job; then
        overall_status=1
    fi
done
exit "${overall_status}"
