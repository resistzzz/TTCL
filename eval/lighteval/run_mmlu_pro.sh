#!/usr/bin/env bash
# MMLU-Pro 5-shot via the project-local, chat-template-safe task.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_benchmark.sh" mmlu_pro "${TASK_SPEC:-mmlu_pro_fixed|5}" "$@"
