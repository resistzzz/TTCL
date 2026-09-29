#!/usr/bin/env bash
# GPQA Diamond with the standard zero-shot setup.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_benchmark.sh" gpqa_diamond "${TASK_SPEC:-gpqa:diamond|0}" "$@"
