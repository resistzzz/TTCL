#!/usr/bin/env bash
# TruthfulQA multiple-choice: reports MC1/MC2-style metrics where provided by Lighteval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_benchmark.sh" truthfulqa "${TASK_SPEC:-truthfulqa:mc|0}" "$@"
