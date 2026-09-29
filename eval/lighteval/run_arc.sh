#!/usr/bin/env bash
# Run both ARC-Easy and ARC-Challenge using the conventional 25-shot setup.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_benchmark.sh" arc "${TASK_SPEC:-arc:easy|25,arc:challenge|25}" "$@"
