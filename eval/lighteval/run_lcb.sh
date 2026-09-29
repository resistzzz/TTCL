#!/usr/bin/env bash
# LiveCodeBench code-generation evaluation. Generated Python is executed by
# Lighteval's scorer on this host; see README before enabling it.
set -euo pipefail

LCB_ALLOW_UNSANDBOXED_EXECUTION="${LCB_ALLOW_UNSANDBOXED_EXECUTION:-false}"
LCB_ALLOW_UNSANDBOXED_EXECUTION="${LCB_ALLOW_UNSANDBOXED_EXECUTION,,}"
if [[ "${LCB_ALLOW_UNSANDBOXED_EXECUTION}" != "true" ]]; then
    echo "Refusing to run LiveCodeBench: its scorer executes model-generated Python on this host." >&2
    echo "Run only in an isolated worker/container, then set LCB_ALLOW_UNSANDBOXED_EXECUTION=true." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_benchmark.sh" lcb "${TASK_SPEC:-lcb:codegeneration_release_v5|0}" "$@"
