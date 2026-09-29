#!/usr/bin/env bash
# Download and materialize the LiveCodeBench release used by run_suite.sh once,
# before any GPU jobs start. This prevents parallel LCB workers from racing on
# the same Hugging Face shards and cache locks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LCB_DATASET_REPO="${LCB_DATASET_REPO:-lighteval/code_generation_lite}"
LCB_DATASET_CONFIG="${LCB_DATASET_CONFIG:-release_v5}"
LCB_DATASET_REVISION="${LCB_DATASET_REVISION:-89e5fc5c2a8e748f50e95bc7235fab2372d49bfa}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Cannot find PYTHON_BIN=${PYTHON_BIN}" >&2
    exit 127
fi

echo "Preparing LCB dataset ${LCB_DATASET_REPO}/${LCB_DATASET_CONFIG}..."
"${PYTHON_BIN}" - <<PY
from datasets import load_dataset
from huggingface_hub import snapshot_download

repo = "${LCB_DATASET_REPO}"
config = "${LCB_DATASET_CONFIG}"
revision = "${LCB_DATASET_REVISION}"

snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    revision=revision,
    allow_patterns=[f"{config}/*"],
)
dataset = load_dataset(repo, config, revision=revision, split="test")
print(f"LCB dataset ready: {len(dataset)} problems ({repo}/{config})")
PY
