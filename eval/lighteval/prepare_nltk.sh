#!/usr/bin/env bash
# Download and validate IFEval's NLTK resources once, before parallel jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NLTK_CACHE_DIR="${NLTK_CACHE_DIR:-${SCRIPT_DIR}/nltk_data}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Cannot find PYTHON_BIN '${PYTHON_BIN}'." >&2
    exit 127
fi

mkdir -p "${NLTK_CACHE_DIR}"
NLTK_DATA="${NLTK_CACHE_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import shutil
from pathlib import Path

import nltk

cache_dir = Path(os.environ["NLTK_DATA"])

for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
        print(f"NLTK resource ready: {resource} ({cache_dir})")
        continue
    except LookupError:
        pass

    # Repair only this resource in the dedicated evaluation cache.
    for target in (cache_dir / "tokenizers" / resource, cache_dir / "tokenizers" / f"{resource}.zip"):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print(f"Downloading NLTK resource: {resource} -> {cache_dir}")
    if not nltk.download(resource, download_dir=str(cache_dir), raise_on_error=True):
        raise RuntimeError(f"Failed to download NLTK resource: {resource}")
    nltk.data.find(f"tokenizers/{resource}")
PY
