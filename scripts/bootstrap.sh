#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mirror="${SUPERMIMIC_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
python scripts/fetch_sources.py

# Run once for a new environment. This script never modifies behaviour.
if ! conda run -n supermimic python -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
    conda create -n supermimic --file configs/conda-linux-64.lock.txt -y
fi

# Fetch only the special Torch wheel from its own index; ordinary dependencies
# come from the chosen PyPI mirror, preventing multi-GB CUDA downloads there.
conda run --no-capture-output -n supermimic python -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
conda run --no-capture-output -n supermimic python -m pip install --no-cache-dir --no-deps \
    --index-url "$mirror" -r configs/requirements-linux-cu128-portable.lock.txt
conda run --no-capture-output -n supermimic python -m pip install --no-cache-dir --no-deps \
    --index-url "$mirror" --build-constraint configs/build-constraints.txt -e vendor/mjlab -e .
conda run --no-capture-output -n supermimic python -m pip check
conda run --no-capture-output -n supermimic python scripts/check_gpu.py
