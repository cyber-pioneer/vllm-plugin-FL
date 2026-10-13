#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

uv pip install --system --break-system-packages --upgrade pip

# Existing CI images may contain an older FlagGems release.
if ! python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("flag-gems")
except PackageNotFoundError:
    installed = None
raise SystemExit(installed != "5.4.0")
PY
then
    uv pip install --system --break-system-packages --no-build-isolation --no-deps \
        "flag-gems @ git+https://github.com/flagos-ai/FlagGems.git@v5.4.0"
fi

# GPU functional tests cover the optional fused package in both switch states.
uv pip install --system --break-system-packages --no-build-isolation -e ".[test,flaggems-vllm]"

python3 - <<'PY'
from importlib.metadata import version

installed = version("flag-gems")
assert installed == "5.4.0", f"CUDA CI requires FlagGems 5.4.0, found {installed}"
print(f"FlagGems: {installed}")
PY
