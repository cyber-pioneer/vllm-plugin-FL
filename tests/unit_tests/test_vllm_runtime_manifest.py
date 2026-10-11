# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image packaging must preserve all installed vLLM files, not just versions."""

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "docker/kunlunxin/verify_vllm_runtime.py"
_spec = importlib.util.spec_from_file_location("verify_vllm_runtime", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_snapshot_retains_runtime_and_non_source_files(tmp_path):
    for name in (
        "benchmarks/serve.py",
        "testing/utils.py",
        "__pycache__/module.pyc",
        "data.bin",
    ):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"original bytes")
    manifest = _module.snapshot(tmp_path)
    assert "benchmarks/serve.py" in manifest["entries"]
    assert "testing/utils.py" in manifest["entries"]
    assert "__pycache__/module.pyc" in manifest["entries"]
    assert "data.bin" in manifest["entries"]
    _module.verify(tmp_path, manifest)


@pytest.mark.parametrize("change", ["missing", "extra", "changed", "permission"])
def test_manifest_rejects_package_changes(tmp_path, change):
    path = tmp_path / "original.py"
    path.write_bytes(b"original bytes")
    manifest = _module.snapshot(tmp_path)
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (tmp_path / "extra.py").touch()
    elif change == "changed":
        path.write_bytes(b"modified bytes")
    else:
        path.chmod(0o600 if path.stat().st_mode & 0o777 != 0o600 else 0o644)
    with pytest.raises(ValueError):
        _module.verify(tmp_path, manifest)


def test_manifest_checks_symlink_target(tmp_path):
    target = tmp_path / "target"
    target.touch()
    link = tmp_path / "link"
    link.symlink_to("target")
    manifest = _module.snapshot(tmp_path)
    _module.verify(tmp_path, manifest)
    link.unlink()
    link.symlink_to("missing")
    with pytest.raises(ValueError):
        _module.verify(tmp_path, manifest)
