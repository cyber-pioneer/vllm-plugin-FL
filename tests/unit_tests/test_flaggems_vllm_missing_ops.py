# Copyright 2026 FlagOS Contributors
"""An unavailable fused operator must not silently use FlagGems."""

import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("name", ["pack_rhs_qsi4c128p", "w4a8_g128_linear"])
def test_unavailable_release_op_is_explicit(monkeypatch, name):
    from vllm_fl.ops.flaggems_vllm_adapters import required_op

    monkeypatch.setitem(sys.modules, "flaggems_vllm", ModuleType("flaggems_vllm"))
    with pytest.raises(NotImplementedError, match=name):
        required_op(name)
