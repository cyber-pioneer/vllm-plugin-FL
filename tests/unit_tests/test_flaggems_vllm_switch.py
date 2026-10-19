# Copyright 2026 FlagOS Contributors

import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize(
    "value, expected", [(None, False), ("0", False), ("1", True), ("TRUE", True)]
)
def test_optional_fused_provider_defaults_off(monkeypatch, value, expected):
    import vllm_fl.envs as envs

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    if value is not None:
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", value)
    assert envs.VLLM_FL_USE_FLAGGEMS_VLLM is expected


def test_default_quantization_does_not_import_optional_package(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import quantization

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    value, expected = object(), (object(), object())
    monkeypatch.setattr(
        quantization,
        "dynamic_per_token_quant_int8_flaggems_triton",
        lambda x: expected if x is value else None,
    )
    assert quantization.dynamic_per_token_quant_int8_flaggems_vllm(value) is expected


@pytest.mark.parametrize("mode", ["0", "1"])
def test_fused_experts_selects_provider_without_changing_arguments(monkeypatch, mode):
    from vllm_fl.quantization.w8a8 import moe_experts

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", mode)
    calls = []
    for name in ("flag_gems", "flaggems_vllm"):
        module = ModuleType(name)

        def invoke(_name=name, **kwargs):
            calls.append((_name, kwargs))
            return kwargs["hidden_states"]

        module.fused_experts_impl = invoke
        monkeypatch.setitem(sys.modules, name, module)
    value = object()
    assert (
        moe_experts._flaggems_fused_experts_impl(hidden_states=value, inplace=False)
        is value
    )
    assert calls == [
        (
            "flaggems_vllm" if mode == "1" else "flag_gems",
            {"hidden_states": value, "inplace": False},
        )
    ]
