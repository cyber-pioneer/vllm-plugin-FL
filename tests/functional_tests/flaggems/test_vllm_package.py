# Copyright 2026 FlagOS Contributors
"""Exercise migrated fused operators through the plugin boundary."""

from types import SimpleNamespace

import pytest
import torch

OPS = (
    "silu_and_mul",
    "silu_and_mul_with_clamp",
    "gelu_and_mul",
    "apply_rotary_pos_emb",
    "flash_attn_varlen_func",
    "reshape_and_cache_flash",
    "flash_mla",
    "moe_align_block_size_triton",
    "topk_softmax",
    "invoke_fused_moe_triton_kernel",
    "grouped_topk",
    "moe_sum",
    "fused_experts_impl",
    "chunk_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_fwd",
)

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to import the operator runtime")


@pytest.mark.parametrize("name", OPS)
def test_release_exports(name):
    import flaggems_vllm

    op = getattr(flaggems_vllm, name)
    assert callable(op)
    assert op.__module__.startswith("flaggems_vllm.")


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_activation_dispatch(monkeypatch, graph, activation):
    import flaggems_vllm

    from vllm_fl.dispatch.backends.flaggems.impl.activation import (
        gelu_and_mul_flaggems,
        silu_and_mul_flaggems,
    )

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    op_name = f"{activation}_and_mul"
    calls = []
    real_op = getattr(flaggems_vllm, op_name)

    def observed(*args, **kwargs):
        calls.append(real_op.__module__)
        return real_op(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, op_name, observed)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    obj = SimpleNamespace(approximate="none")
    wrapper = silu_and_mul_flaggems if activation == "silu" else gelu_and_mul_flaggems
    expected = getattr(torch.nn.functional, activation)(x[:, :128]) * x[:, 128:]
    for _ in range(3):
        actual = wrapper(obj, x)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            actual = wrapper(obj, x)
        capture.replay()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert calls and all(module.startswith("flaggems_vllm.") for module in calls)


@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_inplace_dispatch(monkeypatch, renormalize):
    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import topk_softmax_flaggems

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    logits = torch.randn(16, 64, device="cuda")
    weights = torch.empty(16, 4, device="cuda")
    ids = torch.empty(16, 4, device="cuda", dtype=torch.int32)
    token_ids = torch.empty_like(ids)
    expected_weights, expected_ids = logits.softmax(-1).topk(4)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    result_weights, result_ids = topk_softmax_flaggems(
        weights, ids, token_ids, logits, renormalize
    )
    assert result_weights.data_ptr() == weights.data_ptr()
    assert result_ids.data_ptr() == ids.data_ptr()
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids.long(), expected_ids)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.702, 1.0)])
def test_clamped_activation(monkeypatch, graph, alpha, beta):
    import flaggems_vllm

    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_fl.ops.activation import SiluAndMulWithClampFL

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    calls = []
    real_op = flaggems_vllm.silu_and_mul_with_clamp

    def observed(*args, **kwargs):
        calls.append(real_op.__module__)
        return real_op(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, "silu_and_mul_with_clamp", observed)
    with set_current_vllm_config(VllmConfig()):
        obj = SiluAndMulWithClampFL(2.0, alpha, beta, compile_native=False)
    x = torch.randn(16, 128, device="cuda") * 4
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=2.0)
    expected = gate * torch.sigmoid(alpha * gate) * (up.clamp(-2.0, 2.0) + beta)
    for _ in range(3):
        actual = obj.forward_oot(x)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            actual = obj.forward_oot(x)
        capture.replay()
    torch.testing.assert_close(actual, expected)
    assert bool(calls) == (alpha == 1.0 and beta == 0.0)


@pytest.mark.parametrize("activation", ["silu", "gelu", "clamp"])
def test_compiled_activation_uses_fused_package(monkeypatch, activation):
    import flaggems_vllm

    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_fl.ops.activation import (
        GeluAndMulFL,
        SiluAndMulFL,
        SiluAndMulWithClampFL,
    )

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    name = (
        "silu_and_mul_with_clamp" if activation == "clamp" else f"{activation}_and_mul"
    )
    real = getattr(flaggems_vllm, name)
    calls = []

    def observed(*args, **kwargs):
        calls.append(real.__module__)
        return real(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, name, observed)
    with set_current_vllm_config(VllmConfig()):
        obj = {
            "silu": SiluAndMulFL,
            "gelu": GeluAndMulFL,
            "clamp": lambda: SiluAndMulWithClampFL(2.0, 1.0, 0.0, compile_native=False),
        }[activation]()
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    eager = obj.forward_oot(x)
    calls.clear()
    compiled = torch.compile(obj.forward_oot, fullgraph=True)
    actual = compiled(x)
    torch.testing.assert_close(actual, eager)
    assert calls and all(module.startswith("flaggems_vllm.") for module in calls)


def test_compile_activation_selection_respects_operator_controls(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.ops.activation import enable_fused_activations_for_compile

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul,gelu_and_mul")
    config = SimpleNamespace(custom_ops=["none", "-GeluAndMul"])
    enable_fused_activations_for_compile(config)
    enable_fused_activations_for_compile(config)
    assert config.custom_ops == ["none", "-GeluAndMul", "+SiluAndMul"]
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    untouched = SimpleNamespace(custom_ops=["none"])
    enable_fused_activations_for_compile(untouched)
    assert untouched.custom_ops == ["none"]
