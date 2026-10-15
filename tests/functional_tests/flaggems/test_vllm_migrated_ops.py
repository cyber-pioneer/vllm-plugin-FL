# Copyright 2026 FlagOS Contributors
"""Numerical and buffer-contract checks for the remaining direct imports."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_rms_norm_release_contract(monkeypatch, residual, strided):
    from vllm_fl.dispatch.backends.flaggems.impl.normalization import rms_norm_flaggems

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    x = x[:, ::2] if strided else x[:, :128].contiguous()
    r = torch.randn_like(x) if residual else None
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    obj = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    original = x.clone()
    summed = x.float() + r.float() if residual else x.float()
    expected = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * weight.float()
    result = rms_norm_flaggems(obj, x, r)
    if residual:
        assert result[0] is x and result[1] is r
        torch.testing.assert_close(r, summed.to(r.dtype))
        result = result[0]
    else:
        torch.testing.assert_close(x, original)
    torch.testing.assert_close(result, expected.to(x.dtype), atol=0.04, rtol=0.02)


def test_scaled_mm_release_contract():
    from vllm_fl.ops.flaggems_vllm_adapters import scaled_mm

    a = torch.randint(-32, 32, (7, 64), device="cuda", dtype=torch.int8)
    b = torch.randint(-32, 32, (64, 96), device="cuda", dtype=torch.int8)
    sa = torch.rand(7, 1, device="cuda") * 0.1
    sb = torch.rand(96, device="cuda") * 0.1
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    expected = (a.float() @ b.float()) * sa * sb + bias.float()
    actual = scaled_mm(a, b, sa, sb, bias=bias, out_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=0.125, rtol=0.02)


def _reference(q, k, v, g, beta, initial, lengths, indices=None):
    outputs, states, all_states = [], initial.clone(), []
    start = 0
    for seq, length in enumerate(lengths):
        index = seq if indices is None else int(indices[seq])
        state = states[index].float().transpose(-1, -2)
        for token in range(start, start + length):
            key = k[0, token].float().repeat_interleave(v.shape[2] // k.shape[2], 0)
            query = q[0, token].float().repeat_interleave(v.shape[2] // q.shape[2], 0)
            state = state * g[0, token].float().exp()[:, None, None]
            delta = v[0, token].float() - torch.einsum("hk,hkv->hv", key, state)
            delta = delta * beta[0, token].float()[:, None]
            state = state + key[:, :, None] * delta[:, None, :]
            outputs.append(
                torch.einsum("hk,hkv->hv", query, state) * k.shape[-1] ** -0.5
            )
            all_states.append(state.transpose(-1, -2).clone())
        states[index] = state.transpose(-1, -2)
        start += length
    return torch.stack(outputs)[None].to(v.dtype), states, torch.stack(all_states)


@pytest.mark.parametrize("lengths", [[7], [65], [3, 67]])
@pytest.mark.parametrize("initial_present", [False, True])
def test_chunk_gdn_release_state_layout(lengths, initial_present):
    from vllm_fl.ops.flaggems_vllm_adapters import chunk_gated_delta_rule_fwd

    torch.manual_seed(17)
    total = sum(lengths)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, total, 2, 64, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(1, total, 4, 32, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, total, 4, device="cuda") * 0.1
    beta = torch.rand(1, total, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    state = (
        torch.randn(len(lengths), 4, 32, 64, device="cuda") * 0.1
        if initial_present
        else torch.zeros(len(lengths), 4, 32, 64, device="cuda")
    )
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    expected, final, _ = _reference(q, k, v, g, beta, state, lengths)
    actual = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=64**-0.5,
        initial_state=state if initial_present else None,
        output_final_state=True,
        cu_seqlens=cu,
    )
    torch.testing.assert_close(actual[1], expected, atol=0.02, rtol=0.03)
    torch.testing.assert_close(actual[3], final, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("lengths", [[7], [65], [3, 67]])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_gdn_release_state_layout(lengths, inplace, packed):
    from vllm_fl.ops.flaggems_vllm_adapters import fused_recurrent_gated_delta_rule_fwd

    torch.manual_seed(19)
    total = sum(lengths)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, total, 2, 64, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(1, total, 4, 32, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, total, 4, device="cuda") * 0.1
    beta = torch.rand(1, total, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.randn(len(lengths), 4, 32, 64, device="cuda") * 0.1
    if packed:
        q, k, v = [
            x.as_strided(x.shape, (x.stride(1) + x.stride(2), *x.stride()[1:]))
            for x in (q, k, v)
        ]
    original = state.clone()
    expected, final, all_states = _reference(q, k, v, g, beta, state, lengths)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    actual, actual_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=64**-0.5,
        initial_state=state,
        inplace_final_state=inplace,
        cu_seqlens=cu,
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    if inplace:
        assert actual_state is state
        torch.testing.assert_close(state, final, atol=0.02, rtol=0.03)
    else:
        torch.testing.assert_close(state, original)
        torch.testing.assert_close(actual_state, all_states, atol=0.02, rtol=0.03)


@pytest.mark.parametrize("rms", [False, True])
@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("group_size", [None, 32])
def test_fla_norm_and_gate_release_contract(rms, before, group_size):
    from vllm_fl.ops.flaggems_vllm_adapters import LayerNormFn

    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    w = torch.randn(64, device="cuda")
    b = torch.randn_like(w)
    value = x.float()
    if not before:
        value = value * torch.nn.functional.silu(z.float())
    groups = value.reshape(3, -1, group_size or 64)
    if not rms:
        groups = groups - groups.mean(-1, keepdim=True)
    expected = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected.reshape_as(x) * w + b
    if before:
        expected = expected * torch.nn.functional.silu(z.float())
    actual = LayerNormFn.apply(x, w, b, z, 1e-6, group_size, before, rms)
    torch.testing.assert_close(actual, expected.to(x.dtype), atol=0.04, rtol=0.02)


def test_recurrent_dense_batch_without_initial_state():
    from vllm_fl.ops.flaggems_vllm_adapters import fused_recurrent_gated_delta_rule_fwd

    q, k = [
        torch.nn.functional.normalize(
            torch.randn(2, 3, 2, 64, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(2, 3, 4, 32, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(2, 3, 4, device="cuda") * 0.1
    beta = torch.rand(2, 3, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    flat = [x.reshape(1, 6, *x.shape[2:]) for x in (q, k, v, g, beta)]
    initial = torch.zeros(2, 4, 32, 64, device="cuda")
    expected, final, _ = _reference(*flat, initial, [3, 3])
    actual, actual_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=64**-0.5,
        initial_state=None,
        inplace_final_state=True,
        cu_seqlens=None,
    )
    torch.testing.assert_close(actual, expected.reshape_as(v), atol=0.02, rtol=0.03)
    torch.testing.assert_close(actual_state, final, atol=0.02, rtol=0.03)


def test_recurrent_indexed_state_updates():
    from vllm_fl.ops.flaggems_vllm_adapters import fused_recurrent_gated_delta_rule_fwd

    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, 6, 2, 64, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(1, 6, 4, 32, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, 6, 4, device="cuda") * 0.1
    beta = torch.rand(1, 6, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.randn(4, 4, 32, 64, device="cuda") * 0.1
    indices = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 2, 6], device="cuda", dtype=torch.int32)
    expected, final, _ = _reference(q, k, v, g, beta, state, [2, 4], indices)
    actual, actual_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=64**-0.5,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )
    assert actual_state is state
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    torch.testing.assert_close(state, final, atol=0.02, rtol=0.03)


@pytest.mark.parametrize("residual", [False, True])
def test_rms_norm_cuda_graph(residual):
    from vllm_fl.ops.flaggems_vllm_adapters import rms_norm

    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x) if residual else None
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    for _ in range(3):
        rms_norm(x, r, weight, 1e-6)
    original = x.clone()
    original_r = r.clone() if residual else None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rms_norm(x, r, weight, 1e-6)
    x.copy_(original)
    if residual:
        r.copy_(original_r)
    expected = original.float() + original_r.float() if residual else original.float()
    summed = expected.clone()
    expected = (
        expected
        * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    )
    graph.replay()
    if residual:
        torch.testing.assert_close(actual[1], summed.to(x.dtype))
        actual = actual[0]
    torch.testing.assert_close(actual, expected.to(x.dtype), atol=0.04, rtol=0.02)


@pytest.mark.parametrize("inplace", [False, True])
def test_recurrent_padding_does_not_read_or_update_state_pool(inplace):
    from vllm_fl.ops.flaggems_vllm_adapters import fused_recurrent_gated_delta_rule_fwd

    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, 6, 2, 64, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(1, 6, 4, 32, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, 6, 4, device="cuda") * 0.1
    beta = torch.rand(1, 6, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    state = torch.randn(3, 4, 32, 64, device="cuda") * 0.1
    original = state.clone()
    valid_inputs = [x[:, :2] for x in (q, k, v, g, beta)]
    expected, final, token_states = _reference(*valid_inputs, state, [2], [1])
    expected = torch.cat((expected, torch.zeros_like(v[:, 2:])), dim=1)
    actual, actual_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=64**-0.5,
        initial_state=state,
        inplace_final_state=inplace,
        cu_seqlens=torch.tensor([0, 2, 6], device="cuda", dtype=torch.int32),
        ssm_state_indices=torch.tensor([1, -1], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    if inplace:
        assert actual_state is state
        torch.testing.assert_close(state, final, atol=0.02, rtol=0.03)
    else:
        torch.testing.assert_close(state, original)
        expected_states = torch.cat(
            (token_states, torch.zeros_like(token_states[:1]).expand(4, -1, -1, -1))
        )
        torch.testing.assert_close(actual_state, expected_states, atol=0.02, rtol=0.03)
