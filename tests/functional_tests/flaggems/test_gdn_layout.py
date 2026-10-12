# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch
import triton


@pytest.fixture(autouse=True)
def enable_fused_package(monkeypatch):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")


def _inputs(key_dim, lengths):
    torch.manual_seed(17)
    total = sum(lengths)
    q = torch.nn.functional.normalize(
        torch.randn(1, total, 2, key_dim, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn(1, total, 6, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, total, 6, device="cuda") * 0.2
    beta = torch.rand(1, total, 6, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(
            size, device="cuda", dtype=torch.uint8
        )
    )
    return q, k, v, g, beta, cu


@pytest.mark.parametrize("key_dim", [64, 128])
@pytest.mark.parametrize("with_initial", [False, True])
def test_chunk_gdn_preserves_vllm_state_layout(key_dim, with_initial):
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule_fwd

    from vllm_fl.ops.fla.chunk import ChunkGatedDeltaRuleFunction

    q, k, v, g, beta, cu = _inputs(key_dim, [187, 201])
    state = (
        torch.randn(2, 6, 128, key_dim, device="cuda") * 0.1 if with_initial else None
    )
    expected = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=key_dim**-0.5,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
    )
    actual, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, key_dim**-0.5, state, True, cu, False
    )
    assert final_state.shape == (2, 6, 128, key_dim)
    torch.testing.assert_close(actual, expected[1], atol=0.03, rtol=0.03)
    torch.testing.assert_close(final_state, expected[3], atol=0.03, rtol=0.03)


@pytest.mark.parametrize("key_dim", [64, 128])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("lengths", [[1, 1], [32, 33]])
def test_recurrent_gdn_preserves_state_layout_and_alias(key_dim, inplace, lengths):
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )

    from vllm_fl.ops.fla.fused_recurrent import FusedRecurrentFunction

    q, k, v, g, beta, cu = _inputs(key_dim, lengths)
    state = torch.randn(3, 6, 128, key_dim, device="cuda") * 0.1
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    if inplace and max(lengths) > 1:
        indices = indices[:, None].expand(-1, max(lengths)).contiguous()
    expected, expected_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=key_dim**-0.5,
        initial_state=state.clone(),
        inplace_final_state=inplace,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )
    original_pointer = state.data_ptr()
    actual, final_state = FusedRecurrentFunction.apply(
        q, k, v, g, beta, key_dim**-0.5, state, inplace, cu, indices, None, False
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(final_state, expected_state, atol=0.03, rtol=0.03)
    assert (final_state.data_ptr() == original_pointer) == inplace
