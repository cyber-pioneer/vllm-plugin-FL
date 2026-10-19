# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
    topk_softmax_flaggems,
)


def _install_topk_softmax(monkeypatch, kernel, mode):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", mode)
    provider = "flaggems_vllm" if mode == "1" else "flag_gems"
    module = ModuleType(provider)
    module.topk_softmax = kernel
    monkeypatch.setitem(sys.modules, provider, module)


@pytest.mark.parametrize("mode", ["0", "1"])
def test_topk_softmax_supports_legacy_signature(monkeypatch, mode):
    calls = []

    def legacy_kernel(weights, indices, token_indices, logits):
        calls.append(logits)
        weights.copy_(torch.tensor([[0.125, 0.375]]))
        indices.copy_(torch.tensor([[1, 2]], dtype=indices.dtype))

    _install_topk_softmax(monkeypatch, legacy_kernel, mode)
    weights = torch.empty((1, 2))
    indices = torch.empty((1, 2), dtype=torch.int32)
    logits = torch.zeros((1, 3))

    actual_weights, actual_indices = topk_softmax_flaggems(
        weights, indices, torch.empty_like(indices), logits, renormalize=True
    )

    assert len(calls) == 1
    assert calls[0] is logits
    torch.testing.assert_close(actual_weights, torch.tensor([[0.25, 0.75]]))
    assert actual_indices is indices
    torch.testing.assert_close(indices, torch.tensor([[1, 2]], dtype=torch.int32))


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
@pytest.mark.parametrize("mode", ["0", "1"])
def test_topk_softmax_propagates_backend_errors_without_retry(
    monkeypatch, error_type, mode
):
    calls = []
    error = error_type("topk backend failure")

    def failing_kernel(*args):
        calls.append(args)
        raise error

    _install_topk_softmax(monkeypatch, failing_kernel, mode)

    with pytest.raises(error_type, match="topk backend failure") as caught:
        topk_softmax_flaggems(
            torch.empty((1, 2)),
            torch.empty((1, 2), dtype=torch.int32),
            torch.empty((1, 2), dtype=torch.int32),
            torch.zeros((1, 3)),
            renormalize=True,
        )

    assert caught.value is error
    assert len(calls) == 1
    assert len(calls[0]) == 5
