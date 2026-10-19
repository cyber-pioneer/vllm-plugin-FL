# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul
from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")


@torch.library.custom_op("vllm_fl::silu_and_mul", mutates_args=())
def _silu_and_mul_graph(x: torch.Tensor) -> torch.Tensor:
    return _silu_and_mul(None, x)


@_silu_and_mul_graph.register_fake
def _silu_and_mul_graph_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@torch.library.custom_op("vllm_fl::gelu_and_mul", mutates_args=())
def _gelu_and_mul_graph(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return _gelu_and_mul(SimpleNamespace(approximate=approximate), x)


@_gelu_and_mul_graph.register_fake
def _gelu_and_mul_graph_fake(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def enable_fused_activations_for_compile(compilation_config) -> None:
    """Keep opted-in, dispatch-managed activations in compiled models."""
    import vllm_fl.envs as fl_envs
    from vllm_fl.utils import (
        get_oot_blacklist,
        get_oot_whitelist,
        is_oot_enabled,
        use_flaggems_op,
    )

    if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM or not is_oot_enabled():
        return
    whitelist, blacklist = get_oot_whitelist(), get_oot_blacklist() or []
    for op, name in (("silu_and_mul", "SiluAndMul"), ("gelu_and_mul", "GeluAndMul")):
        if not use_flaggems_op(op) or op in blacklist:
            continue
        if whitelist is not None and op not in whitelist:
            continue
        if any(f"-{alias}" in compilation_config.custom_ops for alias in (op, name)):
            continue
        if f"+{name}" not in compilation_config.custom_ops:
            compilation_config.custom_ops.append(f"+{name}")


class SiluAndMulFL(SiluAndMul):
    def __init__(self, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return _silu_and_mul_graph(x)
        return _silu_and_mul(self, x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_oot(x)


class GeluAndMulFL(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return _gelu_and_mul_graph(x, self.approximate)
        return _gelu_and_mul(self, x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_oot(x)


__all__ = ["SiluAndMulFL", "GeluAndMulFL"]
