# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.activation import (
    GeluAndMul,
    SiluAndMul,
    SiluAndMulWithClamp,
)

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


@torch.library.custom_op("vllm_fl::silu_and_mul_with_clamp", mutates_args=())
def _silu_and_mul_with_clamp_graph(x: torch.Tensor, limit: float) -> torch.Tensor:
    import vllm_fl.envs as fl_envs

    if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        native = SimpleNamespace(swiglu_limit=limit, alpha=1.0, beta=0.0)
        return SiluAndMulWithClamp.forward_native(native, x)

    from flaggems_vllm import silu_and_mul_with_clamp

    gate, up = x.chunk(2, dim=-1)
    return silu_and_mul_with_clamp(gate, up, limit)


@_silu_and_mul_with_clamp_graph.register_fake
def _silu_and_mul_with_clamp_graph_fake(x: torch.Tensor, limit: float) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def enable_fused_activations_for_compile(compilation_config) -> None:
    """Keep selected OOT activations opaque to Inductor and CUDA graph capture."""
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
    for op, name in (
        ("silu_and_mul", "SiluAndMul"),
        ("gelu_and_mul", "GeluAndMul"),
        ("silu_and_mul_with_clamp", "SiluAndMulWithClamp"),
    ):
        if not use_flaggems_op(op) or op in blacklist:
            continue
        if whitelist is not None and op not in whitelist:
            continue
        if any(f"-{alias}" in compilation_config.custom_ops for alias in (op, name)):
            continue
        selected = f"+{name}"
        if selected not in compilation_config.custom_ops:
            compilation_config.custom_ops.append(selected)


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return _silu_and_mul_graph(x)
        return _silu_and_mul(self, x)


class GeluAndMulFL(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return _gelu_and_mul_graph(x, self.approximate)
        return _gelu_and_mul(self, x)


__all__ = ["SiluAndMulFL", "GeluAndMulFL", "SiluAndMulWithClampFL"]


class SiluAndMulWithClampFL(SiluAndMulWithClamp):
    """Use the fused package for the standard clamped SwiGLU operation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import vllm_fl.envs as fl_envs
        from vllm_fl.utils import use_flaggems_op

        self._use_fused_standard_clamp = (
            fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM
            and use_flaggems_op("silu_and_mul_with_clamp")
            and self.alpha == 1.0
            and self.beta == 0.0
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_fused_standard_clamp:
            return self.forward_native(x)
        return _silu_and_mul_with_clamp_graph(x, self.swiglu_limit)
