# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/worker/gpu_model_runner.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
from contextlib import nullcontext, contextmanager
from types import NoneType
from typing import TYPE_CHECKING, Any, Optional, cast, Generator
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
# from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
try:
    from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
except ImportError:
    # deep_gemm may be broken in some environments; provide a fallback
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "kernel_warmup import failed (likely deep_gemm issue), "
        "using no-op kernel_warmup"
    )
    def kernel_warmup(worker):
        pass
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper

from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import GiB_bytes  # , MemorySnapshot, memory_profiling
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

import vllm_fl.envs as fl_envs
from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe.ascend_config import (
    init_ascend_config,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe.parallel_state import (
    init_ascend_model_parallel,
)
from vllm_fl.ops.custom_ops import register_oot_ops
from vllm_fl.utils import get_flag_gems_whitelist_blacklist

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@dataclass
class MemorySnapshot:
    """Platform-agnostic memory snapshot for FL worker."""

    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0
    auto_measure: bool = True

    def __post_init__(self):
        if self.auto_measure:
            self.measure()

    def measure(self):
        import time

        torch_device_fn = current_platform.torch_device_fn

        # Get peak memory stats using platform-agnostic API
        try:
            self.torch_peak = torch_device_fn.memory_stats().get(
                "allocated_bytes.all.peak", 0
            )
        except (AttributeError, RuntimeError):
            self.torch_peak = 0

        # Get free and total memory using platform-agnostic API
        self.free_memory, self.total_memory = torch_device_fn.mem_get_info()
        self.cuda_memory = self.total_memory - self.free_memory

        # Get torch reserved memory
        try:
            self.torch_memory = torch_device_fn.memory_reserved()
        except (AttributeError, RuntimeError):
            self.torch_memory = 0

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        result = MemorySnapshot(auto_measure=False)
        result.torch_peak = self.torch_peak - other.torch_peak
        result.free_memory = self.free_memory - other.free_memory
        result.total_memory = self.total_memory
        result.cuda_memory = self.cuda_memory - other.cuda_memory
        result.torch_memory = self.torch_memory - other.torch_memory
        result.non_torch_memory = self.non_torch_memory - other.non_torch_memory
        result.timestamp = self.timestamp - other.timestamp
        return result


@dataclass
class MemoryProfilingResult:
    """Platform-agnostic memory profiling result."""

    before_create: MemorySnapshot = None
    before_profile: MemorySnapshot = None
    after_profile: MemorySnapshot = None
    weights_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    non_kv_cache_memory: int = 0
    profile_time: float = 0.0

    def __post_init__(self):
        if self.before_profile is None:
            self.before_profile = MemorySnapshot(auto_measure=False)
        if self.after_profile is None:
            self.after_profile = MemorySnapshot(auto_measure=False)


@contextmanager
def memory_profiling_fl(
    baseline_snapshot: MemorySnapshot, weights_memory: int
) -> Generator[MemoryProfilingResult, None, None]:
    """Platform-agnostic memory profiling context manager for FL worker."""
    gc.collect()
    torch_device_fn = current_platform.torch_device_fn
    torch_device_fn.empty_cache()

    # Reset peak memory stats - platform agnostic
    try:
        torch_device_fn.reset_peak_memory_stats()
    except (AttributeError, RuntimeError):
        pass  # Some platforms may not support this

    result = MemoryProfilingResult()
    result.before_create = baseline_snapshot
    result.weights_memory = weights_memory
    result.before_profile.measure()

    yield result

    gc.collect()
    torch_device_fn.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    non_torch_memory = result.non_torch_increase
    peak_activation_memory = result.torch_peak_increase
    result.non_kv_cache_memory = (
        non_torch_memory + peak_activation_memory + result.weights_memory
    )


class WorkerFL(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        init_ascend_config(vllm_config)
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils.import_utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "CUDA"],
            )
        else:
            self.profiler = None

        logger.debug("=== ENVIRONMENT VARIABLES ===")
        for k, v in sorted(os.environ.items()):
            logger.debug("%s=%r", k, v)

        register_oot_ops()

        # Monkey-patch causal_conv1d ops to use Ascend dispatch versions
        # The Triton-Ascend backend cannot compile the upstream Triton
        # causal_conv1d kernels, so redirect to the Ascend vendor impl.
        try:
            from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import (
                causal_conv1d_fn as ascend_causal_conv1d_fn,
                causal_conv1d_update_npu as ascend_causal_conv1d_update,
            )
            import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_mod
            _conv1d_mod.causal_conv1d_fn = ascend_causal_conv1d_fn
            _conv1d_mod.causal_conv1d_update = ascend_causal_conv1d_update
            # Also patch qwen3_next which has its own cached import
            import vllm.model_executor.models.qwen3_next as _qwen3_next_mod
            _qwen3_next_mod.causal_conv1d_fn = ascend_causal_conv1d_fn
            _qwen3_next_mod.causal_conv1d_update = ascend_causal_conv1d_update
            logger.info("Patched causal_conv1d ops to use Ascend vendor implementation")
        except Exception as e:
            logger.warning(f"Failed to patch causal_conv1d ops: {e}")

        # Patch torch.cuda.device to redirect to torch.npu.device on NPU.
        # The FLA chunk_gated_delta_rule uses torch.cuda.device() which
        # fails on NPU since there's no CUDA.
        try:
            _orig_cuda_device = torch.cuda.device
            class _NPUDeviceContext:
                """Redirect torch.cuda.device to torch.npu.device on NPU."""
                def __init__(self, device_index):
                    self._ctx = torch.npu.device(device_index)
                def __enter__(self):
                    return self._ctx.__enter__()
                def __exit__(self, *args):
                    return self._ctx.__exit__(*args)
            torch.cuda.device = _NPUDeviceContext
            logger.info("Patched torch.cuda.device to use NPU device context")
        except Exception as e:
            logger.warning(f"Failed to patch torch.cuda.device for NPU: {e}")

        # Monkey-patch FLA ops (chunk_gated_delta_rule and
        # fused_recurrent_gated_delta_rule) to use Ascend vendor
        # implementations.  The upstream vLLM Triton kernels for these ops
        # fail to compile on the Triton-Ascend MLIR backend.
        try:
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
                chunk_gated_delta_rule as ascend_chunk_gated_delta_rule,
            )
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule as ascend_fused_recurrent_gated_delta_rule,
            )
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule_fwd_kernel as ascend_fused_recurrent_kernel,
            )
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.layernorm_guard import (
                LayerNormFn as ascend_LayerNormFn,
            )
            # Patch the source modules (following reference vllm-ascend approach)
            import vllm.model_executor.layers.fla.ops as _fla_ops_mod
            import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_mod
            import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_mod
            import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_mod
            # Patch chunk_gated_delta_rule (prefill path)
            _fla_ops_mod.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
            _fla_chunk_mod.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
            # Patch fused_recurrent wrapper + inner kernel (decode path)
            _fla_ops_mod.fused_recurrent_gated_delta_rule = ascend_fused_recurrent_gated_delta_rule
            _fla_recurrent_mod.fused_recurrent_gated_delta_rule = ascend_fused_recurrent_gated_delta_rule
            _fla_recurrent_mod.fused_recurrent_gated_delta_rule_fwd_kernel = ascend_fused_recurrent_kernel
            # Patch LayerNormFn
            _fla_layernorm_mod.LayerNormFn = ascend_LayerNormFn
            # Patch qwen3_next which has its own cached imports
            import vllm.model_executor.models.qwen3_next as _qwen3_next_mod
            _qwen3_next_mod.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
            _qwen3_next_mod.fused_recurrent_gated_delta_rule = ascend_fused_recurrent_gated_delta_rule
            logger.info("Patched FLA ops (chunk_gated_delta_rule, fused_recurrent, LayerNormFn) to use Ascend vendor implementations")
        except Exception as e:
            logger.warning(f"Failed to patch FLA ops for NPU: {e}")

        # Patch torch.argsort for the GDN attention backend.
        # NPU's argsort doesn't support bool tensors and without stable=True
        # produces incorrect indices, causing DDR out-of-range errors.
        try:
            import vllm.v1.attention.backends.gdn_attn as gdn_attn

            _orig_argsort = torch.argsort

            def _npu_argsort(tensor, *args, **kwargs):
                if tensor.dtype == torch.bool:
                    kwargs["stable"] = True
                    return _orig_argsort(tensor.to(torch.int32), *args, **kwargs)
                return _orig_argsort(tensor, *args, **kwargs)

            class _TorchWrapper:
                def __init__(self):
                    self._raw_torch = torch
                def __getattr__(self, name):
                    if name == "argsort":
                        return _npu_argsort
                    return getattr(self._raw_torch, name)

            gdn_attn.torch = _TorchWrapper()
            logger.info("Patched torch.argsort for GDN attention (NPU bool support)")
        except Exception as e:
            logger.warning(f"Failed to patch torch.argsort for GDN attention: {e}")

        # Monkey-patch GroupCoordinator to use HCCL PG options and NPU
        # communicator.  Without this, process groups are created without
        # HCCL-specific options (buffer sizes, etc.) and the default
        # all_to_all / all_reduce paths can trigger async DDR errors on
        # Ascend NPU during prefill of longer sequences.
        # Ported from reference vllm-ascend patch_distributed.py.
        try:
            import torch.distributed as dist
            from torch.distributed import Backend
            import torch_npu
            from vllm.distributed.parallel_state import (
                GroupCoordinator,
                _get_unique_name,
                _register_group,
            )
            from vllm.distributed.device_communicators.base_device_communicator import (
                DeviceCommunicatorBase,
            )
            import vllm.distributed.parallel_state as _ps_mod

            _DEFAULT_HCCL_BUFFER_SIZE = 200

            def _create_hccl_pg_options(group_name: str):
                """Create HCCL process group options with buffer config."""
                options = (
                    torch_npu._C._distributed_c10d
                    .ProcessGroupHCCL.Options()
                )
                if group_name and "mc2" in group_name:
                    return options
                options.hccl_config = {
                    "hccl_buffer_size": _DEFAULT_HCCL_BUFFER_SIZE
                }
                return options

            class _NPUCommunicator(DeviceCommunicatorBase):
                """Minimal NPU communicator with all_to_all support."""

                def __init__(self, cpu_group, device=None,
                             device_group=None, unique_name=""):
                    super().__init__(
                        cpu_group, device, device_group, unique_name
                    )
                    self.device = torch.npu.current_device()

                def all_to_all(
                    self,
                    input_: torch.Tensor,
                    scatter_dim: int = 0,
                    gather_dim: int = -1,
                    scatter_sizes: list[int] | None = None,
                    gather_sizes: list[int] | None = None,
                ) -> torch.Tensor:
                    if scatter_dim < 0:
                        scatter_dim += input_.dim()
                    if gather_dim < 0:
                        gather_dim += input_.dim()

                    if (scatter_sizes is not None
                            and gather_sizes is not None):
                        input_list = [
                            t.contiguous()
                            for t in torch.split(
                                input_, scatter_sizes, scatter_dim
                            )
                        ]
                        output_list = []
                        base_shape = input_list[self.rank].size()
                        for i in range(self.world_size):
                            shape = list(base_shape)
                            shape[gather_dim] = gather_sizes[i]
                            output_list.append(torch.empty(
                                shape,
                                dtype=input_.dtype,
                                device=input_.device,
                            ))
                    else:
                        input_list = [
                            t.contiguous()
                            for t in torch.tensor_split(
                                input_, self.world_size, scatter_dim
                            )
                        ]
                        output_list = [
                            torch.empty_like(input_list[i])
                            for i in range(self.world_size)
                        ]

                    dist.all_to_all(
                        output_list, input_list,
                        group=self.device_group,
                    )
                    return torch.cat(
                        output_list, dim=gather_dim
                    ).contiguous()

            class _GroupCoordinatorPatch(GroupCoordinator):
                def __init__(
                    self,
                    group_ranks: list[list[int]],
                    local_rank: int,
                    torch_distributed_backend: str | Backend,
                    use_device_communicator: bool,
                    use_message_queue_broadcaster: bool = False,
                    group_name: str | None = None,
                ):
                    group_name = group_name or "anonymous"
                    self.unique_name = _get_unique_name(group_name)
                    _register_group(self)

                    self.rank = torch.distributed.get_rank()
                    self.local_rank = local_rank

                    self_device_group = None
                    self_cpu_group = None
                    hccl_pg_options = _create_hccl_pg_options(group_name)

                    for ranks in group_ranks:
                        device_group = torch.distributed.new_group(
                            ranks,
                            backend=torch_distributed_backend,
                            pg_options=hccl_pg_options,
                        )
                        cpu_group = torch.distributed.new_group(
                            ranks, backend="gloo"
                        )
                        if self.rank in ranks:
                            self.ranks = ranks
                            self.world_size = len(ranks)
                            self.rank_in_group = ranks.index(self.rank)
                            self_device_group = device_group
                            self_cpu_group = cpu_group

                    assert self_cpu_group is not None
                    assert self_device_group is not None

                    self.cpu_group = self_cpu_group
                    self.device_group = self_device_group
                    self.device = torch.npu.current_device()

                    self.use_device_communicator = use_device_communicator
                    self.device_communicator = None
                    if use_device_communicator and self.world_size > 1:
                        self.device_communicator = _NPUCommunicator(
                            cpu_group=self.cpu_group,
                            device=self.device,
                            device_group=self.device_group,
                            unique_name=self.unique_name,
                        )

                    from vllm.distributed.device_communicators.shm_broadcast import (
                        MessageQueue,
                    )

                    self.mq_broadcaster: MessageQueue | None = None
                    if use_message_queue_broadcaster and self.world_size > 1:
                        self.mq_broadcaster = (
                            MessageQueue.create_from_process_group(
                                self.cpu_group, 1 << 22, 6
                            )
                        )

                    self.use_custom_op_call = False
                    self.use_cpu_custom_send_recv = False

                def all_to_all(
                    self,
                    input_: torch.Tensor,
                    scatter_dim: int = 0,
                    gather_dim: int = -1,
                    scatter_sizes: list[int] | None = None,
                    gather_sizes: list[int] | None = None,
                ) -> torch.Tensor:
                    if self.world_size == 1:
                        return input_
                    assert -input_.dim() <= scatter_dim < input_.dim()
                    assert -input_.dim() <= gather_dim < input_.dim()
                    assert self.device_communicator is not None
                    return self.device_communicator.all_to_all(
                        input_, scatter_dim, gather_dim,
                        scatter_sizes, gather_sizes,
                    )

                def all_reduce(self, input_):
                    if self.world_size == 1:
                        return input_
                    return torch.ops.vllm.all_reduce(
                        input_, group_name=self.unique_name
                    )

            _ps_mod.GroupCoordinator = _GroupCoordinatorPatch
            logger.info(
                "Patched GroupCoordinator with HCCL PG options "
                "and NPU communicator"
            )
        except Exception as e:
            logger.warning(
                f"Failed to patch GroupCoordinator for NPU: {e}"
            )

        # Register unquantized_gemm as a custom op so that the graph
        # compiler can fuse MatmulAllReduceAddRMSNorm.  Without this, the
        # gemm and all_reduce stay as separate async ops which can cause
        # DDR errors on Ascend NPU.
        # Ported from reference vllm-ascend patch_unquantized_gemm.py.
        try:
            import vllm.model_executor.layers.utils as _layer_utils_mod
            from vllm.utils.torch_utils import direct_register_custom_op

            def _unquantized_gemm_npu(
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return torch.nn.functional.linear(x, weight, bias)

            def _unquantized_gemm_fake(
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
            ) -> torch.Tensor:
                output_shape = (x.shape[0], weight.shape[0])
                return torch.empty(
                    output_shape, dtype=x.dtype, device=x.device
                )

            direct_register_custom_op(
                op_name="unquantized_gemm",
                op_func=_unquantized_gemm_npu,
                fake_impl=_unquantized_gemm_fake,
                mutates_args=[],
                dispatch_key="PrivateUse1",
            )

            def _default_unquantized_gemm_patched(
                layer: torch.nn.Module,
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if x.device.type == "npu":
                    return torch.ops.vllm.unquantized_gemm(x, weight, bias)
                else:
                    return torch.nn.functional.linear(x, weight, bias)

            _layer_utils_mod.default_unquantized_gemm = (
                _default_unquantized_gemm_patched
            )
            logger.info(
                "Registered unquantized_gemm as custom op and "
                "patched default_unquantized_gemm for NPU"
            )
        except Exception as e:
            logger.warning(
                f"Failed to register unquantized_gemm custom op: {e}"
            )

        # Monkey-patch Qwen3NextGatedDeltaNet._forward_core to use the
        # fused sigmoid gating delta rule update kernel for the decode-only
        # path.  The upstream code calls fused_gdn_gating + fused_recurrent
        # as two separate Triton kernels which cause DDR errors on Ascend.
        # The fused kernel combines gating + recurrent into a single launch.
        try:
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.sigmoid_gating import (
                fused_sigmoid_gating_delta_rule_update,
            )
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_gdn_gating import (
                fused_gdn_gating_patch,
            )
            from vllm.forward_context import get_forward_context
            from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
            from vllm.model_executor.layers.fla.ops import (
                chunk_gated_delta_rule as _chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule as _fused_recurrent_gated_delta_rule,
            )
            from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                causal_conv1d_fn as _causal_conv1d_fn,
                causal_conv1d_update as _causal_conv1d_update,
            )
            from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet

            def _forward_core_patched(
                self,
                mixed_qkv: torch.Tensor,
                b: torch.Tensor,
                a: torch.Tensor,
                core_attn_out: torch.Tensor,
            ):
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata

                if attn_metadata is None:
                    return

                assert isinstance(attn_metadata, dict)
                attn_metadata = attn_metadata[self.prefix]
                assert isinstance(attn_metadata, GDNAttentionMetadata)
                has_initial_state = attn_metadata.has_initial_state
                spec_query_start_loc = attn_metadata.spec_query_start_loc
                non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
                spec_sequence_masks = attn_metadata.spec_sequence_masks
                spec_token_indx = attn_metadata.spec_token_indx
                non_spec_token_indx = attn_metadata.non_spec_token_indx
                spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
                non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                conv_state = self_kv_cache[0].transpose(-1, -2)
                ssm_state = self_kv_cache[1]
                num_actual_tokens = attn_metadata.num_actual_tokens
                num_accepted_tokens = attn_metadata.num_accepted_tokens

                mixed_qkv = mixed_qkv[:num_actual_tokens]
                b = b[:num_actual_tokens]
                a = a[:num_actual_tokens]

                # 1. Convolution sequence transformation
                conv_weights = self.conv1d.weight.view(
                    self.conv1d.weight.size(0), self.conv1d.weight.size(2)
                )
                if spec_sequence_masks is not None:
                    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                        mixed_qkv_spec = mixed_qkv
                        mixed_qkv_non_spec = None
                    else:
                        mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                        mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
                else:
                    mixed_qkv_spec = None
                    mixed_qkv_non_spec = mixed_qkv

                # 1.1: Process the multi-query part
                if spec_sequence_masks is not None:
                    mixed_qkv_spec = _causal_conv1d_update(
                        mixed_qkv_spec,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                        num_accepted_tokens=num_accepted_tokens,
                        query_start_loc=spec_query_start_loc,
                        max_query_len=spec_state_indices_tensor.size(-1),
                        validate_data=False,
                    )

                # 1.2: Process the remaining part
                if attn_metadata.num_prefills > 0:
                    if mixed_qkv_non_spec is not None:
                        mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                        mixed_qkv_non_spec = _causal_conv1d_fn(
                            mixed_qkv_non_spec_T,
                            conv_weights,
                            self.conv1d.bias,
                            activation=self.activation,
                            conv_states=conv_state,
                            has_initial_state=has_initial_state,
                            cache_indices=non_spec_state_indices_tensor,
                            query_start_loc=non_spec_query_start_loc,
                            metadata=attn_metadata,
                        ).transpose(0, 1)
                elif attn_metadata.num_decodes > 0:
                    mixed_qkv_non_spec = _causal_conv1d_update(
                        mixed_qkv_non_spec,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                        validate_data=True,
                    )
                else:
                    mixed_qkv_non_spec = None

                query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
                query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

                if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
                    # Prefill or spec decoding path: use gating + separate kernels
                    g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
                    if spec_sequence_masks is not None:
                        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                            g_spec = g
                            beta_spec = beta
                            g_non_spec = None
                            beta_non_spec = None
                        else:
                            g_spec = g.index_select(1, spec_token_indx)
                            beta_spec = beta.index_select(1, spec_token_indx)
                            g_non_spec = g.index_select(1, non_spec_token_indx)
                            beta_non_spec = beta.index_select(1, non_spec_token_indx)
                    else:
                        g_spec = None
                        beta_spec = None
                        g_non_spec = g
                        beta_non_spec = beta

                    # 2.1: Process the multi-query part
                    if spec_sequence_masks is not None:
                        core_attn_out_spec, last_recurrent_state = _fused_recurrent_gated_delta_rule(
                            q=query_spec,
                            k=key_spec,
                            v=value_spec,
                            g=g_spec,
                            beta=beta_spec,
                            initial_state=ssm_state,
                            inplace_final_state=True,
                            cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                            ssm_state_indices=spec_state_indices_tensor,
                            num_accepted_tokens=num_accepted_tokens,
                            use_qk_l2norm_in_kernel=True,
                        )
                    else:
                        core_attn_out_spec, last_recurrent_state = None, None

                    # 2.2: Process the remaining part
                    if attn_metadata.num_prefills > 0:
                        initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                        initial_state[~has_initial_state, ...] = 0
                        (
                            core_attn_out_non_spec,
                            last_recurrent_state,
                        ) = _chunk_gated_delta_rule(
                            q=query_non_spec,
                            k=key_non_spec,
                            v=value_non_spec,
                            g=g_non_spec,
                            beta=beta_non_spec,
                            initial_state=initial_state,
                            output_final_state=True,
                            cu_seqlens=non_spec_query_start_loc,
                            head_first=False,
                            use_qk_l2norm_in_kernel=True,
                        )
                        ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
                    elif attn_metadata.num_decodes > 0:
                        core_attn_out_non_spec, last_recurrent_state = _fused_recurrent_gated_delta_rule(
                            q=query_non_spec,
                            k=key_non_spec,
                            v=value_non_spec,
                            g=g_non_spec,
                            beta=beta_non_spec,
                            initial_state=ssm_state,
                            inplace_final_state=True,
                            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                            ssm_state_indices=non_spec_state_indices_tensor,
                            use_qk_l2norm_in_kernel=True,
                        )
                    else:
                        core_attn_out_non_spec, last_recurrent_state = None, None

                elif attn_metadata.num_decodes > 0:
                    # Decode-only path: use fused sigmoid gating + delta rule
                    # update kernel (single kernel launch, avoids DDR errors)
                    core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log.contiguous(),
                        dt_bias=self.dt_bias.contiguous(),
                        q=query_non_spec.contiguous(),
                        k=key_non_spec.contiguous(),
                        v=value_non_spec.contiguous(),
                        a=a.contiguous(),
                        b=b.contiguous(),
                        initial_state_source=ssm_state,
                        initial_state_indices=non_spec_state_indices_tensor,
                        cu_seqlens=non_spec_query_start_loc,
                        use_qk_l2norm_in_kernel=True,
                        softplus_beta=1.0,
                        softplus_threshold=20.0,
                    )

                # 3. Merge core attention output
                if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
                    merged_out = torch.empty(
                        (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                        dtype=core_attn_out_non_spec.dtype,
                        device=core_attn_out_non_spec.device,
                    )
                    merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
                    merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
                    core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
                elif spec_sequence_masks is not None:
                    core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
                else:
                    core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

            Qwen3NextGatedDeltaNet._forward_core = _forward_core_patched
            logger.info("Patched Qwen3NextGatedDeltaNet._forward_core to use fused sigmoid gating for decode-only path")
        except Exception as e:
            logger.warning(f"Failed to patch _forward_core for NPU: {e}")

        # Monkey-patch _update_hybrid_attention_mamba_layout to be a no-op
        # on Ascend.  The upstream vLLM interleaves K/V blocks via
        # as_strided_, which makes kv_cache[0] and kv_cache[1] non-contiguous.
        # torch_npu._npu_reshape_and_cache requires contiguous tensors.
        # Skipping the interleaving keeps the default (2, num_blocks, ...)
        # layout where kv_cache[0] (K) and kv_cache[1] (V) are contiguous
        # slices, matching the reference vllm-ascend approach.
        try:
            from vllm_fl.worker.model_runner import ModelRunnerFL

            def _noop_hybrid_layout(self, kv_caches):
                pass

            ModelRunnerFL._update_hybrid_attention_mamba_layout = (
                _noop_hybrid_layout
            )
            logger.info(
                "Patched _update_hybrid_attention_mamba_layout to no-op "
                "for Ascend (keeps contiguous kv_cache slices)"
            )
        except Exception as e:
            logger.warning(
                f"Failed to patch _update_hybrid_attention_mamba_layout: {e}"
            )

        if fl_envs.USE_FLAGGEMS:
            import flag_gems

            # Get whitelist and blacklist from environment variables
            whitelist, blacklist = get_flag_gems_whitelist_blacklist()

            # Use whitelist if specified (takes precedence over blacklist)
            if whitelist:
                logger.info(f"[FlagGems] Enable only the following ops: {whitelist}")
                flag_gems.only_enable(
                    include=whitelist,
                    record=True,
                    once=True,
                    path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
                )
            elif blacklist:
                logger.info(f"[FlagGems] Disable the following ops: {blacklist}")
                flag_gems.enable(
                    unused=blacklist,
                    record=True,
                    once=True,
                    path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
                )
            else:
                logger.info("[FlagGems] Enable all ops")
                flag_gems.enable(
                    record=True, once=True, path=fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH
                )

    # def sleep(self, level: int = 1) -> None:
    #     TODO(lms): rewrite CuMemAllocator
    #     from vllm.device_allocator.cumem import CuMemAllocator

    #     free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

    #     # Save the buffers before level 2 sleep
    #     if level == 2:
    #         model = self.model_runner.model
    #         self._sleep_saved_buffers = {
    #             name: buffer.cpu().clone()
    #             for name, buffer in model.named_buffers()
    #         }

    #     allocator = CuMemAllocator.get_instance()
    #     allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
    #     free_bytes_after_sleep, total = torch.cuda.mem_get_info()
    #     freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    #     used_bytes = total - free_bytes_after_sleep
    #     assert freed_bytes >= 0, "Memory usage increased after sleeping."
    #     logger.info(
    #         "Sleep mode freed %.2f GiB memory, "
    #         "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
    #         used_bytes / GiB_bytes)

    # def wake_up(self, tags: Optional[list[str]] = None) -> None:
    #     from vllm.device_allocator.cumem import CuMemAllocator

    #     allocator = CuMemAllocator.get_instance()
    #     allocator.wake_up(tags)

    #     # Restore the buffers after level 2 sleep
    #     if len(self._sleep_saved_buffers):
    #         model = self.model_runner.model
    #         for name, buffer in model.named_buffers():
    #             if name in self._sleep_saved_buffers:
    #                 buffer.data.copy_(self._sleep_saved_buffers[name].data)
    #         self._sleep_saved_buffers = {}

    # def _maybe_get_memory_pool_context(self,
    #                                    tag: str) -> AbstractContextManager:
    #     if self.vllm_config.model_config.enable_sleep_mode:
    #         from vllm.device_allocator.cumem import CuMemAllocator

    #         allocator = CuMemAllocator.get_instance()
    #         if tag == "weights":
    #             assert allocator.get_current_usage() == 0, (
    #                 "Sleep mode can only be "
    #                 "used for one instance per process.")
    #         context = allocator.use_memory_pool(tag=tag)
    #     else:
    #         context = nullcontext()
    #     return context

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        # This env var set by Ray causes exceptions with graph building.
        if (
            self.parallel_config.data_parallel_size > 1
            and self.parallel_config.data_parallel_size_local > 0
            and self.parallel_config.distributed_executor_backend
            not in ["ray", "external_launcher"]
            and self.vllm_config.parallel_config.data_parallel_backend != "ray"
            and self.vllm_config.parallel_config.nnodes_within_dp == 1
        ):
            # Use local DP rank if available, otherwise use global DP rank.
            dp_local_rank = self.parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = self.parallel_config.data_parallel_rank

            tp_pp_world_size = (
                self.parallel_config.pipeline_parallel_size
                * self.parallel_config.tensor_parallel_size
            )

            # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
            self.local_rank += dp_local_rank * tp_pp_world_size
            device_count = (
                current_platform.torch_device_fn.device_count()
                if current_platform.torch_device_fn.is_available()
                else 0
            )
            assert self.local_rank < device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds. "
                f"Device count: {device_count}"
            )
            visible_device_count = (
                current_platform.torch_device_fn.device_count()
                if current_platform.torch_device_fn.is_available()
                else 0
            )
            assert self.parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({self.parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )
        self.device = torch.device(f"{current_platform.device_type}:{self.local_rank}")
        current_platform.set_device(self.device)

        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment BEFORE taking
        # memory snapshot
        # This ensures NCCL buffers are allocated before we measure
        # available memory
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        init_ascend_model_parallel(self.parallel_config)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        current_platform.empty_cache()

        ### TODO(lms): patch MemorySnapshot in other platform
        # take current memory snapshot
        self.init_snapshot = MemorySnapshot()
        self.requested_memory = (
            self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        )
        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )
        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        from vllm_fl.worker.model_runner import ModelRunnerFL

        # Construct the model runner
        self.model_runner = ModelRunnerFL(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        ### TODO(lms): support manages a memory pool for device tensors.
        self.model_runner.load_model(eep_scale_up=eep_scale_up)
        # with self._maybe_get_memory_pool_context(tag="weights"):
        #     self.model_runner.load_model(eep_scale_up=eep_scale_up)

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        GiB = lambda b: b / GiB_bytes
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {GiB(self.init_snapshot.free_memory):.2f} "
                f"GiB, reserved {GiB(kv_cache_memory_bytes):.2f} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        current_platform.empty_cache()
        current_platform.torch_device_fn.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling_fl(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), %.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        gc.collect()

        return int(self.available_kv_cache_memory_bytes)

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        ### TODO(lms):
        # if self.vllm_config.model_config.enable_sleep_mode:
        #     from vllm.device_allocator.cumem import CuMemAllocator

        #     allocator = CuMemAllocator.get_instance()
        #     context = allocator.use_memory_pool(tag="kv_cache")
        # else:
        #     context = nullcontext()
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        warmup_sizes = []
        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        ### NOTE(lms): can add gems kernel pretune here
        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.
            GiB = lambda b: round(b / GiB_bytes, 2)

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). "
                f"Actual usage is {GiB(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {GiB(self.peak_activation_memory)} GiB "
                f"for peak activation, {GiB(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {GiB(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{GiB(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # new/cached request numbers in each iteration
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        num_new = len(scheduler_output.scheduled_new_reqs)
        num_cached = len(scheduler_output.scheduled_cached_reqs.req_ids)

        return self.profiler.annotate_context_manager(
            f"execute_new_{num_new}_cached_{num_cached}"
        )

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config

        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # TODO(lucas): This is pretty gross; ideally we should only ever call
            # `_determine_batch_execution_and_padding` once (will get called again
            # in `execute_model`) but this requires a larger refactor of PP.
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                )
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, batch_desc.num_tokens
                )
            }
        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict = get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )
            assert tensor_dict is not None
            intermediate_tensors = IntermediateTensors(tensor_dict)

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(output, (ModelRunnerOutput, NoneType)):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != ("external_launcher")
            and not get_pp_group().is_last_rank
        )

        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiling is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def _eplb_before_scale_down(self, old_ep_size: int, new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info(
                "[Elastic EP] Starting expert resharding before scaling down..."
            )
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            self.model_runner.model,
            execute_shuffle=True,
            global_expert_loads=None,
            rank_mapping=rank_mapping,
        )
        current_platform.torch_device_fn.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
        self,
        old_ep_size: int,
        new_ep_size: int,
        global_expert_loads: Optional[torch.Tensor],
    ) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding after scaling up...")
        rank_mapping = {old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)}
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            execute_shuffle=True,
            global_expert_loads=global_expert_loads,
            rank_mapping=rank_mapping,
        )
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        """
        Update parallel config with provided reconfig_request
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )

    def _reconfigure_moe(
        self, old_ep_size: int, new_ep_size: int
    ) -> Optional[torch.Tensor]:
        """
        Reconfigure MoE modules with provided reconfig_request

        Return the global expert load if new_ep_size > old_ep_size,
        otherwise None
        """
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            prepare_communication_buffer_for_model,
        )
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoE,
            FusedMoEParallelConfig,
        )

        parallel_config = self.vllm_config.parallel_config

        def get_moe_modules(model: torch.nn.Module) -> list[FusedMoE]:
            return [
                module
                for module in model.modules()
                if (
                    module.__class__.__name__ == "FusedMoE"
                    or module.__class__.__name__ == "SharedFusedMoE"
                )
            ]

        def update_moe_modules(moe_modules: list[FusedMoE], num_local_experts: int):
            assert all(
                module.moe_config.num_local_experts == num_local_experts
                for module in moe_modules
            ), "All MoE modules must have the same number of experts"
            for module in moe_modules:
                module.moe_config.num_experts = num_local_experts * new_ep_size
                module.global_num_experts = module.moe_config.num_experts
                module.moe_parallel_config = FusedMoEParallelConfig.make(
                    tp_size_=get_tp_group().world_size,
                    pcp_size_=get_pcp_group().world_size,
                    dp_size_=get_dp_group().world_size,
                    vllm_parallel_config=parallel_config,
                )
                module.moe_config.moe_parallel_config = module.moe_parallel_config
            return moe_modules

        model_moe_modules = get_moe_modules(self.model_runner.model)
        num_local_experts = model_moe_modules[0].moe_config.num_local_experts

        update_moe_modules(model_moe_modules, num_local_experts)
        drafter_model = None
        if hasattr(self.model_runner, "drafter") and hasattr(
            self.model_runner.drafter, "model"
        ):
            drafter_model = self.model_runner.drafter.model
        if drafter_model is not None and is_mixture_of_experts(drafter_model):
            drafter_moe_modules = get_moe_modules(drafter_model)
            # Check if drafter and model have matching configs
            assert (
                drafter_moe_modules[0].moe_config.num_local_experts == num_local_experts
            ), "Drafter and model configs should be the same"
            update_moe_modules(drafter_moe_modules, num_local_experts)

        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = (
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]
            )
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts
                - self.model_runner.eplb_state.logical_replica_count.shape[1]
            )
            global_expert_loads = None
        else:
            num_local_physical_experts_tensor = torch.tensor(
                [num_local_experts], dtype=torch.int32, device="cpu"
            )
            torch.distributed.broadcast(
                num_local_physical_experts_tensor,
                group=get_ep_group().cpu_group,
                group_src=0,
            )
            num_local_physical_experts = int(num_local_physical_experts_tensor.item())
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_loads_any = self.model_runner.eplb_state.rearrange(
                execute_shuffle=False
            )
            global_expert_loads = cast(list[torch.Tensor], global_expert_loads_any)
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts - global_expert_loads[0].shape[1]
            )
        prepare_communication_buffer_for_model(self.model_runner.model)
        if drafter_model is not None:
            prepare_communication_buffer_for_model(drafter_model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
        return global_expert_loads

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
            get_ep_group,
        )

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = (
            reconfig_request.new_data_parallel_size
            * get_tp_group().world_size
            * get_pp_group().world_size
        )
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )

        global_expert_loads = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_loads is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size, global_expert_loads)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    def shutdown(self) -> None:
        if runner := getattr(self, "model_runner", None):
            runner.ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    attention_config = vllm_config.attention_config
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance(attention_config.backend)

    init_method = distributed_init_method or "env://"
    init_distributed_environment(
        parallel_config.world_size, rank, init_method, local_rank, backend
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
