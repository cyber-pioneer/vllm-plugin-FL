# Copyright (c) 2025 BAAI. All rights reserved.

"""
Ascend NPU monkey-patches for vLLM FL plugin.

These patches replace upstream vLLM module-level functions with Ascend NPU
implementations. They are necessary because some upstream code (e.g.
qwen3_next._forward_core) calls module-level functions directly, bypassing
the CustomOp/dispatch system.

Call apply_ascend_patches() once during initialization (from register_oot_ops)
when running on NPU.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_ascend_patches():
    """Apply all Ascend NPU monkey-patches. Idempotent."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    _patch_cuda_device()
    _patch_causal_conv1d()
    _patch_fla_ops()
    _patch_argsort_for_gdn()
    _patch_mm_encoder_attention()
    _patch_group_coordinator()
    _patch_unquantized_gemm()
    _patch_hybrid_attention_layout()


def _patch_cuda_device():
    """Patch torch.cuda.device -> torch.npu.device for NPU compatibility."""
    try:

        class _NPUDeviceContext:
            def __init__(self, device_index):
                self._ctx = torch.npu.device(device_index)

            def __enter__(self):
                return self._ctx.__enter__()

            def __exit__(self, *args):
                return self._ctx.__exit__(*args)

        torch.cuda.device = _NPUDeviceContext
        logger.info("Patched torch.cuda.device for NPU")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda.device: %s", e)


def _patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import (
            causal_conv1d_fn as ascend_causal_conv1d_fn,
            causal_conv1d_update_npu as ascend_causal_conv1d_update,
        )
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_mod
        import vllm.model_executor.models.qwen3_next as _qwen3_next_mod

        _conv1d_mod.causal_conv1d_fn = ascend_causal_conv1d_fn
        _conv1d_mod.causal_conv1d_update = ascend_causal_conv1d_update
        _qwen3_next_mod.causal_conv1d_fn = ascend_causal_conv1d_fn
        _qwen3_next_mod.causal_conv1d_update = ascend_causal_conv1d_update
        logger.info("Patched causal_conv1d ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)


def _patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
            chunk_gated_delta_rule as ascend_chunk_gated_delta_rule,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as ascend_fused_recurrent,
            fused_recurrent_gated_delta_rule_fwd_kernel as ascend_fused_recurrent_kernel,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_gdn_gating import (
            fused_gdn_gating_patch as ascend_fused_gdn_gating,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.layernorm_guard import (
            LayerNormFn as ascend_LayerNormFn,
        )
        import vllm.model_executor.layers.fla.ops as _fla_ops_mod
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_mod
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_mod
        import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_mod
        import vllm.model_executor.models.qwen3_next as _qwen3_next_mod

        _fla_ops_mod.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
        _fla_chunk_mod.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
        _fla_ops_mod.fused_recurrent_gated_delta_rule = ascend_fused_recurrent
        _fla_recurrent_mod.fused_recurrent_gated_delta_rule = ascend_fused_recurrent
        _fla_recurrent_mod.fused_recurrent_gated_delta_rule_fwd_kernel = (
            ascend_fused_recurrent_kernel
        )
        _fla_layernorm_mod.LayerNormFn = ascend_LayerNormFn
        _qwen3_next_mod.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
        _qwen3_next_mod.fused_recurrent_gated_delta_rule = ascend_fused_recurrent
        _qwen3_next_mod.fused_gdn_gating = ascend_fused_gdn_gating
        logger.info("Patched FLA ops + fused_gdn_gating for Ascend")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)


def _patch_argsort_for_gdn():
    """Patch torch.argsort in GDN attention for NPU bool tensor support."""
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
        logger.info("Patched torch.argsort for GDN attention")
    except Exception as e:
        logger.warning("Failed to patch torch.argsort: %s", e)


def _patch_mm_encoder_attention():
    """Patch MMEncoderAttention for NPU flash attention."""
    try:
        import torch_npu as _torch_npu_mme
        import einops as _einops_mme
        from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention

        def _mm_encoder_attn_forward_oot(
            self,
            query,
            key,
            value,
            cu_seqlens=None,
            max_seqlen=None,
        ):
            bsz, q_len = query.size()[:2]
            kv_len = key.size(1)
            is_reshaped = query.dim() == 4
            q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)
            context_layer = torch.empty_like(q)
            if cu_seqlens is None:
                cu_seqlens = torch.arange(
                    0,
                    (bsz + 1) * q_len,
                    step=q_len,
                    dtype=torch.int32,
                    device=query.device,
                )
            seq_len_cpu = torch.diff(cu_seqlens).to("cpu")
            _torch_npu_mme._npu_flash_attention_unpad(
                query=q,
                key=k,
                value=v,
                seq_len=seq_len_cpu,
                scale_value=self.head_size**-0.5,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                out=context_layer,
            )
            fmt = "(b s) h d -> b s h d" if is_reshaped else "(b s) h d -> b s (h d)"
            return _einops_mme.rearrange(context_layer, fmt, b=bsz).contiguous()

        MMEncoderAttention.forward_oot = _mm_encoder_attn_forward_oot
        logger.info("Patched MMEncoderAttention for NPU flash attention")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)


def _patch_group_coordinator():
    """Patch GroupCoordinator with HCCL PG options and NPU communicator."""
    try:
        import torch.distributed as dist
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
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            if group_name and "mc2" in group_name:
                return options
            options.hccl_config = {"hccl_buffer_size": _DEFAULT_HCCL_BUFFER_SIZE}
            return options

        class _NPUCommunicator(DeviceCommunicatorBase):
            def __init__(self, cpu_group, device=None, device_group=None, unique_name=""):
                super().__init__(cpu_group, device, device_group, unique_name)
                self.device = torch.npu.current_device()

            def all_to_all(
                self,
                input_,
                scatter_dim=0,
                gather_dim=-1,
                scatter_sizes=None,
                gather_sizes=None,
            ):
                if scatter_dim < 0:
                    scatter_dim += input_.dim()
                if gather_dim < 0:
                    gather_dim += input_.dim()
                if scatter_sizes is not None and gather_sizes is not None:
                    input_list = [
                        t.contiguous()
                        for t in torch.split(input_, scatter_sizes, scatter_dim)
                    ]
                    output_list = []
                    base_shape = input_list[self.rank].size()
                    for i in range(self.world_size):
                        shape = list(base_shape)
                        shape[gather_dim] = gather_sizes[i]
                        output_list.append(
                            torch.empty(shape, dtype=input_.dtype, device=input_.device)
                        )
                else:
                    input_list = [
                        t.contiguous()
                        for t in torch.tensor_split(input_, self.world_size, scatter_dim)
                    ]
                    output_list = [
                        torch.empty_like(input_list[i]) for i in range(self.world_size)
                    ]
                dist.all_to_all(output_list, input_list, group=self.device_group)
                return torch.cat(output_list, dim=gather_dim).contiguous()

        class _GroupCoordinatorPatch(GroupCoordinator):
            def __init__(
                self,
                group_ranks,
                local_rank,
                torch_distributed_backend,
                use_device_communicator,
                use_message_queue_broadcaster=False,
                group_name=None,
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
                    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
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

                self.mq_broadcaster = None
                if use_message_queue_broadcaster and self.world_size > 1:
                    self.mq_broadcaster = MessageQueue.create_from_process_group(
                        self.cpu_group, 1 << 22, 6
                    )
                self.use_custom_op_call = False
                self.use_cpu_custom_send_recv = False

            def all_to_all(
                self,
                input_,
                scatter_dim=0,
                gather_dim=-1,
                scatter_sizes=None,
                gather_sizes=None,
            ):
                if self.world_size == 1:
                    return input_
                assert -input_.dim() <= scatter_dim < input_.dim()
                assert -input_.dim() <= gather_dim < input_.dim()
                assert self.device_communicator is not None
                return self.device_communicator.all_to_all(
                    input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes
                )

            def all_reduce(self, input_):
                if self.world_size == 1:
                    return input_
                return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)

        _ps_mod.GroupCoordinator = _GroupCoordinatorPatch
        logger.info("Patched GroupCoordinator with HCCL PG options")
    except Exception as e:
        logger.warning("Failed to patch GroupCoordinator: %s", e)


def _patch_unquantized_gemm():
    """Register unquantized_gemm custom op for NPU MatmulAllReduceAddRMSNorm fusion."""
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
            return torch.empty(
                (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
            )

        direct_register_custom_op(
            op_name="unquantized_gemm",
            op_func=_unquantized_gemm_npu,
            fake_impl=_unquantized_gemm_fake,
            mutates_args=[],
            dispatch_key="PrivateUse1",
        )

        def _default_unquantized_gemm_patched(layer, x, weight, bias=None):
            if x.device.type == "npu":
                return torch.ops.vllm.unquantized_gemm(x, weight, bias)
            return torch.nn.functional.linear(x, weight, bias)

        _layer_utils_mod.default_unquantized_gemm = _default_unquantized_gemm_patched
        logger.info("Registered unquantized_gemm custom op for NPU")
    except Exception as e:
        logger.warning("Failed to register unquantized_gemm: %s", e)


def _patch_hybrid_attention_layout():
    """No-op _update_hybrid_attention_mamba_layout to keep contiguous kv_cache."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL

        ModelRunnerFL._update_hybrid_attention_mamba_layout = (
            lambda self, kv_caches: None
        )
        logger.info("Patched _update_hybrid_attention_mamba_layout to no-op")
    except Exception as e:
        logger.warning("Failed to patch _update_hybrid_attention_mamba_layout: %s", e)
