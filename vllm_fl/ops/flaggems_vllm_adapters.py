# Copyright 2026 FlagOS Contributors
"""Adapt the pinned FlagGems-vllm release to the plugin's operator contracts.

These functions are selected only when VLLM_FL_USE_FLAGGEMS_VLLM=1. They do
not fall back to FlagGems when a release operator is unavailable.
"""

import torch


def required_op(name):
    import flaggems_vllm

    op = getattr(flaggems_vllm, name, None)
    if not callable(op):
        raise NotImplementedError(
            f"FlagGems-vllm v0.2.0 does not provide {name}; "
            "VLLM_FL_USE_FLAGGEMS_VLLM=1 cannot use the FlagGems implementation"
        )
    return op


def rms_norm(x, residual, weight, eps):
    shape = (x.shape[-1],)
    if residual is None:
        # The release exports add-RMSNorm rather than standalone RMSNorm.
        return required_op("add_rms_norm")(x, torch.zeros_like(x), shape, weight, eps)
    # vLLM's residual path mutates both caller-owned buffers. The release may
    # materialize contiguous copies, so copy results back into the originals.
    output, summed = required_op("fused_add_rms_norm")(x, residual, shape, weight, eps)
    x.copy_(output)
    residual.copy_(summed)
    return x, residual


def scaled_mm(input, weight, scale_a, scale_b, *, bias=None, out_dtype):
    return required_op("triton_scaled_mm")(
        input, weight, scale_a, scale_b, out_dtype=out_dtype, bias=bias
    )


def chunk_gated_delta_rule_fwd(*, state_v_major=True, **kwargs):
    from vllm.triton_utils.allocation import set_triton_allocator

    set_triton_allocator(kwargs["q"].device)
    op = required_op("chunk_gated_delta_rule_fwd")
    # Generic/Ascend release kernels store K,V; MetaX stores V,K. Do not infer
    # layout from shape: K == V is common and conceals a transposition error.
    kernel_v_major = "._metax." in op.__module__
    transpose = state_v_major != kernel_v_major
    initial = kwargs.get("initial_state")
    if transpose and initial is not None:
        kwargs["initial_state"] = initial.transpose(-1, -2).contiguous()
    result = list(op(**kwargs))
    if transpose and result[3] is not None:
        result[3] = result[3].transpose(-1, -2).contiguous()
    return tuple(result)


def fused_recurrent_gated_delta_rule_fwd(*, state_v_major=True, **kwargs):
    from vllm.triton_utils.allocation import set_triton_allocator

    set_triton_allocator(kwargs["q"].device)
    op = required_op("fused_recurrent_gated_delta_rule_fwd")
    q, k, v = (kwargs[name] for name in ("q", "k", "v"))
    original_state = kwargs.get("initial_state")
    initial = original_state
    if initial is None:
        cu_seqlens = kwargs.get("cu_seqlens")
        count = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        dims = (
            (v.shape[-1], k.shape[-1]) if state_v_major else (k.shape[-1], v.shape[-1])
        )
        initial = torch.zeros(
            count, v.shape[2], *dims, device=q.device, dtype=torch.float32
        )
    # The release's debug logging dereferences cu_seqlens on strided inputs.
    if kwargs.get("cu_seqlens") is None:
        kwargs["cu_seqlens"] = (
            torch.arange(q.shape[0] + 1, device=q.device, dtype=torch.int32)
            * q.shape[1]
        )
        kwargs["q"] = q.reshape(1, -1, *q.shape[2:])
        kwargs["k"] = k.reshape(1, -1, *k.shape[2:])
        kwargs["v"] = v.reshape(1, -1, *v.shape[2:])
        kwargs["g"] = kwargs["g"].reshape(1, -1, *kwargs["g"].shape[2:])
        kwargs["beta"] = kwargs["beta"].reshape(1, -1, *kwargs["beta"].shape[2:])
    # Evaluate the release predicate on the tensors actually passed to it.
    packed = all(
        x.stride(0) == x.stride(1) + x.stride(2)
        for x in (kwargs["q"], kwargs["k"], kwargs["v"])
    )
    kernel_v_major = kwargs["q"].shape[1] > 64 and not packed
    transpose = state_v_major != kernel_v_major
    if transpose:
        initial = initial.transpose(-1, -2).contiguous()
    kwargs["initial_state"] = initial
    # Without explicit indices this release addresses initial states by token
    # offset (bos), whereas the public FLA API supplies one state per sequence.
    # Its in-place stores also read an index for every token. Materialize that
    # mapping rather than allowing a later sequence to read beyond its state.
    indices = kwargs.get("ssm_state_indices")
    explicit_indices = indices is not None
    if indices is None:
        indices = torch.arange(
            kwargs["cu_seqlens"].numel() - 1,
            device=q.device,
            dtype=torch.int32,
        )
    if indices.ndim == 1:
        width = kwargs["q"].shape[1] if kwargs.get("inplace_final_state", True) else 1
        indices = indices[:, None].expand(-1, width).contiguous()
    valid_tokens = None
    if explicit_indices:
        # The release short kernel does not guard PAD_SLOT_ID=-1. Redirect
        # padding to a private scratch state, never to a negative pool address.
        accepted = kwargs.get("num_accepted_tokens")
        start_column = (
            torch.zeros(indices.shape[0], device=q.device, dtype=torch.long)
            if accepted is None
            else (accepted.long() - 1).clamp_min(0)
        )
        valid_sequences = indices.gather(1, start_column[:, None]).squeeze(1) >= 0
        positions = torch.arange(kwargs["q"].shape[1], device=q.device)
        sequence = torch.searchsorted(kwargs["cu_seqlens"][1:], positions, right=True)
        offset = (positions - kwargs["cu_seqlens"][sequence]).clamp_max(
            indices.shape[1] - 1
        )
        valid_tokens = valid_sequences[sequence] & (indices[sequence, offset] >= 0)
        scratch_index = initial.shape[0]
        indices = torch.where(
            valid_sequences[:, None] & (indices >= 0), indices, scratch_index
        ).contiguous()
        initial = torch.cat((initial, torch.zeros_like(initial[:1])), dim=0)
        kwargs["initial_state"] = initial
    kwargs["ssm_state_indices"] = indices
    output, final = op(**kwargs)
    if valid_tokens is not None:
        output = output.masked_fill(~valid_tokens[None, :, None, None], 0)
        if kwargs.get("inplace_final_state", True):
            final = final[:-1]
        else:
            final = final.masked_fill(~valid_tokens[:, None, None, None], 0)
    if final is not None:
        if kernel_v_major and not kwargs.get("inplace_final_state", True):
            # Large-T stores V,K, but the allocation carries K,V metadata.
            final = final.reshape(*final.shape[:-2], v.shape[-1], k.shape[-1])
        if transpose:
            final = final.transpose(-1, -2).contiguous()
        if kwargs.get("inplace_final_state", True) and original_state is not None:
            original_state.copy_(final)
            final = original_state
    return output.reshape(v.shape), final


class LayerNormFn(torch.autograd.Function):
    """Inference-only Ascend FLA norm/gate adapter using release kernels."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        dtype = x.dtype
        value = x.float()
        if z is not None and not norm_before_gate:
            value = value * torch.nn.functional.silu(z.float())
        group_size = group_size or x.shape[-1]
        if x.shape[-1] % group_size:
            raise ValueError("group_size must divide the last dimension")
        outputs = []
        for start in range(0, x.shape[-1], group_size):
            part = value[..., start : start + group_size].contiguous()
            w = weight[start : start + group_size].float()
            b = bias[start : start + group_size].float() if bias is not None else None
            if is_rms_norm:
                out = required_op("add_rms_norm")(
                    part, torch.zeros_like(part), (group_size,), w, eps
                )
                if b is not None:
                    out = out + b
            else:
                out = required_op("skip_layer_norm")(
                    part,
                    torch.zeros_like(part),
                    (group_size,),
                    w,
                    b if b is not None else torch.zeros_like(w),
                    eps,
                )
            outputs.append(out)
        output = torch.cat(outputs, dim=-1)
        if z is not None and norm_before_gate:
            output = output * torch.nn.functional.silu(z.float())
        return output.to(dtype)
