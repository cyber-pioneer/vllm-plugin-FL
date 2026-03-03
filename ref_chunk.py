"""
Pure PyTorch reference implementation of chunk_gated_delta_rule.
Processes token-by-token (equivalent to fused_recurrent) for verification.
"""
import torch
import torch.nn.functional as F
from typing import Optional


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """
    Pure PyTorch reference - processes token by token.
    q: [B, T, H, K]  (key heads)
    k: [B, T, H, K]  (key heads)
    v: [B, T, HV, V] (value heads)
    g: [B, T, HV]    (value heads)
    beta: [B, T, HV] (value heads)
    initial_state: [N, HV, K, V]
    """
    B, T_total, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K ** -0.5

    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Clone initial_state for the output final state
    if initial_state is not None:
        final_state = initial_state.clone()
    else:
        final_state = torch.zeros(N, HV, K, V, dtype=torch.float32, device=q.device)

    o = torch.zeros_like(v)  # [B, T_total, HV, V]

    for n in range(N):
        if cu_seqlens is not None:
            bos = cu_seqlens[n].item()
            eos = cu_seqlens[n + 1].item()
        else:
            bos = n * T_total
            eos = bos + T_total
        T = eos - bos

        if T == 0:
            continue

        for hv in range(HV):
            h_idx = hv // (HV // H)  # GQA key head mapping
            # Load state [K, V]
            h = final_state[n, hv].float()

            for t in range(T):
                abs_t = bos + t
                q_vec = q[0, abs_t, h_idx].float()
                k_vec = k[0, abs_t, h_idx].float()
                v_vec = v[0, abs_t, hv].float()
                g_val = g[0, abs_t, hv].float()
                b_val = beta[0, abs_t, hv].float()

                if use_qk_l2norm_in_kernel:
                    q_vec = F.normalize(q_vec, p=2, dim=-1)
                    k_vec = F.normalize(k_vec, p=2, dim=-1)
                q_vec = q_vec * scale

                # Decay state
                h = h * torch.exp(g_val)
                # Delta rule update
                delta = v_vec - h @ k_vec  # [V] - [K,V]@[K]
                delta = delta * b_val
                h = h + k_vec.unsqueeze(1) * delta.unsqueeze(0)  # [K,V]
                # Output
                o_vec = h.T @ q_vec  # [V,K]@[K] = [V]
                o[0, abs_t, hv] = o_vec.to(o.dtype)

            # Store final state
            final_state[n, hv] = h.to(final_state.dtype)

    if output_final_state:
        return o, final_state
    return o, None


def fused_recurrent_ref(
    q, k, v, g, beta, initial_state=None,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Pure PyTorch reference for fused_recurrent_gated_delta_rule."""
    B, T_total, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    scale = K ** -0.5
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = initial_state.clone()

    o = torch.zeros_like(v)

    for n in range(N):
        if cu_seqlens is not None:
            bos = cu_seqlens[n].item()
            eos = cu_seqlens[n + 1].item()
        else:
            bos = n * T_total
            eos = bos + T_total
        T = eos - bos
        if T == 0:
            continue

        if ssm_state_indices is not None:
            if num_accepted_tokens is not None:
                t_init = num_accepted_tokens[n].item() - 1
            else:
                t_init = 0
            state_idx = (ssm_state_indices[n].item()
                         if ssm_state_indices.ndim == 1
                         else ssm_state_indices[n, t_init].item())
        else:
            state_idx = n

        for hv in range(HV):
            h_idx = hv // (HV // H)
            h = initial_state[state_idx, hv].float()

            for t in range(T):
                abs_t = bos + t
                q_vec = q[0, abs_t, h_idx].float()
                k_vec = k[0, abs_t, h_idx].float()
                v_vec = v[0, abs_t, hv].float()
                g_val = g[0, abs_t, hv].float()
                b_val = beta[0, abs_t, hv].float()

                if use_qk_l2norm_in_kernel:
                    q_vec = F.normalize(q_vec, p=2, dim=-1)
                    k_vec = F.normalize(k_vec, p=2, dim=-1)
                q_vec = q_vec * scale

                h = h * torch.exp(g_val)
                delta = v_vec - h @ k_vec
                delta = delta * b_val
                h = h + k_vec.unsqueeze(1) * delta.unsqueeze(0)
                o_vec = h.T @ q_vec
                o[0, abs_t, hv] = o_vec.to(o.dtype)

            if inplace_final_state:
                initial_state[state_idx, hv] = h.to(initial_state.dtype)

    return o, final_state


def fused_gdn_gating_ref(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Pure PyTorch reference for fused_gdn_gating."""
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x, beta=beta, threshold=threshold)
    beta_out = torch.sigmoid(b.float()).to(b.dtype)
    return g.unsqueeze(0), beta_out.unsqueeze(0)


def causal_conv1d_ref_loop(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """Pure loop-based causal conv1d (no F.conv1d). x: (dim, seqlen)."""
    dtype_in = x.dtype
    dim, seqlen = x.shape
    _, width = weight.shape

    # Build full history: [initial_states | x] or [zeros | x]
    if initial_states is not None:
        full = torch.cat([initial_states.float(), x.float()], dim=-1)
    else:
        full = torch.cat([torch.zeros(dim, width - 1, device=x.device, dtype=torch.float32),
                          x.float()], dim=-1)

    # Compute causal conv1d via loop
    w = weight.float()  # (dim, width)
    out = torch.zeros(dim, seqlen, device=x.device, dtype=torch.float32)
    for t in range(seqlen):
        # window: full[:, t : t + width]
        window = full[:, t : t + width]  # (dim, width)
        out[:, t] = (window * w).sum(dim=-1)
    if bias is not None:
        out = out + bias.float().unsqueeze(-1)

    # Final states: last (width-1) elements from full
    if return_final_states:
        final_states = full[:, -(width - 1):].to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states

    # Activation
    if activation in ["silu", "swish"]:
        out = out * torch.sigmoid(out)
    out = out.to(dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    conv_states: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    metadata=None,
    pad_slot_id: int = -1,
):
    """Pure PyTorch reference for causal_conv1d_fn (prefill path)."""
    from vllm.attention.backends.utils import PAD_SLOT_ID
    if pad_slot_id == -1:
        pad_slot_id = PAD_SLOT_ID

    if x.stride(-1) != 1:
        x = x.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]

    outputs = []
    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        init_s = None
        if has_initial_state[i]:
            init_s = conv_states[cache_indices[i]][..., :(width - 1)]
        out_s, _ = causal_conv1d_ref_loop(
            x_s,
            weight,
            bias,
            initial_states=init_s,
            return_final_states=True,
            final_states_out=conv_states[cache_indices[i]][..., :(width - 1)].unsqueeze(0),
            activation=activation,
        )
        outputs.append(out_s)
    return torch.cat(outputs, dim=-1)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation=None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    validate_data: bool = False,
):
    """Pure PyTorch reference for causal_conv1d_update (decode path)."""
    from vllm.attention.backends.utils import PAD_SLOT_ID
    if pad_slot_id == -1:
        pad_slot_id = PAD_SLOT_ID
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    original_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    _, width = weight.shape

    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    out = torch.empty_like(x)

    for b_idx in range(batch):
        if conv_state_indices is not None:
            state_idx = conv_state_indices[b_idx].item()
            if state_idx == PAD_SLOT_ID:
                continue
        else:
            state_idx = b_idx

        # Get input for this sequence
        if query_start_loc is not None:
            qs = query_start_loc[b_idx].item()
            qe = query_start_loc[b_idx + 1].item()
            x_seq = x[qs:qe, :].T  # (dim, seq_len)
        else:
            x_seq = x[b_idx]  # (dim, seqlen)

        seq_len = x_seq.shape[-1]
        if seq_len == 0:
            continue

        # Read current conv state
        state = conv_state[state_idx, :, :(width - 1)].float()  # (dim, width-1)

        # Process token by token
        w = weight.float()  # (dim, width)
        for t in range(seq_len):
            token = x_seq[:, t:t+1].float()  # (dim, 1)
            full_window = torch.cat([state, token], dim=-1)  # (dim, width)
            val = (full_window * w).sum(dim=-1)  # (dim,)
            if bias is not None:
                val = val + bias.float()
            if activation in ["silu", "swish"]:
                val = val * torch.sigmoid(val)
            # Write output
            if query_start_loc is not None:
                out[qs + t, :] = val.to(out.dtype)
            else:
                out[b_idx, :, t] = val.to(out.dtype)
            # Shift state
            state = torch.cat([state[:, 1:], token], dim=-1)

        # Write back state
        conv_state[state_idx, :, :(width - 1)] = state.to(conv_state.dtype)

    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_dtype)
