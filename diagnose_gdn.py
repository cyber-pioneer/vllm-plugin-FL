"""Diagnostic: wrap Ascend kernels with validation against PyTorch references."""
import os
import sys
import torch
import torch.nn.functional as F
import logging

log = logging.getLogger("gdn_diag")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

_call_count = {"gating": 0, "recurrent": 0, "chunk": 0, "conv1d_fn": 0, "conv1d_upd": 0}
_MAX_CHECKS = 3  # Only validate first N calls to avoid slowing down


def _gating_ref(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Pure PyTorch reference for fused_gdn_gating."""
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x, beta=beta, threshold=threshold)
    beta_out = torch.sigmoid(b.float()).to(b.dtype)
    return g.unsqueeze(0), beta_out.unsqueeze(0)


def make_validating_gating(real_fn):
    def wrapper(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        _call_count["gating"] += 1
        g, beta_out = real_fn(A_log, a, b, dt_bias, beta, threshold)
        if _call_count["gating"] <= _MAX_CHECKS:
            g_ref, beta_ref = _gating_ref(A_log, a, b, dt_bias, beta, threshold)
            g_err = (g.float() - g_ref.float()).abs().max().item()
            beta_err = (beta_out.float() - beta_ref.float()).abs().max().item()
            log.info(
                f"[DIAG gating #{_call_count['gating']}] "
                f"g shape={g.shape} dtype={g.dtype} "
                f"range=[{g.min().item():.4f}, {g.max().item():.4f}] "
                f"nan={g.isnan().any().item()} "
                f"max_err_vs_ref={g_err:.6f} | "
                f"beta shape={beta_out.shape} dtype={beta_out.dtype} "
                f"range=[{beta_out.min().item():.4f}, {beta_out.max().item():.4f}] "
                f"nan={beta_out.isnan().any().item()} "
                f"max_err_vs_ref={beta_err:.6f}"
            )
            if g_err > 0.01 or beta_err > 0.01:
                log.error(f"[DIAG gating] LARGE ERROR! g_err={g_err} beta_err={beta_err}")
        return g, beta_out
    return wrapper


def make_validating_recurrent(real_fn):
    def wrapper(q, k, v, g, beta, initial_state=None, **kwargs):
        _call_count["recurrent"] += 1
        # Run the real kernel
        o, final_state = real_fn(q, k, v, g, beta, initial_state=initial_state, **kwargs)
        if _call_count["recurrent"] <= _MAX_CHECKS:
            log.info(
                f"[DIAG recurrent #{_call_count['recurrent']}] "
                f"q={q.shape} k={k.shape} v={v.shape} g={g.shape} beta={beta.shape} "
                f"o={o.shape} o_range=[{o.min().item():.4f}, {o.max().item():.4f}] "
                f"o_nan={o.isnan().any().item()} o_inf={o.isinf().any().item()} "
                f"state_nan={final_state.isnan().any().item() if final_state is not None else 'N/A'}"
            )
        return o, final_state
    return wrapper


def make_validating_chunk(real_fn):
    def wrapper(q, k, v, g, beta, initial_state=None, **kwargs):
        _call_count["chunk"] += 1
        o, final_state = real_fn(q, k, v, g, beta, initial_state=initial_state, **kwargs)
        if _call_count["chunk"] <= _MAX_CHECKS:
            log.info(
                f"[DIAG chunk #{_call_count['chunk']}] "
                f"q={q.shape} k={k.shape} v={v.shape} g={g.shape} beta={beta.shape} "
                f"initial_state={initial_state.shape if initial_state is not None else None} "
                f"o={o.shape} o_range=[{o.min().item():.4f}, {o.max().item():.4f}] "
                f"o_nan={o.isnan().any().item()} o_inf={o.isinf().any().item()} "
                f"final_state={final_state.shape if final_state is not None else None} "
                f"fs_nan={final_state.isnan().any().item() if final_state is not None else 'N/A'} "
                f"fs_range=[{final_state.min().item():.4f}, {final_state.max().item():.4f}]"
                if final_state is not None else ""
            )
        return o, final_state
    return wrapper


def install_diagnostics():
    """Call this after the Ascend patches are applied to wrap them with validation."""
    import vllm.model_executor.models.qwen3_next as mod

    real_gating = mod.fused_gdn_gating
    mod.fused_gdn_gating = make_validating_gating(real_gating)
    log.info("[DIAG] Wrapped fused_gdn_gating with validation")

    real_recurrent = mod.fused_recurrent_gated_delta_rule
    mod.fused_recurrent_gated_delta_rule = make_validating_recurrent(real_recurrent)
    log.info("[DIAG] Wrapped fused_recurrent_gated_delta_rule with validation")

    real_chunk = mod.chunk_gated_delta_rule
    mod.chunk_gated_delta_rule = make_validating_chunk(real_chunk)
    log.info("[DIAG] Wrapped chunk_gated_delta_rule with validation")


if __name__ == "__main__":
    # Standalone test of fused_gdn_gating_patch vs reference
    print("=== Standalone fused_gdn_gating test ===")
    from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils import init_device_properties_triton
    init_device_properties_triton()
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_gdn_gating import (
        fused_gdn_gating_patch,
    )

    for batch in [1, 4, 8, 32]:
        num_heads = 8  # typical for 35B TP=4
        A_log = torch.randn(num_heads, dtype=torch.bfloat16, device="npu:0")
        a = torch.randn(batch, num_heads, dtype=torch.bfloat16, device="npu:0")
        b = torch.randn(batch, num_heads, dtype=torch.bfloat16, device="npu:0")
        dt_bias = torch.randn(num_heads, dtype=torch.bfloat16, device="npu:0")

        g_kern, beta_kern = fused_gdn_gating_patch(A_log, a, b, dt_bias)
        g_ref, beta_ref = _gating_ref(A_log, a, b, dt_bias)

        g_err = (g_kern.float() - g_ref.float()).abs().max().item()
        beta_err = (beta_kern.float() - beta_ref.float()).abs().max().item()
        print(f"  batch={batch} num_heads={num_heads}: g_max_err={g_err:.6f} beta_max_err={beta_err:.6f}")
        assert g_err < 0.01, f"GATING g ERROR too large: {g_err}"
        assert beta_err < 0.01, f"GATING beta ERROR too large: {beta_err}"

    print("All gating tests passed!")
