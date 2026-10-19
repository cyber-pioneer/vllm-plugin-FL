# Interfaces retained from FlagGems

This inventory targets FlagGems-vllm **v0.2.0**, commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`, on the vLLM 0.28 branch.
An existing export is substituted only when it preserves the current call
contract. The entries below retain their original providers in both switch
modes. Missing interface requests are tracked in
[FlagGems-vllm #932](https://github.com/flagos-ai/FlagGems-vllm/issues/932).

| Interface | Call site | Reason |
| --- | --- | --- |
| `gems_rms_forward` | `vllm_fl/dispatch/backends/flaggems/impl/normalization.py` | The standalone/residual RMSNorm wrapper is not exported. Combining other kernels would require output and mutation-contract adaptation. |
| `pack_rhs_qsi4c128p` | `vllm_fl/quantization/arm_cpu_w4a8.py` | No corresponding ARM W4A8 packing export exists. |
| `w4a8_g128_linear` | `vllm_fl/quantization/arm_cpu_w4a8.py` | No corresponding ARM W4A8 execution export exists. |
| `chunk_gated_delta_rule_fwd` | `vllm_fl/ops/fla/chunk.py` | The export exists, but a compatible state-layout and vendor contract has not been established for the current vLLM integration. Preserve the branch's existing path instead of adding state conversions. |
| `fused_recurrent_gated_delta_rule_fwd` | `vllm_fl/ops/fla/fused_recurrent.py` | The export exists, but state addressing, padding, and layout require separate compatibility validation. |
| Ascend vendor-specific FLA/MoE implementations | `vllm_fl/dispatch/backends/vendor/ascend/` | Preserve the independently validated vLLM 0.28 Ascend implementation. Ascend CI does not enable the optional provider. This is a platform exclusion, not a claim that every generic operator is absent. |

ATen registration (`flag_gems.enable()`), device detection, runtime utilities,
and backend launch/configuration helpers remain in FlagGems. They are not
explicit fused-operator replacements. The existing native CUDA attention
selection is preserved. The NVIDIA policy documents its temporary ATen `mm`
exclusion separately from these explicit-interface limitations.
