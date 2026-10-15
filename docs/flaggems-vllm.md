# FlagGems-vllm fused operators

When enabled, direct FlagGems operator calls dispatch through FlagGems-vllm,
including activation, rotary, attention, MoE, normalization, scaled matrix
multiplication, and GDN. General ATen registration continues to use
`flag_gems.enable()`. Runtime utilities such as device detection and ATen
registration helpers also remain with FlagGems. Existing attention
backend selection is preserved: the default service paths use upstream Triton
or DeepSeek sparse attention unless FlagGems attention is explicitly selected.

Install the pinned release source along with the plugin:

```bash
python3 -m pip install --no-deps --force-reinstall \
  'flaggems-vllm @ git+https://github.com/flagos-ai/FlagGems-vllm.git@b5c2c2249319277ccda4d41aa0f65164bd9b2a19'
python3 -m pip install --no-build-isolation -e .
```

CI installs the v0.2.0 release commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`. That release still declares
`0.1.0` in its package metadata and `__version__`; verify its source commit or
pip's `direct_url.json` instead of interpreting the metadata as the release tag.
The release is https://github.com/flagos-ai/FlagGems-vllm/releases/tag/v0.2.0.

Every hardware platform's CI setup calls the shared installer with its selected
Python interpreter. It verifies the source commit in `direct_url.json` and keeps
the existing Torch, Triton, FlagGems, vLLM, and NumPy versions. NVIDIA CI alone
requires FlagGems 5.4.0. All nine platforms (CUDA, Ascend, Hygon, MetaX, MUSA,
Enflame, Iluvatar, Kunlunxin, and T-Head) are enabled in the CI registry, and
all platform test containers set `VLLM_FL_USE_FLAGGEMS_VLLM=1`. This exercises
the fused integration through each platform's existing unit, functional,
inference, serving, and benchmark gates where configured. CUDA model gates
also run the same inference/serving cases with `0` as a baseline, using the
same dependencies. Platform compatibility is evaluated by CI; enabling the
switch does not establish that every release kernel supports every platform.
The release kernel checks remain scoped to NVIDIA CUDA.

FlagGems-vllm integration is disabled by default, even when the package is
installed. Set `VLLM_FL_USE_FLAGGEMS_VLLM=1` before starting vLLM to enable it.
Leaving the variable unset, or setting it to `0`, selects the existing FlagGems
fused implementations and the in-tree INT8 quantization kernel. General
`flag_gems.enable()` stays enabled by default in both modes. Restart the service
when changing this variable. The enabled path does not silently substitute
the old package when FlagGems-vllm is absent or a fused kernel fails. Existing
operator dispatch policy remains responsible for backend selection. Clamped
SwiGLU uses the fused package for alpha=1 and beta=0; other parameters use
the existing native formula because the release does not implement them.

Run the fused integration checks on CUDA:

```bash
VLLM_PLUGINS=fl VLLM_FL_USE_FLAGGEMS_VLLM=1 \
    python3 -m pytest tests/functional_tests/flaggems/test_vllm_package.py
```

Follow `tools/adaptation-gate-cases/README.md` for the full Qwen3.6 eager and
graph text, image, and mixed request matrix. This main branch targets vLLM
0.24.0; model architectures and the plugin ModelRunnerFL must be compatible
with that vLLM installation. DeepSeek-V4-Flash is text-only, so the gate
image and mixed-image scenarios do not apply to it. Run both gate text
scenarios in eager and graph modes when validating that model.

The plugin enables selected fused activation CustomOps for Inductor so default
CUDA graphs execute FlagGems-vllm kernels instead of compiling the native
activation formula. Opaque Torch operators and fake implementations preserve
full graph tracing. Explicit per-operator disables and OOT whitelist/blacklist
settings remain effective; the CUDA graph mode is unchanged.

DeepSeek-V4-Flash FP8 checkpoints require `--kv-cache-dtype fp8` with this
vLLM version. The validation used tensor parallel size 4. Add `--enforce-eager`
for eager mode and omit it for the default graph mode.

The plugin's GDN adapters select the pinned release's chunk and recurrent
operators. They translate vLLM 0.24 V-major states to the selected release
kernel's layout and preserve in-place state updates. Explicit sequence-to-state
indices avoid token-offset addressing; padding slots use private scratch state
and produce zero output without reading or updating the caller's pool.
Ascend's older K-major
FLA contract is adapted separately. Existing upstream model paths that do not
call a FlagGems operator retain their original backend selection.

The release exposes add-RMSNorm rather than standalone RMSNorm. The standalone
adapter supplies a zero residual; the fused residual adapter preserves the
caller-owned input and residual buffers, including noncontiguous views.
W8A8 linear uses the release's `triton_scaled_mm` interface. Sunrise attention,
MLA, scaled-mm, and MoE configuration imports also follow the fused switch;
Ascend FLA normalization and expert calls use the release adapters.

FlagGems-vllm v0.2.0 does not export ARM `pack_rhs_qsi4c128p` or
`w4a8_g128_linear`. Selecting those ARM paths with the fused switch enabled
raises an explicit unsupported-operator error. They require an upstream release
implementation; the plugin does not silently call the old FlagGems kernels.
With the switch disabled, the existing ARM implementations remain available.

Standard clamped SwiGLU uses the same dispatcher and selection policy as SiLU
and GELU. Its public `custom_ops` name remains `silu_and_mul_with_clamp`; the
replacement also accepts `SiluAndMulWithClamp` as an alias. Unsupported alpha
and beta values retain the upstream native formula before fused execution.
MetaX fused FA2 receives no descales for unquantized `auto` KV cache; quantized
KV cache remains unsupported. A separate MetaX opt-in check validates this path.
When MetaX's existing Triton lacks the optional TLE API, the plugin omits the
release's `persistent_topk` vendor export from its eager operator collection.
The compatibility loader verifies the pinned release's initializer hash;
attention and all remaining vendor imports execute unchanged. Installed package
files and the platform's Triton version are not modified.

NVIDIA compatibility exclusions are loaded automatically from `nvidia.yaml`.
No deployment blacklist append or CUDA/NCCL workaround is required. Validate
both modes with the original model gates when updating FlagGems or the fused
release.
