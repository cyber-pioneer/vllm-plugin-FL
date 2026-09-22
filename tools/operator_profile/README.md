# Operator Profiling

This directory profiles one serving scenario at a time by using vLLM's native
`/start_profile` and `/stop_profile` endpoints. It does not alter vLLM's rank
policy: every rank profiles normally, while the offline extractor reads rank 0
only.

The workload is fixed at 64 concurrent requests with 4096 input tokens and 256
output tokens per request. The first batch is warmup. Only the second batch is
inside the profiling window.

Run all commands from the vllm-plugin-FL repository root and use a new directory
under `/workspace/op_profile` for each scenario.

## Start a server

`serve.sh` is a thin wrapper around `vllm serve`. It adds only the native torch
profiler configuration and output-directory checks. Model selection and other
vLLM arguments remain explicit command-line inputs.

Qwen plugin graph example:

```bash
RUN_DIR=/workspace/op_profile/qwen_plugin_graph_4096_256
VLLM_PLUGINS=fl USE_FLAGTUNE=0 bash tools/operator_profile/serve.sh \
  /models/Qwen3.6-35B-A3B qwen "$RUN_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config \
    '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

Qwen native-vLLM graph baseline:

```bash
RUN_DIR=/workspace/op_profile/qwen_native_graph_4096_256
VLLM_PLUGINS="" bash tools/operator_profile/serve.sh \
  /models/Qwen3.6-35B-A3B qwen "$RUN_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config \
    '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

DeepSeek plugin graph example:

```bash
RUN_DIR=/workspace/op_profile/deepseek_plugin_graph_4096_256
VLLM_PLUGINS=fl USE_FLAGTUNE=0 bash tools/operator_profile/serve.sh \
  /models/DeepSeek-V4-Flash deepseek-v4-flash "$RUN_DIR" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --safetensors-load-strategy prefetch \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config \
    '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

The examples do not set `VLLM_USE_BREAKABLE_CUDAGRAPH`; vLLM retains its native
model- and version-specific selection. Omit `--enforce-eager` for graph mode and
add it only for an eager run.

`USE_FLAGTUNE=0` disables the optional remote FlagTune cost model while keeping
FlagGems kernels and their local Triton autotuning enabled. It makes offline runs
independent of FlagTune model downloads. Remove it when remote FlagTune models
are available and intentionally part of the test.

Before startup, `serve.sh` removes an earlier temporary
`/tmp/flaggems_enable_oplist.txt`. This prevents stale plugin evidence from
entering the current run or a native baseline.

## Collect one profile

Run this in another terminal after the server starts:

```bash
bash tools/operator_profile/profile.sh qwen \
  /workspace/op_profile/qwen_plugin_graph_4096_256
```

The script waits for server health, sends one warmup batch, calls the native
`/start_profile` endpoint, sends the measured batch, calls `/stop_profile`, and
then parses rank 0. An output directory containing an existing trace is rejected.

## Output files

- `operator_list.csv`: normalized operator inventory, stable operator IDs, and
  each row's share of total rank-0 runtime kernel time. `1.23` means `1.23%`;
  the column header carries the `%` unit, and values below `0.01%` are written
  as `<0.01`.
- `kernel_time.csv`: kernel call counts, durations, and runtime time shares.
- `kernel_shape_dtype.csv`: kernel/operator/shape/dtype mapping variants.
- `summary.json`: trace scope, mapping coverage, and conservation checks.
- `flaggems_enable_oplist.txt`: current plugin evidence, when generated.

Missing operator attribution, shape, or dtype is retained as `null` with an
explicit mapping status. No kernel is dropped. Every check under
`summary.json.conservation` must be `true`.

Mapping and grouping rules are isolated in `rule/rule_map.py`. Coverage rules
are isolated in `rule/rule_coverage.py`.

## Generate FlagOS coverage

```bash
python3 tools/operator_profile/generate_flagos_coverage.py \
  --baseline /workspace/op_profile/qwen_native_graph_4096_256/results/operator_list.csv \
  --plugin /workspace/op_profile/qwen_plugin_graph_4096_256/results/operator_list.csv \
  --flaggems-oplist /workspace/op_profile/qwen_plugin_graph_4096_256/results/flaggems_enable_oplist.txt \
  --output /workspace/op_profile/qwen_flagos_coverage/operator_flagos_coverage.csv
```

The current policy counts every observed Triton operation in the numerator.
Other operations require auditable FlagGems evidence. Communication remains in
the denominator and requires FlagCX evidence to enter the numerator. Coverage
is based on operator kinds and is not weighted by calls or execution time.
