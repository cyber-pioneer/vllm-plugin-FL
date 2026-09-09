# Runtime Kernel Profiling

This workflow profiles one fixed workload only: 64 concurrent requests, 4096
input tokens, and 256 output tokens per request. It supports graph and eager
execution with either vLLM-plugin-FL or native vLLM.

The request driver completes one warmup batch before `/start_profile`. Only the
second batch is inside the profiling window. CUDA Graph construction is not
profiled. Reports are generated from the rank-0 runtime trace.

## Models

| Model | Run name | Server script | Request config | TP |
|---|---|---|---|---:|
| Qwen3.6-35B-A3B | `qwen3_6_35b_a3b` | `serve_qwen3_6_35b_a3b.sh` | `qwen3_6_35b_a3b_request_4096_256.json` | 2 |
| DeepSeek-V4-Flash | `deepseek_v4_flash` | `serve_deepseek_v4_flash.sh` | `deepseek_v4_flash_request_4096_256.json` | 8 |

Run commands from `/vllm-workspace/vllm-plugin-FL`.

## Run one scenario

Choose one environment from the table. Use the same `PROFILE_RUN_SUFFIX` in
both terminals so the server trace and request artifacts share one directory.

| Scenario | Server environment | Suggested suffix |
|---|---|---|
| plugin graph | none | `_plugin_graph_4096_256` |
| plugin eager | `PROFILE_EXECUTION_MODE=eager` | `_plugin_eager_4096_256` |
| native graph | `VLLM_PLUGINS=""` | `_native_graph_4096_256` |
| native eager | `VLLM_PLUGINS="" PROFILE_EXECUTION_MODE=eager` | `_native_eager_4096_256` |

Terminal A:

```bash
<server-environment> \
PROFILE_RUN_SUFFIX=<suffix> \
bash tools/graph_operator_profile/<server-script>
```

Terminal B, after `/health` is ready:

```bash
PROFILE_RUN_SUFFIX=<suffix> \
bash tools/graph_operator_profile/profile_request.sh \
  <run-name> \
  tools/graph_operator_profile/<request-config>
```

Example: Qwen plugin graph:

```bash
PROFILE_RUN_SUFFIX=_plugin_graph_4096_256 \
bash tools/graph_operator_profile/serve_qwen3_6_35b_a3b.sh
```

```bash
PROFILE_RUN_SUFFIX=_plugin_graph_4096_256 \
bash tools/graph_operator_profile/profile_request.sh \
  qwen3_6_35b_a3b \
  tools/graph_operator_profile/qwen3_6_35b_a3b_request_4096_256.json
```

Results are written to:

```text
/vllm-workspace/graph_operator_profile_runs/<run-name><suffix>/results/
```

For plugin runs, confirm that `/tmp/flaggems_enable_oplist.txt` belongs to the
current run before moving it into the same `results` directory. For native
runs, confirm that the server process has an explicitly empty `VLLM_PLUGINS`,
the server log has no plugin activation message, and no FlagGems oplist is
created.

## Compare four scenarios

Run all four rows in the scenario table for one model, then compare these pairs
with `compare_kernel_profiles.py --scan-cpu-operators`:

- plugin graph versus native graph
- plugin eager versus native eager
- plugin graph versus plugin eager
- native graph versus native eager

Generate the consolidated report with:

```bash
python3 tools/graph_operator_profile/generate_four_scenario_report.py \
  --model-title <model-title> \
  --tp-size <tp-size> \
  --plugin-graph <plugin-graph-run> \
  --plugin-eager <plugin-eager-run> \
  --native-graph <native-graph-run> \
  --native-eager <native-eager-run> \
  --output-dir <comparison-directory>
```

## Output files

`operator_list.csv` is the deduplicated operator-to-kernel inventory:

- `operator_id`
- `operator_name`
- `operator_kind`
- `kernel_name`

Every non-communication row has a positive integer ID. ATen operators are
numbered first. Pure communication rows use `operator_id=null`, are excluded
from numbering, and appear last. Unattributed kernels are retained with
`operator_name=null`; all `nvjet_tst_*` kernels share one operator ID.

Custom operators are numbered by normalized demangled callable identity.
Function arguments, template arguments, and a leading `void` return type do
not affect the ID. Different specializations of one callable therefore share
an ID. All `moe_align_block_size_stage*` kernels are normalized to
`moe_align_block_size` and share one ID.

`operator_kind` is one of:

- `aten`
- `custom`
- `communication`
- `fused_communication_compute`
- `runtime_operator`
- `triton_compiled`
- `torch_compile`
- `unattributed_nvjet`
- `unattributed`

Communication classification is rule-based. NCCL, vLLM custom all-reduce, and
PyTorch symmetric-memory all-reduce kernels are pure communication. A kernel
that combines a collective with model computation remains numbered as
`fused_communication_compute`.

`kernel_time.csv` is the compact timing aggregate. Each row is one unique
`(operator_name, kernel_name)` relation:

- `operator_name`
- `kernel_name`
- `kernel_call_count`
- `kernel_time_us`
- `percent`

`kernel_time_us` is summed runtime kernel duration. `percent` uses the total
rank-0 runtime kernel duration as its denominator, has three decimal places,
and emits values below `0.001%` as `<0.001%`. Every physical kernel event
contributes to exactly one row, so totals are not duplicated.

`kernel_shape_dtype.csv` is the input-metadata aggregate. Each row represents
one kernel/operator/shape/dtype/mapping-status combination:

- `operator_name`
- `kernel_name`
- `variant_index`
- `mapping_status`
- `input_shapes`
- `input_dtypes`
- `candidate_operators`
- `kernel_event_count`
- `kernel_time_us`

Shapes, dtypes, and candidates are compact JSON stored in CSV cells. Missing
metadata is the literal `null`; it never causes a kernel to be dropped.
`mapping_status` is one of:

- `operator_shape_matched`
- `operator_matched_shape_missing`
- `operator_matched_dtype_missing`
- `operator_matched_metadata_missing`
- `shape_ambiguous`
- `operator_ambiguous`
- `missing_external_id`
- `no_cpu_op_match`

`summary.json` records collection scope, event counts, mapping coverage, and
conservation checks. A valid extraction requires every value in
`conservation` to be `true`. Kernel time is aggregated as integer nanoseconds
and emitted in microseconds.

## Coverage boundary

The reports retain every GPU kernel event emitted in the selected rank-0
runtime trace, including kernels whose operator, shape, or dtype cannot be
recovered. They exclude CUDA Graph construction, CPU-only operators, and
per-event timestamps, process IDs, thread IDs, streams, and External ids.

CUDA Graph replay exposes physical kernels but usually does not replay the
original PyTorch CPU operators. A graph-internal kernel can therefore have no
recoverable operator, input shape, or dtype. It remains in every kernel
inventory with `null` metadata and an explicit mapping status.

Memcpy and memset counts and durations remain in `summary.json`, but they are
not kernel keys. Different requests, batch sizes, sequence lengths, sampling
settings, TP ranks, or MoE routing can activate different kernels and shapes.
Rank 0 is a reproducible single-rank view, not proof that another rank has no
additional activity.
