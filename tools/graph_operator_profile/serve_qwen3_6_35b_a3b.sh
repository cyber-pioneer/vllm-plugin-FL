#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${PROFILE_RUN_ROOT:-/vllm-workspace/graph_operator_profile_runs}
RUN_SUFFIX=${PROFILE_RUN_SUFFIX:-}
RUN_DIR="$RUN_ROOT/qwen3_6_35b_a3b$RUN_SUFFIX"
EXECUTION_MODE=${PROFILE_EXECUTION_MODE:-graph}
case "$EXECUTION_MODE" in
  graph)
    EXECUTION_ARGS=()
    ;;
  eager)
    EXECUTION_ARGS=(--enforce-eager)
    ;;
  *)
    echo "unsupported PROFILE_EXECUTION_MODE: $EXECUTION_MODE" >&2
    exit 2
    ;;
esac
PROFILE_DIR="$RUN_DIR/profile"
if [[ -d "$RUN_DIR" ]]; then
  archive="$RUN_ROOT/archive/qwen3_6_35b_a3b${RUN_SUFFIX}_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$(dirname "$archive")"
  mv "$RUN_DIR" "$archive"
fi
mkdir -p "$PROFILE_DIR"

printf -v PROFILER_CONFIG '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_memory":false,"ignore_frontend":true}' "$PROFILE_DIR"

export VLLM_PLUGINS=${VLLM_PLUGINS-fl}
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

exec vllm serve /models/Qwen3.6-35B-A3B \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  "${EXECUTION_ARGS[@]}" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_num_of_warmups":0}' \
  --profiler-config "$PROFILER_CONFIG" \
  > "$RUN_DIR/serve.log" 2>&1
