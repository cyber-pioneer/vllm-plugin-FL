#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 <model-key> [workload-config.json]" >&2
  exit 2
fi

MODEL_CASE=$1
RUN_ROOT=${PROFILE_RUN_ROOT:-/vllm-workspace/graph_operator_profile_runs}
RUN_SUFFIX=${PROFILE_RUN_SUFFIX:-}
RUN_DIR="$RUN_ROOT/$MODEL_CASE$RUN_SUFFIX"
PROFILE_DIR="$RUN_DIR/profile"
TOOL_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_CONFIG=${PROFILE_MODEL_CONFIG:-$TOOL_DIR/models.json}
REQUEST_CONFIG=${2:-$TOOL_DIR/workload_4096_256.json}
BASE_URL=${PROFILE_BASE_URL:-http://localhost:8000}
HEALTH_TIMEOUT=${PROFILE_HEALTH_TIMEOUT_SECONDS:-3600}
SERVED_MODEL=$(python3 -c '
import json
import sys

models = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    print(models[sys.argv[2]]["served_model_name"])
except KeyError as error:
    raise SystemExit(f"invalid model registry entry: {error}")
' "$MODEL_CONFIG" "$MODEL_CASE")
if [[ ! -d "$PROFILE_DIR" ]]; then
  echo "profile directory does not exist: $PROFILE_DIR" >&2
  echo "start serve.py with the same model key and PROFILE_RUN_SUFFIX first" >&2
  exit 1
fi
active=0
stop_profile() {
  if [[ "$active" -eq 1 ]]; then
    curl -fsS -XPOST "$BASE_URL/stop_profile"
    active=0
  fi
}
trap stop_profile EXIT

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS "$BASE_URL/health" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "server health check timed out after $HEALTH_TIMEOUT seconds" >&2
    exit 1
  fi
  sleep 5
done
python3 "$TOOL_DIR/run_concurrent_requests.py" \
  --config "$REQUEST_CONFIG" \
  --model "$SERVED_MODEL" \
  --base-url "$BASE_URL" \
  --prompt-output "$RUN_DIR/prompt_token_ids.json" \
  --responses "$RUN_DIR/warmup_responses.json" \
  --metrics "$RUN_DIR/warmup_metrics.json"

curl -fsS -XPOST "$BASE_URL/start_profile"
active=1
python3 "$TOOL_DIR/run_concurrent_requests.py" \
  --config "$REQUEST_CONFIG" \
  --model "$SERVED_MODEL" \
  --base-url "$BASE_URL" \
  --prompt-input "$RUN_DIR/prompt_token_ids.json" \
  --responses "$RUN_DIR/profiled_responses.json" \
  --metrics "$RUN_DIR/profiled_metrics.json"
stop_profile

python3 "$TOOL_DIR/extract_operator_shapes.py" \
  --runtime "$PROFILE_DIR" \
  --rank 0 \
  --output-dir "$RUN_DIR/results"
