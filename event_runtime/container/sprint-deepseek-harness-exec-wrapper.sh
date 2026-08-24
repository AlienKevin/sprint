#!/usr/bin/env bash
set -uo pipefail

RUNTIME_DIR=${SPRINT_RUNTIME_DIR:-/run}
AGENT_STATE_DIR="$RUNTIME_DIR/sprint-agent"
AGENT_LOG_DIR=${SPRINT_AGENT_LOG_DIR:-/logs/agent}
DURABLE_DIR=${SPRINT_DURABLE_DIR:-/durable}
RUN_ID=${SPRINT_RUN_ID:-}
ATTEMPT=${SPRINT_CPU_LAUNCH_ATTEMPT:-1}
PROXY_BIN=${SPRINT_OPENROUTER_PROXY_BIN:-/opt/sprint-openrouter-ledger-proxy.py}
PROXY_BASE_URL=${SPRINT_OPENROUTER_PROXY_BASE_URL:-http://127.0.0.1:18080/api/v1}
UPSTREAM_URL=${SPRINT_OPENROUTER_UPSTREAM_URL:-https://openrouter.ai/api/v1}
PROVIDER_ENDPOINT=${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-deepseek}
RUNNER=${SPRINT_DEEPSEEK_HARNESS_RUNNER:-/opt/sprint-deepseek-harness-runner.py}
MODEL=${SPRINT_MODEL:-deepseek/deepseek-v4-flash-vision-exp}
PROCESS_FILE="$AGENT_STATE_DIR/agent-process"
EXPECTED_INTERRUPT="$AGENT_STATE_DIR/expected-interrupt"
STOP_ACK_TIMEOUT_SECONDS=${SPRINT_STOP_ACK_TIMEOUT_SECONDS:-600}

EXPECTED_MODEL=deepseek/deepseek-v4-flash-vision-exp
REQUEST_CONTRACT='{"model":"deepseek/deepseek-v4-flash-vision-exp","stream":true,"temperature":1.0,"top_p":0.95,"max_tokens":384000,"reasoning_effort":"max"}'

[[ -n "$RUN_ID" ]] || { echo "SPRINT_RUN_ID is required" >&2; exit 2; }
[[ "$MODEL" == "$EXPECTED_MODEL" ]] || {
  echo "DeepSeek Harness model is sealed to $EXPECTED_MODEL" >&2
  exit 2
}
[[ "$PROVIDER_ENDPOINT" == "deepseek" ]] || {
  echo "Vision Exp is sealed to the official DeepSeek OpenRouter endpoint" >&2
  exit 2
}
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY is required" >&2
  exit 2
}
[[ -x "$PROXY_BIN" && -x "$RUNNER" ]] || {
  echo "DeepSeek Harness runtime wrapper is incomplete" >&2
  exit 2
}
[[ -r /opt/event_runtime/container/sprint-deepseek-goal-bootstrap.mjs ]] || {
  echo "DeepSeek Harness native goal bootstrap is missing" >&2
  exit 2
}
[[ "$STOP_ACK_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "SPRINT_STOP_ACK_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}

umask 077
mkdir -p "$AGENT_STATE_DIR" "$AGENT_LOG_DIR"
rm -f "$EXPECTED_INTERRUPT"

proxy_pid=""
stop_proxy() {
  [[ -n "$proxy_pid" ]] || return 0
  kill "$proxy_pid" 2>/dev/null || true
  wait "$proxy_pid" 2>/dev/null || true
  proxy_pid=""
}
trap stop_proxy EXIT

ledger_root="$DURABLE_DIR/runs/$RUN_ID/api-usage"
"$PROXY_BIN" \
  --upstream "$UPSTREAM_URL" \
  --ledger-root "$ledger_root" \
  --run-id "$RUN_ID" \
  --cpu-attempt "$ATTEMPT" \
  --runtime-dir "$RUNTIME_DIR" \
  --provider-endpoint "$PROVIDER_ENDPOINT" \
  --request-contract-json "$REQUEST_CONTRACT" \
  >>"$AGENT_LOG_DIR/openrouter-ledger-proxy.log" 2>&1 &
proxy_pid=$!

ready=0
for _ in $(seq 1 600); do
  kill -0 "$proxy_pid" 2>/dev/null || break
  if python3 - "$PROXY_BASE_URL" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
from urllib.parse import urlsplit

base = urlsplit(sys.argv[1])
with urllib.request.urlopen(f"{base.scheme}://{base.netloc}/healthz", timeout=1) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    ready=1
    break
  fi
  sleep 0.1
done
((ready == 1)) || { echo "OpenRouter ledger proxy failed to start" >&2; exit 1; }

session_root="$DURABLE_DIR/runs/$RUN_ID/deepseek-harness/sessions"
events="$AGENT_LOG_DIR/deepseek-harness-events.jsonl"
session_id="$RUN_ID"

setsid "$RUNNER" \
  --workspace /app \
  --session-root "$session_root" \
  --session-id "$session_id" \
  --events "$events" \
  --base-url "$PROXY_BASE_URL" \
  --stop-file "$RUNTIME_DIR/sprint-stop" \
  "$@" &
agent_pid=$!

agent_pgid=""
start_time=""
for _ in {1..100}; do
  [[ -r "/proc/$agent_pid/stat" ]] || break
  agent_pgid=$(ps -o pgid= -p "$agent_pid" | tr -d '[:space:]')
  start_time=$(awk '{print $22}' "/proc/$agent_pid/stat" 2>/dev/null || true)
  [[ "$agent_pgid" == "$agent_pid" && -n "$start_time" ]] && break
  sleep 0.01
done
if [[ -z "$agent_pgid" || "$agent_pgid" != "$agent_pid" || -z "$start_time" ]]; then
  kill -TERM "$agent_pid" 2>/dev/null || true
  wait "$agent_pid" 2>/dev/null || true
  echo "failed to create an isolated DeepSeek Harness process group" >&2
  exit 1
fi

process_tmp="$PROCESS_FILE.$$"
printf '%s %s %s\n' "$agent_pid" "$agent_pgid" "$start_time" >"$process_tmp"
mv "$process_tmp" "$PROCESS_FILE"

wait "$agent_pid"
rc=$?
rm -f "$PROCESS_FILE"

if [[ -f "$EXPECTED_INTERRUPT" ]]; then
  expected_reason=$(tr -d '\r\n' <"$EXPECTED_INTERRUPT" 2>/dev/null || true)
  expected_reason=${expected_reason:-operator_stop}
  stop_ack="$DURABLE_DIR/runs/$RUN_ID/STOP_ACK"
  deadline=$((SECONDS + STOP_ACK_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if python3 - "$stop_ack" "$RUN_ID" "$expected_reason" <<'PY'
import json
import pathlib
import sys

try:
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
valid = (
    payload.get("run_id") == sys.argv[2]
    and payload.get("reason") == sys.argv[3]
    and bool(payload.get("final_snapshot_id"))
)
raise SystemExit(0 if valid else 1)
PY
    then
      exit 0
    fi
    sleep 1
  done
  echo "timed out waiting for durable $expected_reason acknowledgement" >&2
  exit 75
fi

exit "$rc"
