#!/usr/bin/env bash
set -uo pipefail

REAL_CODEX=${1:?missing real Codex path}
shift

# Harbor also calls `codex --version` during setup. Only an agent run needs its
# own process group and stop-status handling.
if [[ "${1:-}" != "exec" ]]; then
  exec "$REAL_CODEX" "$@"
fi

RUNTIME_DIR=${SPRINT_RUNTIME_DIR:-/run}
AGENT_STATE_DIR="$RUNTIME_DIR/sprint-agent"
AGENT_LOG_DIR=${SPRINT_AGENT_LOG_DIR:-/logs/agent}
CODEX_HOME_DIR=${CODEX_HOME:-/tmp/codex-home}
PROCESS_FILE="$AGENT_STATE_DIR/codex-process"
EXPECTED_INTERRUPT="$AGENT_STATE_DIR/expected-interrupt"
DURABLE_DIR=${SPRINT_DURABLE_DIR:-/durable}
RUN_ID=${SPRINT_RUN_ID:-}
STOP_ACK_TIMEOUT_SECONDS=${SPRINT_STOP_ACK_TIMEOUT_SECONDS:-600}
OPENROUTER_PROXY_DRAIN_TIMEOUT_SECONDS=${SPRINT_OPENROUTER_PROXY_DRAIN_TIMEOUT_SECONDS:-900}
OPENROUTER_PROXY_RECOVERY_TIMEOUT_SECONDS=${SPRINT_OPENROUTER_PROXY_RECOVERY_TIMEOUT_SECONDS:-300}
CODEX_EXECUTABLE=${SPRINT_CODEX_EXECUTABLE:-}
OPENROUTER_PROXY_BIN=${SPRINT_OPENROUTER_PROXY_BIN:-/opt/sprint-openrouter-ledger-proxy.py}
OPENROUTER_PROXY_BASE_URL=${SPRINT_OPENROUTER_PROXY_BASE_URL:-http://127.0.0.1:18080/api/v1}
OPENROUTER_UPSTREAM_URL=${SPRINT_OPENROUTER_UPSTREAM_URL:-https://openrouter.ai/api/v1}
OPENROUTER_PROVIDER_ENDPOINT=${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}
OPENROUTER_QUANTIZATION=${SPRINT_OPENROUTER_QUANTIZATION:-}
OPENROUTER_REQUEST_CONTRACT_JSON=${SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON:-}
OPENROUTER_ALLOWED_INFERENCE_PATH=${SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH:-responses}

if [[ ! "$STOP_ACK_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPRINT_STOP_ACK_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
for timeout_name in OPENROUTER_PROXY_DRAIN_TIMEOUT_SECONDS OPENROUTER_PROXY_RECOVERY_TIMEOUT_SECONDS; do
  if [[ ! "${!timeout_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$timeout_name must be a positive integer" >&2
    exit 2
  fi
done

umask 077
mkdir -p "$AGENT_STATE_DIR" "$AGENT_LOG_DIR" "$CODEX_HOME_DIR"
rm -f "$EXPECTED_INTERRUPT"

proxy_pid=""
proxy_drain_on_exit=1
fail_closed_openrouter_recovery() {
  python3 - "$DURABLE_DIR" "$RUN_ID" "$RUNTIME_DIR" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import sys

durable_dir, run_id, runtime_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
payload = {
    "schema_version": 2,
    "run_id": run_id,
    "reason": "budget_telemetry_unavailable",
    "status": "fail_closed",
    "detail": "OpenRouter request charge did not reconcile before proxy shutdown",
    "requested_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
marker = durable_dir / "runs" / run_id / "BUDGET_STOP_REQUESTED.json"
items = []
if not marker.exists():
    items.append((marker, json.dumps(payload, indent=2, sort_keys=True) + "\n"))
items.append((runtime_dir / "sprint-stop", "budget_telemetry_unavailable\n"))
for path, content in items:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
PY
}

stop_openrouter_proxy() {
  [[ -n "$proxy_pid" ]] || return 0
  if ((proxy_drain_on_exit)); then
    local deadline=$((SECONDS + OPENROUTER_PROXY_DRAIN_TIMEOUT_SECONDS))
    local status=1
    while kill -0 "$proxy_pid" 2>/dev/null && ((SECONDS < deadline)); do
      python3 - "$OPENROUTER_PROXY_BASE_URL" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request
from urllib.parse import urlsplit

base = urlsplit(sys.argv[1])
url = f"{base.scheme}://{base.netloc}/ledger-status"
with urllib.request.urlopen(url, timeout=7) as response:
    payload = json.load(response)
raise SystemExit(0 if payload.get("pending_request_count") == 0 else 2)
PY
      status=$?
      ((status == 0)) && break
      sleep 0.25
    done
    if ((status != 0)) && kill -0 "$proxy_pid" 2>/dev/null; then
      fail_closed_openrouter_recovery
    fi
  fi
  kill "$proxy_pid" 2>/dev/null || true
  wait "$proxy_pid" 2>/dev/null || true
  rm -f "$AGENT_STATE_DIR/openrouter-proxy.pid"
  proxy_pid=""
}
trap stop_openrouter_proxy EXIT

start_openrouter_proxy() {
  [[ -n "$RUN_ID" ]] || {
    echo "OpenRouter ledger proxy requires SPRINT_RUN_ID" >&2
    exit 1
  }
  [[ -x "$OPENROUTER_PROXY_BIN" ]] || {
    echo "OpenRouter ledger proxy missing: $OPENROUTER_PROXY_BIN" >&2
    exit 1
  }
  local upstream_api_key=${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}
  [[ ${#upstream_api_key} -ge 16 ]] || {
    echo "OpenRouter ledger proxy requires a sealed upstream key" >&2
    exit 1
  }
  local ledger_root="$DURABLE_DIR/runs/$RUN_ID/api-usage"
  local attempt=${SPRINT_CPU_LAUNCH_ATTEMPT:-1}
  local -a route_args=()
  if [[ -n "$OPENROUTER_PROVIDER_ENDPOINT" ]]; then
    route_args+=(--provider-endpoint "$OPENROUTER_PROVIDER_ENDPOINT")
  fi
  if [[ -n "$OPENROUTER_QUANTIZATION" ]]; then
    route_args+=(--quantization "$OPENROUTER_QUANTIZATION")
  fi
  if [[ -n "$OPENROUTER_REQUEST_CONTRACT_JSON" ]]; then
    route_args+=(--request-contract-json "$OPENROUTER_REQUEST_CONTRACT_JSON")
  fi
  route_args+=(--allowed-inference-path "$OPENROUTER_ALLOWED_INFERENCE_PATH")
  "$OPENROUTER_PROXY_BIN" \
    --upstream "$OPENROUTER_UPSTREAM_URL" \
    --ledger-root "$ledger_root" \
    --run-id "$RUN_ID" \
    --cpu-attempt "$attempt" \
    --runtime-dir "$RUNTIME_DIR" \
    --upstream-api-key-stdin \
    "${route_args[@]}" \
    >>"$AGENT_LOG_DIR/openrouter-ledger-proxy.log" 2>&1 \
    <<<"$upstream_api_key" &
  proxy_pid=$!
  upstream_api_key=""
  unset OPENROUTER_API_KEY
  # Codex requires an API-key-shaped value, but the trusted proxy ignores it.
  # This token has no value at OpenRouter and cannot be used for direct calls.
  export OPENAI_API_KEY=sprint-local-proxy-token
  printf '%s\n' "$proxy_pid" >"$AGENT_STATE_DIR/openrouter-proxy.pid"
  local ready=0
  # A legacy run may need one bounded ledger migration before the proxy can
  # serve health checks. New rollup-backed restarts are constant-time.
  local recovery_deadline=$((SECONDS + OPENROUTER_PROXY_RECOVERY_TIMEOUT_SECONDS))
  while ((SECONDS < recovery_deadline)); do
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
      break
    fi
    if python3 - "$OPENROUTER_PROXY_BASE_URL" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
from urllib.parse import urlsplit

base = urlsplit(sys.argv[1])
url = f"{base.scheme}://{base.netloc}/healthz"
with urllib.request.urlopen(url, timeout=7) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    then
      ready=1
      break
    fi
    sleep 1
  done
  if ((ready == 0)); then
    proxy_drain_on_exit=0
    fail_closed_openrouter_recovery
    echo "OpenRouter ledger proxy failed to become ready" >&2
    exit 1
  fi
  export SPRINT_CODEX_DEEPSEEK_BASE_URL="$OPENROUTER_PROXY_BASE_URL"
  export SPRINT_CODEX_LUNA_BASE_URL="$OPENROUTER_PROXY_BASE_URL"
  export SPRINT_CODEX_OPENAI_BASE_URL="$OPENROUTER_PROXY_BASE_URL"
  export OPENAI_BASE_URL="$OPENROUTER_PROXY_BASE_URL"
}

if [[ "${SPRINT_OPENROUTER_LEDGER_REQUIRED:-0}" == "1" ]]; then
  start_openrouter_proxy
fi

# Install a static model catalog for every controlled comparison model.
# DeepSeek needs its custom provider definition; OpenAI models use their exact
# Codex 0.149.1 entries so backend refreshes cannot change tool or collaboration
# semantics during the experiment.
want_deepseek=0
want_openai=0
case "${SPRINT_CODEX_PROVIDER:-}" in
  deepseek|DeepSeek|DEEPSEEK) want_deepseek=1 ;;
esac
case "${OPENAI_BASE_URL:-}" in
  *api.deepseek.com*) want_deepseek=1 ;;
esac
case "${SPRINT_MODEL:-}" in
  */gpt-5.6-luna|gpt-5.6-luna|*/gpt-5.6-sol|gpt-5.6-sol) want_openai=1 ;;
esac
previous=
for argument in "$@"; do
  if [[ "$previous" == "--model" || "$previous" == "-m" ]]; then
    [[ "$argument" == "gpt-5.6-luna" || "$argument" == "gpt-5.6-sol" ]] && want_openai=1
    previous=
    continue
  fi
  case "$argument" in
    --model=gpt-5.6-luna|--model=gpt-5.6-sol) want_openai=1 ;;
    --model|-m) previous=$argument ;;
  esac
done
if ((want_deepseek)); then
  if [[ -n "${SPRINT_CODEX_DEEPSEEK_MODEL:-}" ]]; then
    rewritten=()
    replace_next_model=0
    for argument in "$@"; do
      if ((replace_next_model)); then
        rewritten+=("$SPRINT_CODEX_DEEPSEEK_MODEL")
        replace_next_model=0
        continue
      fi
      case "$argument" in
        --model|-m)
          rewritten+=("$argument")
          replace_next_model=1
          ;;
        --model=*) rewritten+=("--model=$SPRINT_CODEX_DEEPSEEK_MODEL") ;;
        *) rewritten+=("$argument") ;;
      esac
    done
    set -- "${rewritten[@]}"
  fi
  apply=${SPRINT_APPLY_DEEPSEEK_CODEX_CONFIG:-/opt/sprint-apply-deepseek-codex-config.sh}
  if [[ ! -x "$apply" && -f "$apply" ]]; then
    chmod +x "$apply" 2>/dev/null || true
  fi
  if [[ -f "$apply" ]]; then
    # shellcheck disable=SC1090
    bash "$apply" || {
      echo "failed to apply DeepSeek Codex config via $apply" >&2
      exit 1
    }
  else
    echo "DeepSeek Codex apply script missing: $apply" >&2
    exit 1
  fi
elif ((want_openai)); then
  # Harbor selects the public model id on the command line.  The pinned Codex
  # catalog deliberately stores the same contract under a private preset slug,
  # so rewrite only the local CLI selector.  The trusted OpenRouter proxy still
  # overwrites and verifies the actual upstream model independently.
  if [[ -n "${SPRINT_CODEX_OPENAI_MODEL:-}" ]]; then
    rewritten=()
    replace_next_model=0
    for argument in "$@"; do
      if ((replace_next_model)); then
        rewritten+=("$SPRINT_CODEX_OPENAI_MODEL")
        replace_next_model=0
        continue
      fi
      case "$argument" in
        --model|-m)
          rewritten+=("$argument")
          replace_next_model=1
          ;;
        --model=*) rewritten+=("--model=$SPRINT_CODEX_OPENAI_MODEL") ;;
        *) rewritten+=("$argument") ;;
      esac
    done
    set -- "${rewritten[@]}"
  fi
  apply=${SPRINT_APPLY_OPENAI_CODEX_CONFIG:-/opt/sprint-apply-openai-codex-config.sh}
  if [[ ! -x "$apply" && -f "$apply" ]]; then
    chmod +x "$apply" 2>/dev/null || true
  fi
  if [[ -f "$apply" ]]; then
    bash "$apply" || {
      echo "failed to apply pinned OpenAI Codex config via $apply" >&2
      exit 1
    }
  else
    echo "pinned OpenAI Codex apply script missing: $apply" >&2
    exit 1
  fi
fi

group_has_live_processes() {
  local pgid=$1
  ps -eo pgid=,stat= | awk -v wanted="$pgid" \
    '$1 == wanted && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

copy_codex_state() {
  local destination="$AGENT_LOG_DIR/codex-state"
  local staging="$AGENT_LOG_DIR/.codex-state.$$"
  local path relative
  local -a fixed=(
    config.toml
    models.json
    history.jsonl
    session_index.jsonl
    sessions
    archived_sessions
    log
    logs
    app-server-control
  )

  rm -rf "$staging"
  mkdir -p "$staging"
  for relative in "${fixed[@]}"; do
    path="$CODEX_HOME_DIR/$relative"
    [[ -e "$path" && ! -L "$path" ]] || continue
    cp -a -- "$path" "$staging/$relative"
  done
  for path in "$CODEX_HOME_DIR"/state_*.sqlite*; do
    [[ -f "$path" && ! -L "$path" ]] || continue
    cp -a -- "$path" "$staging/"
  done

  # Never copy auth.json, the secrets directory, model/plugin caches, or other
  # unlisted Codex files. The allowlist above holds the resumable trace state.
  rm -rf "$destination"
  mv "$staging" "$destination"
}

if [[ -z "$CODEX_EXECUTABLE" ]]; then
  launcher=$(readlink -f "$REAL_CODEX")
  package_root=$(dirname "$(dirname "$launcher")")
  native_candidates=()
  for candidate in \
    "$package_root"/node_modules/@openai/codex-*/vendor/*/bin/codex \
    "$package_root"/vendor/*/bin/codex; do
    [[ -x "$candidate" ]] && native_candidates+=("$candidate")
  done
  if ((${#native_candidates[@]} != 1)); then
    echo "could not resolve one native Codex executable from $REAL_CODEX" >&2
    exit 1
  fi
  CODEX_EXECUTABLE=${native_candidates[0]}
  export CODEX_MANAGED_PACKAGE_ROOT="$package_root"
  export CODEX_MANAGED_BY_NPM=1
  unset CODEX_MANAGED_BY_BUN CODEX_MANAGED_BY_PNPM
fi

# Run the native CLI as the group leader. This retains npm's package metadata
# while avoiding an extra signal-relay process between the watcher and Codex.
setsid "$CODEX_EXECUTABLE" "$@" &
codex_pid=$!

# The shell can resume before the new child has executed setsid(2).  Reading
# its PGID immediately can therefore observe the parent's process group and
# falsely classify a healthy launch as unsafe.  Wait briefly for the child to
# become its own group leader, while still failing closed if it exits first.
codex_pgid=
start_time=
for _ in {1..100}; do
  [[ -r "/proc/$codex_pid/stat" ]] || break
  codex_pgid=$(ps -o pgid= -p "$codex_pid" | tr -d '[:space:]')
  start_time=$(awk '{print $22}' "/proc/$codex_pid/stat" 2>/dev/null || true)
  if [[ "$codex_pgid" == "$codex_pid" && -n "$start_time" ]]; then
    break
  fi
  sleep 0.01
done

if [[ -z "$codex_pgid" || "$codex_pgid" != "$codex_pid" || -z "$start_time" ]]; then
  kill -TERM "$codex_pid" 2>/dev/null || true
  wait "$codex_pid" 2>/dev/null || true
  echo "failed to create an isolated Codex process group" >&2
  exit 1
fi

process_tmp="$PROCESS_FILE.$$"
printf '%s %s %s\n' "$codex_pid" "$codex_pgid" "$start_time" >"$process_tmp"
mv "$process_tmp" "$PROCESS_FILE"

wait "$codex_pid"
rc=$?

# After a requested interrupt, retain metadata while the watcher checks and,
# if needed, terminates any child left in the isolated group.
if [[ -f "$EXPECTED_INTERRUPT" ]]; then
  deadline=$((SECONDS + 120))
  while group_has_live_processes "$codex_pgid" && ((SECONDS < deadline)); do
    sleep 1
  done
fi

copy_codex_state
rm -f "$PROCESS_FILE"

if [[ -f "$EXPECTED_INTERRUPT" ]]; then
  if [[ -z "$RUN_ID" ]]; then
    echo "requested interrupt is missing SPRINT_RUN_ID" >&2
    exit 75
  fi
  stop_ack="$DURABLE_DIR/runs/$RUN_ID/STOP_ACK"
  expected_reason=$(tr -d '\r\n' <"$EXPECTED_INTERRUPT" 2>/dev/null || true)
  expected_reason=${expected_reason:-operator_stop}
  deadline=$((SECONDS + STOP_ACK_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if python3 - "$stop_ack" "$RUN_ID" "$expected_reason" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
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
      # Codex can report exit 1 when SIGINT invalidates an in-flight
      # unified_exec process. The trusted interrupt marker plus a checksummed
      # durable STOP_ACK prove this was the requested stop, not an agent crash.
      exit 0
    fi
    sleep 1
  done
  echo "timed out waiting for durable $expected_reason acknowledgement" >&2
  exit 75
fi
exit "$rc"
