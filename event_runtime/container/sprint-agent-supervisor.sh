#!/usr/bin/env bash
set -uo pipefail

RUN_ID=""
AGENT_KIND=""
POLL_SECONDS=2
TERM_GRACE_SECONDS=90
DURABLE_DIR=/durable
RUNTIME_DIR=/run
AGENT_LOG_DIR=/logs/agent
ARTIFACT_LOG_DIR=/logs/artifacts
CODEX_HOME_DIR=/tmp/codex-home
TRACE_MIRROR_BIN=/opt/sprint-trace-mirror.py
BUDGET_WATCHDOG_BIN=/opt/sprint-budget-watchdog.py
CODEX_PATTERN=""
EXIT_AFTER_ACK=0
TRACE_MIRROR_PID=""

usage() {
  cat <<'EOF'
Usage: sprint-agent-supervisor.sh --run-id ID --agent-kind KIND [options]

Options:
  --agent-kind KIND        codex or deepseek-harness.
  --poll-seconds N         Watch interval (default: 2).
  --term-grace-seconds N   SIGINT grace period (default: 90).
  --budget-watchdog-bin PATH  In-sandbox budget watchdog executable.
EOF
}

while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --agent-kind) AGENT_KIND=${2:?}; shift 2 ;;
    --poll-seconds) POLL_SECONDS=${2:?}; shift 2 ;;
    --term-grace-seconds) TERM_GRACE_SECONDS=${2:?}; shift 2 ;;
    --durable-dir) DURABLE_DIR=${2:?}; shift 2 ;;
    --runtime-dir) RUNTIME_DIR=${2:?}; shift 2 ;;
    --agent-log-dir) AGENT_LOG_DIR=${2:?}; shift 2 ;;
    --artifact-log-dir) ARTIFACT_LOG_DIR=${2:?}; shift 2 ;;
    --codex-home-dir) CODEX_HOME_DIR=${2:?}; shift 2 ;;
    --budget-watchdog-bin) BUDGET_WATCHDOG_BIN=${2:?}; shift 2 ;;
    --codex-pattern) CODEX_PATTERN=${2:?}; shift 2 ;;
    --exit-after-ack) EXIT_AFTER_ACK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$ ]]; then
  echo "invalid or missing --run-id" >&2
  exit 2
fi
if [[ "$AGENT_KIND" != "codex" && "$AGENT_KIND" != "deepseek-harness" ]]; then
  echo "--agent-kind must be codex or deepseek-harness" >&2
  exit 2
fi
for value in "$POLL_SECONDS" "$TERM_GRACE_SECONDS"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "time values must be whole seconds" >&2
    exit 2
  fi
done

RUN_ROOT="$DURABLE_DIR/runs/$RUN_ID"
STATE_DIR="$RUN_ROOT/supervisor"
LOG_DIR="$STATE_DIR/logs"
HEARTBEAT="$STATE_DIR/heartbeat.json"
STOP_ACK="$RUN_ROOT/STOP_ACK"
STOP_FILE="$RUNTIME_DIR/sprint-stop"
STOP_SIGNALLED="$RUNTIME_DIR/sprint-stop-signalled"
if [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
  FIRST_SEEN="$STATE_DIR/first-deepseek-harness-seen"
else
  FIRST_SEEN="$STATE_DIR/first-codex-seen"
fi
AGENT_STATE_DIR="$RUNTIME_DIR/sprint-agent"
CODEX_PROCESS_FILE="$AGENT_STATE_DIR/codex-process"
HARNESS_PROCESS_FILE="$AGENT_STATE_DIR/agent-process"
EXPECTED_INTERRUPT="$AGENT_STATE_DIR/expected-interrupt"
AGENT_PID=""
AGENT_PGID=""
SAW_AGENT=0

mkdir -p "$LOG_DIR" "$RUNTIME_DIR" "$AGENT_STATE_DIR"
touch "$LOG_DIR/agent-supervisor.log"
chmod 0600 "$LOG_DIR/agent-supervisor.log" 2>/dev/null || true

now_iso() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  local line
  line="$(now_iso) $*"
  printf '%s\n' "$line" >>"$LOG_DIR/agent-supervisor.log"
  printf '%s\n' "$line"
}

atomic_text() {
  local path=$1 value=$2
  python3 - "$path" "$value" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(sys.argv[2])
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
}

write_heartbeat() {
  local state=$1
  python3 - "$HEARTBEAT" "$RUN_ID" "$AGENT_KIND" "$state" "$AGENT_PID" \
    "$AGENT_PGID" "$SAW_AGENT" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "run_id": sys.argv[2],
    "agent_kind": sys.argv[3],
    "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "state": sys.argv[4],
    "agent_pid": int(sys.argv[5]) if sys.argv[5].isdigit() else None,
    "agent_pgid": int(sys.argv[6]) if sys.argv[6].isdigit() else None,
    "agent_seen": sys.argv[7] == "1",
}
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
}

pid_running() {
  local pid=$1 state
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/stat" ]] || return 1
  state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
  [[ "$state" != "Z" && -n "$state" ]]
}

group_running() {
  local pgid=$1
  ps -eo pgid=,stat= | awk -v wanted="$pgid" \
    '$1 == wanted && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

pid_is_codex() {
  local pid=$1 row base
  local -a args=()
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  row=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  if [[ -n "$CODEX_PATTERN" ]]; then
    [[ "$row" =~ $CODEX_PATTERN ]]
    return
  fi
  mapfile -d '' -t args <"/proc/$pid/cmdline"
  ((${#args[@]} >= 2)) || return 1
  base=${args[0]##*/}
  if [[ "$base" == "codex" && "${args[1]}" == "exec" ]]; then
    return 0
  fi
  [[ "$base" == "node" || "$base" == "nodejs" ]] || return 1
  ((${#args[@]} >= 3)) || return 1
  [[ "${args[1]}" == */codex || "${args[1]}" == */codex.js ]] || return 1
  [[ "${args[2]}" == "exec" ]]
}

pid_is_deepseek_harness() {
  local pid=$1 row
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  row=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$row" == *sprint-deepseek-harness-runner.py* ]]
}

find_deepseek_harness_identity() {
  local pid pgid recorded_start actual_start actual_pgid watcher_pgid
  [[ -r "$HARNESS_PROCESS_FILE" ]] || return 1
  read -r pid pgid recorded_start <"$HARNESS_PROCESS_FILE" || return 1
  [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ && "$recorded_start" =~ ^[0-9]+$ ]] \
    || return 1
  ((pid > 1 && pgid > 1 && pid == pgid)) || return 1
  watcher_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')
  [[ "$pgid" != "$watcher_pgid" ]] || return 1
  if ! pid_running "$pid"; then
    group_running "$pgid" || return 1
    printf '%s %s\n' "$pid" "$pgid"
    return 0
  fi
  actual_start=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)
  [[ "$actual_start" == "$recorded_start" ]] || return 1
  actual_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
  [[ "$actual_pgid" == "$pgid" ]] || return 1
  pid_is_deepseek_harness "$pid" || return 1
  printf '%s %s\n' "$pid" "$pgid"
}

find_codex_identity() {
  local pid pgid recorded_start actual_start actual_pgid watcher_pgid
  [[ -r "$CODEX_PROCESS_FILE" ]] || return 1
  read -r pid pgid recorded_start <"$CODEX_PROCESS_FILE" || return 1
  [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ && "$recorded_start" =~ ^[0-9]+$ ]] \
    || return 1
  ((pid > 1 && pgid > 1 && pid == pgid)) || return 1
  watcher_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')
  [[ "$pgid" != "$watcher_pgid" ]] || return 1
  if ! pid_running "$pid"; then
    group_running "$pgid" || return 1
    printf '%s %s\n' "$pid" "$pgid"
    return 0
  fi
  actual_start=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)
  [[ "$actual_start" == "$recorded_start" ]] || return 1
  actual_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
  [[ "$actual_pgid" == "$pgid" ]] || return 1
  pid_is_codex "$pid" || return 1
  printf '%s %s\n' "$pid" "$pgid"
}

find_agent_identity() {
  if [[ "$AGENT_KIND" == "codex" ]]; then
    find_codex_identity
  else
    find_deepseek_harness_identity
  fi
}

write_ack() {
  local reason=$1
  python3 - "$STOP_ACK" "$RUN_ID" "$AGENT_KIND" "$reason" \
    "$AGENT_PID" "$AGENT_PGID" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "run_id": sys.argv[2],
    "agent_kind": sys.argv[3],
    "acknowledged_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "reason": sys.argv[4],
    "agent_pid": int(sys.argv[5]) if sys.argv[5].isdigit() else None,
    "agent_pgid": int(sys.argv[6]) if sys.argv[6].isdigit() else None,
}
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
  sync "$STOP_ACK" 2>/dev/null || true
}

stop_signal_watch() {
  local running
  [[ -n "${SIGNAL_WATCH_PID:-}" ]] || return
  for running in $(jobs -pr); do
    if [[ "$running" == "$SIGNAL_WATCH_PID" ]]; then
      kill "$SIGNAL_WATCH_PID" 2>/dev/null || true
    fi
  done
  wait "$SIGNAL_WATCH_PID" 2>/dev/null || true
  SIGNAL_WATCH_PID=""
}

finalize_and_ack() {
  local reason=$1
  stop_signal_watch
  stop_trace_mirror
  mirror_trace_once
  write_heartbeat finalizing
  write_ack "$reason"
  write_heartbeat stop_acknowledged
  log "STOP_ACK written reason=$reason"
  # The supervisor is Modal's sandbox keepalive.  Exiting it here tears down
  # the entire sandbox before Harbor can restore its network policy and copy
  # final artifacts.  Only explicitly supervised test/one-shot callers may
  # request that behavior; production Harbor trials keep the sandbox alive and
  # let Harbor own environment teardown after the agent command returns.
  if ((EXIT_AFTER_ACK)); then
    exit 0
  fi
  while true; do
    write_heartbeat stop_acknowledged
    sleep 30
  done
}

mirror_trace_once() {
  [[ -x "$TRACE_MIRROR_BIN" ]] || return 0
  "$TRACE_MIRROR_BIN" --once \
    --run-id "$RUN_ID" \
    --agent-kind "$AGENT_KIND" \
    --cpu-attempt 1 \
    --codex-home "$CODEX_HOME_DIR" \
    --agent-log-dir "$AGENT_LOG_DIR" \
    --durable-dir "$DURABLE_DIR" \
    >>"$LOG_DIR/trace-mirror.log" 2>&1 || \
    log "final trace mirror failed (non-fatal)"
}

start_trace_mirror() {
  [[ -x "$TRACE_MIRROR_BIN" ]] || return 0
  "$TRACE_MIRROR_BIN" \
    --run-id "$RUN_ID" \
    --agent-kind "$AGENT_KIND" \
    --cpu-attempt 1 \
    --codex-home "$CODEX_HOME_DIR" \
    --agent-log-dir "$AGENT_LOG_DIR" \
    --durable-dir "$DURABLE_DIR" \
    --interval-seconds "${SPRINT_TRACE_MIRROR_INTERVAL:-5}" \
    >>"$LOG_DIR/trace-mirror.log" 2>&1 &
  TRACE_MIRROR_PID=$!
}

stop_trace_mirror() {
  [[ -n "$TRACE_MIRROR_PID" ]] || return 0
  kill "$TRACE_MIRROR_PID" 2>/dev/null || true
  wait "$TRACE_MIRROR_PID" 2>/dev/null || true
  TRACE_MIRROR_PID=""
}

cleanup_watchers() {
  stop_signal_watch
  stop_trace_mirror
}

signal_watch() {
  local identity pid pgid deadline watcher_pgid stop_reason
  while true; do
    identity=$(find_agent_identity || true)
    read -r pid pgid <<<"$identity"
    if [[ -n "$pid" && ! -f "$FIRST_SEEN" ]]; then
      atomic_text "$FIRST_SEEN" "$(date +%s)"$'\n'
    fi
    if [[ -e "$STOP_FILE" && -n "$pid" && ! -e "$STOP_SIGNALLED" ]]; then
      stop_reason=$(tr -d '\r\n' <"$STOP_FILE" 2>/dev/null || true)
      stop_reason=${stop_reason:-operator_stop}
      atomic_text "$STOP_SIGNALLED" "$stop_reason"$'\n'
      atomic_text "$EXPECTED_INTERRUPT" "$stop_reason"$'\n'
      log "stopping $AGENT_KIND only reason=$stop_reason pid=$pid${pgid:+ pgid=$pgid}"
      kill -INT "$pid" 2>/dev/null || true
      deadline=$(($(date +%s) + TERM_GRACE_SECONDS))
      while group_running "$pgid" && (($(date +%s) < deadline)); do
        sleep 1
      done
      if group_running "$pgid"; then
        watcher_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')
        if [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 && "$pgid" != "$watcher_pgid" ]]; then
          log "$AGENT_KIND group still running after grace period; sending SIGTERM pgid=$pgid"
          kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
      fi
    while group_running "$pgid"; do
      sleep 1
    done
    return
    fi
    sleep 1
  done
}

if [[ -f "$FIRST_SEEN" ]]; then
  first_epoch=$(tr -dc '0-9' <"$FIRST_SEEN")
  [[ -n "$first_epoch" ]] && SAW_AGENT=1
fi

start_telemetry() {
  # Harbor convention mount: /logs/artifacts → trial artifacts/logs/artifacts.
  # The telemetry helper writes its live copy directly to the durable volume.
  local telemetry_bin=/opt/sprint-telemetry.sh
  local out_dir="$ARTIFACT_LOG_DIR/telemetry"
  [[ -x "$telemetry_bin" ]] || return 0
  mkdir -p "$out_dir" 2>/dev/null || true
  SPRINT_RUN_ID="$RUN_ID" "$telemetry_bin" \
    --role cpu-agent \
    --run-id "$RUN_ID" \
    --out-dir "$out_dir" \
    --durable-dir "$DURABLE_DIR" \
    --interval-seconds "${SPRINT_TELEMETRY_INTERVAL:-20}" \
    --pidfile "$RUNTIME_DIR/sprint-telemetry.pid" \
    >>"$LOG_DIR/telemetry-start.log" 2>&1 || \
    log "telemetry start failed (non-fatal)"
}

budget_watchdog_once() {
  [[ -x "$BUDGET_WATCHDOG_BIN" ]] || {
    log "budget watchdog missing; failing closed"
    atomic_text "$STOP_FILE" "budget_telemetry_unavailable"$'\n'
    return 20
  }
  timeout --signal=KILL 15 "$BUDGET_WATCHDOG_BIN" \
    --run-id "$RUN_ID" \
    --durable-dir "$DURABLE_DIR" \
    --runtime-dir "$RUNTIME_DIR" \
    --codex-home "$CODEX_HOME_DIR" \
    >>"$LOG_DIR/budget-watchdog.log" 2>&1
  local status=$?
  case "$status" in
    0|10|20) return "$status" ;;
    *)
      log "budget watchdog crashed status=$status; failing closed"
      atomic_text "$STOP_FILE" "budget_telemetry_unavailable"$'\n'
      return 20
      ;;
  esac
}

signal_watch &
SIGNAL_WATCH_PID=$!
trap cleanup_watchers EXIT

start_telemetry
start_trace_mirror
budget_watchdog_once || true

last_heartbeat_epoch=0
last_budget_epoch=0
write_heartbeat waiting_for_agent

while true; do
  now=$(date +%s)
  if ((SAW_AGENT == 0)) && [[ -f "$FIRST_SEEN" ]]; then
    first_epoch=$(tr -dc '0-9' <"$FIRST_SEEN")
    [[ -n "$first_epoch" ]] && SAW_AGENT=1
  fi
  found=$(find_agent_identity || true)

  if [[ -n "$found" ]]; then
    read -r AGENT_PID AGENT_PGID <<<"$found"
    if ((SAW_AGENT == 0)); then
      SAW_AGENT=1
      atomic_text "$FIRST_SEEN" "$now"$'\n'
      log "first $AGENT_KIND process observed pid=$AGENT_PID${AGENT_PGID:+ pgid=$AGENT_PGID}"
    fi
  elif ((SAW_AGENT)); then
    final_reason=$(tr -d '\r\n' <"$STOP_SIGNALLED" 2>/dev/null || true)
    finalize_and_ack "${final_reason:-agent_exit}"
  elif [[ -e "$STOP_FILE" ]]; then
    final_reason=$(tr -d '\r\n' <"$STOP_FILE" 2>/dev/null || true)
    finalize_and_ack "${final_reason:-budget_telemetry_unavailable}"
  fi

  # Check process identity before refreshing the in-sandbox ledger.  The
  # wrapper removes its authoritative process record as soon as the agent
  # exits, then drains and stops the local proxy.  Running the watchdog first
  # can observe that normal teardown window as "agent without proxy", create a
  # false fail-closed stop, and overwrite a clean agent_exit classification.
  if ((now - last_budget_epoch >= 5)); then
    budget_watchdog_once || true
    last_budget_epoch=$now
  fi

  if ((now - last_heartbeat_epoch >= 20)); then
    if ((SAW_AGENT)); then
      write_heartbeat running
    else
      write_heartbeat waiting_for_agent
    fi
    last_heartbeat_epoch=$now
  fi
  sleep "$POLL_SECONDS"
done
