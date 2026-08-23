#!/usr/bin/env bash
set -uo pipefail

RUN_ID=""
AGENT_KIND=""
SNAPSHOT_SECONDS=300
POLL_SECONDS=2
TERM_GRACE_SECONDS=90
DURABLE_DIR=/durable
RUNTIME_DIR=/run
APP_DIR=/app
AGENT_LOG_DIR=/logs/agent
ARTIFACT_LOG_DIR=/logs/artifacts
ROOT_DIR=/root
CODEX_HOME_DIR=/tmp/codex-home
PASSWORD_FILE=""
RESTIC_BIN=restic
TRACE_MIRROR_BIN=/opt/sprint-trace-mirror.py
BUDGET_WATCHDOG_BIN=/opt/sprint-budget-watchdog.py
CLAUDE_PATTERN=""
CODEX_PATTERN=""
EXIT_AFTER_ACK=0
TRACE_MIRROR_PID=""

usage() {
  cat <<'EOF'
Usage: sprint-snapshot-loop.sh --run-id ID --agent-kind KIND [options]

Options:
  --agent-kind KIND        claude-code, codex, or deepseek-harness.
  --password-file PATH     Restic password file on the durable volume.
  --snapshot-seconds N     Periodic snapshot interval (default: 300).
  --poll-seconds N         Watch interval (default: 2).
  --term-grace-seconds N   SIGINT grace period (default: 90).
  --budget-watchdog-bin PATH  In-sandbox budget watchdog executable.
EOF
}

while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --agent-kind) AGENT_KIND=${2:?}; shift 2 ;;
    --password-file) PASSWORD_FILE=${2:?}; shift 2 ;;
    --snapshot-seconds) SNAPSHOT_SECONDS=${2:?}; shift 2 ;;
    --poll-seconds) POLL_SECONDS=${2:?}; shift 2 ;;
    --term-grace-seconds) TERM_GRACE_SECONDS=${2:?}; shift 2 ;;
    --durable-dir) DURABLE_DIR=${2:?}; shift 2 ;;
    --runtime-dir) RUNTIME_DIR=${2:?}; shift 2 ;;
    --app-dir) APP_DIR=${2:?}; shift 2 ;;
    --agent-log-dir) AGENT_LOG_DIR=${2:?}; shift 2 ;;
    --artifact-log-dir) ARTIFACT_LOG_DIR=${2:?}; shift 2 ;;
    --root-dir) ROOT_DIR=${2:?}; shift 2 ;;
    --codex-home-dir) CODEX_HOME_DIR=${2:?}; shift 2 ;;
    --restic-bin) RESTIC_BIN=${2:?}; shift 2 ;;
    --budget-watchdog-bin) BUDGET_WATCHDOG_BIN=${2:?}; shift 2 ;;
    --claude-pattern) CLAUDE_PATTERN=${2:?}; shift 2 ;;
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
if [[ "$AGENT_KIND" != "claude-code" && "$AGENT_KIND" != "codex" \
      && "$AGENT_KIND" != "deepseek-harness" ]]; then
  echo "--agent-kind must be claude-code, codex, or deepseek-harness" >&2
  exit 2
fi
for value in "$SNAPSHOT_SECONDS" "$POLL_SECONDS" "$TERM_GRACE_SECONDS"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "time values must be whole seconds" >&2
    exit 2
  fi
done

RUN_ROOT="$DURABLE_DIR/runs/$RUN_ID"
REPO="$RUN_ROOT/restic"
STATE_DIR="$RUN_ROOT/snapshot"
LOG_DIR="$STATE_DIR/logs"
GEN_FILE="$STATE_DIR/generations.jsonl"
HEARTBEAT="$STATE_DIR/heartbeat.json"
STOP_ACK="$RUN_ROOT/STOP_ACK"
STOP_FILE="$RUNTIME_DIR/sprint-stop"
STOP_SIGNALLED="$RUNTIME_DIR/sprint-stop-signalled"
CPU_ATTEMPT="${SPRINT_CPU_LAUNCH_ATTEMPT:-1}"
if [[ ! "$CPU_ATTEMPT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPRINT_CPU_LAUNCH_ATTEMPT must be a positive integer" >&2
  exit 2
fi
ATTEMPT_TAG=$(printf '%03d' "$CPU_ATTEMPT")
if [[ "$AGENT_KIND" == "claude-code" ]]; then
  FIRST_SEEN="$STATE_DIR/first-claude-seen.attempt-$ATTEMPT_TAG"
elif [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
  FIRST_SEEN="$STATE_DIR/first-deepseek-harness-seen.attempt-$ATTEMPT_TAG"
else
  FIRST_SEEN="$STATE_DIR/first-codex-seen.attempt-$ATTEMPT_TAG"
fi
SNAPSHOT_LOCK="$STATE_DIR/snapshot.lock"
AGENT_STATE_DIR="$RUNTIME_DIR/sprint-agent"
CODEX_PROCESS_FILE="$AGENT_STATE_DIR/codex-process"
HARNESS_PROCESS_FILE="$AGENT_STATE_DIR/agent-process"
EXPECTED_INTERRUPT="$AGENT_STATE_DIR/expected-interrupt"
PASSWORD_FILE=${PASSWORD_FILE:-"$RUN_ROOT/secrets/restic-password"}
LAST_SNAPSHOT_ID=""
LAST_REASON=""
AGENT_PID=""
AGENT_PGID=""
SAW_AGENT=0

mkdir -p "$REPO" "$LOG_DIR" "$RUNTIME_DIR" "$AGENT_STATE_DIR"
touch "$LOG_DIR/snapshot-loop.log"
chmod 0600 "$LOG_DIR/snapshot-loop.log" 2>/dev/null || true

now_iso() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  local line
  line="$(now_iso) $*"
  printf '%s\n' "$line" >>"$LOG_DIR/snapshot-loop.log"
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
    "$AGENT_PGID" "$LAST_SNAPSHOT_ID" "$LAST_REASON" "$SAW_AGENT" <<'PY'
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
    "last_snapshot_id": sys.argv[7] or None,
    "last_snapshot_reason": sys.argv[8] or None,
    "agent_seen": sys.argv[9] == "1",
}
if payload["agent_kind"] == "claude-code":
    payload["claude_pid"] = payload["agent_pid"]
    payload["claude_seen"] = payload["agent_seen"]
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
}

find_claude_pid() {
  if [[ -n "$CLAUDE_PATTERN" ]]; then
    ps -eo pid=,stat=,comm=,args= | awk -v pattern="$CLAUDE_PATTERN" \
      -v self="$BASHPID" -v parent="$$" \
      '$1 != self && $1 != parent && $2 !~ /^Z/ && $3 != "awk" && $0 ~ pattern {print $1; exit}'
  else
    ps -eo pid=,stat=,comm=,args= | awk \
      '$2 !~ /^Z/ && ($3 == "claude" || ($3 == "node" && $0 ~ /[/]claude .*--output-format=stream-json/)) {print $1; exit}'
  fi
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

pid_is_claude() {
  local pid=$1 row
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  row=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  if [[ -n "$CLAUDE_PATTERN" ]]; then
    [[ "$row" =~ $CLAUDE_PATTERN ]]
  else
    [[ "$row" == *claude* && "$row" == *--output-format=stream-json* ]]
  fi
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
  local pid
  if [[ "$AGENT_KIND" == "claude-code" ]]; then
    pid=$(find_claude_pid || true)
    [[ -n "$pid" ]] && printf '%s\n' "$pid"
  elif [[ "$AGENT_KIND" == "codex" ]]; then
    find_codex_identity
  else
    find_deepseek_harness_identity
  fi
}

queue_fingerprint() {
  local submissions="$APP_DIR/submissions"
  if [[ ! -d "$submissions" ]]; then
    printf '%s\n' missing
    return
  fi
  find "$submissions" -maxdepth 2 -type f -printf '%P:%s:%T@\n' 2>/dev/null \
    | sort | sha256sum | awk '{print $1}'
}

ensure_repo() {
  [[ -s "$PASSWORD_FILE" ]] || {
    log "snapshot deferred: password file is unavailable"
    return 1
  }
  chmod 0600 "$PASSWORD_FILE" 2>/dev/null || true
  if [[ ! -f "$REPO/config" ]]; then
    "$RESTIC_BIN" -r "$REPO" --password-file "$PASSWORD_FILE" init \
      >>"$LOG_DIR/restic-init.log" 2>&1 || return 1
  fi
}

snapshot() {
  local reason=$1 output snapshot_id rc=0
  output=$(mktemp /tmp/sprint-restic.XXXXXX)
  (
    flock -x 9
    ensure_repo || exit 1
    local -a sources=()
    local path relative
    for path in "$APP_DIR" "$AGENT_LOG_DIR" "$ARTIFACT_LOG_DIR"; do
      [[ -e "$path" ]] && sources+=("$path")
    done
    if [[ "$AGENT_KIND" == "codex" ]]; then
      for relative in config.toml history.jsonl session_index.jsonl sessions \
        archived_sessions log logs app-server-control; do
        path="$CODEX_HOME_DIR/$relative"
        [[ -e "$path" && ! -L "$path" ]] && sources+=("$path")
      done
      for path in "$CODEX_HOME_DIR"/state_*.sqlite*; do
        [[ -f "$path" && ! -L "$path" ]] && sources+=("$path")
      done
    elif [[ "$AGENT_KIND" == "claude-code" && -e "$ROOT_DIR" ]]; then
      # Preserve the existing Claude Code recovery set. Codex uses an explicit
      # allowlist above so unrelated root credentials cannot enter its backup.
      sources+=("$ROOT_DIR")
    fi
    ((${#sources[@]})) || exit 1
    "$RESTIC_BIN" -r "$REPO" --password-file "$PASSWORD_FILE" backup \
      --json --tag "sprint:$RUN_ID" --tag "reason:$reason" \
      --exclude "$APP_DIR/.cache" \
      --exclude "$APP_DIR/**/__pycache__" \
      --exclude "$APP_DIR/**/.pytest_cache" \
      --exclude "$ROOT_DIR/.cache" \
      --exclude "$ROOT_DIR/.npm" \
      --exclude "$ROOT_DIR/.local/share/uv" \
      --exclude "$ROOT_DIR/.local/bin" \
      --exclude "$ROOT_DIR/.codex/auth.json" \
      --exclude "$CODEX_HOME_DIR/auth.json" \
      --exclude "$CODEX_HOME_DIR/cache" \
      --exclude "$CODEX_HOME_DIR/.cache" \
      --exclude "$CODEX_HOME_DIR/plugins/cache" \
      --exclude "$CODEX_HOME_DIR/models_cache.json" \
      --exclude "/tmp/codex-secrets" \
      "${sources[@]}" >"$output" 2>&1
  ) 9>"$SNAPSHOT_LOCK" || rc=$?

  if ((rc != 0)); then
    log "snapshot failed reason=$reason rc=$rc"
    if [[ -s "$output" ]]; then
      tr '\n' ' ' <"$output" | cut -c1-500 >>"$LOG_DIR/snapshot-errors.log"
      printf '\n' >>"$LOG_DIR/snapshot-errors.log"
    fi
    rm -f "$output"
    return 1
  fi

  snapshot_id=$(python3 - "$output" <<'PY'
import json
import sys

snapshot_id = ""
for raw in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    snapshot_id = event.get("snapshot_id") or snapshot_id
print(snapshot_id)
PY
)
  rm -f "$output"
  [[ -n "$snapshot_id" ]] || snapshot_id=unknown
  LAST_SNAPSHOT_ID=$snapshot_id
  LAST_REASON=$reason

  python3 - "$GEN_FILE" "$RUN_ID" "$AGENT_KIND" "$reason" "$snapshot_id" \
    "$AGENT_PID" "$AGENT_PGID" "$(queue_fingerprint)" <<'PY'
import datetime
import json
import os
import sys

path = sys.argv[1]
row = {
    "schema_version": 2,
    "run_id": sys.argv[2],
    "agent_kind": sys.argv[3],
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "reason": sys.argv[4],
    "snapshot_id": sys.argv[5],
    "agent_pid": int(sys.argv[6]) if sys.argv[6].isdigit() else None,
    "agent_pgid": int(sys.argv[7]) if sys.argv[7].isdigit() else None,
    "queue_fingerprint": sys.argv[8],
}
if row["agent_kind"] == "claude-code":
    row["claude_pid"] = row["agent_pid"]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
with os.fdopen(fd, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  log "snapshot complete reason=$reason id=$snapshot_id"
  return 0
}

write_ack() {
  local reason=$1
  python3 - "$STOP_ACK" "$RUN_ID" "$AGENT_KIND" "$reason" \
    "$LAST_SNAPSHOT_ID" "$AGENT_PID" "$AGENT_PGID" <<'PY'
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
    "final_snapshot_id": sys.argv[5],
    "agent_pid": int(sys.argv[6]) if sys.argv[6].isdigit() else None,
    "agent_pgid": int(sys.argv[7]) if sys.argv[7].isdigit() else None,
}
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
  sync "$STOP_ACK" "$GEN_FILE" 2>/dev/null || true
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

final_snapshot_and_ack() {
  local reason=$1
  stop_signal_watch
  stop_trace_mirror
  mirror_trace_once
  budget_watchdog_once || true
  write_heartbeat final_snapshot
  until snapshot "$reason"; do
    sleep 10
  done
  write_ack "$reason"
  write_heartbeat stop_acknowledged
  log "STOP_ACK written reason=$reason snapshot=$LAST_SNAPSHOT_ID"
  if ((EXIT_AFTER_ACK)) || [[ "$reason" == "agent_cost_budget_exhausted" ]] || \
    [[ "$reason" == "budget_telemetry_unavailable" ]]; then
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
    --cpu-attempt "${SPRINT_CPU_LAUNCH_ATTEMPT:-1}" \
    --codex-home "$CODEX_HOME_DIR" \
    --claude-home "$ROOT_DIR/.claude" \
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
    --cpu-attempt "${SPRINT_CPU_LAUNCH_ATTEMPT:-1}" \
    --codex-home "$CODEX_HOME_DIR" \
    --claude-home "$ROOT_DIR/.claude" \
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
      if [[ "$AGENT_KIND" != "claude-code" ]]; then
        atomic_text "$EXPECTED_INTERRUPT" "$stop_reason"$'\n'
      fi
      log "stopping $AGENT_KIND only reason=$stop_reason pid=$pid${pgid:+ pgid=$pgid}"
      kill -INT "$pid" 2>/dev/null || true
      deadline=$(($(date +%s) + TERM_GRACE_SECONDS))
      if [[ "$AGENT_KIND" != "claude-code" ]]; then
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
      else
        while pid_running "$pid" && (($(date +%s) < deadline)); do
          sleep 1
        done
        if pid_running "$pid" && pid_is_claude "$pid"; then
          log "Claude still running after grace period; sending SIGTERM pid=$pid"
          kill -TERM "$pid" 2>/dev/null || true
        fi
        while pid_running "$pid"; do
          sleep 1
        done
      fi
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
  # Restic already backs up ARTIFACT_LOG_DIR, so telemetry rides in snapshots.
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

last_snapshot_epoch=0
last_heartbeat_epoch=0
last_budget_epoch=0
last_queue=$(queue_fingerprint)
snapshot startup && last_snapshot_epoch=$(date +%s)
write_heartbeat waiting_for_agent

while true; do
  now=$(date +%s)
  if ((now - last_budget_epoch >= 5)); then
    budget_watchdog_once || true
    last_budget_epoch=$now
  fi
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
    final_snapshot_and_ack "${final_reason:-agent_exit}"
  elif [[ -e "$STOP_FILE" ]]; then
    final_reason=$(tr -d '\r\n' <"$STOP_FILE" 2>/dev/null || true)
    final_snapshot_and_ack "${final_reason:-budget_telemetry_unavailable}"
  fi

  queue=$(queue_fingerprint)
  if [[ "$queue" != "$last_queue" ]]; then
    if snapshot queue_change; then
      last_snapshot_epoch=$now
    fi
    last_queue=$queue
  elif ((now - last_snapshot_epoch >= SNAPSHOT_SECONDS)); then
    if snapshot periodic; then
      last_snapshot_epoch=$now
    fi
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
