#!/usr/bin/env bash
# Launch (or one-shot) GPU/CPU telemetry for G1 sprint sandboxes.
# Default: daemonize into /logs/artifacts/telemetry so Harbor Hub collects it.
set -uo pipefail

ROLE=agent
RUN_ID=${QWOP_RUN_ID:-}
OUT_DIR=${QWOP_TELEMETRY_DIR:-/logs/artifacts/telemetry}
DURABLE_DIR=/durable
INTERVAL=${QWOP_TELEMETRY_INTERVAL:-20}
ONCE=0
FORCE=0
PIDFILE=/run/qwop-telemetry.pid
PYTHON_BIN=${QWOP_TELEMETRY_PYTHON:-python3}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PY=${QWOP_TELEMETRY_PY:-$SCRIPT_DIR/qwop-telemetry.py}

usage() {
  cat <<'EOF'
Usage: qwop-telemetry.sh [options]

Options:
  --role ROLE              agent|verifier|host (default: agent)
  --run-id ID              Durable run id (also mirrored under /durable)
  --out-dir PATH           Primary output dir (default: /logs/artifacts/telemetry)
  --durable-dir PATH       Durable volume root (default: /durable)
  --interval-seconds N     Sample interval (default: 20)
  --once                   Single sample then exit
  --pidfile PATH           Daemon pidfile (default: /run/qwop-telemetry.pid)
  --force                  Start even if another daemon holds the pidfile
  --daemon                 Explicit daemon mode (default unless --once)
EOF
}

while (($#)); do
  case "$1" in
    --role) ROLE=${2:?}; shift 2 ;;
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --out-dir) OUT_DIR=${2:?}; shift 2 ;;
    --durable-dir) DURABLE_DIR=${2:?}; shift 2 ;;
    --interval-seconds) INTERVAL=${2:?}; shift 2 ;;
    --pidfile) PIDFILE=${2:?}; shift 2 ;;
    --once) ONCE=1; shift ;;
    --force) FORCE=1; shift ;;
    --daemon) ONCE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$PY" ]]; then
  echo "missing telemetry python: $PY" >&2
  exit 1
fi
if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]] || ((INTERVAL < 5)); then
  echo "--interval-seconds must be an integer >= 5" >&2
  exit 2
fi

umask 077
mkdir -p "$OUT_DIR" "$(dirname "$PIDFILE")" 2>/dev/null || true

ARGS=(
  "$PY"
  --role "$ROLE"
  --out-dir "$OUT_DIR"
  --durable-dir "$DURABLE_DIR"
  --interval-seconds "$INTERVAL"
  --pidfile "$PIDFILE"
)
[[ -n "$RUN_ID" ]] && ARGS+=(--run-id "$RUN_ID")
((FORCE)) && ARGS+=(--force)

if ((ONCE)); then
  exec "$PYTHON_BIN" "${ARGS[@]}" --once
fi

# Idempotent background start. Parent returns immediately for keepalive use.
if [[ -f "$PIDFILE" ]]; then
  old=$(tr -dc '0-9' <"$PIDFILE" || true)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    if tr '\0' ' ' <"/proc/$old/cmdline" 2>/dev/null | grep -q qwop-telemetry; then
      echo "telemetry already running pid=$old"
      exit 0
    fi
  fi
fi

nohup "$PYTHON_BIN" "${ARGS[@]}" \
  >>"$OUT_DIR/telemetry-daemon.log" 2>&1 </dev/null &
echo $! >"$PIDFILE"
chmod 0600 "$PIDFILE" "$OUT_DIR/telemetry-daemon.log" 2>/dev/null || true
echo "telemetry started pid=$! out=$OUT_DIR"
exit 0
