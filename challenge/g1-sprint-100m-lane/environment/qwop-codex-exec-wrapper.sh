#!/usr/bin/env bash
set -uo pipefail

REAL_CODEX=${1:?missing real Codex path}
shift

# Harbor also calls `codex --version` during setup. Only an agent run needs its
# own process group and stop-status handling.
if [[ "${1:-}" != "exec" ]]; then
  exec "$REAL_CODEX" "$@"
fi

RUNTIME_DIR=${QWOP_RUNTIME_DIR:-/run}
AGENT_STATE_DIR="$RUNTIME_DIR/qwop-agent"
AGENT_LOG_DIR=${QWOP_AGENT_LOG_DIR:-/logs/agent}
CODEX_HOME_DIR=${CODEX_HOME:-/tmp/codex-home}
PROCESS_FILE="$AGENT_STATE_DIR/codex-process"
EXPECTED_INTERRUPT="$AGENT_STATE_DIR/expected-interrupt"
CODEX_EXECUTABLE=${QWOP_CODEX_EXECUTABLE:-}

umask 077
mkdir -p "$AGENT_STATE_DIR" "$AGENT_LOG_DIR"
rm -f "$EXPECTED_INTERRUPT"

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
codex_pgid=$(ps -o pgid= -p "$codex_pid" | tr -d '[:space:]')
start_time=$(awk '{print $22}' "/proc/$codex_pid/stat" 2>/dev/null || true)

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

if [[ -f "$EXPECTED_INTERRUPT" && ( "$rc" == "130" || "$rc" == "143" ) ]]; then
  exit 0
fi
exit "$rc"
