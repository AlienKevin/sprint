# shellcheck shell=bash
# Sourced by non-interactive Bash through BASH_ENV.
#
# Harbor installs the agent CLIs before this file becomes active. Once their
# real executables exist, route inference runs through the durable wrappers.

# Keepalive writes these so container-exec / agent shells share the run id
# even when Harbor --ae vars are only on the Codex process.
if [[ -z "${SPRINT_RUN_ID:-}" && -r /run/sprint-run-id ]]; then
  SPRINT_RUN_ID="$(tr -d '[:space:]' </run/sprint-run-id)"
  export SPRINT_RUN_ID
fi

SPRINT_REAL_CLAUDE=""
if [[ -x /usr/local/bin/claude ]]; then
  SPRINT_REAL_CLAUDE=/usr/local/bin/claude
else
  for candidate in "$HOME"/.local/bin/claude "$HOME"/.nvm/versions/node/*/bin/claude; do
    [[ -x "$candidate" ]] && SPRINT_REAL_CLAUDE=$candidate
  done
fi

if [[ "${SPRINT_AGENT_KIND:-}" == "claude-code" && -n "$SPRINT_REAL_CLAUDE" ]]; then
  export SPRINT_REAL_CLAUDE
  claude() {
    if [[ "${1:-}" == "--version" || "${1:-}" == "-v" ]]; then
      "$SPRINT_REAL_CLAUDE" "$@"
    else
      /opt/sprint-claude-code-exec-wrapper.sh "$SPRINT_REAL_CLAUDE" "$@"
    fi
  }
  export -f claude
fi
if [[ -z "${SPRINT_GPU_JOBS_ROOT:-}" && -r /run/sprint-gpu-jobs-root ]]; then
  SPRINT_GPU_JOBS_ROOT="$(tr -d '[:space:]' </run/sprint-gpu-jobs-root)"
  export SPRINT_GPU_JOBS_ROOT
elif [[ -z "${SPRINT_GPU_JOBS_ROOT:-}" && -n "${SPRINT_RUN_ID:-}" ]]; then
  export SPRINT_GPU_JOBS_ROOT="/durable/runs/${SPRINT_RUN_ID}/gpu-jobs"
fi

SPRINT_REAL_CODEX=""
if [[ -x /usr/local/bin/codex ]]; then
  SPRINT_REAL_CODEX=/usr/local/bin/codex
else
  for candidate in "$HOME"/.nvm/versions/node/*/bin/codex; do
    [[ -x "$candidate" ]] && SPRINT_REAL_CODEX=$candidate
  done
fi

if [[ "${SPRINT_AGENT_KIND:-}" == "codex" && -n "$SPRINT_REAL_CODEX" ]]; then
  export SPRINT_REAL_CODEX
  codex() {
    if [[ "${1:-}" == "exec" ]]; then
      /opt/sprint-codex-exec-wrapper.sh "$SPRINT_REAL_CODEX" "$@"
    else
      "$SPRINT_REAL_CODEX" "$@"
    fi
  }
  export -f codex
fi
