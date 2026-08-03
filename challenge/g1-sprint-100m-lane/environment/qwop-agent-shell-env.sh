# Sourced by non-interactive Bash through BASH_ENV.
#
# Harbor installs Codex before this function becomes active. Once the real
# executable exists, route only `codex exec` through the durable stop wrapper.
QWOP_REAL_CODEX=""
if [[ -x /usr/local/bin/codex ]]; then
  QWOP_REAL_CODEX=/usr/local/bin/codex
else
  for candidate in "$HOME"/.nvm/versions/node/*/bin/codex; do
    [[ -x "$candidate" ]] && QWOP_REAL_CODEX=$candidate
  done
fi

if [[ "${QWOP_AGENT_KIND:-}" == "codex" && -n "$QWOP_REAL_CODEX" ]]; then
  export QWOP_REAL_CODEX
  codex() {
    if [[ "${1:-}" == "exec" ]]; then
      /opt/qwop-codex-exec-wrapper.sh "$QWOP_REAL_CODEX" "$@"
    else
      "$QWOP_REAL_CODEX" "$@"
    fi
  }
  export -f codex
fi
