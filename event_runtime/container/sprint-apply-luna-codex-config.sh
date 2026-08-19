#!/usr/bin/env bash
# Compatibility entrypoint; all pinned OpenAI Codex models share one installer.
set -euo pipefail

export SPRINT_CODEX_OPENAI_MODEL_LOCK="${SPRINT_CODEX_OPENAI_MODEL_LOCK:-${SPRINT_CODEX_LUNA_MODEL_LOCK:-/opt/sprint-codex-luna-model-lock.json}}"
export SPRINT_CODEX_OPENAI_MODEL_ID="${SPRINT_CODEX_OPENAI_MODEL_ID:-gpt-5.6-luna}"
export SPRINT_CODEX_OPENAI_MODEL="${SPRINT_CODEX_OPENAI_MODEL:-${SPRINT_CODEX_LUNA_MODEL:-@preset/sprint-gpt-5-6-luna-openai-standard}}"
export SPRINT_CODEX_OPENAI_BASE_URL="${SPRINT_CODEX_OPENAI_BASE_URL:-${SPRINT_CODEX_LUNA_BASE_URL:-http://127.0.0.1:18080/api/v1}}"
exec bash "${SPRINT_APPLY_OPENAI_CODEX_CONFIG:-/opt/sprint-apply-openai-codex-config.sh}"
