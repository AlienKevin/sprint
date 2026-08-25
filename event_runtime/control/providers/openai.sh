#!/usr/bin/env bash
# Harbor Codex through a pinned OpenRouter/OpenAI preset and durable Modal lane.
# Does NOT launch unless CONFIRM_LAUNCH=1.
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL="${MODEL:?MODEL is required}"
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_VERSION="${CODEX_VERSION:-0.149.1}"
RUN_ID="${RUN_ID:-lane-openai-$(date -u +%Y%m%dT%H%M%SZ)}"
GOAL="$ROOT/event_runtime/control/templates/codex.j2"

case "${MODEL#*/}" in
  gpt-5.6-luna)
    OPENROUTER_PRESET="${OPENROUTER_PRESET:-@preset/sprint-gpt-5-6-luna-openai-standard}"
    ;;
  gpt-5.6-sol)
    OPENROUTER_PRESET="${OPENROUTER_PRESET:-@preset/sprint-gpt-5-6-sol-openai-standard}"
    ;;
  *)
    echo "unsupported controlled OpenAI model: $MODEL" >&2
    exit 2
    ;;
esac

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
export SPRINT_CODEX_OPENAI_MODEL="$OPENROUTER_PRESET"
export SPRINT_OPENROUTER_PRESET="$OPENROUTER_PRESET"

echo "launcher: providers/openai.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT"
echo "endpoint: $ENDPOINT"
echo "provider: OpenRouter preset -> OpenAI standard only (no fallback)"
echo "preset:   $OPENROUTER_PRESET"
echo "goal:     $GOAL"
echo "profile:  $MODAL_PROFILE"
echo "auth:     OPENROUTER_API_KEY via env-file"

DRY_ARGS=(
  --dry-run
  --run-id "$RUN_ID"
  --agent-kind codex
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --reasoning-effort "$REASONING_EFFORT"
  --codex-version "$CODEX_VERSION"
)

echo "--- dry-run ---"
"$ROOT/event_runtime/control/launch.sh" "${DRY_ARGS[@]}"

if [[ "${CONFIRM_LAUNCH:-}" != "1" ]]; then
  echo "dry-run only. Set CONFIRM_LAUNCH=1 to launch." >&2
  exit 0
fi

echo "--- launch ---"
LAUNCH_ARGS=(
  "$ROOT/event_runtime/control/launch.sh"
  --run-id "$RUN_ID"
  --agent-kind codex
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --reasoning-effort "$REASONING_EFFORT"
  --codex-version "$CODEX_VERSION"
)
LAUNCH_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LAUNCH_ARGS[@]}")
python3 "$ROOT/event_runtime/control/start_trial.py" \
  --run-id "$RUN_ID" \
  --batch-id "${SPRINT_BATCH_ID:-}" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env OPENROUTER_API_KEY
echo "trial unit=sprint-trial-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
