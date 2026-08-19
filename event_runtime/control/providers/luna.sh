#!/usr/bin/env bash
# Harbor Codex + gpt-5.6-luna through pinned OpenRouter/OpenAI, durable Modal lane.
# Does NOT launch unless CONFIRM_LAUNCH=1.
#
# Usage:
#   CONFIRM_LAUNCH=1 event_runtime/control/providers/luna.sh
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL="${MODEL:-openai/gpt-5.6-luna}"
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"
OPENROUTER_PRESET="${OPENROUTER_PRESET:-@preset/sprint-gpt-5-6-luna-openai-standard}"
# Codex/Responses accepts max for Luna (Chat Completions does not). See REASONING_EFFORT_PROBE.md.
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# Same immutable harness pin as every competitor.
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
RUN_ID="${RUN_ID:-lane-luna-$(date -u +%Y%m%dT%H%M%SZ)}"

GOAL="$ROOT/event_runtime/control/templates/codex.j2"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export SPRINT_CODEX_LUNA_MODEL="$OPENROUTER_PRESET"

echo "launcher: providers/luna.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION  (Harbor --ak version=...)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT  (Codex/Responses max; see REASONING_EFFORT_PROBE.md)"
echo "endpoint: $ENDPOINT"
echo "provider: OpenRouter preset -> OpenAI standard only (no fallback)"
echo "preset:   $OPENROUTER_PRESET"
echo "goal:     $GOAL"
echo "profile:  $MODAL_PROFILE"
echo "auth:     OPENAI_API_KEY=[OpenRouter key] via durable env-file (not --ae)"

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
  --supervised-launch
  --run-id "$RUN_ID" \
  --agent-kind codex \
  --model "$MODEL" \
  --endpoint "$ENDPOINT" \
  --reasoning-effort "$REASONING_EFFORT" \
  --codex-version "$CODEX_VERSION"
)
LAUNCH_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LAUNCH_ARGS[@]}")
python3 "$ROOT/event_runtime/control/start_supervisor.py" \
  --run-id "$RUN_ID" \
  --batch-id "${SPRINT_BATCH_ID:-}" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env OPENAI_API_KEY \
  --max-restarts "${CPU_MAX_RESTARTS:-50}" \
  --min-backoff-s "${CPU_MIN_BACKOFF_S:-30}" \
  --max-backoff-s "${CPU_MAX_BACKOFF_S:-600}"
echo "supervisor unit=sprint-lane-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
