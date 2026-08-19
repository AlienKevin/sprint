#!/usr/bin/env bash
# Harbor Codex + a DeepSeek model through a sealed OpenRouter endpoint.
# Does NOT launch unless CONFIRM_LAUNCH=1.
#
# Official docs: https://api-docs.deepseek.com/quick_start/agent_integrations/codex/
# Harbor still passes --endpoint (writes openai_base_url); the sandbox wrapper
# then rewrites CODEX_HOME to model_provider=deepseek + wire_api=responses +
# DeepSeek models.json (1M context).
#
# Usage:
#   CONFIRM_LAUNCH=1 event_runtime/control/providers/deepseek.sh
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"
OPENROUTER_PRESET="${OPENROUTER_PRESET:-@preset/sprint-deepseek-v4-flash-0731-official}"
if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" ]]; then
  OPENROUTER_MODEL="${OPENROUTER_MODEL:-$MODEL}"
else
  OPENROUTER_MODEL="${OPENROUTER_MODEL:-$OPENROUTER_PRESET}"
fi
REASONING_EFFORT="${REASONING_EFFORT:-max}"  # API-max for DeepSeek Flash
# Same immutable harness pin as every competitor.
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
RUN_ID="${RUN_ID:-lane-deepseek-$(date -u +%Y%m%dT%H%M%SZ)}"

GOAL="$ROOT/event_runtime/control/templates/codex.j2"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export SPRINT_CODEX_DEEPSEEK_BASE_URL="$ENDPOINT"
export SPRINT_CODEX_DEEPSEEK_MODEL="$OPENROUTER_MODEL"
if [[ "$MODEL" == "deepseek/deepseek-v4-flash" \
      && -z "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
      && -z "${SPRINT_DEEPSEEK_PRICING_SNAPSHOT:-}" ]]; then
  SPRINT_DEEPSEEK_PRICING_SNAPSHOT=$(PYTHONPATH="$ROOT" python3 - \
    "$OPENROUTER_API_KEY" <<'PY'
import json
import sys

from event_runtime.control.deepseek_pricing import fetch_openrouter_snapshot

print(
    json.dumps(
        fetch_openrouter_snapshot(sys.argv[1]),
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
  )
  export SPRINT_DEEPSEEK_PRICING_SNAPSHOT
fi

echo "launcher: providers/deepseek.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION  (Harbor --ak version=...)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT"
echo "endpoint: $ENDPOINT"
echo "wire:     $OPENROUTER_MODEL"
echo "provider: ${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-OpenRouter controlled preset} (no fallback)"
echo "quant:    ${SPRINT_OPENROUTER_QUANTIZATION:-provider default}"
echo "wire_api: responses (via [model_providers.deepseek]; not openai_base_url alone)"
echo "catalog:  DeepSeek models.json (${SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW:-1048576} context, auto_compact_token_limit=null)"
echo "goal:     $GOAL"
echo "profile:  $MODAL_PROFILE"
echo "auth:     OPENAI_API_KEY=[OpenRouter key] via env-file (not --ae)"

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
SUPERVISOR_ARGS=(
  "$ROOT/event_runtime/control/start_supervisor.py"
  --run-id "$RUN_ID" \
  --batch-id "${SPRINT_BATCH_ID:-}" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env OPENAI_API_KEY \
  --max-restarts "${CPU_MAX_RESTARTS:-50}" \
  --min-backoff-s "${CPU_MIN_BACKOFF_S:-30}" \
  --max-backoff-s "${CPU_MAX_BACKOFF_S:-600}"
)
if [[ -n "${SPRINT_DEEPSEEK_PRICING_SNAPSHOT:-}" ]]; then
  SUPERVISOR_ARGS+=(--secret-env SPRINT_DEEPSEEK_PRICING_SNAPSHOT)
fi
python3 "${SUPERVISOR_ARGS[@]}"
echo "supervisor unit=sprint-lane-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
