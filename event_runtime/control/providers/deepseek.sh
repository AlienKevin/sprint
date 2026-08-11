#!/usr/bin/env bash
# Harbor Codex + deepseek-v4-flash (official DeepSeek Codex harness).
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
ENDPOINT="${ENDPOINT:-https://api.deepseek.com}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"  # API-max for DeepSeek Flash
# Same immutable harness pin as every competitor.
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
RUN_ID="${RUN_ID:-lane-deepseek-$(date -u +%Y%m%dT%H%M%SZ)}"

GOAL="$ROOT/event_runtime/control/templates/codex.j2"

if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY (or compatible OPENAI_API_KEY) is required" >&2
  exit 1
fi

echo "launcher: providers/deepseek.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION  (Harbor --ak version=...)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT  (DeepSeek Flash API-max; xhigh maps to high)"
echo "endpoint: $ENDPOINT"
echo "provider: deepseek (official Codex config; SPRINT_CODEX_PROVIDER=deepseek)"
echo "wire_api: responses (via [model_providers.deepseek]; not openai_base_url alone)"
echo "catalog:  DeepSeek models.json (1M context, auto_compact_token_limit=null)"
echo "goal:     $GOAL"
echo "profile:  $MODAL_PROFILE"
echo "auth:     OPENAI_API_KEY=[configured] via env-file (not --ae)"

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
