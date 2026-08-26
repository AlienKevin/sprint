#!/usr/bin/env bash
# DeepSeek V4 Flash Vision Exp through the official DeepSeek OpenRouter route
# using DeepSeek's benchmark harness in its published minimal configuration.
# Does NOT launch unless CONFIRM_LAUNCH=1.
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL=deepseek/deepseek-v4-flash-vision-exp
ENDPOINT=https://openrouter.ai/api/v1
REASONING_EFFORT="${REASONING_EFFORT:-max}"
RUN_ID="${RUN_ID:-lane-deepseek-vision-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
      && "$SPRINT_OPENROUTER_PROVIDER_ENDPOINT" != "deepseek" ]]; then
  echo "Vision Exp is locked to the official DeepSeek OpenRouter endpoint" >&2
  exit 2
fi
if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" ]]; then
  echo "Vision Exp official endpoint does not expose a sealed quantization" >&2
  exit 2
fi
export SPRINT_OPENROUTER_PROVIDER_ENDPOINT=deepseek
export OPENROUTER_MODEL="$MODEL"
unset SPRINT_OPENROUTER_QUANTIZATION

LAUNCH_ARGS=(
  --run-id "$RUN_ID"
  --agent-kind deepseek-harness
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --reasoning-effort "$REASONING_EFFORT"
)

echo "launcher: providers/deepseek_harness.sh"
echo "run_id:   $RUN_ID"
echo "agent:    deepseek-harness (minimal benchmark mode)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT"
echo "endpoint: $ENDPOINT"
echo "provider: $SPRINT_OPENROUTER_PROVIDER_ENDPOINT (only; no fallback)"
echo "wire_api: chat/completions"
echo "auth:     OPENROUTER_API_KEY via env-file"
echo "--- dry-run ---"
"$ROOT/event_runtime/control/launch.sh" --dry-run "${LAUNCH_ARGS[@]}"

if [[ "${CONFIRM_LAUNCH:-}" != "1" ]]; then
  echo "dry-run only. Set CONFIRM_LAUNCH=1 to launch." >&2
  exit 0
fi

echo "--- launch ---"
LAUNCH_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
  "$ROOT/event_runtime/control/launch.sh" "${LAUNCH_ARGS[@]}")
python3 "$ROOT/event_runtime/control/start_trial.py" \
  --run-id "$RUN_ID" \
  --batch-id "${SPRINT_BATCH_ID:-}" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env OPENROUTER_API_KEY \
  --launch-env OPENROUTER_MODEL \
  --launch-env SPRINT_OPENROUTER_PROVIDER_ENDPOINT
echo "trial unit=sprint-trial-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
