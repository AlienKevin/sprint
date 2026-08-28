#!/usr/bin/env bash
# GLM-5.3-Flash at max effort through the latest pinned Claude Code harness,
# using OpenRouter's Anthropic skin and Z.AI's first-party FP8 route only.
# Does NOT launch unless CONFIRM_LAUNCH=1.
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL=z-ai/glm-5.3-flash
ENDPOINT=https://openrouter.ai/api
REASONING_EFFORT=max
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.248}"
RUN_ID="${RUN_ID:-lane-glm-5-3-flash-max-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
      && "$SPRINT_OPENROUTER_PROVIDER_ENDPOINT" != "z-ai/fp8" ]]; then
  echo "GLM-5.3-Flash is locked to the official Z.AI OpenRouter endpoint" >&2
  exit 2
fi
if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" \
      && "$SPRINT_OPENROUTER_QUANTIZATION" != "fp8" ]]; then
  echo "GLM-5.3-Flash is locked to the official Z.AI FP8 endpoint" >&2
  exit 2
fi
export SPRINT_OPENROUTER_PROVIDER_ENDPOINT=z-ai/fp8
export SPRINT_OPENROUTER_QUANTIZATION=fp8

LAUNCH_ARGS=(
  --run-id "$RUN_ID"
  --agent-kind claude-code
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --reasoning-effort "$REASONING_EFFORT"
  --claude-code-version "$CLAUDE_CODE_VERSION"
)

echo "launcher: providers/z_ai_claude_code.sh"
echo "run_id:   $RUN_ID"
echo "agent:    claude-code (native goal mode)"
echo "claude:   $CLAUDE_CODE_VERSION"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT"
echo "endpoint: $ENDPOINT"
echo "provider: z-ai/fp8 (only; no fallback)"
echo "wire_api: Anthropic Messages"
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
  --launch-env SPRINT_OPENROUTER_PROVIDER_ENDPOINT \
  --launch-env SPRINT_OPENROUTER_QUANTIZATION
echo "trial unit=sprint-trial-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
