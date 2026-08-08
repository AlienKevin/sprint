#!/usr/bin/env bash
# Sprint durable lane: Codex + Terra through OpenAI. Dry-runs unless confirmed.
set -euo pipefail

ROOT="${SPRINT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"
MODEL="${MODEL:-openai/gpt-5.6-terra}"
REASONING_EFFORT="${REASONING_EFFORT:-ultra}"
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
RUN_ID="${RUN_ID:-lane-terra-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required" >&2
  exit 1
fi

DRY_ARGS=(
  --dry-run
  --run-id "$RUN_ID"
  --agent-kind codex
  --model "$MODEL"
  --reasoning-effort "$REASONING_EFFORT"
  --codex-version "$CODEX_VERSION"
)

echo "run_id:   $RUN_ID"
echo "agent:    codex $CODEX_VERSION"
echo "model:    $MODEL"
echo "endpoint: https://api.openai.com (official OpenAI API)"
echo "cpu:      4 physical cores, 16 GiB, GPU-free, 24-hour sandbox timeout"
echo "auth:     private OPENAI_API_KEY env-file"
echo "--- dry-run ---"
"$ROOT/runs/run-lane-durable.sh" "${DRY_ARGS[@]}"

if [[ "${CONFIRM_LAUNCH:-}" != "1" ]]; then
  echo "dry-run only. Set CONFIRM_LAUNCH=1 to launch." >&2
  exit 0
fi

LAUNCH_ARGS=(
  "$ROOT/runs/run-lane-durable.sh"
  --supervised-launch
  --run-id "$RUN_ID"
  --agent-kind codex
  --model "$MODEL"
  --reasoning-effort "$REASONING_EFFORT"
  --codex-version "$CODEX_VERSION"
)
LAUNCH_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LAUNCH_ARGS[@]}")
python3 "$ROOT/runs/ops/start_lane_supervisor.py" \
  --run-id "$RUN_ID" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env OPENAI_API_KEY \
  --max-restarts "${CPU_MAX_RESTARTS:-50}" \
  --min-backoff-s "${CPU_MIN_BACKOFF_S:-30}" \
  --max-backoff-s "${CPU_MAX_BACKOFF_S:-600}"
echo "supervisor unit=sprint-lane-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
