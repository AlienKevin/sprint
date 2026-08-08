#!/usr/bin/env bash
# Sprint durable lane: Claude Code + Opus 5. Dry-runs unless CONFIRM_LAUNCH=1.
set -euo pipefail

ROOT="${SPRINT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"
MODEL="${MODEL:-claude-opus-5}"
RUN_ID="${RUN_ID:-lane-opus-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  echo "CLAUDE_CODE_OAUTH_TOKEN is required; API-key fallback is disabled" >&2
  exit 1
fi

DRY_ARGS=(
  --dry-run
  --run-id "$RUN_ID"
  --agent-kind claude-code
  --model "$MODEL"
  --reasoning-effort max
)

echo "run_id:  $RUN_ID"
echo "agent:   claude-code"
echo "model:   $MODEL"
echo "cpu:     4 physical cores, 16 GiB, GPU-free, 24-hour sandbox timeout"
echo "auth:    CLAUDE_CODE_OAUTH_TOKEN via private env-file"
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
  --agent-kind claude-code
  --model "$MODEL"
  --reasoning-effort max
)
LAUNCH_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LAUNCH_ARGS[@]}")
python3 "$ROOT/runs/ops/start_lane_supervisor.py" \
  --run-id "$RUN_ID" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env CLAUDE_CODE_OAUTH_TOKEN \
  --max-restarts "${CPU_MAX_RESTARTS:-50}" \
  --min-backoff-s "${CPU_MIN_BACKOFF_S:-30}" \
  --max-backoff-s "${CPU_MAX_BACKOFF_S:-600}"
echo "supervisor unit=sprint-lane-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
