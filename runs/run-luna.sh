#!/usr/bin/env bash
# Harbor Codex + gpt-5.6-luna (official OpenAI), durable Modal lane.
# Does NOT launch unless CONFIRM_LAUNCH=1.
#
# Usage:
#   set -a; source runs/.secrets/openai-luna.env; set +a
#   runs/run-luna.sh                  # dry-run only
#   CONFIRM_LAUNCH=1 runs/run-luna.sh # real launch
set -euo pipefail

ROOT="${SPRINT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL="${MODEL:-openai/gpt-5.6-luna}"
# Codex/Responses accepts max for Luna (Chat Completions does not). See REASONING_EFFORT_PROBE.md.
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# Same pin as the current eval launchers; see runs/CODEX_PIN.md.
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
RUN_ID="${RUN_ID:-lane-luna-$(date -u +%Y%m%dT%H%M%SZ)}"

# Fair /goal template: durable launcher still reads codex-goal.j2.
# Before a real launch, sync: cp runs/codex-goal-slash.j2 runs/codex-goal.j2
GOAL_SRC="$ROOT/runs/codex-goal-slash.j2"
GOAL_DST="$ROOT/runs/codex-goal.j2"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY unset; source runs/.secrets/openai-luna.env" >&2
  exit 1
fi

echo "launcher: run-luna.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION  (Harbor --ak version=...)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT  (Codex/Responses max; see REASONING_EFFORT_PROBE.md)"
echo "endpoint: (default OpenAI)"
echo "goal:     $GOAL_SRC -> sync to $GOAL_DST before launch"
echo "profile:  $MODAL_PROFILE"
echo "auth:     OPENAI_API_KEY=[configured] via durable env-file (not --ae)"

DRY_ARGS=(
  --dry-run
  --run-id "$RUN_ID"
  --agent-kind codex
  --model "$MODEL"
  --reasoning-effort "$REASONING_EFFORT"
  --codex-version "$CODEX_VERSION"
)

echo "--- dry-run ---"
"$ROOT/runs/run-lane-durable.sh" "${DRY_ARGS[@]}"

if [[ "${CONFIRM_LAUNCH:-}" != "1" ]]; then
  echo "dry-run only. Set CONFIRM_LAUNCH=1 to launch." >&2
  exit 0
fi

if ! cmp -s "$GOAL_SRC" "$GOAL_DST"; then
  echo "Refusing launch: $GOAL_DST does not match /goal template $GOAL_SRC" >&2
  echo "Sync with: cp $GOAL_SRC $GOAL_DST" >&2
  exit 2
fi

echo "--- launch ---"
LAUNCH_ARGS=(
  "$ROOT/runs/run-lane-durable.sh"
  --supervised-launch
  --run-id "$RUN_ID" \
  --agent-kind codex \
  --model "$MODEL" \
  --reasoning-effort "$REASONING_EFFORT" \
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
