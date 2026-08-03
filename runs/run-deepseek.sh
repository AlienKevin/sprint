#!/usr/bin/env bash
# DRAFT — Harbor Codex + deepseek-v4-flash (official DeepSeek API), durable Modal lane.
# Does NOT launch unless CONFIRM_LAUNCH=1.
#
# Usage:
#   set -a; source runs/.secrets/deepseek.env; set +a
#   runs/run-deepseek.sh                  # dry-run only
#   CONFIRM_LAUNCH=1 runs/run-deepseek.sh # real launch
set -euo pipefail

ROOT=/data/qwop-bench
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
ENDPOINT="${ENDPOINT:-https://api.deepseek.com}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"  # API-max for DeepSeek Flash
# Same pin as Luna; see runs/CODEX_PIN.md (npm @openai/codex@latest = 0.146.0).
CODEX_VERSION="${CODEX_VERSION:-0.146.0}"
RUN_ID="${RUN_ID:-lane-deepseek-$(date -u +%Y%m%dT%H%M%SZ)}"

GOAL_SRC="$ROOT/runs/codex-goal-slash.j2"
GOAL_DST="$ROOT/runs/codex-goal.j2"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY unset; source runs/.secrets/deepseek.env (maps DeepSeek key)" >&2
  exit 1
fi

echo "draft:    run-deepseek.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION  (Harbor --ak version=...)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT  (DeepSeek Flash API-max; xhigh maps to high)"
echo "endpoint: $ENDPOINT"
echo "goal:     $GOAL_SRC -> sync to $GOAL_DST before launch"
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
"$ROOT/runs/run-lane-durable.sh" "${DRY_ARGS[@]}"

if [[ "${CONFIRM_LAUNCH:-}" != "1" ]]; then
  echo "dry-run only. Set CONFIRM_LAUNCH=1 after approving BLINDSPOTS_LUNA_DEEPSEEK.md" >&2
  exit 0
fi

if ! cmp -s "$GOAL_SRC" "$GOAL_DST"; then
  echo "Refusing launch: $GOAL_DST does not match /goal template $GOAL_SRC" >&2
  echo "Sync with: cp $GOAL_SRC $GOAL_DST" >&2
  exit 2
fi

echo "--- launch ---"
nohup "$ROOT/runs/run-lane-durable.sh" \
  --run-id "$RUN_ID" \
  --agent-kind codex \
  --model "$MODEL" \
  --endpoint "$ENDPOINT" \
  --reasoning-effort "$REASONING_EFFORT" \
  --codex-version "$CODEX_VERSION" \
  >"/data/qwop-launch-${RUN_ID}.log" 2>&1 </dev/null &
echo "launched pid=$! log=/data/qwop-launch-${RUN_ID}.log"
