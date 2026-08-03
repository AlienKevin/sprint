#!/usr/bin/env bash
# DRAFT — Harbor Codex + gpt-5.6-luna (official OpenAI), durable Modal lane.
# Does NOT launch unless CONFIRM_LAUNCH=1.
#
# Usage:
#   set -a; source runs/.secrets/openai-luna.env; set +a
#   runs/run-luna.sh                  # dry-run only
#   CONFIRM_LAUNCH=1 runs/run-luna.sh # real launch
set -euo pipefail

ROOT=/data/qwop-bench
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"

MODEL="${MODEL:-openai/gpt-5.6-luna}"
# Codex/Responses accepts max for Luna (Chat Completions does not). See REASONING_EFFORT_PROBE.md.
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# Same pin as DeepSeek; see runs/CODEX_PIN.md (npm @openai/codex@latest = 0.146.0).
CODEX_VERSION="${CODEX_VERSION:-0.146.0}"
RUN_ID="${RUN_ID:-lane-luna-$(date -u +%Y%m%dT%H%M%SZ)}"

# Fair /goal template: durable launcher still reads codex-goal.j2.
# Before a real launch, sync: cp runs/codex-goal-slash.j2 runs/codex-goal.j2
GOAL_SRC="$ROOT/runs/codex-goal-slash.j2"
GOAL_DST="$ROOT/runs/codex-goal.j2"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY unset; source runs/.secrets/openai-luna.env" >&2
  exit 1
fi

echo "draft:    run-luna.sh"
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
  echo "dry-run only. Set CONFIRM_LAUNCH=1 after approving BLINDSPOTS_LUNA_DEEPSEEK.md" >&2
  exit 0
fi

if ! cmp -s "$GOAL_SRC" "$GOAL_DST"; then
  echo "Refusing launch: $GOAL_DST does not match /goal template $GOAL_SRC" >&2
  echo "Sync with: cp $GOAL_SRC $GOAL_DST" >&2
  exit 2
fi

echo "--- launch ---"
# Prefer nohup so the shell can exit; monitor starts inside durable launcher.
nohup "$ROOT/runs/run-lane-durable.sh" \
  --run-id "$RUN_ID" \
  --agent-kind codex \
  --model "$MODEL" \
  --reasoning-effort "$REASONING_EFFORT" \
  --codex-version "$CODEX_VERSION" \
  >"/data/qwop-launch-${RUN_ID}.log" 2>&1 </dev/null &
echo "launched pid=$! log=/data/qwop-launch-${RUN_ID}.log"
