#!/usr/bin/env bash
# Short DeepSeek smoke for automatic GPU-worker recovery.
# Does NOT launch unless CONFIRM_LAUNCH=1.
set -euo pipefail

ROOT="${SPRINT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"
touch /data/.keepalive

MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
ENDPOINT="${ENDPOINT:-https://api.deepseek.com}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
RUN_ID="${RUN_ID:-smoke-cpu-gpu-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_DIR="$ROOT/runs/ops/$RUN_ID"
REPORT="$ROOT/runs/SMOKE_GPU_RECOVERY_$(date -u +%Y%m%dT%H%M%SZ).md"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f "$ROOT/runs/.secrets/deepseek.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/runs/.secrets/deepseek.env"
    set +a
  fi
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY unset; source runs/.secrets/deepseek.env" >&2
  exit 1
fi

echo "run_id=$RUN_ID"
echo "report=$REPORT"
echo "model=$MODEL codex=$CODEX_VERSION effort=$REASONING_EFFORT"

if [[ "${CONFIRM_LAUNCH:-}" != "1" ]]; then
  "$ROOT/runs/run-lane-durable.sh" --dry-run \
    --run-id "$RUN_ID" --agent-kind codex \
    --model "$MODEL" --endpoint "$ENDPOINT" \
    --reasoning-effort "$REASONING_EFFORT" \
    --prompt-template "$ROOT/runs/codex-recovery-smoke-goal.j2" \
    --codex-version "$CODEX_VERSION"
  echo "dry-run only. CONFIRM_LAUNCH=1 to launch + drive smoke." >&2
  exit 0
fi

# Keep /goal template in sync for DeepSeek.
cp "$ROOT/runs/codex-goal-slash.j2" "$ROOT/runs/codex-goal.j2"

nohup "$ROOT/runs/run-lane-durable.sh" \
  --run-id "$RUN_ID" \
  --agent-kind codex \
  --model "$MODEL" \
  --endpoint "$ENDPOINT" \
  --reasoning-effort "$REASONING_EFFORT" \
  --prompt-template "$ROOT/runs/codex-recovery-smoke-goal.j2" \
  --codex-version "$CODEX_VERSION" \
  >"/data/sprint-launch-${RUN_ID}.log" 2>&1 </dev/null &
LAUNCH_PID=$!
echo "launched pid=$LAUNCH_PID log=/data/sprint-launch-${RUN_ID}.log"

# Drive the checklist from the host once the agent sandbox is up.
exec run-heavy python3 -u "$ROOT/runs/ops/smoke_gpu_recovery.py" \
  --run-id "$RUN_ID" \
  --report "$REPORT"
