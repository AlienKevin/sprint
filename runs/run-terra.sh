#!/usr/bin/env bash
# Harbor Codex run: gpt-5.6-terra via OpenRouter (OpenAI provider pinned).
#
# Requires:
#   export OPENROUTER_API_KEY=sk-or-v1-...
#
# Optional:
#   OPENROUTER_PROVIDER_PIN=openai   # default; header X-OpenRouter-Provider
#   MODEL=openai/gpt-5.6-terra       # Harbor strips to last path segment
#
# Auth: OPENROUTER key is written to a mode-0600 --env-file (never --ae KEY=value).
# Jobs land under /data/qwop-bench/runs/jobs-terra (durable for Hub upload
# after secrets are scrubbed). Does not touch Opus/Kimi job dirs.
set -euo pipefail

cd /tmp/harbor-fork 2>/dev/null || cd /data/harbor-continuous
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"
unset HARBOR_API_KEY

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi

MODEL="${MODEL:-openai/gpt-5.6-terra}"
PROVIDER_PIN="${OPENROUTER_PROVIDER_PIN:-openai}"
JOBS_DIR="${JOBS_DIR:-/data/qwop-bench/runs/jobs-terra}"
TASK="${TASK:-/data/qwop-bench/challenge/g1-sprint-100m-lane}"
REASONING_EFFORT="${REASONING_EFFORT:-ultra}"
LAUNCH_LABEL="${LAUNCH_LABEL:-terra-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$JOBS_DIR"
KEEPALIVE_JSON=$(python3 /data/qwop-bench/runs/ops/telemetry_keepalive.py \
  --run-id "terra-${MODEL##*/}")

ENV_FILE=$(python3 /data/qwop-bench/runs/ops/write_harbor_env_file.py \
  --label "$LAUNCH_LABEL" \
  "OPENAI_API_KEY=${OPENROUTER_API_KEY}" \
  "OPENAI_BASE_URL=https://openrouter.ai/api/v1" \
  "OPENROUTER_PROVIDER_PIN=${PROVIDER_PIN}")
# Drop host copies so argv and child shells do not inherit the raw key.
unset OPENROUTER_API_KEY OPENAI_API_KEY

echo "task:     $TASK"
echo "agent:    codex"
echo "model:    $MODEL  (codex uses last path segment)"
echo "effort:   $REASONING_EFFORT"
echo "base:     https://openrouter.ai/api/v1"
echo "pin:      X-OpenRouter-Provider=$PROVIDER_PIN"
echo "goal:     prompt template runs/codex-goal.j2 (Codex has no /goal slash-command)"
echo "jobs-dir: $JOBS_DIR"
echo "auth:     OPENROUTER_API_KEY -> env-file OPENAI_API_KEY (not --ae)"
echo "env-file: $ENV_FILE"
echo "telemetry: Modal keepalive starts /opt/qwop-telemetry.sh -> /logs/artifacts/telemetry"

exec uv run harbor run \
  --path "$TASK" \
  --agent codex \
  --model "$MODEL" \
  --ak "reasoning_effort=$REASONING_EFFORT" \
  --ak "prompt_template_path=/data/qwop-bench/runs/codex-goal.j2" \
  --env modal \
  -n 1 \
  --jobs-dir "$JOBS_DIR" \
  --env-file "$ENV_FILE" \
  --ek "keepalive=$KEEPALIVE_JSON" \
  "$@"
