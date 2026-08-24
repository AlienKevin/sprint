#!/usr/bin/env bash
# Harbor Codex + a DeepSeek model through a sealed OpenRouter endpoint.
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

MODEL="${MODEL:-deepseek/deepseek-v4-flash-0731}"
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"
if [[ "${MODEL#*/}" == "deepseek-v4-flash-vision-exp" ]]; then
  if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
        && "$SPRINT_OPENROUTER_PROVIDER_ENDPOINT" != "deepseek" ]]; then
    echo "DeepSeek V4 Flash Vision Exp is locked to the official DeepSeek endpoint" >&2
    exit 2
  fi
  if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" ]]; then
    echo "DeepSeek V4 Flash Vision Exp official endpoint has no sealed quantization" >&2
    exit 2
  fi
  export SPRINT_OPENROUTER_PROVIDER_ENDPOINT=deepseek
  export SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW=1048576
  unset SPRINT_OPENROUTER_QUANTIZATION
elif [[ "${MODEL#*/}" == deepseek-v4-flash* ]]; then
  if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
        && "$SPRINT_OPENROUTER_PROVIDER_ENDPOINT" != "baidu/fp8" ]]; then
    echo "DeepSeek V4 Flash provider is locked to baidu/fp8" >&2
    exit 2
  fi
  if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" \
        && "$SPRINT_OPENROUTER_QUANTIZATION" != "fp8" ]]; then
    echo "DeepSeek V4 Flash quantization is locked to fp8" >&2
    exit 2
  fi
  export SPRINT_OPENROUTER_PROVIDER_ENDPOINT=baidu/fp8
  export SPRINT_OPENROUTER_QUANTIZATION=fp8
  export SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW=1048576
fi
if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" ]]; then
  OPENROUTER_MODEL="${OPENROUTER_MODEL:-$MODEL}"
else
  echo "DeepSeek routes must pin one OpenRouter provider endpoint" >&2
  exit 2
fi
REASONING_EFFORT="${REASONING_EFFORT:-max}"  # API-max for DeepSeek Flash
# Same immutable harness pin as every competitor.
CODEX_VERSION="${CODEX_VERSION:-0.149.1}"
RUN_ID="${RUN_ID:-lane-deepseek-$(date -u +%Y%m%dT%H%M%SZ)}"

GOAL="$ROOT/event_runtime/control/templates/codex.j2"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export SPRINT_CODEX_DEEPSEEK_BASE_URL="$ENDPOINT"
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-$MODEL}"
export SPRINT_CODEX_DEEPSEEK_MODEL="$OPENROUTER_MODEL"
echo "launcher: providers/deepseek.sh"
echo "run_id:   $RUN_ID"
echo "agent:    codex"
echo "codex:    $CODEX_VERSION  (Harbor --ak version=...)"
echo "model:    $MODEL"
echo "effort:   $REASONING_EFFORT"
echo "endpoint: $ENDPOINT"
echo "wire:     $OPENROUTER_MODEL"
echo "provider: $SPRINT_OPENROUTER_PROVIDER_ENDPOINT (only; no fallback)"
echo "quant:    ${SPRINT_OPENROUTER_QUANTIZATION:-provider default}"
echo "wire_api: responses (via [model_providers.deepseek]; not openai_base_url alone)"
echo "catalog:  DeepSeek models.json (${SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW:-1048576} context, completion-aware auto compaction)"
echo "goal:     $GOAL"
echo "profile:  $MODAL_PROFILE"
echo "auth:     OPENAI_API_KEY=[OpenRouter key] via env-file (not --ae)"

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
  --run-id "$RUN_ID" \
  --agent-kind codex \
  --model "$MODEL" \
  --endpoint "$ENDPOINT" \
  --reasoning-effort "$REASONING_EFFORT" \
  --codex-version "$CODEX_VERSION"
)
LAUNCH_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LAUNCH_ARGS[@]}")
TRIAL_ARGS=(
  "$ROOT/event_runtime/control/start_trial.py"
  --run-id "$RUN_ID" \
  --batch-id "${SPRINT_BATCH_ID:-}" \
  --launch-argv-json "$LAUNCH_JSON" \
  --secret-env OPENAI_API_KEY
)
if [[ -n "${SPRINT_DEEPSEEK_PRICING_SNAPSHOT:-}" ]]; then
  TRIAL_ARGS+=(--secret-env SPRINT_DEEPSEEK_PRICING_SNAPSHOT)
fi
if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" ]]; then
  TRIAL_ARGS+=(
    --launch-env OPENROUTER_MODEL
    --launch-env SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW
    --launch-env SPRINT_CODEX_DEEPSEEK_MODEL
    --launch-env SPRINT_OPENROUTER_PROVIDER_ENDPOINT
  )
  if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" ]]; then
    TRIAL_ARGS+=(--launch-env SPRINT_OPENROUTER_QUANTIZATION)
  fi
fi
python3 "${TRIAL_ARGS[@]}"
echo "trial unit=sprint-trial-${RUN_ID}.service log=/data/sprint-launch-${RUN_ID}.log"
