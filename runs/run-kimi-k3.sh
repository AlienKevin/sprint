#!/usr/bin/env bash
# Harbor Kimi Code run: kimi-k3 (max thinking) on the lane sprint task.
#
# Inference: Modal Shared Endpoint on profile kevinli020508
#   endpoint: kimi-k3-shared (ep-MYgY9vnoAsB7IkHwu5quZA)
#   base:     https://kevinli020508--ep-kimi-k3-shared-server.us-west.modal.direct/v1
#   auth:     Authorization: Bearer wk-<id>.ws-<secret>
#             (same proxy token also works as Modal-Key / Modal-Secret headers)
#   model:    moonshotai/Kimi-K3
#
# Requires proxy token in runs/.secrets/modal-kimi-k3.env (or env vars):
#   MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET
#   or KIMI_MODEL_API_KEY=wk-....ws-....
#
# Auth: Bearer token is written to a mode-0600 --env-file (never --ae KEY=value).
# Non-secret Kimi knobs stay on --ae.
#
# Optional:
#   MODEL=moonshotai/Kimi-K3
#   TASK=/data/qwop-bench/challenge/g1-sprint-100m-lane
#   JOBS_DIR=/data/qwop-bench/runs/jobs-kimi-k3
#
# Jobs land under jobs-kimi-k3 (durable for Hub upload after secrets scrub).
# Does not touch Opus / Terra job dirs.
set -euo pipefail

cd /tmp/harbor-fork 2>/dev/null || cd /data/harbor-continuous
export MODAL_PROFILE="${MODAL_PROFILE:-kevinli020508}"
unset HARBOR_API_KEY

SECRETS_ENV="${KIMI_SECRETS_ENV:-/data/qwop-bench/runs/.secrets/modal-kimi-k3.env}"
if [ -f "$SECRETS_ENV" ]; then
  # shellcheck disable=SC1090
  set -a
  source "$SECRETS_ENV"
  set +a
fi

if [ -z "${KIMI_MODEL_API_KEY:-}" ]; then
  if [ -n "${MODAL_PROXY_TOKEN_ID:-}" ] && [ -n "${MODAL_PROXY_TOKEN_SECRET:-}" ]; then
    KIMI_MODEL_API_KEY="${MODAL_PROXY_TOKEN_ID}.${MODAL_PROXY_TOKEN_SECRET}"
  fi
fi
if [ -z "${KIMI_MODEL_API_KEY:-}" ]; then
  echo "KIMI_MODEL_API_KEY or MODAL_PROXY_TOKEN_ID/SECRET required (see $SECRETS_ENV)" >&2
  exit 1
fi

# Strip accidental whitespace/newlines (empty Bearer -> Modal "proxy auth required").
KIMI_MODEL_API_KEY="$(printf '%s' "$KIMI_MODEL_API_KEY" | tr -d '[:space:]')"
case "$KIMI_MODEL_API_KEY" in
  wk-*.ws-*) ;;
  *)
    echo "KIMI_MODEL_API_KEY must be wk-<id>.ws-<secret> (Modal Shared Bearer form)" >&2
    exit 1
    ;;
esac

MODAL_PROXY_TOKEN_ID="${MODAL_PROXY_TOKEN_ID:-${KIMI_MODEL_API_KEY%%.*}}"
MODAL_PROXY_TOKEN_SECRET="${MODAL_PROXY_TOKEN_SECRET:-${KIMI_MODEL_API_KEY#*.}}"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
JOBS_DIR="${JOBS_DIR:-/data/qwop-bench/runs/jobs-kimi-k3}"
TASK="${TASK:-/data/qwop-bench/challenge/g1-sprint-100m-lane}"
BASE_URL="${KIMI_MODEL_BASE_URL:-https://kevinli020508--ep-kimi-k3-shared-server.us-west.modal.direct/v1}"
HOST_ALLOW="${KIMI_ALLOW_HOST:-kevinli020508--ep-kimi-k3-shared-server.us-west.modal.direct}"
# kimi-code maps this onto OpenAI max_tokens. Modal Shared's context budget is
# input+completion <= 1048576, so full 1M max_tokens 400s once the system
# prompt/tools are counted. Leave headroom.
MAX_CTX="${KIMI_MODEL_MAX_CONTEXT_SIZE:-1000000}"
LAUNCH_LABEL="${LAUNCH_LABEL:-kimi-k3-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$JOBS_DIR"

echo "task:     $TASK"
echo "agent:    kimi-code"
echo "model:    $MODEL"
echo "effort:   KIMI_MODEL_THINKING_EFFORT=max"
echo "base:     $BASE_URL"
echo "max_ctx:  $MAX_CTX (Modal Shared input+completion cap 1048576)"
echo "goal:     prompt template runs/claude-code-goal.j2 (/goal {{ instruction }})"
echo "jobs-dir: $JOBS_DIR"
echo "auth:     Modal proxy token via env-file KIMI_MODEL_API_KEY (not --ae)"
echo "modal:    MODAL_PROFILE=$MODAL_PROFILE (Isaac / sandboxes + Shared Endpoint)"
echo "token_id: ${MODAL_PROXY_TOKEN_ID}"

# Preflight: Modal returns bare {"error":"proxy auth required"} when Authorization
# is missing/empty; wrong secret returns "Webhook token secret validation failure".
# Fail launch if Shared auth or chat is broken (cold 502s get a short retry).
preflight() {
  local code body
  code=$(curl -sS -o /tmp/kimi-preflight-models.json -w '%{http_code}' \
    -H "Authorization: Bearer ${KIMI_MODEL_API_KEY}" \
    "${BASE_URL}/models" --max-time 30 || echo ERR)
  if [ "$code" != "200" ]; then
    echo "preflight /models failed HTTP=$code" >&2
    head -c 200 /tmp/kimi-preflight-models.json >&2 || true
    echo >&2
    return 1
  fi
  local attempt=1
  while [ "$attempt" -le 4 ]; do
    code=$(curl -sS -o /tmp/kimi-preflight-chat.json -w '%{http_code}' \
      -H "Authorization: Bearer ${KIMI_MODEL_API_KEY}" \
      -H "Content-Type: application/json" \
      "${BASE_URL}/chat/completions" --max-time 180 \
      -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8}" \
      || echo ERR)
    if [ "$code" = "200" ]; then
      echo "preflight: /models 200, /chat/completions 200 (attempt $attempt)"
      return 0
    fi
    body=$(head -c 160 /tmp/kimi-preflight-chat.json 2>/dev/null || true)
    echo "preflight chat HTTP=$code attempt=$attempt body=${body}" >&2
    case "$body" in
      *proxy\ auth*|*Webhook\ token*)
        echo "preflight auth failure; refusing to launch" >&2
        return 1
        ;;
    esac
    attempt=$((attempt + 1))
    sleep 5
  done
  echo "preflight chat never became healthy" >&2
  return 1
}
preflight

ENV_FILE=$(python3 /data/qwop-bench/runs/ops/write_harbor_env_file.py \
  --label "$LAUNCH_LABEL" \
  "KIMI_MODEL_API_KEY=${KIMI_MODEL_API_KEY}")
# Keep ID for logs; drop secret material from the shell before harbor argv.
unset KIMI_MODEL_API_KEY MODAL_PROXY_TOKEN_SECRET
# Harbor sensitive-env templatize may still resolve ${KIMI_MODEL_API_KEY} from
# the env-file-loaded process environ after --env-file.

KEEPALIVE_JSON=$(python3 /data/qwop-bench/runs/ops/telemetry_keepalive.py --run-id kimi-k3)
echo "telemetry: Modal keepalive starts /opt/qwop-telemetry.sh -> /logs/artifacts/telemetry"
echo "env-file: $ENV_FILE"

exec uv run harbor run \
  --path "$TASK" \
  --agent kimi-code \
  --model "$MODEL" \
  --ak "prompt_template_path=/data/qwop-bench/runs/claude-code-goal.j2" \
  --allow-agent-host "$HOST_ALLOW" \
  --env modal \
  -n 1 \
  --jobs-dir "$JOBS_DIR" \
  --env-file "$ENV_FILE" \
  --ek "keepalive=$KEEPALIVE_JSON" \
  --ae "KIMI_MODEL_BASE_URL=$BASE_URL" \
  --ae "KIMI_MODEL_MAX_CONTEXT_SIZE=$MAX_CTX" \
  --ae "KIMI_MODEL_CAPABILITIES=image_in,thinking" \
  --ae "KIMI_MODEL_THINKING_EFFORT=max" \
  --ae "KIMI_CODE_EXPERIMENTAL_FLAG=true" \
  "$@"
