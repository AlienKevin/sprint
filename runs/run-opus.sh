#!/usr/bin/env bash
# The real run: Opus 5 trains a G1 sprint policy for as long as it is given.
#
# Needs one of:
#   export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat...   (preferred: subscription usage)
#   export ANTHROPIC_API_KEY=sk-ant-api...         (falls back to API billing)
# Get the first with `claude setup-token`, which needs a browser sign-in.
#
# Optional:
#   export HARBOR_API_KEY=sk-harbor-...            (uploads the job, privately)
#
# Auth secrets go in a mode-0600 --env-file (never --ae TOKEN=value on cmdline).
# Non-secret toggles (CLAUDE_FORCE_OAUTH=1) may still use --ae.
#
# The task allows the agent 24 h. To stop sooner, pass --timeout-multiplier:
#   ./run-opus.sh --timeout-multiplier 0.3         (about 7 h)
set -euo pipefail

cd /tmp/harbor-fork 2>/dev/null || cd /data/harbor-continuous
# The Isaac images are cached in this Modal workspace and image caches are
# per-workspace, so another profile rebuilds 20 GB before the agent starts.
export MODAL_PROFILE=kevinli020508

LAUNCH_LABEL="${LAUNCH_LABEL:-opus-$(date -u +%Y%m%dT%H%M%SZ)}"
AGENT_ENV=()
ENV_ASSIGNMENTS=()

if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    # CLAUDE_FORCE_OAUTH drops the API key so the CLI bills the subscription
    # rather than API credits. Non-secret; safe on --ae.
    AGENT_ENV+=(--ae "CLAUDE_FORCE_OAUTH=1")
    ENV_ASSIGNMENTS+=("CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}")
    echo "auth: subscription (OAuth token via env-file)"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ENV_ASSIGNMENTS+=("ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
    echo "auth: API key via env-file (billed as API usage)"
else
    echo "No Anthropic credential. Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY." >&2
    exit 1
fi

ENV_FILE=$(python3 /data/qwop-bench/runs/ops/write_harbor_env_file.py \
  --label "$LAUNCH_LABEL" \
  "${ENV_ASSIGNMENTS[@]}")
unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN

# The key is exercised now rather than trusted. Harbor checks Hub auth before
# the run, so a dead key would abort at the start, but checking here says which
# credential is the problem instead of failing on a generic "not logged in".
UPLOAD=()
if [ -n "${HARBOR_API_KEY:-}" ]; then
    if uv run python -c "
import asyncio, sys
from harbor.upload.db_client import UploadDB
try:
    asyncio.run(UploadDB().get_user_id())
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
        UPLOAD+=(--upload --private)
        echo "hub: uploading privately"
    else
        echo "hub: HARBOR_API_KEY is set but rejected; running without --upload." >&2
        echo "     Scrub+upload later via: python3 runs/hub_track_upload.py --once" >&2
        echo "     Artifacts still land under /data/qwop-bench/runs/jobs." >&2
    fi
fi

KEEPALIVE_JSON=$(python3 /data/qwop-bench/runs/ops/telemetry_keepalive.py --run-id opus)
echo "telemetry: Modal keepalive starts /opt/qwop-telemetry.sh -> /logs/artifacts/telemetry"
echo "env-file: $ENV_FILE"

exec uv run harbor run \
    --path /data/qwop-bench/challenge/g1-sprint-100m \
    --agent claude-code \
    --model claude-opus-5 \
    --ak "prompt_template_path=/data/qwop-bench/runs/claude-code-goal.j2" \
    --env modal \
    -n 1 \
    --jobs-dir /data/qwop-bench/runs/jobs \
    --env-file "$ENV_FILE" \
    --ek "keepalive=$KEEPALIVE_JSON" \
    "${AGENT_ENV[@]}" \
    "${UPLOAD[@]}" \
    "$@"
