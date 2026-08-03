#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/qwop-bench
HARBOR=/data/harbor-continuous
HARBOR_COMMIT=c6d90cb50ff4a19e725e4724e004c21ecbd96f49
HARBOR_BRANCH=continuous-verification
UV=/home/ubuntu/.local/bin/uv
CONTROL="$ROOT/runs/ops/qwopctl.py"
TASK="$ROOT/challenge/g1-sprint-100m-lane"
MODAL_PROFILE=${MODAL_PROFILE:-kevinli020508}
SANDBOX_TIMEOUT_SECONDS=86400
DEPLOY_DEBOUNCE_SECONDS=300
# npm @openai/codex@latest as of 2026-08-02 (also host `codex --version`).
# Harbor installs via `npm install -g @openai/codex@$CODEX_VERSION` when set.
CODEX_VERSION=${CODEX_VERSION:-0.146.0}
RUN_ID=""
AGENT_KIND=claude-code
MODEL=""
ENDPOINT=""
REASONING_EFFORT=""
DRY_RUN=0
START_MONITOR=1
EXTRA=()

usage() {
  cat <<'EOF'
Usage: runs/run-lane-durable.sh [options] [-- HARBOR_ARGS...]

Options:
  --run-id ID                Explicit unique run ID.
  --agent-kind KIND          claude-code or codex (default claude-code).
  --model MODEL              Agent model; required for codex.
  --endpoint HTTPS_URL       Optional Codex API endpoint (no credentials/query).
  --reasoning-effort VALUE   Agent reasoning effort.
  --codex-version VERSION    Pin @openai/codex npm version (codex only; default 0.146.0).
  --dry-run                  Print redacted configuration; launch nothing.
  --no-monitor               Do not start the host monitor automatically.
EOF
}

while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --agent-kind) AGENT_KIND=${2:?}; shift 2 ;;
    --model) MODEL=${2:?}; shift 2 ;;
    --endpoint) ENDPOINT=${2:?}; shift 2 ;;
    --reasoning-effort) REASONING_EFFORT=${2:?}; shift 2 ;;
    --codex-version) CODEX_VERSION=${2:?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-monitor) START_MONITOR=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown launcher argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="lane-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,48}$ ]]; then
  echo "run ID must be 3-49 safe filename characters" >&2
  exit 2
fi
if [[ "$AGENT_KIND" != "claude-code" && "$AGENT_KIND" != "codex" ]]; then
  echo "--agent-kind must be claude-code or codex" >&2
  exit 2
fi
if [[ -z "$MODEL" ]]; then
  if [[ "$AGENT_KIND" == "claude-code" ]]; then
    MODEL=claude-opus-5
  else
    echo "--model is required for codex" >&2
    exit 2
  fi
fi
if [[ -z "$REASONING_EFFORT" ]]; then
  if [[ "$AGENT_KIND" == "claude-code" ]]; then
    REASONING_EFFORT=max
  else
    REASONING_EFFORT=high
  fi
fi
if [[ "$MODEL" == -* || "$MODEL" =~ [[:space:][:cntrl:]] ]]; then
  echo "--model must be one non-option value" >&2
  exit 2
fi
if [[ "$REASONING_EFFORT" == -* || "$REASONING_EFFORT" =~ [[:space:][:cntrl:]] ]]; then
  echo "--reasoning-effort must be one non-option value" >&2
  exit 2
fi
if [[ "$CODEX_VERSION" == -* || "$CODEX_VERSION" =~ [[:space:][:cntrl:]@] || -z "$CODEX_VERSION" ]]; then
  echo "--codex-version must be a non-empty npm version (no @ prefix)" >&2
  exit 2
fi
if [[ -n "$ENDPOINT" ]]; then
  python3 - "$ENDPOINT" <<'PY'
import sys
from urllib.parse import urlsplit

value = sys.argv[1]
url = urlsplit(value)
if (
    url.scheme != "https"
    or not url.hostname
    or url.username
    or url.password
    or url.query
    or url.fragment
    or any(ord(char) < 32 for char in value)
):
    raise SystemExit("--endpoint must be a credential-free HTTPS URL without query or fragment")
PY
fi
for item in "${EXTRA[@]}"; do
  case "$item" in
    --agent|--agent=*|--model|--model=*|\
    --ak|--ak=*|--agent-kwarg|--agent-kwarg=*|\
    --ae|--ae=*|--agent-env|--agent-env=*|--env-file|--env-file=*|\
    --ek|--ek=*|--environment-kwarg|--environment-kwarg=*|\
    --jobs-dir|--jobs-dir=*|--job-name|--job-name=*|\
    --timeout-multiplier|--timeout-multiplier=*|\
    --agent-timeout-multiplier|--agent-timeout-multiplier=*|\
    --env|--env=*|--path|--path=*|CLAUDE_FORCE_OAUTH*)
      echo "unsafe Harbor override rejected: $item" >&2
      exit 2
      ;;
  esac
done

[[ -x "$UV" ]] || { echo "missing uv: $UV" >&2; exit 1; }
[[ -d "$HARBOR/.git" ]] || { echo "missing stable Harbor checkout: $HARBOR" >&2; exit 1; }
origin_url=$(git -C "$HARBOR" remote get-url origin)
[[ "$origin_url" == *AlienKevin/harbor* ]] || {
  echo "Harbor origin is not AlienKevin/harbor: $origin_url" >&2
  exit 1
}
actual_commit=$(git -C "$HARBOR" rev-parse HEAD)
[[ "$actual_commit" == "$HARBOR_COMMIT" ]] || {
  echo "Harbor HEAD $actual_commit does not match pin $HARBOR_COMMIT" >&2
  exit 1
}
actual_branch=$(git -C "$HARBOR" branch --show-current)
[[ "$actual_branch" == "$HARBOR_BRANCH" ]] || {
  echo "Harbor branch $actual_branch does not match $HARBOR_BRANCH" >&2
  exit 1
}
[[ -z "$(git -C "$HARBOR" status --porcelain)" ]] || {
  echo "stable Harbor checkout is dirty: $HARBOR" >&2
  exit 1
}

if [[ "$AGENT_KIND" == "claude-code" ]]; then
  if [[ -n "$ENDPOINT" ]]; then
    echo "--endpoint is supported only for codex" >&2
    exit 2
  fi
  AGENT_SECRET_NAME=CLAUDE_CODE_OAUTH_TOKEN
  AGENT_SECRET=${CLAUDE_CODE_OAUTH_TOKEN:-}
  if [[ -z "$AGENT_SECRET" ]]; then
    echo "CLAUDE_CODE_OAUTH_TOKEN is required; API-key fallback is disabled" >&2
    exit 1
  fi
else
  AGENT_SECRET_NAME=OPENAI_API_KEY
  AGENT_SECRET=${OPENAI_API_KEY:-}
  if [[ -z "$AGENT_SECRET" ]]; then
    echo "OPENAI_API_KEY is required for codex" >&2
    exit 1
  fi
fi
if [[ "$AGENT_SECRET" == *$'\n'* ]]; then
  echo "$AGENT_SECRET_NAME contains a newline" >&2
  exit 1
fi
if ((${#AGENT_SECRET} < 16)); then
  echo "$AGENT_SECRET_NAME is too short for safe Harbor redaction" >&2
  exit 1
fi

APP_NAME="qwop-$RUN_ID"
VOLUME_NAME="qwop-$RUN_ID"
STATE_DIR="$ROOT/runs/ops/$RUN_ID"
JOBS_ROOT="$STATE_DIR/harbor-jobs"
SECRET_DIR="/data/qwop-run-secrets/$RUN_ID"
PASSWORD_FILE="$SECRET_DIR/restic-password"
ENV_FILE="$SECRET_DIR/harbor.env"
REMOTE_PASSWORD="/durable/runs/$RUN_ID/secrets/restic-password"

VOLUMES_JSON=$(python3 - "$VOLUME_NAME" <<'PY'
import json
import sys
print(json.dumps({"/durable": sys.argv[1]}, separators=(",", ":")))
PY
)
LABELS_JSON=$(python3 - "$RUN_ID" "$AGENT_KIND" <<'PY'
import json
import sys
print(json.dumps(
    {"qwop.run_id": sys.argv[1], "qwop.agent_kind": sys.argv[2]},
    separators=(",", ":"),
))
PY
)
KEEPALIVE_JSON=$(python3 - "$RUN_ID" "$AGENT_KIND" "$REMOTE_PASSWORD" <<'PY'
import json
import shlex
import sys

run_id, agent_kind, password = sys.argv[1:]
watcher = [
    "/opt/qwop-snapshot-loop.sh",
    "--run-id", run_id,
    "--agent-kind", agent_kind,
    "--password-file", password,
]
# Start GPU/CPU telemetry before the durable snapshot loop. Snapshot-loop also
# starts telemetry idempotently; this covers the sleep-infinity fallback path.
telemetry = (
    "umask 077; mkdir -p /logs/artifacts/telemetry /run; "
    "if [ -x /opt/qwop-telemetry.sh ]; then "
    "QWOP_RUN_ID="
    + shlex.quote(run_id)
    + " /opt/qwop-telemetry.sh --role agent --run-id "
    + shlex.quote(run_id)
    + " --out-dir /logs/artifacts/telemetry "
    "--interval-seconds ${QWOP_TELEMETRY_INTERVAL:-20} "
    "--pidfile /run/qwop-telemetry.pid || true; "
    "fi; "
)
command = (
    telemetry
    + "if [ -x /opt/qwop-snapshot-loop.sh ]; then exec "
    + " ".join(shlex.quote(part) for part in watcher)
    + "; else exec sleep infinity; fi"
)
print(json.dumps(["sh", "-c", command], separators=(",", ":")))
PY
)

print_config() {
  python3 - "$RUN_ID" "$APP_NAME" "$VOLUME_NAME" "$STATE_DIR" "$JOBS_ROOT" \
    "$AGENT_KIND" "$MODEL" "$ENDPOINT" "$REASONING_EFFORT" "$CODEX_VERSION" \
    "$AGENT_SECRET_NAME" \
    "$SANDBOX_TIMEOUT_SECONDS" "$MODAL_PROFILE" "$HARBOR" "$HARBOR_COMMIT" \
    "$HARBOR_BRANCH" "$VOLUMES_JSON" "$KEEPALIVE_JSON" <<'PY'
import json
import sys

(run_id, app, volume, state, jobs, agent_kind, model, endpoint, effort,
 codex_version, auth_name, sandbox_timeout, profile, harbor, commit, branch,
 volumes, keepalive) = sys.argv[1:]
print(json.dumps({
    "run_id": run_id,
    "app_name": app,
    "volume_name": volume,
    "state_dir": state,
    "jobs_root": jobs,
    "agent_kind": agent_kind,
    "model": model,
    "endpoint": endpoint or None,
    "reasoning_effort": effort,
    "codex_version": codex_version if agent_kind == "codex" else None,
    "automatic_stop": False,
    "sandbox_timeout_seconds": int(sandbox_timeout),
    "sandbox_timeout_role": "infrastructure_orphan_backstop",
    "modal_profile": profile,
    "harbor_path": harbor,
    "harbor_commit": commit,
    "harbor_branch": branch,
    "volumes": json.loads(volumes),
    "keepalive": json.loads(keepalive),
    "auth": f"{auth_name}=[configured]",
    "launch": False,
}, indent=2, sort_keys=True))
PY
}

if ((DRY_RUN)); then
  print_config
  exit 0
fi

if [[ -e "$STATE_DIR" || -e "$SECRET_DIR" ]]; then
  echo "run ID already exists: $RUN_ID" >&2
  exit 1
fi

umask 077
mkdir -p "$STATE_DIR" "$JOBS_ROOT" "$SECRET_DIR"
printf '%s=%s\n' "$AGENT_SECRET_NAME" "$AGENT_SECRET" >"$ENV_FILE"
if [[ -n "$ENDPOINT" ]]; then
  printf 'OPENAI_BASE_URL=%s\n' "$ENDPOINT" >>"$ENV_FILE"
fi
openssl rand -hex 32 >"$PASSWORD_FILE"
chmod 0600 "$ENV_FILE" "$PASSWORD_FILE"

python3 - "$STATE_DIR/run.json" "$RUN_ID" "$APP_NAME" "$VOLUME_NAME" \
  "$STATE_DIR" "$JOBS_ROOT" "$SECRET_DIR" "$MODAL_PROFILE" \
  "$AGENT_KIND" "$MODEL" "$ENDPOINT" "$REASONING_EFFORT" "$CODEX_VERSION" \
  "$SANDBOX_TIMEOUT_SECONDS" "$DEPLOY_DEBOUNCE_SECONDS" "$HARBOR" \
  "$HARBOR_COMMIT" "$HARBOR_BRANCH" <<'PY'
import datetime
import json
import os
import pathlib
import sys

(path, run_id, app, volume, state, jobs, secrets, profile, agent_kind, model,
 endpoint, effort, codex_version, sandbox_timeout, debounce, harbor, commit,
 branch) = sys.argv[1:]
payload = {
    "schema_version": 2,
    "run_id": run_id,
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "app_name": app,
    "volume_name": volume,
    "state_dir": state,
    "jobs_root": jobs,
    "expected_job_path": str(pathlib.Path(jobs) / run_id),
    "secret_dir": secrets,
    "modal_profile": profile,
    "agent_kind": agent_kind,
    "model": model,
    "endpoint": endpoint or None,
    "reasoning_effort": effort,
    "codex_version": codex_version if agent_kind == "codex" else None,
    "automatic_stop": False,
    "sandbox_timeout_seconds": int(sandbox_timeout),
    "sandbox_timeout_role": "infrastructure_orphan_backstop",
    "deploy_debounce_seconds": int(debounce),
    "harbor_path": harbor,
    "harbor_commit": commit,
    "harbor_branch": branch,
    "task_path": "/data/qwop-bench/challenge/g1-sprint-100m-lane",
    "site_dir": "/data/qwop-bench/sprint-web",
    "vercel_project_id": "prj_dgvTovRNwdSDcefYmo6oXfju9M3p",
    "vercel_org_id": "team_SNgoAcFfHYXYdUIXhj16bGek",
    "vercel_scope": "alienkevins-projects",
    "production_alias": "https://g1-sprint.vercel.app",
}
target = pathlib.Path(path)
tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, target)
PY

if [[ "$AGENT_KIND" == "claude-code" ]]; then
  PROMPT_TEMPLATE="$ROOT/runs/claude-code-goal.j2"
  AGENT_HARBOR_ARGS=()
else
  PROMPT_TEMPLATE="$ROOT/runs/codex-goal.j2"
  AGENT_HARBOR_ARGS=(
    --ak "version=$CODEX_VERSION"
    --ae "BASH_ENV=/opt/qwop-agent-shell-env.sh"
    --ae "QWOP_AGENT_KIND=codex"
    --ae "QWOP_RUNTIME_DIR=/run"
    --ae "QWOP_AGENT_LOG_DIR=/logs/agent"
  )
fi

unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN \
  CLAUDE_FORCE_OAUTH OPENAI_API_KEY OPENAI_BASE_URL HARBOR_API_KEY AGENT_SECRET
export MODAL_PROFILE
python3 -m modal volume create --version 1 "$VOLUME_NAME"
python3 -m modal volume put "$VOLUME_NAME" "$PASSWORD_FILE" \
  "runs/$RUN_ID/secrets/restic-password"
python3 -m modal volume put "$VOLUME_NAME" "$STATE_DIR/run.json" \
  "runs/$RUN_ID/state/run.json"

if ((START_MONITOR)); then
  nohup python3 "$CONTROL" monitor --run-id "$RUN_ID" \
    >"$STATE_DIR/monitor.log" 2>&1 </dev/null &
  printf '%s\n' "$!" >"$STATE_DIR/monitor.pid"
fi

printf '%s\n' "$$" >"$STATE_DIR/harbor.pid"
printf 'run_id=%s\nagent_kind=%s\nstate=%s\napp=%s\nvolume=%s\n' \
  "$RUN_ID" "$AGENT_KIND" "$STATE_DIR" "$APP_NAME" "$VOLUME_NAME"

cd "$HARBOR"
exec run-heavy "$UV" run --frozen --extra modal harbor run \
  --path "$TASK" \
  --agent "$AGENT_KIND" --model "$MODEL" \
  --ak "prompt_template_path=$PROMPT_TEMPLATE" \
  --ak "reasoning_effort=$REASONING_EFFORT" \
  "${AGENT_HARBOR_ARGS[@]}" \
  --env modal -n 1 -y \
  --job-name "$RUN_ID" \
  --jobs-dir "$JOBS_ROOT" \
  --env-file "$ENV_FILE" \
  --ek "app_name=$APP_NAME" \
  --ek "volumes=$VOLUMES_JSON" \
  --ek "labels=$LABELS_JSON" \
  --ek "sandbox_timeout_secs=$SANDBOX_TIMEOUT_SECONDS" \
  --ek "keepalive=$KEEPALIVE_JSON" \
  "${EXTRA[@]}"
