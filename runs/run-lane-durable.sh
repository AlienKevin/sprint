#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPRINT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
HARBOR="${HARBOR_PATH:-$ROOT/harbor}"
HARBOR_COMMIT=5e5ddfcab7e40746d887b220e1280b2dae747a94
HARBOR_BRANCH=continuous-verification
UV="${UV:-$(command -v uv || true)}"
if [[ -z "$UV" && -x /home/ubuntu/.local/bin/uv ]]; then
  UV=/home/ubuntu/.local/bin/uv
fi
CONTROL="$ROOT/runs/ops/sprintctl.py"
TASK="$ROOT/challenge/g1-sprint-100m-lane"
MODAL_PROFILE=${MODAL_PROFILE:-kevinli020508}
SANDBOX_TIMEOUT_SECONDS=86400
DEPLOY_DEBOUNCE_SECONDS=300
# These exact CLI versions are baked into the task image. Harbor verifies them
# locally during offline agent setup and skips installation.
CODEX_VERSION=${CODEX_VERSION:-0.147.0}
# The optional Claude Code adapter remains pinned for reproducibility even
# though the active comparison uses Codex for both model families.
CLAUDE_VERSION=${CLAUDE_VERSION:-2.1.220}
BAKED_CODEX_VERSION=0.147.0
BAKED_CLAUDE_VERSION=2.1.220
# Hold one dedicated A10G for the run's lifetime instead of spawning a worker
# per training job. Gives each arm its own GPU at all times (fair comparison)
# and survives Modal preempting it, at the cost of paying for an idle GPU
# between jobs. Off by default. See runs/ops/gpu_worker.ensure_standing_sandbox.
STANDING_GPU=${STANDING_GPU:-0}
BATCH_ID=${SPRINT_BATCH_ID:-}
RUN_ID=""
AGENT_KIND=claude-code
MODEL=""
ENDPOINT=""
REASONING_EFFORT=""
PROMPT_TEMPLATE_OVERRIDE=""
DRY_RUN=0
START_MONITOR=1
SUPERVISED_LAUNCH=0

usage() {
  cat <<'EOF'
Usage: runs/run-lane-durable.sh [options]

Options:
  --run-id ID                Explicit unique run ID.
  --agent-kind KIND          claude-code or codex (default claude-code).
  --model MODEL              Agent model; required for codex.
  --endpoint HTTPS_URL       Optional Codex API endpoint (no credentials/query).
  --reasoning-effort VALUE   Agent reasoning effort.
  --codex-version VERSION    Pin @openai/codex npm version (codex only; default 0.147.0).
  --prompt-template PATH     Goal template under this repository's runs directory.
  --standing-gpu             Hold a dedicated A10G for the whole run.
  --dry-run                  Print redacted configuration; launch nothing.
  --no-monitor               Do not start the host monitor automatically.
  --supervised-launch        Create or resume this run on the same Volume.
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
    --prompt-template) PROMPT_TEMPLATE_OVERRIDE=${2:?}; shift 2 ;;
    --standing-gpu) STANDING_GPU=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-monitor) START_MONITOR=0; shift ;;
    --supervised-launch) SUPERVISED_LAUNCH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      (($# == 0)) || {
        echo "extra Harbor arguments are disabled by the evaluation security policy" >&2
        exit 2
      }
      break
      ;;
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
if [[ -n "$BATCH_ID" && ! "$BATCH_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,48}$ ]]; then
  echo "SPRINT_BATCH_ID must be 3-49 safe filename characters" >&2
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
if [[ -n "$PROMPT_TEMPLATE_OVERRIDE" ]]; then
  PROMPT_TEMPLATE_OVERRIDE=$(realpath "$PROMPT_TEMPLATE_OVERRIDE")
  [[ "$PROMPT_TEMPLATE_OVERRIDE" == "$ROOT/runs/"* && -f "$PROMPT_TEMPLATE_OVERRIDE" ]] || {
    echo "--prompt-template must be a file under $ROOT/runs" >&2
    exit 2
  }
fi
if [[ -n "$ENDPOINT" ]]; then
  MODEL_API_HOST=$(python3 - "$ENDPOINT" <<'PY'
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
print(url.hostname.lower())
PY
  )
elif [[ "$AGENT_KIND" == "claude-code" ]]; then
  MODEL_API_HOST=api.anthropic.com
else
  MODEL_API_HOST=api.openai.com
fi
case "$MODEL_API_HOST" in
  api.anthropic.com|api.deepseek.com|api.openai.com|openrouter.ai) ;;
  *)
    echo "model endpoint host is not in the audited egress allowlist: $MODEL_API_HOST" >&2
    exit 2
    ;;
esac
if [[ "$CODEX_VERSION" != "$BAKED_CODEX_VERSION" ]]; then
  echo "Codex $CODEX_VERSION is not baked into the offline image (expected $BAKED_CODEX_VERSION)" >&2
  exit 2
fi
if [[ "$CLAUDE_VERSION" != "$BAKED_CLAUDE_VERSION" ]]; then
  echo "Claude Code $CLAUDE_VERSION is not baked into the offline image (expected $BAKED_CLAUDE_VERSION)" >&2
  exit 2
fi
[[ -n "$UV" && -x "$UV" ]] || {
  echo "uv is required (set UV to its absolute executable path)" >&2
  exit 1
}
[[ -d "$HARBOR/src/harbor" ]] || { echo "missing vendored Harbor: $HARBOR" >&2; exit 1; }
[[ -f "$HARBOR/.sprint-upstream-commit" ]] || {
  echo "missing vendored Harbor provenance: $HARBOR/.sprint-upstream-commit" >&2
  exit 1
}
actual_commit=$(tr -d '[:space:]' <"$HARBOR/.sprint-upstream-commit")
[[ "$actual_commit" == "$HARBOR_COMMIT" ]] || {
  echo "vendored Harbor $actual_commit does not match pin $HARBOR_COMMIT" >&2
  exit 1
}

export SPRINT_SHARED_STATE_DIR="$ROOT/runs/ops/blind-verifier"

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

STATE_DIR="$ROOT/runs/ops/$RUN_ID"
APP_NAME="sprint-$RUN_ID"
TRAINING_APP_NAME="sprint-$RUN_ID-training"
VERIFIER_APP_NAME="sprint-$RUN_ID-verifier"
VOLUME_NAME="sprint-$RUN_ID-volume"
JOBS_ROOT="$STATE_DIR/harbor-jobs"
SECRET_DIR="/data/sprint-run-secrets/$RUN_ID"
PASSWORD_FILE="$SECRET_DIR/restic-password"
ENV_FILE="$SECRET_DIR/harbor.env"
REMOTE_PASSWORD="/durable/runs/$RUN_ID/secrets/restic-password"
CPU_LAUNCH_ATTEMPT=1

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
    {
        "sprint.run_id": sys.argv[1],
        "sprint.agent_kind": sys.argv[2],
        "sprint.role": "cpu-agent",
    },
    separators=(",", ":"),
))
PY
)
VERIFIER_LABELS_JSON=$(python3 - "$RUN_ID" "$AGENT_KIND" <<'PY'
import json
import sys
print(json.dumps(
    {
        "sprint.run_id": sys.argv[1],
        "sprint.agent_kind": sys.argv[2],
        "sprint.role": "verifier-gpu",
    },
    separators=(",", ":"),
))
PY
)
make_keepalive_json() {
python3 - "$RUN_ID" "$AGENT_KIND" "$REMOTE_PASSWORD" \
  "$CPU_LAUNCH_ATTEMPT" <<'PY'
import json
import shlex
import sys

run_id, agent_kind, password, cpu_attempt = sys.argv[1:]
watcher = [
    "/opt/sprint-snapshot-loop.sh",
    "--run-id", run_id,
    "--agent-kind", agent_kind,
    "--password-file", password,
]
# Start GPU/CPU telemetry before the durable snapshot loop. Snapshot-loop also
# starts telemetry idempotently; this covers the sleep-infinity fallback path.
telemetry = (
    "umask 077; mkdir -p /logs/artifacts/telemetry /run; "
    "printf '%s\\n' "
    + shlex.quote(run_id)
    + " > /run/sprint-run-id; "
    "printf '%s\\n' "
    + shlex.quote(f"/durable/runs/{run_id}/gpu-jobs")
    + " > /run/sprint-gpu-jobs-root; "
    "printf '%s\\n' cpu-agent > /run/sprint-role; "
    "if [ -x /opt/sprint-telemetry.sh ]; then "
    "SPRINT_RUN_ID="
    + shlex.quote(run_id)
    + " SPRINT_CPU_LAUNCH_ATTEMPT="
    + shlex.quote(cpu_attempt)
    + " /opt/sprint-telemetry.sh --role cpu-agent --run-id "
    + shlex.quote(run_id)
    + " --out-dir /logs/artifacts/telemetry "
    "--interval-seconds ${SPRINT_TELEMETRY_INTERVAL:-20} "
    "--pidfile /run/sprint-telemetry.pid || true; "
    "fi; "
)
command = (
    telemetry
    + "export SPRINT_CPU_LAUNCH_ATTEMPT="
    + shlex.quote(cpu_attempt)
    + "; if [ -x /opt/sprint-snapshot-loop.sh ]; then exec "
    + " ".join(shlex.quote(part) for part in watcher)
    + "; else exec sleep infinity; fi"
)
print(json.dumps(["sh", "-c", command], separators=(",", ":")))
PY
}
KEEPALIVE_JSON=$(make_keepalive_json)

print_config() {
  python3 - "$RUN_ID" "$APP_NAME" "$TRAINING_APP_NAME" "$VERIFIER_APP_NAME" \
    "$VOLUME_NAME" "$STATE_DIR" "$JOBS_ROOT" \
    "$AGENT_KIND" "$MODEL" "$ENDPOINT" "$REASONING_EFFORT" "$CODEX_VERSION" \
    "$AGENT_SECRET_NAME" \
    "$SANDBOX_TIMEOUT_SECONDS" "$MODEL_API_HOST" "$MODAL_PROFILE" "$HARBOR" "$HARBOR_COMMIT" \
    "$HARBOR_BRANCH" "$VOLUMES_JSON" "$KEEPALIVE_JSON" <<'PY'
import json
import sys

(run_id, app, training_app, verifier_app, volume, state, jobs, agent_kind, model, endpoint, effort,
 codex_version, auth_name, sandbox_timeout, model_api_host, profile, harbor, commit, branch,
 volumes, keepalive) = sys.argv[1:]
payload = {
    "run_id": run_id,
    "app_name": app,
    "training_app_name": training_app,
    "verifier_app_name": verifier_app,
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
    "sandbox_timeout_role": "modal_maximum_lifetime",
    "cpu_agent": {
        "physical_cpu_cores": 4,
        "vcpus_equivalent": 8,
        "memory_mb": 16384,
        "gpus": 0,
    },
    "cgroup_telemetry_required": True,
    "agent_network_policy": "model-api-only",
    "agent_allowed_host": model_api_host,
    "hosted_model_tools_policy": "disabled" if agent_kind == "codex" else None,
    "service_tier": (
        "default"
        if agent_kind == "codex"
        and model.split("/", 1)[-1] in {"gpt-5.6-terra", "gpt-5.6-luna"}
        else None
    ),
    "usage_audit_required": (
        agent_kind == "codex"
        and model.split("/", 1)[-1]
        in {"gpt-5.6-terra", "gpt-5.6-luna", "deepseek-v4-flash"}
    ),
    "modal_profile": profile,
    "harbor_path": harbor,
    "harbor_commit": commit,
    "harbor_branch": branch,
    "volumes": json.loads(volumes),
    "keepalive": json.loads(keepalive),
    "auth": f"{auth_name}=[configured]",
    "launch": False,
}
if endpoint and "api.deepseek.com" in endpoint:
    payload["sprint_codex_provider"] = "deepseek"
    payload["codex_wire_api"] = "responses"
    payload["codex_model_catalog"] = "deepseek-official-models.json"
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if ((DRY_RUN)); then
  print_config
  exit 0
fi

# Real evaluations only start from image definitions that were eagerly built
# and exercised on Modal. The content hashes make this fail closed after any
# agent/training/verifier image change rather than paying a surprise lazy build
# during the first model's run.
python3 "$ROOT/runs/ops/check_modal_image_warmup.py"
read -r AGENT_TRAINING_IMAGE_ID VERIFIER_IMAGE_ID < <(
  python3 - "$ROOT/runs/ops/modal-image-warmup.json" <<'PY'
import json
import re
import sys

payload = json.load(open(sys.argv[1]))
agent = payload["contexts"]["agent_training"]["image_id"]
verifier = payload["contexts"]["verifier"]["image_id"]
if not all(isinstance(value, str) and re.fullmatch(r"im-[A-Za-z0-9]+", value)
           for value in (agent, verifier)):
    raise SystemExit("warm-up manifest contains an invalid Modal image ID")
print(agent, verifier)
PY
)

umask 077
RESUMING=0
if [[ -f "$STATE_DIR/run.json" ]]; then
  if (( ! SUPERVISED_LAUNCH )); then
    echo "run ID already exists: $RUN_ID" >&2
    exit 1
  fi
  RESUMING=1
  CPU_LAUNCH_ATTEMPT=$(python3 - "$STATE_DIR/run.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
print(int(payload.get("cpu_launch_attempt") or 1) + 1)
PY
)
  JOBS_ROOT="$STATE_DIR/cpu-attempts/$(printf '%02d' "$CPU_LAUNCH_ATTEMPT")/harbor-jobs"
  [[ -f "$PASSWORD_FILE" && -f "$ENV_FILE" ]] || {
    echo "cannot resume without run secret files: $RUN_ID" >&2
    exit 1
  }
elif [[ -e "$SECRET_DIR" || -e "$JOBS_ROOT" ]]; then
  echo "partial run state already exists: $RUN_ID" >&2
  exit 1
fi
# The Modal keepalive is the snapshot watcher's process environment, separate
# from Harbor's agent-exec environment. Rebuild it after determining the
# generation so a replacement cannot reuse attempt 1's first-seen sentinel.
KEEPALIVE_JSON=$(make_keepalive_json)

mkdir -p "$STATE_DIR" "$JOBS_ROOT" "$SECRET_DIR"
printf '%s=%s\n' "$AGENT_SECRET_NAME" "$AGENT_SECRET" >"$ENV_FILE"
if [[ -n "$ENDPOINT" ]]; then
  printf 'OPENAI_BASE_URL=%s\n' "$ENDPOINT" >>"$ENV_FILE"
fi
# Official DeepSeek Codex provider+catalog (not Harbor openai_base_url alone).
# Luna / default OpenAI endpoints leave this unset.
if [[ "$ENDPOINT" == *api.deepseek.com* ]]; then
  printf 'SPRINT_CODEX_PROVIDER=deepseek\n' >>"$ENV_FILE"
fi
if (( ! RESUMING )); then
  openssl rand -hex 32 >"$PASSWORD_FILE"
fi
chmod 0600 "$ENV_FILE" "$PASSWORD_FILE"

# The host needs Modal credentials to create the sandbox, but the untrusted
# agent must never receive a cloud control-plane credential. Keep its env file
# on an explicit allowlist rather than relying on ambient inheritance.
python3 "$ROOT/runs/ops/validate_agent_env.py" "$ENV_FILE" "$AGENT_SECRET_NAME"

if [[ "$AGENT_KIND" == "claude-code" ]]; then
  PROMPT_TEMPLATE="${PROMPT_TEMPLATE_OVERRIDE:-$ROOT/runs/claude-code-goal.j2}"
else
  PROMPT_TEMPLATE="${PROMPT_TEMPLATE_OVERRIDE:-$ROOT/runs/codex-goal.j2}"
fi

python3 - "$STATE_DIR/run.json" "$RUN_ID" "$APP_NAME" "$TRAINING_APP_NAME" \
  "$VERIFIER_APP_NAME" "$VOLUME_NAME" \
  "$STATE_DIR" "$JOBS_ROOT" "$SECRET_DIR" "$MODAL_PROFILE" \
  "$AGENT_KIND" "$MODEL" "$ENDPOINT" "$REASONING_EFFORT" "$CODEX_VERSION" \
  "$SANDBOX_TIMEOUT_SECONDS" "$DEPLOY_DEBOUNCE_SECONDS" "$HARBOR" \
  "$HARBOR_COMMIT" "$HARBOR_BRANCH" "$RESUMING" "$CPU_LAUNCH_ATTEMPT" \
  "$SUPERVISED_LAUNCH" "$STANDING_GPU" "$MODEL_API_HOST" \
  "$PROMPT_TEMPLATE" "$ROOT/runs/ops/modal-image-warmup.json" "$ROOT" "$BATCH_ID" <<'PY'
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import sys

(path, run_id, app, training_app, verifier_app, volume, state, jobs, secrets, profile, agent_kind, model,
 endpoint, effort, codex_version, sandbox_timeout, debounce, harbor, commit,
 branch, resuming, cpu_attempt, supervised, standing_gpu_flag,
 model_api_host, prompt_template, warmup_manifest_path, root, batch_id) = sys.argv[1:]
standing_gpu = standing_gpu_flag == "1"
target = pathlib.Path(path)
root_path = pathlib.Path(root)
task_root = root_path / "challenge/g1-sprint-100m-lane"
prompt_path = pathlib.Path(prompt_template)
warmup_manifest = json.loads(pathlib.Path(warmup_manifest_path).read_text())

def sha256_file(file_path):
    return hashlib.sha256(pathlib.Path(file_path).read_bytes()).hexdigest()

lock_fd = os.open(target.with_name("run.json.lock"), os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(lock_fd, fcntl.LOCK_EX)
base = {
    "schema_version": 2,
    "run_id": run_id,
    "batch_id": batch_id or None,
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "app_name": app,
    "training_app_name": training_app,
    "verifier_app_name": verifier_app,
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
    "resolved_model_version": (
        "DeepSeek-V4-Flash-0731"
        if model.split("/", 1)[-1] == "deepseek-v4-flash"
        else model.split("/", 1)[-1]
    ),
    "codex_version": codex_version if agent_kind == "codex" else None,
    "automatic_stop": False,
    "sandbox_timeout_seconds": int(sandbox_timeout),
    "sandbox_timeout_role": "modal_maximum_lifetime",
    "deploy_debounce_seconds": int(debounce),
    "harbor_path": harbor,
    "harbor_commit": commit,
    "harbor_branch": branch,
    "task_path": str(task_root),
    "evaluation_provenance": {
        "prompt_template_path": prompt_template,
        "prompt_template_sha256": sha256_file(prompt_path),
        "task_toml_sha256": sha256_file(task_root / "task.toml"),
        "agent_training_context_sha256": warmup_manifest["contexts"]["agent_training"]["sha256"],
        "agent_training_image_id": warmup_manifest["contexts"]["agent_training"]["image_id"],
        "verifier_context_sha256": warmup_manifest["contexts"]["verifier"]["sha256"],
        "verifier_image_id": warmup_manifest["contexts"]["verifier"]["image_id"],
        "image_warmup_completed_at_epoch_s": warmup_manifest["completed_at_epoch_s"],
    },
    "site_dir": str(root_path / "sprint-web"),
    "vercel_project_id": "prj_dgvTovRNwdSDcefYmo6oXfju9M3p",
    "vercel_org_id": "team_SNgoAcFfHYXYdUIXhj16bGek",
    "vercel_scope": "alienkevins-projects",
    "production_alias": "https://g1-sprint.vercel.app",
    # Agent sandbox is CPU-only (task.toml gpus=0); host monitor dispatches
    # preemptible A10G workers for sprint-gpu-train jobs on the same volume.
    "cpu_agent_gpu_worker": True,
    "standing_gpu_worker": standing_gpu,
    "agent_cpu_instances": 1,
    "training_max_concurrent_per_run": 1,
    "agent_gpus": 0,
    "gpu_worker_gpu": "A10G",
    "gpu_worker_gpus_per_job": 1,
    "resource_contract": {
        "cpu_agent": {
            "instances": 1,
            "physical_cpu_cores": 4,
            "vcpus_equivalent": 8,
            "memory_mb": 16384,
            "gpus": 0,
        },
        "training_worker": {
            "max_concurrent": 1,
            "physical_cpu_cores": 8,
            "vcpus_equivalent": 16,
            "memory_mb": 32768,
            "gpu_count": 1,
            "gpu_type": "A10G",
        },
        "verifier": {
            "max_concurrent": 1,
            "physical_cpu_cores": 8,
            "vcpus_equivalent": 16,
            "memory_mb": 32768,
            "gpu_count": 1,
            "gpu_type": "A10G",
        },
    },
    "agent_cloud_control_plane_credentials_injected": False,
    "agent_network_policy": "model-api-only",
    "agent_allowed_host": model_api_host,
    "gpu_worker_network_policy": "no-network",
    "verifier_network_policy": "no-network",
    # Finalization is gated on a web-safe single-clock export containing the
    # trace, CPU/GPU samples, lifecycle, and every submitted artifact.
    "unified_timeline_required": True,
    "modal_billing_required": True,
    "cgroup_telemetry_required": True,
    # OpenAI cost is accepted only from Harbor's per-request, checksummed usage
    # audit. Aggregate cached/uncached counters cannot recover long-context or
    # cache-write pricing correctly.
    "usage_audit_required": agent_kind == "codex" and model.split("/", 1)[-1] in {"gpt-5.6-terra", "gpt-5.6-luna", "deepseek-v4-flash"},
    "hosted_model_tools_policy": "disabled" if agent_kind == "codex" else None,
    "timeline_bucket_seconds": 60,
    "telemetry_cpu_max_gap_seconds": 45,
    "telemetry_gpu_max_gap_seconds": 45,
    "telemetry_resource_roles": ["cpu-agent", "training-gpu", "verifier-gpu"],
    # Per-run immutable queues feed one cross-process trusted verifier lease.
    # No verifier state is returned to an agent, so shared queue latency cannot
    # influence policy development.
    "scoring_queue_scope": "central_blind_per_run_queues",
    "scoring_queue_key": run_id,
    "scoring_max_concurrent": 1,
    "scoring_global_max_concurrent": 1,
    "scoring_feedback_policy": "sealed_until_agent_exit",
    "scoring_drain_policy": "all_accepted_submissions",
    "scoring_deduplication_key": "task_fingerprint_plus_policy_sha256",
    "evaluation_result_policy": "all_blind_submissions",
    "verifier_cost_attribution": "measurement_overhead_separate_from_agent_cost",
    "cpu_supervised": supervised == "1",
    "cpu_launch_attempt": int(cpu_attempt),
}
if resuming == "1":
    payload = json.loads(target.read_text())
    expected = {
        "run_id": run_id,
        "app_name": app,
        "training_app_name": training_app,
        "verifier_app_name": verifier_app,
        "volume_name": volume,
        "agent_kind": agent_kind,
        "model": model,
        "endpoint": endpoint or None,
        "reasoning_effort": effort,
        "codex_version": codex_version if agent_kind == "codex" else None,
        "harbor_commit": commit,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"resume configuration mismatch: {mismatches}")
    recorded_provenance = payload.get("evaluation_provenance") or {}
    current_provenance = base["evaluation_provenance"]
    provenance_keys = (
        "prompt_template_sha256",
        "task_toml_sha256",
        "agent_training_context_sha256",
        "agent_training_image_id",
        "verifier_context_sha256",
        "verifier_image_id",
    )
    provenance_mismatches = {
        key: (recorded_provenance.get(key), current_provenance.get(key))
        for key in provenance_keys
        if recorded_provenance.get(key) != current_provenance.get(key)
    }
    if provenance_mismatches:
        raise SystemExit(
            f"resume evaluation provenance mismatch: {provenance_mismatches}"
        )
    history = list(payload.get("cpu_launch_history") or [])
    history.append({
        "attempt": int(cpu_attempt),
        "launched_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "jobs_root": jobs,
    })
    payload.update({
        "jobs_root": jobs,
        "expected_job_path": str(pathlib.Path(jobs) / run_id),
        "cpu_launch_attempt": int(cpu_attempt),
        "cpu_launch_history": history,
        "cpu_supervised": supervised == "1",
        # Refresh security metadata when an older durable run is resumed under
        # the current launcher. The actual policy comes from task.toml and the
        # Harbor arguments below; keeping run.json current makes the audit trail
        # accurately describe the resumed sandbox.
        "agent_cloud_control_plane_credentials_injected": False,
        "agent_network_policy": "model-api-only",
        "agent_allowed_host": model_api_host,
        "gpu_worker_network_policy": "no-network",
        "verifier_network_policy": "no-network",
        "modal_billing_required": True,
        "cgroup_telemetry_required": True,
        "resource_contract": base["resource_contract"],
        "telemetry_gpu_max_gap_seconds": 45,
        "telemetry_cpu_max_gap_seconds": 45,
        "telemetry_resource_roles": ["cpu-agent", "training-gpu", "verifier-gpu"],
    })
else:
    payload = base
    payload["cpu_launch_history"] = [{
        "attempt": int(cpu_attempt),
        "launched_at": payload["created_at"],
        "jobs_root": jobs,
    }]
tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, target)
PY

SHARED_AGENT_ENV=(
  --ae "SPRINT_RUN_ID=$RUN_ID"
  --ae "SPRINT_GPU_JOBS_ROOT=/durable/runs/$RUN_ID/gpu-jobs"
  --ae "SPRINT_CPU_LAUNCH_ATTEMPT=$CPU_LAUNCH_ATTEMPT"
  --ae "SPRINT_MODEL=$MODEL"
  --ae "SPRINT_SCORING_QUEUE_KEY=$RUN_ID"
  --ae "SPRINT_REQUESTED_CPU_CORES=4"
  --ae "SPRINT_REQUESTED_MEMORY_MIB=16384"
)
if [[ "$AGENT_KIND" == "claude-code" ]]; then
  AGENT_HARBOR_ARGS=(
    --ak "version=$CLAUDE_VERSION"
    # The codex branch captures a full agent transcript to /logs/agent; this
    # branch did not, so a claude-code arm that dies leaves nothing to read.
    # lane-opus-20260803T081650Z exited ~8 min after launch, repeatedly, and
    # could not be diagnosed at all for exactly this reason -- while Luna's
    # codex arm produced a 2.9 MB codex.txt that made last week's root cause
    # analysis possible. Two arms with different forensic capability is itself
    # a fairness problem.
    --ae "SPRINT_AGENT_KIND=claude-code"
    --ae "SPRINT_AGENT_LOG_DIR=/logs/agent"
    --ae "SPRINT_RUNTIME_DIR=/run"
    "${SHARED_AGENT_ENV[@]}"
  )
else
  AGENT_HARBOR_ARGS=(
    --ak "version=$CODEX_VERSION"
    # Hosted search has a separate per-call price and would also bypass the
    # benchmark's no-general-internet policy through the provider.
    --ak "web_search=disabled"
    --ae "BASH_ENV=/opt/sprint-agent-shell-env.sh"
    --ae "SPRINT_AGENT_KIND=codex"
    --ae "SPRINT_RUNTIME_DIR=/run"
    --ae "SPRINT_AGENT_LOG_DIR=/logs/agent"
    "${SHARED_AGENT_ENV[@]}"
  )
  if [[ "$ENDPOINT" == *api.deepseek.com* ]]; then
    AGENT_HARBOR_ARGS+=(--ae "SPRINT_CODEX_PROVIDER=deepseek")
  fi
  if [[ "${MODEL#*/}" == "gpt-5.6-terra" || "${MODEL#*/}" == "gpt-5.6-luna" ]]; then
    # Pin standard pricing. Leaving this unset lets Codex/project defaults pick
    # another service tier, which cannot be reconstructed from token counts.
    AGENT_HARBOR_ARGS+=(--ak "service_tier=default")
  fi
fi

unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN \
  CLAUDE_FORCE_OAUTH OPENAI_API_KEY OPENAI_BASE_URL HARBOR_API_KEY AGENT_SECRET
export MODAL_PROFILE
if (( ! RESUMING )); then
  python3 -m modal volume create --version 1 "$VOLUME_NAME"
  python3 -m modal volume put "$VOLUME_NAME" "$PASSWORD_FILE" \
    "runs/$RUN_ID/secrets/restic-password"
else
  # A resume against a vanished Volume is unrecoverable: every durable path
  # (gpu-jobs, checkpoints, telemetry, restic snapshots) lives on it, and this
  # branch never recreates it. Left unchecked the launcher just exits 1 and the
  # supervisor relaunches into the same wall until max_restarts burns out, which
  # is exactly how lane-luna-20260803T035544Z died silently after 15 restarts
  # (see runs/BAKEOFF_20260803_FINDINGS.md D0). Fail loudly and immediately
  # instead, so the operator sees the cause rather than a restart-budget
  # exhaustion several hours later.
  if ! python3 -m modal volume list 2>/dev/null | grep -qF "$VOLUME_NAME"; then
    echo "FATAL: resume requested but Volume '$VOLUME_NAME' does not exist." >&2
    echo "       Durable state for $RUN_ID is gone; this run cannot be resumed." >&2
    echo "       Relaunch as a NEW run id (fresh volume) instead of resuming." >&2
    printf '{"at":"%s","run_id":"%s","reason":"durable_volume_missing","volume":"%s"}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$VOLUME_NAME" \
      >> "$STATE_DIR/controller-errors.jsonl" 2>/dev/null || true
    exit 78   # EX_CONFIG: unrecoverable configuration, not a transient failure
  fi
fi
# Resume relaunches rewrite the same remote path; Modal requires --force.
python3 -m modal volume put -f "$VOLUME_NAME" "$STATE_DIR/run.json" \
  "runs/$RUN_ID/state/run.json"

if ((START_MONITOR)); then
  monitor_alive=0
  if [[ -f "$STATE_DIR/monitor.pid" ]]; then
    monitor_pid=$(tr -dc '0-9' <"$STATE_DIR/monitor.pid" || true)
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
      monitor_alive=1
    fi
  fi
  if (( ! monitor_alive )); then
    nohup python3 "$CONTROL" monitor --run-id "$RUN_ID" \
      >>"$STATE_DIR/monitor.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$STATE_DIR/monitor.pid"
  fi
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
  --allow-agent-host "$MODEL_API_HOST" \
  --env modal -n 1 -y \
  --job-name "$RUN_ID" \
  --jobs-dir "$JOBS_ROOT" \
  --env-file "$ENV_FILE" \
  --ek "app_name=$APP_NAME" \
  --ek "modal_image_id=$AGENT_TRAINING_IMAGE_ID" \
  --ek "verifier_app_name=$VERIFIER_APP_NAME" \
  --ek "verifier_image_id=$VERIFIER_IMAGE_ID" \
  --ek "verifier_labels=$VERIFIER_LABELS_JSON" \
  --ek "volumes=$VOLUMES_JSON" \
  --ek "labels=$LABELS_JSON" \
  --ek "sandbox_timeout_secs=$SANDBOX_TIMEOUT_SECONDS" \
  --ek "keepalive=$KEEPALIVE_JSON"
