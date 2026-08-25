#!/usr/bin/env bash
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
SOURCE_ROOT="$ROOT"
HARBOR="${HARBOR_PATH:-$SOURCE_ROOT/harbor}"
HARBOR_COMMIT=dafb1387151e1c32702963d44fe6c3cea66cf8cb
HARBOR_BRANCH=continuous-verification
UV="${UV:-$(command -v uv || true)}"
if [[ -z "$UV" && -x /home/ubuntu/.local/bin/uv ]]; then
  UV=/home/ubuntu/.local/bin/uv
fi
CONTROL="$ROOT/event_runtime/control/run.py"
MODAL_PROFILE=${MODAL_PROFILE:-kevinli020508}
SANDBOX_TIMEOUT_SECONDS=86400
DEPLOY_DEBOUNCE_SECONDS=300
# These exact CLI versions are baked into the task image. Harbor verifies them
# locally during offline agent setup and skips installation.
CODEX_VERSION=${CODEX_VERSION:-0.149.1}
DEEPSEEK_HARNESS_VERSION=${DEEPSEEK_HARNESS_VERSION:-0.1.1-rc.2}
DEEPSEEK_HARNESS_SDK_VERSION=${DEEPSEEK_HARNESS_SDK_VERSION:-0.1.1rc1}
BAKED_CODEX_VERSION=0.149.1
BAKED_DEEPSEEK_HARNESS_VERSION=0.1.1-rc.2
export DEEPSEEK_HARNESS_VERSION
export DEEPSEEK_HARNESS_SDK_VERSION
# Hold one dedicated A10G for the run's lifetime instead of spawning a worker
# per training job. Gives each arm its own GPU at all times (fair comparison)
# and survives Modal preempting it, at the cost of paying for an idle GPU
# between jobs. Off by default. See event_runtime.compute.worker.
STANDING_GPU=${STANDING_GPU:-0}
BATCH_ID=${SPRINT_BATCH_ID:-}
RUN_ID=""
AGENT_KIND=codex
MODEL=""
ENDPOINT=""
REASONING_EFFORT=""
PROMPT_TEMPLATE_OVERRIDE=""
DRY_RUN=0
START_MONITOR=1
OPENAI_OPENROUTER_PRESET=${SPRINT_OPENROUTER_PRESET:-}

usage() {
  cat <<'EOF'
Usage: event_runtime/control/launch.sh [options]

Options:
  --run-id ID                Explicit unique run ID.
  --agent-kind KIND          codex or deepseek-harness.
  --model MODEL              Agent model (required).
  --endpoint HTTPS_URL       Optional Codex API endpoint (no credentials/query).
  --reasoning-effort VALUE   Agent reasoning effort.
  --codex-version VERSION    Pin @openai/codex npm version (codex only; default 0.149.1).
  --prompt-template PATH     Goal template under event_runtime/control/templates.
  --standing-gpu             Hold a dedicated A10G for the whole run.
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
    --prompt-template) PROMPT_TEMPLATE_OVERRIDE=${2:?}; shift 2 ;;
    --standing-gpu) STANDING_GPU=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-monitor) START_MONITOR=0; shift ;;
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
if [[ "$AGENT_KIND" != "codex" && "$AGENT_KIND" != "deepseek-harness" ]]; then
  echo "--agent-kind must be codex or deepseek-harness" >&2
  exit 2
fi
if [[ -z "$MODEL" ]]; then
  echo "--model is required for $AGENT_KIND" >&2
  exit 2
fi
if [[ -z "$REASONING_EFFORT" ]]; then
  if [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
    REASONING_EFFORT=max
  else
    REASONING_EFFORT=high
  fi
fi
if [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
  [[ "$MODEL" == "deepseek/deepseek-v4-flash-vision-exp" ]] || {
    echo "DeepSeek Harness is sealed to deepseek/deepseek-v4-flash-vision-exp" >&2
    exit 2
  }
  [[ "$REASONING_EFFORT" == "max" ]] || {
    echo "DeepSeek Harness benchmark reasoning effort is sealed to max" >&2
    exit 2
  }
  ENDPOINT=${ENDPOINT:-https://openrouter.ai/api/v1}
elif [[ "${MODEL#*/}" == deepseek-* ]]; then
  echo "DeepSeek models must use the pinned deepseek-harness adapter" >&2
  exit 2
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
  [[ "$PROMPT_TEMPLATE_OVERRIDE" == "$ROOT/event_runtime/control/templates/"* && -f "$PROMPT_TEMPLATE_OVERRIDE" ]] || {
    echo "--prompt-template must be under event_runtime/control/templates" >&2
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
else
  echo "--endpoint is required and must use the audited OpenRouter gateway" >&2
  exit 2
fi
case "$MODEL_API_HOST" in
  openrouter.ai) ;;
  *)
    echo "model endpoint host is not in the audited egress allowlist: $MODEL_API_HOST" >&2
    exit 2
    ;;
esac
if [[ "${MODEL#*/}" == deepseek-v4-flash* && "$MODEL_API_HOST" != "openrouter.ai" ]]; then
  echo "DeepSeek V4 Flash is locked to the audited OpenRouter endpoint" >&2
  exit 2
fi
if [[ "${MODEL#*/}" == "gpt-5.6-luna" && "$MODEL_API_HOST" != "openrouter.ai" ]]; then
  echo "GPT-5.6 Luna is locked to the audited OpenRouter/OpenAI endpoint" >&2
  exit 2
fi
if [[ "$CODEX_VERSION" != "$BAKED_CODEX_VERSION" ]]; then
  echo "Codex $CODEX_VERSION is not baked into the offline image (expected $BAKED_CODEX_VERSION)" >&2
  exit 2
fi
if [[ "$DEEPSEEK_HARNESS_VERSION" != "$BAKED_DEEPSEEK_HARNESS_VERSION" ]]; then
  echo "DeepSeek Harness $DEEPSEEK_HARNESS_VERSION is not baked into the offline image (expected $BAKED_DEEPSEEK_HARNESS_VERSION)" >&2
  exit 2
fi
[[ -n "$UV" && -x "$UV" ]] || {
  echo "uv is required (set UV to its absolute executable path)" >&2
  exit 1
}
export SPRINT_SHARED_STATE_DIR="$ROOT/runs/ops/feedback-verifier/${BATCH_ID:-standalone}"
export SPRINT_SHARED_CACHE_DIR="$ROOT/runs/ops/feedback-verifier/result-cache"

AGENT_SECRET_NAME=OPENROUTER_API_KEY
AGENT_SECRET=${OPENROUTER_API_KEY:-}
if [[ -z "$AGENT_SECRET" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
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
ENV_FILE="$SECRET_DIR/harbor.env"

WARMUP_MANIFEST_PATH="$ROOT/runs/ops/modal-image-warmup.json"
if [[ -f "$STATE_DIR/run.json" ]]; then
  echo "run ID already exists and CPU-agent resume is forbidden: $RUN_ID" >&2
  echo "use a new run ID to restart the whole trial from scratch" >&2
  exit 1
fi

# Vision Exp is available only from DeepSeek's official OpenRouter endpoint.
if [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
  if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
        && "$SPRINT_OPENROUTER_PROVIDER_ENDPOINT" != "deepseek" ]]; then
    echo "Vision Exp is locked to the official DeepSeek OpenRouter endpoint" >&2
    exit 2
  fi
  if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" ]]; then
    echo "Vision Exp official endpoint does not expose a sealed quantization" >&2
    exit 2
  fi
  MODEL=deepseek/deepseek-v4-flash-vision-exp
  export SPRINT_OPENROUTER_PROVIDER_ENDPOINT=deepseek
  unset SPRINT_OPENROUTER_QUANTIZATION

fi

# OpenAI comparison arms use the official OpenAI provider with no fallback.
# The request contract below also replaces the caller's model slug, but route
# pinning remains an independent defense against provider drift.
if [[ "$AGENT_KIND" == "codex" \
      && "$MODEL_API_HOST" == "openrouter.ai" \
      && "${MODEL#*/}" == "gpt-5.6-luna" ]]; then
  if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" \
        && "$SPRINT_OPENROUTER_PROVIDER_ENDPOINT" != "openai" ]]; then
    echo "GPT-5.6 Luna is locked to the official OpenAI endpoint" >&2
    exit 2
  fi
  if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" ]]; then
    echo "GPT-5.6 Luna official endpoint has no quantization override" >&2
    exit 2
  fi
  export SPRINT_OPENROUTER_PROVIDER_ENDPOINT=openai
  unset SPRINT_OPENROUTER_QUANTIZATION
fi

# Freeze every model-side benchmark knob that has an authoritative value. The
# inactive DeepSeek Harness path keeps its Chat Completions field names; active
# Codex arms use Responses API field names.
if [[ "$MODEL_API_HOST" == "openrouter.ai" ]]; then
  case "$AGENT_KIND:${MODEL#*/}" in
    deepseek-harness:deepseek-v4-flash-vision-exp)
      SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON='{"max_tokens":384000,"model":"deepseek/deepseek-v4-flash-vision-exp","reasoning_effort":"max","stream":true,"temperature":1.0,"top_p":0.95}'
      ;;
    codex:deepseek-v4-flash-vision-exp)
      SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON='{"max_output_tokens":384000,"model":"deepseek/deepseek-v4-flash-vision-exp","reasoning":{"effort":"max"},"temperature":1.0,"top_p":0.95}'
      ;;
    codex:gpt-5.6-luna)
      # The official OpenAI route advertises neither sampling overrides nor
      # text verbosity through OpenRouter. With require_parameters=true those
      # fields disqualify the only allowed endpoint instead of being ignored.
      SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON='{"max_output_tokens":128000,"model":"openai/gpt-5.6-luna","reasoning":{"effort":"max","summary":"auto"},"service_tier":"default"}'
      ;;
    codex:gpt-5.6-sol)
      SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON='{"max_output_tokens":128000,"model":"openai/gpt-5.6-sol","reasoning":{"effort":"max","summary":"auto"},"service_tier":"default"}'
      ;;
    *)
      SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON=''
      ;;
  esac
else
  SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON=''
fi
export SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON
if [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
  SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH=chat_completions
else
  SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH=responses
fi
export SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH

TASK_SOURCE="$SOURCE_ROOT/events/g1-100-metres"
TASK="$STATE_DIR/rendered-task"
BUDGET_CONFIG="$SOURCE_ROOT/event_runtime/control/budget.env"
[[ -f "$BUDGET_CONFIG" ]] || {
  echo "missing global event budget configuration: $BUDGET_CONFIG" >&2
  exit 1
}
# shellcheck source=/dev/null
source "$BUDGET_CONFIG"
AGENT_COST_BUDGET_USD=$(python3 - "$AGENT_COST_BUDGET_USD" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0:
    raise SystemExit("AGENT_COST_BUDGET_USD must be a positive finite number")
print(format(value, "g"))
PY
)
read -r AGENT_COST_SHUTDOWN_RESERVE_USD MINIMUM_SAFE_SHUTDOWN_RESERVE_USD < <(python3 - \
  "$AGENT_COST_SHUTDOWN_RESERVE_USD" "$AGENT_COST_BUDGET_USD" <<'PY'
import math
import sys

configured = sys.argv[1]
budget = float(sys.argv[2])
# The benchmark spends the complete cap before teardown. The OpenRouter proxy
# serializes requests and stops admitting paid work after the crossing, so only
# the request already in flight can add model cost beyond it.
minimum = 0.0
reserve = minimum if configured == "auto" else float(configured)
if not math.isfinite(reserve) or reserve < 0 or reserve >= budget:
    raise SystemExit(
        "AGENT_COST_SHUTDOWN_RESERVE_USD must be finite, non-negative, and below the budget"
    )
print(format(reserve, "g"), format(minimum, "g"))
PY
)
# Modal enforces this deadline in its control plane. It is intentionally based
# on the always-on CPU allocation alone: even if both controller and in-sandbox
# watchdog disappear, the run cannot remain billable past its CPU-only budget.
SANDBOX_TIMEOUT_SECONDS=$(python3 - "$SANDBOX_TIMEOUT_SECONDS" \
  "$AGENT_COST_BUDGET_USD" "$AGENT_COST_SHUTDOWN_RESERVE_USD" <<'PY'
import math
import sys

configured = int(sys.argv[1])
budget = float(sys.argv[2])
reserve = float(sys.argv[3])
cpu_rate = 2 * 0.00003942 + 8 * 0.00000667
budget_deadline = max(60, math.floor((budget - reserve) / cpu_rate))
print(min(configured, budget_deadline))
PY
)

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

DEEPSEEK_PRICING_SNAPSHOT_JSON=${SPRINT_DEEPSEEK_PRICING_SNAPSHOT:-}
if [[ "${MODEL#*/}" == "deepseek-v4-flash" ]]; then
  [[ -n "$DEEPSEEK_PRICING_SNAPSHOT_JSON" ]] || {
    echo "DeepSeek launch requires its provider pricing snapshot from batch preflight" >&2
    exit 1
  }
  DEEPSEEK_PRICING_SNAPSHOT_JSON=$(python3 - \
    "$HARBOR/src/harbor/agents/installed/codex_cost.py" \
    "$DEEPSEEK_PRICING_SNAPSHOT_JSON" <<'PY'
import importlib.util
import json
import os
import sys

path, raw = sys.argv[1:]
snapshot = json.loads(raw)
os.environ["SPRINT_DEEPSEEK_PRICING_SNAPSHOT"] = json.dumps(
    snapshot, separators=(",", ":"), sort_keys=True
)
spec = importlib.util.spec_from_file_location("launch_codex_cost", path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load the frozen pricing validator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
selected = module.pricing_snapshot_for_request(
    "deepseek-v4-flash", snapshot["captured_at"]
)
if not isinstance(selected, dict) or selected.get("source_sha256") != snapshot.get("source_sha256"):
    raise SystemExit("DeepSeek pricing snapshot did not survive validation")
print(os.environ["SPRINT_DEEPSEEK_PRICING_SNAPSHOT"])
PY
  )
  export SPRINT_DEEPSEEK_PRICING_SNAPSHOT="$DEEPSEEK_PRICING_SNAPSHOT_JSON"
fi

MODEL_API_COST_BASIS=$(python3 - \
  "$ROOT/event_runtime/container/sprint_openrouter_pricing.py" \
  "$MODEL" "$MODEL_API_HOST" <<'PY'
import importlib.util
import sys

path, model, host = sys.argv[1:]
if host != "openrouter.ai":
    print("published_standard_list_price")
    raise SystemExit
spec = importlib.util.spec_from_file_location("sprint_openrouter_pricing", path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load canonical OpenRouter pricing policy")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.benchmark_cost_basis_for_model(model))
PY
)

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
python3 - "$RUN_ID" "$AGENT_KIND" <<'PY'
import json
import shlex
import sys

run_id, agent_kind = sys.argv[1:]
watcher = [
    "/opt/sprint-agent-supervisor.sh",
    "--run-id", run_id,
    "--agent-kind", agent_kind,
]
# Start GPU/CPU telemetry before the agent supervisor. The supervisor also
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
    + " /opt/sprint-telemetry.sh --role cpu-agent --run-id "
    + shlex.quote(run_id)
    + " --out-dir /logs/artifacts/telemetry "
    "--interval-seconds ${SPRINT_TELEMETRY_INTERVAL:-20} "
    "--pidfile /run/sprint-telemetry.pid || true; "
    "fi; "
)
command = (
    telemetry
    + "if [ -x /opt/sprint-agent-supervisor.sh ]; then exec "
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
  "$HARBOR_BRANCH" "$VOLUMES_JSON" "$KEEPALIVE_JSON" "$AGENT_COST_BUDGET_USD" \
  "$AGENT_COST_SHUTDOWN_RESERVE_USD" "$MINIMUM_SAFE_SHUTDOWN_RESERVE_USD" \
  "$DEEPSEEK_PRICING_SNAPSHOT_JSON" "$MODEL_API_COST_BASIS" <<'PY'
import json
import os
import sys

(run_id, app, training_app, verifier_app, volume, state, jobs, agent_kind, model, endpoint, effort,
 codex_version, auth_name, sandbox_timeout, model_api_host, profile, harbor, commit, branch,
 volumes, keepalive, agent_cost_budget, shutdown_reserve, minimum_reserve,
 pricing_snapshot_json, model_api_cost_basis) = sys.argv[1:]
pricing_snapshot = json.loads(pricing_snapshot_json) if pricing_snapshot_json else None
provider_endpoint = os.environ.get("SPRINT_OPENROUTER_PROVIDER_ENDPOINT")
quantization = os.environ.get("SPRINT_OPENROUTER_QUANTIZATION")
openrouter_route = (
    {
        "only": [provider_endpoint],
        "order": [provider_endpoint],
        "allow_fallbacks": False,
        "require_parameters": True,
        "quantizations": [quantization] if quantization else [],
    }
    if model_api_host == "openrouter.ai" and provider_endpoint
    else None
)
contract_json = os.environ.get("SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON")
openrouter_request_contract = json.loads(contract_json) if contract_json else None
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
    "deepseek_harness_version": (
        os.environ.get("DEEPSEEK_HARNESS_VERSION")
        if agent_kind == "deepseek-harness"
        else None
    ),
    "deepseek_harness_sdk_version": (
        os.environ.get("DEEPSEEK_HARNESS_SDK_VERSION")
        if agent_kind == "deepseek-harness"
        else None
    ),
    "automatic_stop": True,
    "automatic_stop_reason": "agent_cost_budget_exhausted",
    "agent_cost_budget_usd": float(agent_cost_budget),
    "api_pricing_snapshot": pricing_snapshot,
    "openrouter_route": openrouter_route,
    "openrouter_request_contract": openrouter_request_contract,
    "provider_usage_ledger_required": model_api_host == "openrouter.ai",
    "budget_enforcement": {
        "controller_watchdog": True,
        "in_sandbox_watchdog": True,
        "uncertainty_policy": "fail_closed",
        "api_cost_source": (
            "openrouter_reported_per_request"
            if model_api_host == "openrouter.ai"
            else "token_rate_reconstruction"
        ),
        "api_budget_cost_basis": model_api_cost_basis,
        "shutdown_reserve_usd": float(shutdown_reserve),
        "minimum_safe_shutdown_reserve_usd": float(minimum_reserve),
        "durable_stop_marker": "BUDGET_STOP_REQUESTED.json",
    },
    "sandbox_timeout_seconds": int(sandbox_timeout),
    "sandbox_timeout_role": "modal_server_side_budget_backstop",
    "cpu_agent": {
        "physical_cpu_cores": 2,
        "vcpus_equivalent": 4,
        "memory_mb": 8192,
        "gpus": 0,
    },
    "cgroup_telemetry_required": True,
    "gpu_pipeline_telemetry_required": True,
    "agent_network_policy": "model-api-only",
    "agent_allowed_host": model_api_host,
    "hosted_model_tools_policy": (
        "disabled" if agent_kind in {"codex", "deepseek-harness"} else None
    ),
    "service_tier": (
        "default"
        if agent_kind == "codex"
        and model.split("/", 1)[-1]
        in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
        else None
    ),
    "usage_audit_required": (
        model_api_host == "openrouter.ai"
        or (
            agent_kind == "codex"
            and model.split("/", 1)[-1]
            in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
        )
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
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if ((DRY_RUN)); then
  print_config
  exit 0
fi

# Evaluations only start from image definitions that were eagerly built and
# exercised on Modal. A stopped CPU trial is never resumed in place.
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- event_runtime events harbor)" ]]; then
  echo "new evaluations require committed benchmark, Harbor, and launcher source" >&2
  exit 1
fi
python3 "$ROOT/event_runtime/preflight/check_images.py"
read -r AGENT_TRAINING_IMAGE_ID VERIFIER_IMAGE_ID < <(
  python3 - "$WARMUP_MANIFEST_PATH" <<'PY'
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
SPRINT_SOURCE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)

umask 077
if [[ -e "$SECRET_DIR" || -e "$JOBS_ROOT" ]]; then
  echo "partial run state already exists: $RUN_ID" >&2
  exit 1
fi
# The Modal keepalive is the agent supervisor's process environment, separate
# from Harbor's agent-exec environment. Rebuild it after determining the
# generation so a replacement cannot reuse attempt 1's first-seen sentinel.
KEEPALIVE_JSON=$(make_keepalive_json)

mkdir -p "$STATE_DIR" "$JOBS_ROOT" "$SECRET_DIR"
python3 "$SOURCE_ROOT/event_runtime/control/render_task.py" \
  --source "$TASK_SOURCE" \
  --destination "$TASK" \
  --budget "$AGENT_COST_BUDGET_USD"
printf '%s=%s\n' "$AGENT_SECRET_NAME" "$AGENT_SECRET" >"$ENV_FILE"
if [[ -n "$ENDPOINT" && "$AGENT_KIND" == "codex" ]]; then
  printf 'OPENAI_BASE_URL=%s\n' "$ENDPOINT" >>"$ENV_FILE"
fi
chmod 0600 "$ENV_FILE"

# The host needs Modal credentials to create the sandbox, but the untrusted
# agent must never receive a cloud control-plane credential. Keep its env file
# on an explicit allowlist rather than relying on ambient inheritance.
python3 "$ROOT/event_runtime/control/credentials.py" "$ENV_FILE" "$AGENT_SECRET_NAME"

if [[ "$AGENT_KIND" == "deepseek-harness" ]]; then
  PROMPT_TEMPLATE="${PROMPT_TEMPLATE_OVERRIDE:-$SOURCE_ROOT/event_runtime/control/templates/deepseek-harness.j2}"
else
  PROMPT_TEMPLATE="${PROMPT_TEMPLATE_OVERRIDE:-$SOURCE_ROOT/event_runtime/control/templates/codex.j2}"
fi

python3 - "$STATE_DIR/run.json" "$RUN_ID" "$APP_NAME" "$TRAINING_APP_NAME" \
  "$VERIFIER_APP_NAME" "$VOLUME_NAME" \
  "$STATE_DIR" "$JOBS_ROOT" "$SECRET_DIR" "$MODAL_PROFILE" \
  "$AGENT_KIND" "$MODEL" "$ENDPOINT" "$REASONING_EFFORT" "$CODEX_VERSION" \
  "$SANDBOX_TIMEOUT_SECONDS" "$DEPLOY_DEBOUNCE_SECONDS" "$HARBOR" \
  "$HARBOR_COMMIT" "$HARBOR_BRANCH" \
  "$STANDING_GPU" "$MODEL_API_HOST" \
  "$PROMPT_TEMPLATE" "$WARMUP_MANIFEST_PATH" "$ROOT" "$BATCH_ID" \
  "$SOURCE_ROOT" "$SPRINT_SOURCE_COMMIT" "$TASK" \
  "$AGENT_COST_BUDGET_USD" "$AGENT_COST_SHUTDOWN_RESERVE_USD" \
  "$MINIMUM_SAFE_SHUTDOWN_RESERVE_USD" \
  "$DEEPSEEK_PRICING_SNAPSHOT_JSON" "$MODEL_API_COST_BASIS" <<'PY'
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import sys

(path, run_id, app, training_app, verifier_app, volume, state, jobs, secrets, profile, agent_kind, model,
 endpoint, effort, codex_version, sandbox_timeout, debounce, harbor, commit,
 branch, standing_gpu_flag,
 model_api_host, prompt_template, warmup_manifest_path, root, batch_id, source_root,
 sprint_source_commit, rendered_task_root, agent_cost_budget, shutdown_reserve, minimum_reserve,
 pricing_snapshot_json, model_api_cost_basis) = sys.argv[1:]
standing_gpu = standing_gpu_flag == "1"
pricing_snapshot = json.loads(pricing_snapshot_json) if pricing_snapshot_json else None
provider_endpoint = os.environ.get("SPRINT_OPENROUTER_PROVIDER_ENDPOINT")
quantization = os.environ.get("SPRINT_OPENROUTER_QUANTIZATION")
openrouter_route = (
    {
        "only": [provider_endpoint],
        "order": [provider_endpoint],
        "allow_fallbacks": False,
        "require_parameters": True,
        "quantizations": [quantization] if quantization else [],
    }
    if model_api_host == "openrouter.ai" and provider_endpoint
    else None
)
contract_json = os.environ.get("SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON")
openrouter_request_contract = json.loads(contract_json) if contract_json else None
target = pathlib.Path(path)
root_path = pathlib.Path(root)
task_source_root = pathlib.Path(source_root) / "events/g1-100-metres"
task_root = pathlib.Path(rendered_task_root)
prompt_path = pathlib.Path(prompt_template)
warmup_manifest = json.loads(pathlib.Path(warmup_manifest_path).read_text())

def sha256_file(file_path):
    return hashlib.sha256(pathlib.Path(file_path).read_bytes()).hexdigest()

lock_fd = os.open(target.with_name("run.json.lock"), os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(lock_fd, fcntl.LOCK_EX)
payload = {
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
        pricing_snapshot["model_version"]
        if pricing_snapshot is not None
        else model.split("/", 1)[-1]
    ),
    "codex_version": codex_version if agent_kind == "codex" else None,
    "deepseek_harness_version": (
        os.environ.get("DEEPSEEK_HARNESS_VERSION")
        if agent_kind == "deepseek-harness"
        else None
    ),
    "deepseek_harness_sdk_version": (
        os.environ.get("DEEPSEEK_HARNESS_SDK_VERSION")
        if agent_kind == "deepseek-harness"
        else None
    ),
    "deepseek_harness_config_sha256": (
        sha256_file(
            root_path
            / "event_runtime/container/deepseek-harness-minimal.cordis.yml"
        )
        if agent_kind == "deepseek-harness"
        else None
    ),
    "automatic_stop": True,
    "automatic_stop_reason": "agent_cost_budget_exhausted",
    "agent_cost_budget_usd": float(agent_cost_budget),
    "api_pricing_snapshot": pricing_snapshot,
    "openrouter_route": openrouter_route,
    "openrouter_request_contract": openrouter_request_contract,
    "provider_usage_ledger_required": model_api_host == "openrouter.ai",
    "budget_enforcement": {
        "controller_watchdog": True,
        "in_sandbox_watchdog": True,
        "uncertainty_policy": "fail_closed",
        "api_cost_source": (
            "openrouter_reported_per_request"
            if model_api_host == "openrouter.ai"
            else "token_rate_reconstruction"
        ),
        "api_budget_cost_basis": model_api_cost_basis,
        "shutdown_reserve_usd": float(shutdown_reserve),
        "minimum_safe_shutdown_reserve_usd": float(minimum_reserve),
        "durable_stop_marker": "BUDGET_STOP_REQUESTED.json",
    },
    "sandbox_timeout_seconds": int(sandbox_timeout),
    "sandbox_timeout_role": "modal_server_side_budget_backstop",
    "deploy_debounce_seconds": int(debounce),
    "harbor_path": harbor,
    "harbor_commit": commit,
    "harbor_branch": branch,
    "sprint_source_commit": sprint_source_commit,
    "task_path": str(task_root),
    "task_source_path": str(task_source_root),
    "evaluation_provenance": {
        "prompt_template_path": prompt_template,
        "prompt_template_sha256": sha256_file(prompt_path),
        "task_toml_sha256": sha256_file(task_root / "task.toml"),
        "instruction_template_sha256": sha256_file(
            task_source_root / "instruction.md"
        ),
        "rendered_instruction_sha256": sha256_file(task_root / "instruction.md"),
        "agent_training_context_sha256": warmup_manifest["contexts"]["agent_training"]["sha256"],
        "agent_training_image_id": warmup_manifest["contexts"]["agent_training"]["image_id"],
        "verifier_context_sha256": warmup_manifest["contexts"]["verifier"]["sha256"],
        "verifier_image_id": warmup_manifest["contexts"]["verifier"]["image_id"],
        "image_warmup_completed_at_epoch_s": warmup_manifest["completed_at_epoch_s"],
    },
    "site_dir": str(root_path / "web"),
    "vercel_project_id": "prj_dgvTovRNwdSDcefYmo6oXfju9M3p",
    "vercel_org_id": "team_SNgoAcFfHYXYdUIXhj16bGek",
    "vercel_scope": "alienkevins-projects",
    "production_alias": "https://g1-sprint.vercel.app",
    # Agent sandbox is CPU-only (task.toml gpus=0); host monitor dispatches
    # preemptible A10G workers for event-gpu jobs on the same volume.
    "cpu_agent_gpu_worker": True,
    # New runs publish one exact-name index for job/cancellation discovery.
    # This avoids account-wide Modal VolumeListFiles polling by every lane.
    "standing_gpu_worker": standing_gpu,
    "agent_cpu_instances": 1,
    "training_max_concurrent_per_run": 1,
    "agent_gpus": 0,
    "gpu_worker_gpu": "A10G",
    "gpu_worker_gpus_per_job": 1,
    "resource_contract": {
        "cpu_agent": {
            "instances": 1,
            "physical_cpu_cores": 2,
            "vcpus_equivalent": 4,
            "memory_mb": 8192,
            "gpus": 0,
        },
        "training_worker": {
            "max_concurrent": 1,
            "physical_cpu_cores": 6,
            "vcpus_equivalent": 12,
            "memory_mb": 12288,
            "gpu_count": 1,
            "gpu_type": "A10G",
        },
        "verifier": {
            "max_concurrent": 1,
            "physical_cpu_cores": 4,
            "vcpus_equivalent": 8,
            "memory_mb": 10240,
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
    "gpu_pipeline_telemetry_required": True,
    # OpenRouter cost is accepted only from the trusted per-request ledger.
    # Aggregate cached/uncached counters cannot recover dynamic provider,
    # long-context, cache-write, promotion, or peak-floor pricing correctly.
    "usage_audit_required": (
        model_api_host == "openrouter.ai"
        or (
            agent_kind == "codex"
            and model.split("/", 1)[-1]
            in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
        )
    ),
    "hosted_model_tools_policy": (
        "disabled" if agent_kind in {"codex", "deepseek-harness"} else None
    ),
    "service_tier": (
        "default"
        if agent_kind == "codex"
        and model.split("/", 1)[-1]
        in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
        else None
    ),
    "timeline_bucket_seconds": 60,
    "telemetry_cpu_max_gap_seconds": 45,
    "telemetry_gpu_max_gap_seconds": 45,
    "telemetry_gpu_pipeline_max_gap_seconds": 45,
    "telemetry_resource_roles": ["cpu-agent", "training-gpu", "verifier-gpu"],
    # Per-trial admission is rate-limited, then every run in this batch feeds
    # one crash-safe blind official verifier lane.
    "scoring_queue_scope": "shared_blind_archival_queue",
    "scoring_queue_key": batch_id or "standalone",
    "scoring_max_concurrent_per_trial": 1,
    "scoring_cross_trial_lease": True,
    "scoring_minimum_submission_interval_sec": 300,
    "scoring_max_outstanding_submissions_per_trial": 1,
    "scoring_feedback_policy": "official_results_hidden_agent_local_verification",
    "scoring_drain_policy": "all_accepted_submissions",
    "scoring_deduplication_key": "task_fingerprint_plus_policy_sha256",
    "evaluation_result_policy": "all_blind_archival_submissions",
    "verifier_cost_attribution": "measurement_overhead_separate_from_agent_cost",
    "cpu_execution_policy": "single_process_no_resume",
}
tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, target)
PY

SHARED_AGENT_ENV=(
  --ae "SPRINT_RUN_ID=$RUN_ID"
  --ae "SPRINT_GPU_JOBS_ROOT=/durable/runs/$RUN_ID/gpu-jobs"
  --ae "SPRINT_SUBMISSIONS_ROOT=/durable/submissions"
  --ae "SPRINT_SUBMISSION_MIN_INTERVAL_SEC=300"
  --ae "SPRINT_MODEL=$MODEL"
  --ae "SPRINT_SCORING_QUEUE_KEY=${BATCH_ID:-standalone}"
  --ae "SPRINT_REQUESTED_CPU_CORES=2"
  --ae "SPRINT_REQUESTED_MEMORY_MIB=8192"
)
if [[ "$AGENT_KIND" == "codex" ]]; then
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
  if [[ "$MODEL_API_HOST" == "openrouter.ai" ]]; then
    AGENT_HARBOR_ARGS+=(
      --ae "SPRINT_OPENROUTER_LEDGER_REQUIRED=1"
      --ae "SPRINT_OPENROUTER_UPSTREAM_URL=https://openrouter.ai/api/v1"
    )
    if [[ -n "${SPRINT_OPENROUTER_PROVIDER_ENDPOINT:-}" ]]; then
      AGENT_HARBOR_ARGS+=(
        --ae "SPRINT_OPENROUTER_PROVIDER_ENDPOINT=$SPRINT_OPENROUTER_PROVIDER_ENDPOINT"
      )
    fi
    if [[ -n "${SPRINT_OPENROUTER_QUANTIZATION:-}" ]]; then
      AGENT_HARBOR_ARGS+=(
        --ae "SPRINT_OPENROUTER_QUANTIZATION=$SPRINT_OPENROUTER_QUANTIZATION"
      )
    fi
    if [[ -n "$SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON" ]]; then
      AGENT_HARBOR_ARGS+=(
        --ae "SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON=$SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON"
      )
    fi
    AGENT_HARBOR_ARGS+=(
      --ae "SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH=$SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH"
    )
  fi
  case "${MODEL#*/}" in
    gpt-5.6-luna)
      OPENAI_OPENROUTER_PRESET=${OPENAI_OPENROUTER_PRESET:-@preset/sprint-gpt-5-6-luna-openai-standard}
      AGENT_HARBOR_ARGS+=(
        --ae "SPRINT_CODEX_OPENAI_MODEL=$OPENAI_OPENROUTER_PRESET"
        --ae "SPRINT_CODEX_OPENAI_MODEL_ID=gpt-5.6-luna"
        --ae "SPRINT_CODEX_OPENAI_MODEL_LOCK=/opt/sprint-codex-luna-model-lock.json"
        --ae "CODEX_GOAL_BOOTSTRAP_MODEL=$OPENAI_OPENROUTER_PRESET"
        --ae "CODEX_GOAL_BOOTSTRAP_EXPECTED_PROVIDER=sprint_openrouter"
        --ae "CODEX_GOAL_BOOTSTRAP_PREPARE_SCRIPT=/opt/sprint-apply-openai-codex-config.sh"
      )
      ;;
    gpt-5.6-sol)
      OPENAI_OPENROUTER_PRESET=${OPENAI_OPENROUTER_PRESET:-@preset/sprint-gpt-5-6-sol-openai-standard}
      AGENT_HARBOR_ARGS+=(
        --ae "SPRINT_CODEX_OPENAI_MODEL=$OPENAI_OPENROUTER_PRESET"
        --ae "SPRINT_CODEX_OPENAI_MODEL_ID=gpt-5.6-sol"
        --ae "SPRINT_CODEX_OPENAI_MODEL_LOCK=/opt/sprint-codex-sol-model-lock.json"
        --ae "CODEX_GOAL_BOOTSTRAP_MODEL=$OPENAI_OPENROUTER_PRESET"
        --ae "CODEX_GOAL_BOOTSTRAP_EXPECTED_PROVIDER=sprint_openrouter"
        --ae "CODEX_GOAL_BOOTSTRAP_PREPARE_SCRIPT=/opt/sprint-apply-openai-codex-config.sh"
      )
      ;;
  esac
  if [[ "${MODEL#*/}" == "gpt-5.6-sol" || "${MODEL#*/}" == "gpt-5.6-terra" || "${MODEL#*/}" == "gpt-5.6-luna" ]]; then
    # Pin standard pricing. Leaving this unset lets Codex/project defaults pick
    # another service tier, which cannot be reconstructed from token counts.
    AGENT_HARBOR_ARGS+=(--ak "service_tier=default")
  fi
else
  AGENT_HARBOR_ARGS=(
    --ak "version=$DEEPSEEK_HARNESS_VERSION"
    --ae "SPRINT_AGENT_KIND=deepseek-harness"
    --ae "SPRINT_RUNTIME_DIR=/run"
    --ae "SPRINT_AGENT_LOG_DIR=/logs/agent"
    --ae "SPRINT_OPENROUTER_LEDGER_REQUIRED=1"
    --ae "SPRINT_OPENROUTER_UPSTREAM_URL=https://openrouter.ai/api/v1"
    --ae "SPRINT_OPENROUTER_PROVIDER_ENDPOINT=deepseek"
    "${SHARED_AGENT_ENV[@]}"
  )
fi

unset OPENROUTER_API_KEY OPENAI_API_KEY OPENAI_BASE_URL HARBOR_API_KEY AGENT_SECRET
export MODAL_PROFILE
python3 -m modal volume create --version 1 "$VOLUME_NAME"
python3 -m modal volume put "$VOLUME_NAME" "$STATE_DIR/run.json" \
  "runs/$RUN_ID/state/run.json"

start_controller_worker() {
  local kind=$1
  local log_path=$2
  local pid_path=$3
  shift 3
  local unit="sprint-${kind}-${RUN_ID}.service"
  local worker_alive=0
  local worker_pid=
  if [[ -f "$pid_path" ]]; then
    worker_pid=$(tr -dc '0-9' <"$pid_path" || true)
    if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
      worker_alive=1
    fi
  fi
  if ((worker_alive)); then
    return 0
  fi

  # A controller reboot or terminal disconnect must not remove the independent
  # budget watchdog. Prefer a restartable user service; retain nohup only for
  # environments without a reachable user-systemd manager.
  if command -v systemd-run >/dev/null 2>&1 \
    && systemctl --user show-environment >/dev/null 2>&1; then
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
    systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
    systemd-run --user \
      --unit="$unit" \
      --property=Restart=on-failure \
      --property=RestartSec=5s \
      --property="WorkingDirectory=$ROOT" \
      --property="StandardOutput=append:$log_path" \
      --property="StandardError=append:$log_path" \
      /usr/bin/python3 "$CONTROL" "$@" >/dev/null
    return 0
  fi

  nohup python3 "$CONTROL" "$@" >>"$log_path" 2>&1 </dev/null &
  printf '%s\n' "$!" >"$pid_path"
}

if ((START_MONITOR)); then
  start_controller_worker \
    monitor "$STATE_DIR/monitor.log" "$STATE_DIR/monitor.pid" \
    monitor --run-id "$RUN_ID"
  start_controller_worker \
    pulse "$STATE_DIR/budget-pulse.log" "$STATE_DIR/budget-pulse.pid" \
    budget-pulse --run-id "$RUN_ID" --poll-seconds 15
  start_controller_worker \
    gpu-dispatch "$STATE_DIR/gpu-dispatch-loop.log" \
    "$STATE_DIR/gpu-dispatch-loop.pid" \
    gpu-dispatch-loop --run-id "$RUN_ID" --poll-seconds 10
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
