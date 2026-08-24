#!/usr/bin/env python3
"""Reproducible event batch launch, monitoring, and website publishing."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
SCRIPT_DIR = ROOT / "runs" / "ops"
PREFLIGHT_DIR = ROOT / "event_runtime" / "preflight"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from event_runtime.export import frontier as frontier_update  # noqa: E402
from event_runtime.export import performance as performance_export  # noqa: E402
from event_runtime.control import run as sprintctl  # noqa: E402
from event_runtime.control.openrouter_credentials import (  # noqa: E402
    OpenRouterManagementClient,
    OpenRouterManagementError,
    TrialCredentialSpec,
    provision_trial_credentials,
    revoke_trial_credentials,
)
from event_runtime.container.sprint_openrouter_pricing import (  # noqa: E402
    BENCHMARK_COST_BASIS,
    PROVIDER_COST_BASIS,
    OpenRouterPricingError,
    benchmark_cost_usd,
    undiscounted_cost_usd,
)
from event_runtime.container.sprint_openrouter_usage import (  # noqa: E402
    add_token_usage,
    empty_token_usage,
    generation_usage_payload,
)


UV = Path(os.environ.get("UV", "/home/ubuntu/.local/bin/uv"))
HARBOR_PYTHON = ROOT / "harbor/.venv/bin/python3"
WEB = ROOT / "web"
BATCH_ROOT = SCRIPT_DIR / "batches"
WARMUP_MANIFEST = SCRIPT_DIR / "modal-image-warmup.json"
FUNCTIONAL_CANARY_REPORT = SCRIPT_DIR / "training-gpu-canary.json"
FUNCTIONAL_CANARY_FIXTURE = (
    PREFLIGHT_DIR / "training_canary" / "train_sprint.py"
)
BUDGET_CONFIG = MODULE_DIR / "budget.env"
HARBOR_REVISION = "dafb1387151e1c32702963d44fe6c3cea66cf8cb"
CODEX_VERSION = "0.149.1"
TRIALS_PER_MODEL = 3
DEFAULT_FAMILIES = ("deepseek", "luna")
# Production experiments must use OpenRouter with the model author's official
# provider. Historical Baidu/Alibaba arms remain in the specs so archived runs
# can still be rendered, but they are deliberately not launchable.
SUPPORTED_FAMILIES = ("deepseek", "luna", "sol")
OPENAI_FAMILY_SPECS: dict[str, dict[str, str]] = {
    "luna": {
        "model": "openai/gpt-5.6-luna",
        "model_id": "gpt-5.6-luna",
        "preset": "@preset/sprint-gpt-5-6-luna-openai-standard",
        "preset_id": "f06bb802-6122-4e11-8e22-a3476113a3b1",
        "resolved_model": "openai/gpt-5.6-luna-20260709",
        "provider": "OpenAI",
        "provider_endpoint": "openai",
    },
    "sol": {
        "model": "openai/gpt-5.6-sol",
        "model_id": "gpt-5.6-sol",
        "preset": "@preset/sprint-gpt-5-6-sol-openai-standard",
        "preset_id": "798803cc-8d68-4249-b437-c6eb509df833",
        "resolved_model": "openai/gpt-5.6-sol-20260709",
        "provider": "OpenAI",
        "provider_endpoint": "openai",
    },
}
DEEPSEEK_ROUTED_FAMILY_SPECS: dict[str, dict[str, str]] = {
    # `deepseek` is the current public/default family and uses DeepSeek's own
    # pinned minimal benchmark harness with its native persisted goal mode.
    # Keep the older provider comparisons on Codex so historical arms remain
    # reproducible rather than silently changing harnesses.
    "deepseek": {
        "model": "deepseek/deepseek-v4-flash-vision-exp",
        "resolved_model": "deepseek/deepseek-v4-flash-vision-exp-20260821",
        "provider": "DeepSeek",
        "provider_endpoint": "deepseek",
        "quantization": "unknown",
        "context_window": "1048576",
        "wrapper": "deepseek_harness.sh",
        "wire_api": "chat_completions",
        "agent_kind": "deepseek-harness",
        "goal_mode": "deepseek_native_goal",
    },
    "flash-baidu": {
        "model": "deepseek/deepseek-v4-flash-0731",
        "resolved_model": "Baidu | deepseek/deepseek-v4-flash-20260731",
        "provider": "Baidu",
        "provider_endpoint": "baidu/fp8",
        "quantization": "fp8",
        "context_window": "1048576",
        "wrapper": "deepseek.sh",
        "wire_api": "responses",
        "agent_kind": "codex",
        "goal_mode": "codex_session_goal",
    },
    "pro-alibaba": {
        "model": "deepseek/deepseek-v4-pro-0813",
        "resolved_model": "Alibaba | deepseek/deepseek-v4-pro-20260813",
        "provider": "Alibaba",
        "provider_endpoint": "alibaba",
        "quantization": "unknown",
        "context_window": "1000000",
        "wrapper": "deepseek.sh",
        "wire_api": "responses",
        "agent_kind": "codex",
        "goal_mode": "codex_session_goal",
    },
}
REASONING_EFFORT = "max"
RUN_HOURS: float | None = None
POLL_SECONDS = 30
# The task permits 15 minutes for one sealed verification. Leave five minutes
# for sandbox startup and result archival before declaring one run's accepted
# work stuck in the shared verifier lane; otherwise a legitimate timeout can
# race the watchdog.
VERIFIER_LANE_STALL_SECONDS = 20 * 60
# The production Vercel team is on Hobby. A 20-minute rolling publication
# cadence caps publication at 72 deployments per day, leaving headroom
# below the 100/day Hobby allowance for warmups/manual releases. The final
# completed site still bypasses this delay below.
LIVE_SITE_DEPLOY_SECONDS = 20 * 60
PROVIDER_DISCOVERY_ATTEMPTS = 3
PROVIDER_DISCOVERY_RETRY_SECONDS = 1.0
PROVIDER_INFERENCE_ATTEMPTS = 10
PROVIDER_INFERENCE_RETRY_SECONDS = 5.0
PROVIDER_GENERATION_AUDIT_ATTEMPTS = 6
OPENROUTER_CREDIT_SAFETY_FACTOR = 1.05
OPENROUTER_USAGE_AUDIT_GRACE_SECONDS = 120
OPENROUTER_USAGE_AUDIT_TOLERANCE_USD = 0.01
VERCEL_DAILY_QUOTA_BACKOFF_SECONDS = 24 * 60 * 60
VERCEL_DAILY_QUOTA_CODE = "api-deployments-free-per-day"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,48}$")
ALERT_PATTERNS = {
    "provider_rate_limit": re.compile(
        r"\b(?:429|rate.?limit|too many requests)\b", re.I
    ),
    "provider_quota": re.compile(
        r"\b(?:insufficient_quota|quota exceeded|billing limit|spend limit)\b", re.I
    ),
    "provider_auth": re.compile(
        r"(?:\b(?:invalid api key|authentication failed)\b|"
        r"(?<![A-Za-z0-9.])(?:401|403)(?![A-Za-z0-9]))",
        re.I,
    ),
    "modal_infrastructure": re.compile(
        r"\b(?:modal.*(?:internal|connection|timeout)|sandbox.*failed|container.*lost)\b",
        re.I,
    ),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic_json(path: Path, payload: Any, mode: int = 0o600) -> None:
    frontier_update.atomic_write_json(path, payload, mode=mode)


def _public_material_state(value: Any) -> Any:
    """Remove observer-only clocks before deciding whether to republish."""
    if isinstance(value, dict):
        return {
            key: _public_material_state(item)
            for key, item in value.items()
            if key not in {"updated_at", "last_monitor_at"}
        }
    if isinstance(value, list):
        return [_public_material_state(item) for item in value]
    return value


def atomic_public_json(path: Path, payload: Any) -> bool:
    """Write a public snapshot only when user-visible state has changed."""
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        previous = None
    if previous is not None and _public_material_state(
        previous
    ) == _public_material_state(payload):
        return False
    atomic_json(path, payload, mode=0o644)
    return True


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if (
            name.strip()
            in {
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENROUTER_MANAGEMENT_KEY",
            }
            and value
        ):
            values[name.strip()] = value
    return values


def configured_agent_budget_usd() -> float:
    """Read the same global budget default consumed by ``launch.sh``."""
    raw = os.environ.get("AGENT_COST_BUDGET_USD")
    if raw is None:
        match = re.search(
            r"^AGENT_COST_BUDGET_USD=\$\{AGENT_COST_BUDGET_USD:-([^}]+)\}$",
            BUDGET_CONFIG.read_text(),
            re.MULTILINE,
        )
        if not match:
            raise RuntimeError("global agent budget default is unreadable")
        raw = match.group(1)
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("global agent budget must be positive and finite")
    return value


def openrouter_credit_requirement(trial_count: int) -> dict[str, Any]:
    """Return conservative provider credit needed for one planned matrix."""
    if trial_count <= 0:
        raise ValueError("trial count must be positive")
    budget = configured_agent_budget_usd()
    maximum_budget = budget * trial_count
    required = maximum_budget * OPENROUTER_CREDIT_SAFETY_FACTOR
    return {
        "per_trial_budget_usd": budget,
        "trial_count": trial_count,
        "maximum_combined_budget_usd": maximum_budget,
        "safety_factor": OPENROUTER_CREDIT_SAFETY_FACTOR,
        "required_credit_usd": required,
    }


def fetch_openrouter_credit(api_key: str) -> dict[str, float]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"OpenRouter credit request failed: HTTP {exc.code}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise RuntimeError(f"OpenRouter credit request failed: {reason}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    try:
        total = float(data["total_credits"])
        usage = float(data["total_usage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter credit response was invalid") from exc
    remaining = total - usage
    if not all(
        math.isfinite(value) and value >= 0 for value in (total, usage, remaining)
    ):
        raise RuntimeError("OpenRouter credit totals were invalid")
    return {
        "total_credits_usd": total,
        "total_usage_usd": usage,
        "remaining_credit_usd": remaining,
    }


def verify_openrouter_management_access(management_key: str) -> None:
    """Confirm the management credential without mutating account state."""
    OpenRouterManagementClient(management_key).request("GET", "/keys?limit=1")


def functional_gpu_canary_ready() -> bool:
    try:
        warmup = json.loads(WARMUP_MANIFEST.read_text())
        canary = json.loads(FUNCTIONAL_CANARY_REPORT.read_text())
        fixture_sha256 = hashlib.sha256(
            FUNCTIONAL_CANARY_FIXTURE.read_bytes()
        ).hexdigest()
    except (OSError, json.JSONDecodeError):
        return False
    contexts = warmup.get("contexts", {})
    cost_equivalence = canary.get("cost_equivalence") or {}
    cost_comparison = cost_equivalence.get("comparison") or {}
    compared_components = set((cost_comparison.get("comparisons") or {}).keys())
    tolerance = cost_comparison.get("tolerance_usd")
    maximum_delta = cost_comparison.get("max_absolute_delta_usd")
    cost_proof_valid = bool(
        cost_equivalence.get("completed")
        and cost_comparison.get("verified")
        and isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and isinstance(maximum_delta, (int, float))
        and not isinstance(maximum_delta, bool)
        and 0 <= maximum_delta <= tolerance <= 1e-9
        and compared_components
        == {
            "model_api_usd",
            "cpu_agent_usd",
            "training_sandboxes_usd",
            "total_usd",
        }
    )
    return bool(
        warmup.get("completed")
        and canary.get("schema_version") == 5
        and canary.get("completed")
        and canary.get("full_path_verified")
        and canary.get("verifier_equivalence_verified")
        and canary.get("cost_equivalence_verified")
        and canary.get("gpu_budget_mirror_verified")
        and (canary.get("gpu_budget_mirror") or {}).get("completed")
        and (canary.get("gpu_budget_mirror") or {}).get("updates_verified") == 2
        and (canary.get("gpu_budget_mirror") or {}).get("observed_sequences") == [1, 2]
        and cost_proof_valid
        and canary.get("image_id") == contexts.get("agent_training", {}).get("image_id")
        and canary.get("verifier_image_id")
        == contexts.get("verifier", {}).get("image_id")
        and canary.get("training_fixture_sha256") == fixture_sha256
    )


def training_gpu_fleet_probe(
    *, batch_id: str, modal_profile: str, worker_ids: list[str]
) -> tuple[bool, dict[str, Any]]:
    """Require one concurrent AppLauncher success per planned trial lane."""
    if not worker_ids or len(set(worker_ids)) != len(worker_ids):
        return False, {"error": "worker IDs must be non-empty and unique"}
    with tempfile.TemporaryDirectory(prefix="sprint-fleet-probe-") as temporary:
        report_path = Path(temporary) / "report.json"
        command = [
            str(HARBOR_PYTHON),
            str(PREFLIGHT_DIR / "fleet.py"),
            "--report",
            str(report_path),
            "--batch-id",
            batch_id,
        ]
        for worker_id in worker_ids:
            command.extend(("--worker-id", worker_id))
        env = dict(os.environ)
        env["MODAL_PROFILE"] = modal_profile
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            report = {
                "error": "fleet probe did not write a valid report",
                "controller_output_tail": completed.stdout[-4000:],
            }
        expected_image_id = None
        try:
            expected_image_id = json.loads(WARMUP_MANIFEST.read_text())["contexts"][
                "agent_training"
            ]["image_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
        ready = bool(
            completed.returncode == 0
            and report.get("completed") is True
            and report.get("worker_ids") == worker_ids
            and len(report.get("workers", [])) == len(worker_ids)
            and all(row.get("ready") is True for row in report.get("workers", []))
            and report.get("image_id") == expected_image_id
        )
        if not ready and "controller_output_tail" not in report:
            report["controller_output_tail"] = completed.stdout[-4000:]
        return ready, report


def matrix(
    batch_id: str,
    trials_per_model: int = TRIALS_PER_MODEL,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> list[dict[str, Any]]:
    if isinstance(trials_per_model, bool) or not 1 <= trials_per_model <= 50:
        raise ValueError("trials per model must be between 1 and 50")
    arms: list[dict[str, Any]] = []
    specs = {
        **{
            family: {
                "model": spec["model"],
                "wrapper": "openai.sh",
                "resolved_model_version": spec["resolved_model"],
                "openrouter_preset": spec["preset"],
                "provider": spec["provider"],
                "provider_endpoint": spec["provider_endpoint"],
                "quantization": "unknown",
                "agent_kind": "codex",
                "goal_mode": "codex_session_goal",
            }
            for family, spec in OPENAI_FAMILY_SPECS.items()
        },
        **{
            family: {
                **spec,
                "wrapper": spec["wrapper"],
                "resolved_model_version": spec["resolved_model"],
            }
            for family, spec in DEEPSEEK_ROUTED_FAMILY_SPECS.items()
        },
    }
    selected = tuple(dict.fromkeys(families))
    unknown = sorted(set(selected) - set(SUPPORTED_FAMILIES))
    if not selected or unknown:
        raise ValueError(f"invalid model families: {unknown or list(selected)}")
    for family in selected:
        spec = specs[family]
        model_owner = str(spec["model"]).split("/", 1)[0]
        provider_endpoint = str(spec.get("provider_endpoint") or "")
        if model_owner not in {"deepseek", "openai"} or provider_endpoint != model_owner:
            raise ValueError(
                f"{family} must use its model author's official OpenRouter provider"
            )
        for trial in range(1, trials_per_model + 1):
            run_id = f"{batch_id}-{family}-{trial}"
            if not RUN_ID_RE.fullmatch(run_id):
                raise ValueError(f"generated run ID is invalid: {run_id}")
            arms.append(
                {
                    "run_id": run_id,
                    "family": family,
                    "model": spec["model"],
                    "resolved_model_version": spec["resolved_model_version"],
                    "reasoning_effort": REASONING_EFFORT,
                    "agent_kind": spec["agent_kind"],
                    "goal_mode": spec["goal_mode"],
                    "codex_version": CODEX_VERSION,
                    "wrapper": str(MODULE_DIR / "providers" / spec["wrapper"]),
                    "trial": trial,
                    "status": "planned",
                    **(
                        {"openrouter_preset": spec["openrouter_preset"]}
                        if "openrouter_preset" in spec
                        else {}
                    ),
                    **(
                        {
                            key: spec[key]
                            for key in (
                                "provider",
                                "provider_endpoint",
                                "quantization",
                                "context_window",
                            )
                            if key in spec
                        }
                        if "provider_endpoint" in spec
                        else {}
                    ),
                }
            )
    return arms


def batch_dir(batch_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(batch_id):
        raise ValueError("batch ID must be 3-49 safe filename characters")
    return BATCH_ROOT / batch_id


def batch_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "batch.json"


def credential_journal_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "openrouter-credentials.json"


def read_batch(batch_id: str) -> dict[str, Any]:
    path = batch_path(batch_id)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"unknown batch: {batch_id}") from exc
    if payload.get("batch_id") != batch_id:
        raise ValueError("batch state has mismatched ID")
    return payload


def _local_provider_billed_cost(run_id: str) -> tuple[float, int, str | None]:
    state_dir = SCRIPT_DIR / run_id
    candidates: list[tuple[float, int, str]] = []

    path = state_dir / "telemetry" / "agent-cost.json"
    try:
        payload = json.loads(path.read_text())
        model_api = payload["components"]["model_api"]
        cost = float(model_api["provider_billed_cost_usd"])
        pending = int(model_api.get("pending_request_count", 0) or 0)
        as_of = payload.get("as_of")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    else:
        if math.isfinite(cost) and cost >= 0 and pending >= 0 and as_of:
            candidates.append((cost, pending, str(as_of)))

    # The website exporter may atomically replace the live cost mirror with a
    # timeline-derived snapshot that intentionally omits provider-billed
    # metadata.  The proxy's durable summary is the authoritative independent
    # record of completed OpenRouter charges, so it must remain a valid audit
    # source across that replacement and during orderly shutdown.
    summary_path = state_dir / "provider-api-usage" / "api-usage" / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        if summary.get("schema_version") != 3 or summary.get("run_id") != run_id:
            raise ValueError("provider summary identity mismatch")
        summary_cost = float(summary["provider_billed_model_api_usd"])
        summary_pending = max(
            int(summary.get("pending_request_count", 0) or 0),
            int(summary.get("in_flight_request_count", 0) or 0),
        )
        summary_as_of = summary.get("updated_at")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    else:
        if (
            math.isfinite(summary_cost)
            and summary_cost >= 0
            and summary_pending >= 0
            and summary_as_of
        ):
            candidates.append((summary_cost, summary_pending, str(summary_as_of)))

    if not candidates:
        return 0.0, 0, None
    return (
        max(item[0] for item in candidates),
        max(item[1] for item in candidates),
        max(item[2] for item in candidates),
    )


def _rebuild_provider_summary(
    run_id: str,
    request_paths: list[Path],
    replacements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark_total = 0.0
    provider_total = 0.0
    completed = 0
    in_flight: list[str] = []
    recovery: list[str] = []
    token_usage = empty_token_usage()
    for path in request_paths:
        record = replacements.get(path.stem)
        if record is None:
            record = json.loads(path.read_text())
        if record.get("run_id") != run_id:
            raise OpenRouterManagementError("provider ledger identity mismatch")
        request_id = str(record.get("ledger_request_id") or path.stem)
        provider_cost = record.get("provider_reported_cost_usd")
        benchmark_cost = record.get(
            "benchmark_cost_usd",
            record.get("undiscounted_cost_usd", provider_cost),
        )
        if (
            isinstance(provider_cost, (int, float))
            and not isinstance(provider_cost, bool)
            and isinstance(benchmark_cost, (int, float))
            and not isinstance(benchmark_cost, bool)
        ):
            provider_value = float(provider_cost)
            benchmark_value = float(benchmark_cost)
            if (
                not math.isfinite(provider_value)
                or provider_value < 0
                or not math.isfinite(benchmark_value)
                or benchmark_value + 1e-12 < provider_value
            ):
                raise OpenRouterManagementError("provider ledger cost was invalid")
            provider_total += provider_value
            benchmark_total += benchmark_value
            completed += 1
            usage = record.get("usage")
            if usage is not None:
                try:
                    add_token_usage(token_usage, usage)
                except ValueError as exc:
                    raise OpenRouterManagementError(
                        "provider ledger token usage was invalid"
                    ) from exc
        elif record.get("state") == "in_flight":
            in_flight.append(request_id)
        elif record.get("state") == "cost_recovery_required":
            recovery.append(request_id)
        else:
            raise OpenRouterManagementError("provider ledger state was invalid")
    pending = len(in_flight) + len(recovery)
    return {
        "schema_version": 3,
        "run_id": run_id,
        "updated_at": utc_now(),
        "model_api_usd": benchmark_total,
        "provider_billed_model_api_usd": provider_total,
        "promotion_savings_usd": benchmark_total - provider_total,
        "model_api_cost_basis": BENCHMARK_COST_BASIS,
        "provider_billed_cost_basis": PROVIDER_COST_BASIS,
        "completed_request_count": completed,
        "token_usage": token_usage,
        "pending_request_count": pending,
        "in_flight_request_count": len(in_flight),
        "cost_recovery_required_count": len(recovery),
        "in_flight_request_ids": sorted(in_flight),
        "cost_recovery_required_request_ids": sorted(recovery),
    }


def reconcile_openrouter_child_ledger(
    arm: dict[str, Any],
    client: OpenRouterManagementClient,
    *,
    upstream_usage: float,
) -> bool:
    """Settle interrupted requests using generation and child-key evidence.

    The per-trial child key is unique, so its cumulative usage is an
    independent upper-level checksum over every provider charge. Staged files
    are uploaded record-first and summary-last; local mirrors change only
    after both durable uploads succeed.
    """
    run_id = str(arm["run_id"])
    state_dir = SCRIPT_DIR / run_id
    ledger_dir = state_dir / "provider-api-usage" / "api-usage"
    summary_path = ledger_dir / "summary.json"
    requests_dir = ledger_dir / "requests"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenRouterManagementError(
            "provider ledger summary was unavailable"
        ) from exc
    if summary.get("schema_version") != 3 or summary.get("run_id") != run_id:
        raise OpenRouterManagementError("provider ledger summary identity mismatch")
    request_paths = sorted(requests_dir.glob("*.json"))
    pending_records: list[tuple[Path, dict[str, Any]]] = []
    for path in request_paths:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OpenRouterManagementError(
                "provider ledger record was invalid"
            ) from exc
        if record.get("state") in {"in_flight", "cost_recovery_required"}:
            pending_records.append((path, record))
    if not pending_records:
        return False

    local_provider = float(summary.get("provider_billed_model_api_usd") or 0.0)
    # Named generations must be recovered before deciding whether any
    # generation-less request was billed. The child-key delta after all named
    # charges is the only sound evidence that those unnamed requests cost zero.
    pending_records.sort(key=lambda item: 0 if item[1].get("generation_id") else 1)
    replacements: dict[str, dict[str, Any]] = {}
    recovered_provider_total = 0.0
    for path, original in pending_records:
        record = dict(original)
        generation_id = record.get("generation_id")
        if isinstance(generation_id, str) and generation_id:
            generation = client.generation_usage(generation_id)
            try:
                provider_cost = float(generation["total_cost"])
            except (KeyError, TypeError, ValueError) as exc:
                raise OpenRouterManagementError(
                    "generation audit lacked a valid total cost"
                ) from exc
            if not math.isfinite(provider_cost) or provider_cost < 0:
                raise OpenRouterManagementError("generation audit cost was invalid")
            response_model = generation.get("model")
            expected_models = {
                str(arm.get("model") or ""),
                str(arm.get("resolved_model_version") or ""),
            }
            provider = generation.get("provider") or generation.get("provider_name")
            if response_model not in expected_models:
                raise OpenRouterManagementError("generation audit model mismatch")
            if str(provider).casefold() != str(arm.get("provider") or "").casefold():
                raise OpenRouterManagementError("generation audit provider mismatch")
            try:
                usage = generation_usage_payload(generation)
            except ValueError as exc:
                raise OpenRouterManagementError(
                    "generation audit had invalid native token counts"
                ) from exc
            try:
                list_cost = undiscounted_cost_usd(
                    provider_cost, record.get("promotion_snapshot")
                )
                benchmark_cost = benchmark_cost_usd(
                    provider_cost, record.get("promotion_snapshot"), generation
                )
            except OpenRouterPricingError as exc:
                raise OpenRouterManagementError(
                    "generation audit pricing reconstruction failed"
                ) from exc
            record.update(
                {
                    "state": "recovered_complete",
                    "completed_at": utc_now(),
                    "response_id": generation_id,
                    "response_model": response_model,
                    "response_status": "completed",
                    "provider": provider,
                    "route_identity_verified": True,
                    "provider_reported_cost_usd": provider_cost,
                    "undiscounted_cost_usd": list_cost,
                    "benchmark_cost_usd": benchmark_cost,
                    "promotion_adjustment_usd": list_cost - provider_cost,
                    "deepseek_peak_adjustment_usd": benchmark_cost - list_cost,
                    "benchmark_adjustment_usd": benchmark_cost - provider_cost,
                    "promotion_discount_fraction": (
                        record.get("promotion_snapshot") or {}
                    ).get("discount_fraction"),
                    "cost_basis": (record.get("promotion_snapshot") or {}).get(
                        "cost_basis", BENCHMARK_COST_BASIS
                    ),
                    "provider_cost_basis": PROVIDER_COST_BASIS,
                    "usage": usage,
                    "generation_audit": generation,
                    "recovered_after_controller_shutdown": True,
                }
            )
            recovered_provider_total += provider_cost
        elif math.isclose(
            upstream_usage,
            local_provider + recovered_provider_total,
            abs_tol=OPENROUTER_USAGE_AUDIT_TOLERANCE_USD,
        ):
            record.update(
                {
                    "state": "rejected_not_billed",
                    "completed_at": utc_now(),
                    "provider_reported_cost_usd": 0.0,
                    "undiscounted_cost_usd": 0.0,
                    "benchmark_cost_usd": 0.0,
                    "promotion_adjustment_usd": 0.0,
                    "deepseek_peak_adjustment_usd": 0.0,
                    "benchmark_adjustment_usd": 0.0,
                    "provider_cost_basis": PROVIDER_COST_BASIS,
                    "reconciled_from_child_key_total": True,
                    "recovered_after_controller_shutdown": True,
                }
            )
        else:
            raise OpenRouterManagementError(
                "pending provider charge has no recoverable generation ID"
            )
        replacements[path.stem] = record

    rebuilt = _rebuild_provider_summary(run_id, request_paths, replacements)
    if rebuilt["pending_request_count"] != 0 or not math.isclose(
        float(rebuilt["provider_billed_model_api_usd"]),
        upstream_usage,
        abs_tol=OPENROUTER_USAGE_AUDIT_TOLERANCE_USD,
    ):
        raise OpenRouterManagementError(
            "recovered ledger did not reconcile to child-key usage"
        )

    _, run = sprintctl.load_run(run_id)
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-ledger-") as raw_temp:
        temp = Path(raw_temp)
        staged_records: list[tuple[Path, Path]] = []
        for stem, record in replacements.items():
            staged = temp / f"{stem}.json"
            atomic_json(staged, record)
            staged_records.append((requests_dir / f"{stem}.json", staged))
            sprintctl.volume_upload(
                run, staged, f"runs/{run_id}/api-usage/requests/{stem}.json"
            )
        staged_summary = temp / "summary.json"
        atomic_json(staged_summary, rebuilt)
        sprintctl.volume_upload(
            run, staged_summary, f"runs/{run_id}/api-usage/summary.json"
        )
        for destination, staged in staged_records:
            atomic_json(destination, json.loads(staged.read_text()))
        atomic_json(summary_path, rebuilt)
    return True


def audit_openrouter_child_usage(
    payload: dict[str, Any],
    client: OpenRouterManagementClient,
    *,
    now: dt.datetime,
) -> list[dict[str, str]]:
    """Detect use of a child key that bypassed the trusted ledger proxy."""
    alerts: list[dict[str, str]] = []
    for arm in payload.get("arms", []):
        credential = arm.get("openrouter_credential") or {}
        key_hash = credential.get("key_hash")
        if not key_hash:
            continue
        try:
            key = client.key_usage(str(key_hash))
            upstream_usage = float(key.get("usage", 0.0) or 0.0)
            local_usage, pending, local_as_of = _local_provider_billed_cost(
                arm["run_id"]
            )
            if local_as_of is None:
                launched_at = arm.get("launched_at")
                try:
                    startup_age = (now - parse_time(str(launched_at))).total_seconds()
                except (TypeError, ValueError):
                    startup_age = None
                if (
                    arm.get("status") in {"launching", "launched", "running"}
                    and startup_age is not None
                    and 0 <= startup_age < OPENROUTER_USAGE_AUDIT_GRACE_SECONDS
                ):
                    audit = arm.setdefault("openrouter_usage_audit", {})
                    audit.update(
                        {
                            "checked_at": utc_now(),
                            "key_usage_usd": upstream_usage,
                            "ledger_provider_billed_usd": None,
                            "unreconciled_usd": upstream_usage,
                            "pending_request_count": None,
                            "ledger_as_of": None,
                            "reconciliation_deferred": (
                                "trusted_proxy_ledger_starting"
                            ),
                            "startup_age_seconds": round(startup_age, 3),
                        }
                    )
                    continue
                raise OpenRouterManagementError(
                    "trusted proxy usage ledger was unavailable"
                )
            if not math.isfinite(upstream_usage) or upstream_usage < 0:
                raise OpenRouterManagementError("API key usage was invalid")
            if (
                pending > 0
                and arm_terminal(arm)
                and reconcile_openrouter_child_ledger(
                    arm, client, upstream_usage=upstream_usage
                )
            ):
                local_usage, pending, local_as_of = _local_provider_billed_cost(
                    arm["run_id"]
                )
        except (OpenRouterManagementError, TypeError, ValueError) as exc:
            alerts.append(
                {
                    "run_id": arm["run_id"],
                    "kind": "openrouter_key_usage_audit",
                    "source": type(exc).__name__,
                    "count_in_tail": "1",
                }
            )
            continue
        delta = upstream_usage - local_usage
        audit = arm.setdefault("openrouter_usage_audit", {})
        audit.update(
            {
                "checked_at": utc_now(),
                "key_usage_usd": upstream_usage,
                "ledger_provider_billed_usd": local_usage,
                "unreconciled_usd": max(0.0, delta),
                "pending_request_count": pending,
                "ledger_as_of": local_as_of,
            }
        )
        if delta <= OPENROUTER_USAGE_AUDIT_TOLERANCE_USD:
            audit.pop("mismatch_first_seen_at", None)
            audit.pop("reconciliation_deferred", None)
            audit.pop("startup_age_seconds", None)
            resolve_alerts(
                payload,
                run_id=arm["run_id"],
                kind="openrouter_key_usage_audit",
                resolution="trusted child-key and proxy usage audit is healthy",
            )
            resolve_alerts(
                payload,
                run_id=arm["run_id"],
                kind="openrouter_proxy_bypass",
                resolution="child-key usage reconciled with trusted proxy ledger",
            )
            continue
        # OpenRouter can publish the charge immediately before the trusted
        # proxy commits its terminal usage record.  A proxy-tracked in-flight
        # request is therefore evidence of incomplete reconciliation, not a
        # bypass.  Do not let that normal window consume the bypass grace
        # period; a request made outside the proxy cannot increment this
        # trusted counter.
        if pending > 0:
            audit.pop("mismatch_first_seen_at", None)
            audit["reconciliation_deferred"] = "trusted_proxy_request_in_flight"
            continue
        audit.pop("reconciliation_deferred", None)
        first = audit.setdefault("mismatch_first_seen_at", utc_now())
        try:
            mismatch_age = (now - parse_time(first)).total_seconds()
        except (TypeError, ValueError):
            mismatch_age = 0
            audit["mismatch_first_seen_at"] = utc_now()
        if mismatch_age < OPENROUTER_USAGE_AUDIT_GRACE_SECONDS:
            continue
        alerts.append(
            {
                "run_id": arm["run_id"],
                "kind": "openrouter_proxy_bypass",
                "source": "child_key_usage",
                "count_in_tail": "1",
            }
        )
        if arm.get("stop_requested_at") is None:
            sprintctl.persist_stop_request(
                arm["run_id"], reason="openrouter_proxy_bypass_detected"
            )
            arm["stop_requested_at"] = utc_now()
            arm["status"] = "stopping"
    return alerts


def revoke_batch_credentials(payload: dict[str, Any], env_file: Path) -> list[str]:
    if payload.get("credential_status") == "revoked":
        return list(payload.get("credential_cleanup_errors", []))
    keys = load_env(env_file)
    client = OpenRouterManagementClient(keys["OPENROUTER_MANAGEMENT_KEY"])
    errors = revoke_trial_credentials(
        client,
        payload.get("openrouter_credentials", []),
        journal_path=credential_journal_path(payload["batch_id"]),
    )
    payload["credential_status"] = "cleanup_error" if errors else "revoked"
    payload["credentials_revoked_at"] = utc_now()
    payload["credential_cleanup_errors"] = errors
    return errors


def provider_models(url: str, key: str) -> set[str]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    payload: dict[str, Any] | None = None
    for attempt in range(PROVIDER_DISCOVERY_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600 and attempt + 1 < PROVIDER_DISCOVERY_ATTEMPTS:
                time.sleep(PROVIDER_DISCOVERY_RETRY_SECONDS * (2**attempt))
                continue
            raise RuntimeError(
                f"provider model-list request failed: HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 < PROVIDER_DISCOVERY_ATTEMPTS:
                time.sleep(PROVIDER_DISCOVERY_RETRY_SECONDS * (2**attempt))
                continue
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"provider model-list request failed: {reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("provider model-list response was not an object")
    return {
        str(row.get("id"))
        for row in payload.get("data", [])
        if isinstance(row, dict) and row.get("id")
    }


def provider_inference_probe(
    url: str,
    key: str,
    payload: dict[str, Any],
    *,
    generation_audit_url: str | None = None,
) -> dict[str, Any]:
    """Make a tiny paid request and retain only non-secret serving evidence."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    result: Any = None
    generation_id: str | None = None
    for attempt in range(PROVIDER_INFERENCE_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                headers = getattr(response, "headers", None)
                raw_generation_id = (
                    headers.get("X-Generation-Id") if headers is not None else None
                )
                if raw_generation_id:
                    generation_id = str(raw_generation_id).strip() or None
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt + 1 < PROVIDER_INFERENCE_ATTEMPTS:
                time.sleep(min(PROVIDER_INFERENCE_RETRY_SECONDS * (2**attempt), 30.0))
                continue
            detail = ""
            try:
                error_payload = json.loads(exc.read(4096))
                error = error_payload.get("error", error_payload)
                if isinstance(error, dict):
                    parts = [
                        error.get("code"),
                        error.get("type"),
                        error.get("message"),
                    ]
                    detail = ": ".join(str(part) for part in parts if part)
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                pass
            detail = " ".join(detail.split())[:500]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"provider inference request failed: HTTP {exc.code}{suffix}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 < PROVIDER_INFERENCE_ATTEMPTS:
                time.sleep(min(PROVIDER_INFERENCE_RETRY_SECONDS * (2**attempt), 30.0))
                continue
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"provider inference request failed: {reason}") from exc
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("provider inference response lacked a request ID")
    generation: dict[str, Any] = {}
    if generation_audit_url:
        audit_id = generation_id or str(result["id"])
        audit_url = (
            generation_audit_url + "?" + urllib.parse.urlencode({"id": audit_id})
        )
        audit_request = urllib.request.Request(
            audit_url,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
        )
        for attempt in range(PROVIDER_GENERATION_AUDIT_ATTEMPTS):
            try:
                with urllib.request.urlopen(audit_request, timeout=30) as response:
                    audit_payload = json.load(response)
                candidate = audit_payload.get("data")
                if isinstance(candidate, dict):
                    generation = candidate
                    break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {404, 429} or 500 <= exc.code < 600
                if retryable and attempt + 1 < PROVIDER_GENERATION_AUDIT_ATTEMPTS:
                    time.sleep(PROVIDER_DISCOVERY_RETRY_SECONDS * (2**attempt))
                    continue
                raise RuntimeError(
                    f"provider generation audit failed: HTTP {exc.code}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt + 1 < PROVIDER_GENERATION_AUDIT_ATTEMPTS:
                    time.sleep(PROVIDER_DISCOVERY_RETRY_SECONDS * (2**attempt))
                    continue
                reason = getattr(exc, "reason", str(exc))
                raise RuntimeError(
                    f"provider generation audit failed: {reason}"
                ) from exc
        if not generation:
            raise RuntimeError("provider generation audit response was incomplete")
    usage = result.get("usage")
    return {
        "checked_at": utc_now(),
        "endpoint": url,
        "request_id": str(result["id"]),
        "generation_id": generation_id,
        "response_model": str(result.get("model", "")),
        "status": str(result.get("status", result.get("object", ""))),
        "service_tier": result.get("service_tier"),
        "provider": generation.get("provider_name", result.get("provider")),
        "resolved_model": generation.get("model"),
        "preset_id": generation.get("preset_id"),
        "provider_responses": generation.get("provider_responses", []),
        "provider_reported_total_cost_usd": generation.get(
            "total_cost", (usage or {}).get("cost") if isinstance(usage, dict) else None
        ),
        "usage": usage if isinstance(usage, dict) else {},
    }


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout


def sprint_modal_resource_audit(
    *, modal_profile: str, allowed_app_prefixes: tuple[str, ...] = ()
) -> tuple[bool, dict[str, Any]]:
    """Fail closed when an older Sprint App or container is still live.

    Modal's App list retains stopped history, so stopped Sprint Apps are valid
    evidence of cleanup.  A deployed App with zero tasks is still rejected:
    besides keeping the dashboard unambiguous, stopping it prevents a later
    name lookup from accidentally reusing stale App state.  Non-Sprint Apps
    are deliberately outside this benchmark's ownership boundary.
    """
    env = dict(os.environ)
    env["MODAL_PROFILE"] = modal_profile
    try:
        apps = json.loads(
            run_checked(
                [str(HARBOR_PYTHON), "-m", "modal", "app", "list", "--json"],
                env=env,
            )
        )
        containers = json.loads(
            run_checked(
                [
                    str(HARBOR_PYTHON),
                    "-m",
                    "modal",
                    "container",
                    "list",
                    "--json",
                ],
                env=env,
            )
        )
    except (subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        return False, {
            "checked_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(apps, list) or not isinstance(containers, list):
        return False, {
            "checked_at": utc_now(),
            "error": "Modal resource listings were not arrays",
        }

    def allowed(name: object) -> bool:
        return any(
            str(name or "").startswith(prefix) for prefix in allowed_app_prefixes
        )

    live_apps = [
        {
            "app_id": row.get("app_id"),
            "description": row.get("description"),
            "state": row.get("state"),
            "tasks": row.get("tasks"),
        }
        for row in apps
        if isinstance(row, dict)
        and str(row.get("description") or "").startswith("sprint-")
        and str(row.get("state") or "").lower() != "stopped"
        and not allowed(row.get("description"))
    ]
    live_containers = [
        {
            "container_id": row.get("container_id"),
            "app_id": row.get("app_id"),
            "app_name": row.get("app_name"),
            "start_time": row.get("start_time"),
        }
        for row in containers
        if isinstance(row, dict)
        and str(row.get("app_name") or "").startswith("sprint-")
        and not allowed(row.get("app_name"))
    ]
    report = {
        "checked_at": utc_now(),
        "modal_profile": modal_profile,
        "live_sprint_apps": live_apps,
        "live_sprint_containers": live_containers,
        "allowed_live_prefixes": list(allowed_app_prefixes),
        "unrelated_resources_ignored": True,
    }
    return not live_apps and not live_containers, report


def vercel_project_link_ready() -> bool:
    """Accept Vercel CLI metadata while requiring the exact Sprint project."""
    try:
        frontier_update.verify_project_link(WEB)
    except RuntimeError:
        return False
    return True


def deployment_retry_due(
    deploy_state: dict[str, Any], *, now: dt.datetime | None = None
) -> bool:
    retry_not_before = deploy_state.get("retry_not_before")
    if not isinstance(retry_not_before, str) or not retry_not_before:
        return True
    try:
        boundary = parse_time(retry_not_before)
    except (TypeError, ValueError):
        return True
    return (now or dt.datetime.now(dt.timezone.utc)) >= boundary


def record_deployment_error(
    deploy_state: dict[str, Any],
    exc: Exception,
    *,
    now: dt.datetime | None = None,
) -> None:
    observed = now or dt.datetime.now(dt.timezone.utc)
    message = " ".join(str(exc).split())[-2000:]
    deploy_state["last_error"] = {
        "at": observed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type": type(exc).__name__,
        "message": message,
    }
    if VERCEL_DAILY_QUOTA_CODE in message:
        deploy_state["site_status"] = "quota_limited"
        deploy_state["retry_not_before"] = (
            observed + dt.timedelta(seconds=VERCEL_DAILY_QUOTA_BACKOFF_SECONDS)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        deploy_state["quota_code"] = VERCEL_DAILY_QUOTA_CODE
    else:
        deploy_state["site_status"] = "error"


def clear_deployment_error(deploy_state: dict[str, Any]) -> None:
    for key in ("last_error", "retry_not_before", "quota_code"):
        deploy_state.pop(key, None)


def preflight(
    *,
    batch_id: str,
    env_file: Path,
    modal_profile: str,
    require_fresh: bool = True,
    check_providers: bool = True,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    trials_per_model: int = TRIALS_PER_MODEL,
    probe_training_fleet: bool = False,
    coexist_batch_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    provider_probes: dict[str, Any] = {}
    provider_errors: dict[str, str] = {}
    openrouter_credit_snapshot: dict[str, Any] | None = None
    sprint_resource_report: dict[str, Any] | None = None
    keys = load_env(env_file)
    planned = matrix(
        batch_id,
        trials_per_model=trials_per_model,
        families=families,
    )
    required_keys = {
        "deepseek": "OPENROUTER_API_KEY",
        "luna": "OPENROUTER_API_KEY",
        "sol": "OPENROUTER_API_KEY",
        "flash-baidu": "OPENROUTER_API_KEY",
        "pro-alibaba": "OPENROUTER_API_KEY",
    }
    for family in families:
        name = required_keys[family]
        checks[f"secret_{name.lower()}"] = len(keys.get(name, "")) >= 16
    checks["secret_openrouter_management_key"] = (
        len(keys.get("OPENROUTER_MANAGEMENT_KEY", "")) >= 16
    )
    if check_providers and checks["secret_openrouter_management_key"]:
        try:
            verify_openrouter_management_access(keys["OPENROUTER_MANAGEMENT_KEY"])
            checks["openrouter_management_access"] = True
        except (OpenRouterManagementError, ValueError) as exc:
            checks["openrouter_management_access"] = False
            provider_errors["openrouter_management"] = str(exc)
    else:
        checks["openrouter_management_access"] = not check_providers
    auto_recharge_confirmed = os.environ.get(
        "SPRINT_OPENROUTER_AUTO_RECHARGE_CONFIRMED", ""
    ).strip().lower() in {"1", "true", "yes"}
    if check_providers and checks.get("secret_openrouter_api_key"):
        requirement = openrouter_credit_requirement(len(planned))
        try:
            openrouter_credit_snapshot = {
                **fetch_openrouter_credit(keys["OPENROUTER_API_KEY"]),
                **requirement,
                "auto_recharge_confirmed": auto_recharge_confirmed,
            }
            checks["openrouter_credit_query"] = True
            checks["openrouter_credit_headroom"] = bool(
                auto_recharge_confirmed
                or openrouter_credit_snapshot["remaining_credit_usd"]
                >= requirement["required_credit_usd"]
            )
        except RuntimeError as exc:
            checks["openrouter_credit_query"] = False
            checks["openrouter_credit_headroom"] = False
            provider_errors["openrouter_credit"] = str(exc)
    else:
        checks["openrouter_credit_query"] = not check_providers
        checks["openrouter_credit_headroom"] = not check_providers
    checks["harbor_revision"] = (
        ROOT / "harbor/.sprint-upstream-commit"
    ).read_text().strip() == HARBOR_REVISION
    goal_template = MODULE_DIR / "templates" / "codex.j2"
    checks["goal_template"] = bool(
        goal_template.is_file()
        and goal_template.read_text().startswith("/goal ")
        and "{{ instruction }}" in goal_template.read_text()
    )
    checks["warm_images"] = (
        subprocess.run(
            [sys.executable, str(PREFLIGHT_DIR / "check_images.py")],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    checks["functional_gpu_canary"] = functional_gpu_canary_ready()
    checks["vercel_project_link"] = vercel_project_link_ready()
    checks["controller_runtime"] = (
        HARBOR_PYTHON.is_file()
        and os.access(HARBOR_PYTHON, os.X_OK)
        and subprocess.run(
            [str(HARBOR_PYTHON), "-c", "import modal"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    command_env = dict(os.environ)
    command_env["MODAL_PROFILE"] = modal_profile
    checks["modal_auth"] = False
    try:
        json.loads(
            run_checked(
                [
                    str(HARBOR_PYTHON),
                    "-m",
                    "modal",
                    "app",
                    "list",
                    "--json",
                ],
                env=command_env,
            )
        )
        checks["modal_auth"] = True
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        pass
    if checks["modal_auth"]:
        valid_coexistence = all(
            RUN_ID_RE.fullmatch(existing_id) and batch_path(existing_id).is_file()
            for existing_id in coexist_batch_ids
        )
        checks["coexisting_batches_valid"] = valid_coexistence
        allowed_prefixes = (
            tuple(f"sprint-{existing_id}-" for existing_id in coexist_batch_ids)
            if valid_coexistence
            else ()
        )
        checks["no_live_sprint_resources"], sprint_resource_report = (
            sprint_modal_resource_audit(
                modal_profile=modal_profile,
                allowed_app_prefixes=allowed_prefixes,
            )
        )
    else:
        checks["no_live_sprint_resources"] = False
    checks["vercel_auth"] = (
        subprocess.run(
            ["vercel", "whoami"],
            cwd=WEB,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    selected_openai_families = tuple(
        family for family in families if family in OPENAI_FAMILY_SPECS
    )
    if (
        selected_openai_families
        and check_providers
        and checks["secret_openrouter_api_key"]
    ):
        models = provider_models(
            "https://openrouter.ai/api/v1/models", keys["OPENROUTER_API_KEY"]
        )
        for family in selected_openai_families:
            spec = OPENAI_FAMILY_SPECS[family]
            key = f"openai_{family}"
            checks[f"{key}_visible"] = spec["model"] in models
            try:
                provider_probes[key] = provider_inference_probe(
                    "https://openrouter.ai/api/v1/responses",
                    keys["OPENROUTER_API_KEY"],
                    {
                        # Exercise the raw model route exactly as the trusted
                        # proxy does. A preset probe can hide unsupported
                        # parameters because its routing policy differs from
                        # our fail-closed provider contract.
                        "model": spec["model"],
                        "input": "Return OK.",
                        "provider": {
                            "only": [spec["provider_endpoint"]],
                            "order": [spec["provider_endpoint"]],
                            "allow_fallbacks": False,
                            "require_parameters": True,
                        },
                        "reasoning": {"effort": REASONING_EFFORT},
                        "max_output_tokens": 128_000,
                        "service_tier": "default",
                        "tools": [
                            {
                                "type": "function",
                                "name": "sprint_preflight_noop",
                                "description": "Preflight-only no-op tool.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": False,
                                },
                            }
                        ],
                        "tool_choice": "auto",
                        "include": ["reasoning.encrypted_content"],
                        "prompt_cache_key": "sprint-provider-preflight",
                        "store": False,
                    },
                    generation_audit_url="https://openrouter.ai/api/v1/generation",
                )
                probe = provider_probes[key]
                checks[f"{key}_inference"] = (
                    probe.get("provider") == "OpenAI"
                    and probe.get("resolved_model") == spec["resolved_model"]
                    and probe.get("service_tier") == "default"
                )
                if not checks[f"{key}_inference"]:
                    provider_errors[key] = (
                        "controlled preset response did not identify OpenAI standard"
                    )
            except RuntimeError as exc:
                checks[f"{key}_inference"] = False
                provider_errors[key] = str(exc)
    else:
        for family in selected_openai_families:
            key = f"openai_{family}"
            checks[f"{key}_visible"] = not check_providers
            checks[f"{key}_inference"] = not check_providers
    selected_routed_deepseek_families = tuple(
        family for family in families if family in DEEPSEEK_ROUTED_FAMILY_SPECS
    )
    if (
        selected_routed_deepseek_families
        and check_providers
        and checks["secret_openrouter_api_key"]
    ):
        models = provider_models(
            "https://openrouter.ai/api/v1/models", keys["OPENROUTER_API_KEY"]
        )
        for family in selected_routed_deepseek_families:
            spec = DEEPSEEK_ROUTED_FAMILY_SPECS[family]
            key = f"deepseek_{family.replace('-', '_')}"
            checks[f"{key}_visible"] = spec["model"] in models
            provider = {
                "only": [spec["provider_endpoint"]],
                "order": [spec["provider_endpoint"]],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            if spec["quantization"] != "unknown":
                provider["quantizations"] = [spec["quantization"]]
            if spec["wire_api"] == "chat_completions":
                inference_url = "https://openrouter.ai/api/v1/chat/completions"
                inference_payload: dict[str, Any] = {
                    "model": spec["model"],
                    "messages": [{"role": "user", "content": "Return OK."}],
                    "provider": provider,
                    "reasoning_effort": REASONING_EFFORT,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 16,
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "sprint_preflight_noop",
                                "description": "Preflight-only no-op tool.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"ack": {"type": "string"}},
                                    "required": ["ack"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                }
            else:
                inference_url = "https://openrouter.ai/api/v1/responses"
                inference_payload = {
                    "model": spec["model"],
                    "input": "Return OK.",
                    "provider": provider,
                    "reasoning": {"effort": REASONING_EFFORT},
                    "max_output_tokens": 16,
                    # Exercise the same Responses features Codex adds to a
                    # real agent turn. A text.verbosity field is
                    # intentionally absent because routed DeepSeek endpoints
                    # do not advertise or accept it under strict routing.
                    "tools": [
                        {
                            "type": "function",
                            "name": "sprint_preflight_noop",
                            "description": "Preflight-only no-op tool.",
                            "parameters": {
                                "type": "object",
                                "properties": {"ack": {"type": "string"}},
                                "required": ["ack"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "include": ["reasoning.encrypted_content"],
                    "prompt_cache_key": "sprint-provider-preflight",
                    "store": False,
                }
            try:
                provider_probes[key] = provider_inference_probe(
                    inference_url,
                    keys["OPENROUTER_API_KEY"],
                    inference_payload,
                    generation_audit_url="https://openrouter.ai/api/v1/generation",
                )
                probe = provider_probes[key]
                checks[f"{key}_inference"] = (
                    probe.get("provider") == spec["provider"]
                    and probe.get("response_model") == spec["model"]
                )
                if not checks[f"{key}_inference"]:
                    provider_errors[key] = (
                        "sealed route response did not identify the requested provider"
                    )
            except RuntimeError as exc:
                checks[f"{key}_inference"] = False
                provider_errors[key] = str(exc)
    else:
        for family in selected_routed_deepseek_families:
            key = f"deepseek_{family.replace('-', '_')}"
            checks[f"{key}_visible"] = not check_providers
            checks[f"{key}_inference"] = not check_providers
    checks["unique_runs"] = len({arm["run_id"] for arm in planned}) == len(planned)
    checks["fresh_run_ids"] = not any(
        (SCRIPT_DIR / arm["run_id"]).exists() for arm in planned
    )
    fleet_probe: dict[str, Any] | None = None
    if probe_training_fleet:
        checks["training_gpu_fleet"], fleet_probe = training_gpu_fleet_probe(
            batch_id=batch_id,
            modal_profile=modal_profile,
            worker_ids=[arm["run_id"] for arm in planned],
        )
    if require_fresh:
        checks["fresh_batch_id"] = not batch_path(batch_id).exists()
    ready = all(bool(value) for value in checks.values())
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "batch_id": batch_id,
        "modal_profile": modal_profile,
        "families": list(families),
        "trials_per_model": trials_per_model,
        "coexist_batch_ids": list(coexist_batch_ids),
        "env_file": str(env_file),
        "checks": checks,
        "provider_probes": provider_probes,
        "provider_errors": provider_errors,
        "openrouter_credit_snapshot": openrouter_credit_snapshot,
        "sprint_modal_resource_audit": sprint_resource_report,
        "training_gpu_fleet_probe": fleet_probe,
        "ready": ready,
    }


def start_monitor_service(batch_id: str, env_file: Path, modal_profile: str) -> None:
    vercel = shutil.which("vercel")
    if not vercel:
        raise RuntimeError("vercel CLI is not available for the batch monitor")
    service_path = os.pathsep.join(
        dict.fromkeys(
            [
                str(HARBOR_PYTHON.parent),
                str(UV.resolve().parent),
                str(Path(vercel).resolve().parent),
                *os.environ.get("PATH", "").split(os.pathsep),
            ]
        )
    )
    unit = f"sprint-batch-{batch_id}-monitor"
    unit_name = f"{unit}.service"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    unit_path = config_home / "systemd" / "user" / unit_name

    def quote(value: str | Path) -> str:
        raw = str(value)
        if "\n" in raw or "\r" in raw:
            raise ValueError("systemd unit values cannot contain newlines")
        escaped = raw.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
        return f'"{escaped}"'

    monitor_command = [
        HARBOR_PYTHON,
        Path(__file__).resolve(),
        "monitor",
        "--batch-id",
        batch_id,
        "--env-file",
        env_file.resolve(),
        "--modal-profile",
        modal_profile,
        "--loop",
    ]
    unit_text = "\n".join(
        [
            "[Unit]",
            f"Description=Sprint batch monitor for {batch_id}",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={ROOT}",
            f"Environment={quote(f'PATH={service_path}')}",
            f"Environment={quote(f'UV={UV.resolve()}')}",
            f"Environment={quote(f'MODAL_PROFILE={modal_profile}')}",
            "ExecStart=" + " ".join(quote(item) for item in monitor_command),
            "Restart=on-failure",
            "RestartSec=30",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    subprocess.run(
        ["systemctl", "--user", "stop", unit_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    frontier_update.atomic_write_text(unit_path, unit_text, mode=0o600)
    run_checked(["systemctl", "--user", "daemon-reload"])
    run_checked(["systemctl", "--user", "enable", unit_name])
    run_checked(["systemctl", "--user", "restart", unit_name])


def launch(
    batch_id: str,
    env_file: Path,
    modal_profile: str,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    trials_per_model: int = TRIALS_PER_MODEL,
    coexist_batch_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    report = preflight(
        batch_id=batch_id,
        env_file=env_file,
        modal_profile=modal_profile,
        families=families,
        trials_per_model=trials_per_model,
        probe_training_fleet=True,
        coexist_batch_ids=coexist_batch_ids,
    )
    if not report["ready"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError("preflight failed: " + ", ".join(failed))
    state_dir = batch_dir(batch_id)
    state_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    payload = {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": started,
        "reasoning_effort": REASONING_EFFORT,
        "codex_version": CODEX_VERSION,
        "trials_per_model": trials_per_model,
        "families": list(families),
        "coexist_batch_ids": list(coexist_batch_ids),
        "run_hours": RUN_HOURS,
        "site_deploy_interval_seconds": LIVE_SITE_DEPLOY_SECONDS,
        "modal_profile": modal_profile,
        "env_file": str(env_file.resolve()),
        "preflight": report,
        "arms": matrix(
            batch_id,
            trials_per_model=trials_per_model,
            families=families,
        ),
        "alerts": [],
        "deploy": {},
        "status": "launching",
    }
    atomic_json(batch_path(batch_id), payload)
    keys = load_env(env_file)
    management_client = OpenRouterManagementClient(keys["OPENROUTER_MANAGEMENT_KEY"])
    credential_specs = [
        TrialCredentialSpec(
            run_id=arm["run_id"],
            model=arm["model"],
            resolved_model=arm["resolved_model_version"],
            provider=arm["provider_endpoint"],
            budget_usd=configured_agent_budget_usd(),
        )
        for arm in payload["arms"]
    ]
    credentials = provision_trial_credentials(
        management_client,
        credential_specs,
        journal_path=credential_journal_path(batch_id),
    )
    credentials_by_run = {credential.run_id: credential for credential in credentials}
    payload["openrouter_credentials"] = [
        credential.public_metadata() for credential in credentials
    ]
    payload["credential_status"] = "active"
    for arm in payload["arms"]:
        arm["openrouter_credential"] = credentials_by_run[
            arm["run_id"]
        ].public_metadata()
    atomic_json(batch_path(batch_id), payload)
    base_env = dict(os.environ)
    base_env.update(keys)
    # Management authority and the unrestricted workspace key must never enter
    # an untrusted agent sandbox. Each arm receives only its sealed child key.
    base_env.pop("OPENROUTER_MANAGEMENT_KEY", None)
    base_env.pop("OPENROUTER_API_KEY", None)
    base_env.pop("OPENAI_API_KEY", None)
    base_env.update(
        {
            "MODAL_PROFILE": modal_profile,
            "CONFIRM_LAUNCH": "1",
            "REASONING_EFFORT": REASONING_EFFORT,
            "CODEX_VERSION": CODEX_VERSION,
            "SPRINT_BATCH_ID": batch_id,
            "UV": str(UV),
        }
    )
    launched: list[dict[str, Any]] = []
    try:
        for arm in payload["arms"]:
            arm["launch_started_at"] = utc_now()
            atomic_json(batch_path(batch_id), payload)
            env = dict(base_env)
            child_key = credentials_by_run[arm["run_id"]].api_key
            env["OPENROUTER_API_KEY"] = child_key
            env["OPENAI_API_KEY"] = child_key
            env["RUN_ID"] = arm["run_id"]
            env["MODEL"] = arm["model"]
            if arm.get("openrouter_preset"):
                env["OPENROUTER_PRESET"] = arm["openrouter_preset"]
            if arm.get("provider_endpoint"):
                env["OPENROUTER_MODEL"] = arm["model"]
                env["SPRINT_OPENROUTER_PROVIDER_ENDPOINT"] = arm["provider_endpoint"]
                if arm.get("quantization") not in {None, "unknown"}:
                    env["SPRINT_OPENROUTER_QUANTIZATION"] = arm["quantization"]
                if arm["wrapper"].endswith("deepseek.sh"):
                    env["SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW"] = arm["context_window"]
            output = run_checked([arm["wrapper"]], env=env)
            arm["status"] = "launched"
            arm["launch_output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
            arm["launched_at"] = utc_now()
            arm["deadline_at"] = (
                (
                    parse_time(arm["launched_at"]) + dt.timedelta(hours=RUN_HOURS)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                if RUN_HOURS is not None
                else None
            )
            launched.append(arm)
            atomic_json(batch_path(batch_id), payload)
            time.sleep(2)
    except Exception as exc:
        arm["status"] = "launch_error"
        arm["launch_error"] = f"{type(exc).__name__}: {exc}"
        payload["status"] = "rolling_back_partial_launch"
        for started_arm in launched:
            try:
                sprintctl.request_stop(
                    started_arm["run_id"], reason="partial_batch_launch_rollback"
                )
                started_arm["status"] = "stopping_after_launch_rollback"
                started_arm["stop_requested_at"] = utc_now()
            except Exception as stop_exc:  # noqa: BLE001
                started_arm["rollback_error"] = f"{type(stop_exc).__name__}: {stop_exc}"
        payload["status"] = "launch_error"
        payload["credential_cleanup_errors"] = revoke_trial_credentials(
            management_client,
            payload.get("openrouter_credentials", []),
            journal_path=credential_journal_path(batch_id),
        )
        payload["credential_status"] = (
            "cleanup_error" if payload["credential_cleanup_errors"] else "revoked"
        )
        atomic_json(batch_path(batch_id), payload)
        raise
    payload["status"] = "running"
    payload["launched_at"] = utc_now()
    atomic_json(batch_path(batch_id), payload)
    try:
        start_monitor_service(batch_id, env_file, modal_profile)
    except Exception:
        for started_arm in launched:
            try:
                sprintctl.request_stop(
                    started_arm["run_id"], reason="batch_monitor_start_failed"
                )
            except Exception:  # noqa: BLE001 - persisted state documents cleanup
                pass
        payload["status"] = "launch_error"
        payload["credential_cleanup_errors"] = revoke_trial_credentials(
            management_client,
            payload.get("openrouter_credentials", []),
            journal_path=credential_journal_path(batch_id),
        )
        payload["credential_status"] = (
            "cleanup_error" if payload["credential_cleanup_errors"] else "revoked"
        )
        atomic_json(batch_path(batch_id), payload)
        raise
    return payload


def log_alerts(run_id: str) -> list[dict[str, str]]:
    paths = [
        Path(f"/data/sprint-launch-{run_id}.log"),
        SCRIPT_DIR / run_id / "monitor.log",
        SCRIPT_DIR / run_id / "frontier-worker.log",
        SCRIPT_DIR / run_id / "controller-errors.jsonl",
    ]
    alerts: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 256_000))
                text = handle.read().decode(errors="replace")
        except OSError:
            continue
        for kind, pattern in ALERT_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                alerts.append(
                    {
                        "run_id": run_id,
                        "kind": kind,
                        "source": path.name,
                        "count_in_tail": str(len(matches)),
                    }
                )
    return alerts


def verifier_lane_stall_alerts(
    payload: dict[str, Any], *, now: dt.datetime | None = None
) -> list[dict[str, str]]:
    """Alert independently when one trial's accepted verifier work stalls."""
    current = now or dt.datetime.now(dt.timezone.utc)
    alerts: list[dict[str, str]] = []
    for arm in payload.get("arms", []):
        run_id = arm.get("run_id")
        if not isinstance(run_id, str):
            continue
        pending = sum(
            int(arm.get("ledger", {}).get(state, 0) or 0)
            for state in ("queued", "running")
        )
        if pending == 0:
            continue
        progress: list[dt.datetime] = []
        run_dir = SCRIPT_DIR / run_id
        ledgers = list(
            (run_dir / "harbor-jobs").glob("*/*/artifacts/continuous/ledger.jsonl")
        )
        ledgers.extend(
            (run_dir / "cpu-attempts").glob(
                "*/harbor-jobs/*/*/artifacts/continuous/ledger.jsonl"
            )
        )
        for ledger in ledgers:
            try:
                rows = ledger.read_text().splitlines()
            except OSError:
                continue
            for raw in rows:
                try:
                    row = json.loads(raw)
                    if row.get("accepted", True) is False:
                        continue
                    for field in (
                        "submitted_at",
                        "accepted_at",
                        "started_at",
                        "scheduler_acquired_at",
                        "verification_started_at",
                        "finished_at",
                    ):
                        if row.get(field):
                            progress.append(parse_time(str(row[field])))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        if not progress:
            continue
        last_progress = max(progress)
        age_seconds = (current - last_progress).total_seconds()
        if age_seconds <= VERIFIER_LANE_STALL_SECONDS:
            continue
        alerts.append(
            {
                "run_id": run_id,
                "kind": "verifier_lane_stalled",
                "source": "continuous/ledger.jsonl",
                "count_in_tail": str(pending),
                "pending_submissions": str(pending),
                "last_progress_at": last_progress.isoformat(),
                "progress_age_seconds": str(round(age_seconds)),
            }
        )
    return alerts


def continuous_ledger_error_alerts(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Surface accepted policies that terminated without a trusted score."""
    alerts: list[dict[str, str]] = []
    for arm in payload.get("arms", []):
        count = int(arm.get("ledger", {}).get("error", 0) or 0)
        run_id = arm.get("run_id")
        if count and isinstance(run_id, str):
            alerts.append(
                {
                    "run_id": run_id,
                    "kind": "continuous_ledger_error",
                    "source": "continuous/ledger.jsonl",
                    "count_in_tail": str(count),
                }
            )
    return alerts


def live_run_monitor_status(run_id: str) -> dict[str, Any] | None:
    """Read the independent run monitor instead of duplicating its work.

    Each lane owns a long-lived ``sprintctl monitor`` process that performs
    telemetry, GPU dispatch, trace synchronization, and frontier updates.  The
    batch monitor is an observer/supervisor of those lane monitors; running a
    second full ``monitor_once`` serially here both wastes Modal API calls and
    delays later arms in the batch.
    """
    state_dir = SCRIPT_DIR / run_id
    # A stopped lane's monitor normally exits before the batch monitor's next
    # cycle. Its STOP_ACK is already authoritative local state; do not fall
    # back to a synchronous Modal Volume download that can hang publication
    # after the remote App has gone away.
    if sprintctl.terminal_stop_acknowledged(state_dir):
        try:
            _, run = sprintctl.load_run(run_id)
            return sprintctl.status_snapshot(state_dir, run, include_remote=False)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    try:
        pid = int((state_dir / "monitor.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    if not sprintctl.process_alive(pid, f"control/run.py monitor --run-id {run_id}"):
        return None
    try:
        payload = json.loads((state_dir / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        _, run = sprintctl.load_run(run_id)
        return sprintctl.status_snapshot(state_dir, run, include_remote=False)
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        return None
    return payload


def supervisor_active(run_id: str) -> bool:
    """Return whether the durable lane supervisor still owns its run lock."""
    lock_path = SCRIPT_DIR / run_id / "supervise.lock"
    if not lock_path.is_file():
        return False
    try:
        with lock_path.open("r+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        # Fail closed: an unreadable supervisor lock must not authorize
        # finalization while a replacement CPU attempt may still be starting.
        return True
    return False


def arm_terminal(arm: dict[str, Any]) -> bool:
    """Distinguish a terminal run from a supervised provider-retry boundary."""
    run_id = str(arm.get("run_id") or "")
    state_dir = SCRIPT_DIR / run_id
    if sprintctl.terminal_stop_acknowledged(state_dir):
        return True
    ack = arm.get("stop_ack")
    if ack:
        if not isinstance(ack, dict):
            return True
        if str(ack.get("reason") or "") != "agent_exit":
            return True
    return arm.get("harbor_alive") is False and not supervisor_active(run_id)


def public_batch(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batch_id": payload["batch_id"],
        "updated_at": payload.get("updated_at"),
        "status": payload.get("status"),
        "invalid_trial_count": payload.get("invalid_trial_count", 0),
        "replacement_required": bool(payload.get("replacement_required")),
        "reasoning_effort": payload["reasoning_effort"],
        "codex_version": payload["codex_version"],
        "run_hours": payload["run_hours"],
        "arms": [
            {
                key: arm.get(key)
                for key in (
                    "run_id",
                    "family",
                    "model",
                    "resolved_model_version",
                    "trial",
                    "status",
                    "launched_at",
                    "deadline_at",
                    "finalized_at",
                    "last_monitor_at",
                    "stop_requested_at",
                    "ledger",
                    "snapshot_heartbeat_ok",
                    "frontier_worker_alive",
                    "harbor_alive",
                    "benchmark_valid",
                    "replacement_required",
                    "invalidated_reason",
                    "integrity",
                )
            }
            for arm in payload["arms"]
        ],
        "alerts": payload.get("alerts", [])[-100:],
    }


def public_tracking_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """Combine explicitly coexisting batches for the active site pointer.

    Replacement lanes are deliberately launched in a new immutable batch so a
    failed lane remains auditable.  The website's active pointer should still
    show the replacement beside the comparison it belongs to.  Only batch IDs
    explicitly admitted by preflight are included, and each historical batch
    record remains unchanged.
    """

    current = public_batch(payload)
    coexist_ids = payload.get("coexist_batch_ids") or []
    if not coexist_ids:
        return current

    batch_payloads: list[dict[str, Any]] = []
    for batch_id in coexist_ids:
        if not isinstance(batch_id, str) or not batch_id:
            continue
        try:
            batch_payloads.append(read_batch(batch_id))
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
    batch_payloads.append(payload)

    arms_by_run: dict[str, dict[str, Any]] = {}
    excluded_arms: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    for batch_payload in batch_payloads:
        batch = public_batch(batch_payload)
        raw_arms = {
            arm.get("run_id"): arm
            for arm in batch_payload.get("arms", [])
            if isinstance(arm, dict) and arm.get("run_id")
        }
        for arm in batch.get("arms", []):
            run_id = arm.get("run_id")
            if isinstance(run_id, str) and run_id:
                invalidated_reason = raw_arms.get(run_id, {}).get("invalidated_reason")
                stop_reason = (raw_arms.get(run_id, {}).get("stop_ack") or {}).get(
                    "reason"
                )
                exclusion_reason = invalidated_reason or (
                    stop_reason
                    if stop_reason == "budget_telemetry_unavailable"
                    else None
                )
                if exclusion_reason:
                    excluded_arms.append(
                        {
                            "run_id": run_id,
                            "reason": exclusion_reason,
                            "source_batch_id": batch.get("batch_id"),
                        }
                    )
                    continue
                arms_by_run[run_id] = arm
        for alert in batch.get("alerts", []):
            alert_run_id = alert.get("run_id")
            if (
                alert.get("kind") == "finalization"
                and isinstance(alert_run_id, str)
                and (arms_by_run.get(alert_run_id) or {}).get("status") == "finalized"
            ):
                # A transient finalizer exception is no longer actionable once
                # the durable FINALIZED marker exists. Source batch monitors
                # may already be retired, so the replacement tracking owner
                # must suppress that recovered alert as well.
                continue
            # A replacement batch owns the combined performance snapshot and
            # production deployment.  Site-level alerts from a coexisting
            # source batch describe that source controller's obsolete view of
            # the website, not the tracked cohort.  Keep all per-run alerts,
            # and keep site alerts raised by the active tracking owner itself.
            if (
                batch_payload is not payload
                and alert.get("run_id") == "batch"
                and alert.get("kind")
                in {"performance_export", "website_deploy", "final_site_deploy"}
            ):
                continue
            alerts.append(alert)

    current["tracked_batch_ids"] = [
        batch["batch_id"]
        for batch in map(public_batch, batch_payloads)
        if batch.get("batch_id")
    ]
    current["arms"] = list(arms_by_run.values())
    current["excluded_arms"] = excluded_arms
    current["alerts"] = alerts[-100:]
    return current


def write_public_batch(payload: dict[str, Any], *, update_current: bool = True) -> Path:
    """Publish the batch record and, for its owning monitor, the active pointer."""
    public = public_batch(payload)
    public_path = WEB / "data" / "batches" / f"{payload['batch_id']}.json"
    atomic_public_json(public_path, public)
    if update_current:
        atomic_public_json(
            WEB / "data" / "batches" / "current.json",
            public_tracking_batch(payload),
        )
    return public_path


def resolve_alerts(
    payload: dict[str, Any], *, run_id: str, kind: str, resolution: str
) -> None:
    """Move a recovered active alert to durable history."""
    active: list[dict[str, Any]] = []
    resolved = payload.setdefault("resolved_alerts", [])
    for alert in payload.get("alerts", []):
        if alert.get("run_id") != run_id or alert.get("kind") != kind:
            active.append(alert)
            continue
        archived = dict(alert)
        archived["resolved_at"] = utc_now()
        archived["resolution"] = resolution
        resolved.append(archived)
    payload["alerts"] = active


def resolve_recovered_verifier_stalls(
    payload: dict[str, Any], active_stalls: list[dict[str, str]]
) -> None:
    """Archive verifier-stall alerts once that lane progresses or drains."""
    stalled_run_ids = {alert.get("run_id") for alert in active_stalls}
    for arm in payload.get("arms", []):
        run_id = arm.get("run_id")
        if isinstance(run_id, str) and run_id not in stalled_run_ids:
            resolve_alerts(
                payload,
                run_id=run_id,
                kind="verifier_lane_stalled",
                resolution="verifier lane progressed or cleared its pending queue",
            )


def deployment_debounce_seconds(payload: dict[str, Any]) -> int:
    """Publish immediately after every lane has reached a terminal state."""
    return (
        0
        if all(arm_terminal(arm) for arm in payload["arms"])
        else LIVE_SITE_DEPLOY_SECONDS
    )


def deployed_batch_current(payload: dict[str, Any]) -> bool:
    """Check only this batch's public files against the deployed manifest."""
    deploy_state = payload.get("deploy") or {}
    manifest = deploy_state.get("last_deployed_public_artifacts")
    if deploy_state.get("site_status") not in {"deployed", "noop"} or not isinstance(
        manifest, dict
    ):
        return False
    paths = [
        WEB / "data" / "batches" / f"{payload['batch_id']}.json",
        WEB / "data" / "batches" / "current.json",
    ]
    paths.append(WEB / "data" / "performance" / "current.json")
    for arm in payload["arms"]:
        run_id = arm["run_id"]
        policy = WEB / "data" / "policies" / f"{run_id}.json"
        paths.append(
            policy
            if policy.is_file()
            else WEB / "data" / "timelines" / f"{run_id}.json"
        )
    return all(
        path.is_file()
        and manifest.get(path.relative_to(WEB).as_posix())
        == frontier_update.sha256_file(path)
        for path in paths
    )


def refresh_performance_snapshot(payload: dict[str, Any]) -> None:
    """Publish the active batch's aggregate chart before a site deployment."""

    if payload.get("coexist_batch_ids"):
        tracking = public_tracking_batch(payload)
        run_ids = [arm["run_id"] for arm in tracking["arms"]]
    else:
        run_ids = [
            arm["run_id"] for arm in payload.get("arms", []) if arm.get("run_id")
        ]
    credit_snapshot = (payload.get("preflight") or {}).get(
        "openrouter_credit_snapshot"
    ) or {}
    per_trial_cost_cap = credit_snapshot.get("per_trial_budget_usd")
    if per_trial_cost_cap is None:
        per_trial_cost_cap = configured_agent_budget_usd()
    performance_export.build(
        f"{payload['batch_id']}-",
        WEB / "data" / "performance" / "current.json",
        cost_cap=float(per_trial_cost_cap),
        run_ids=run_ids,
    )


def _frontier_ready_for_publish(state: dict[str, Any]) -> bool:
    if any(
        item.get("status") in {"queued", "running"}
        for item in state.get("capture_queue", [])
    ):
        return False
    captures = state.get("captures", {})
    return all(
        not policy.get("replay_path")
        or captures.get(policy_hash, {}).get("valid") is True
        for policy_hash, policy in state.get("policies", {}).items()
    )


def mark_deployed_runs(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Persist durable proof that each run's current public artifact was deployed."""
    alerts: list[dict[str, str]] = []
    deploy_state = payload.get("deploy") or {}
    deployed_artifacts = deploy_state.get("last_deployed_public_artifacts")
    if deploy_state.get("site_status") not in {"deployed", "noop"} or not isinstance(
        deployed_artifacts, dict
    ):
        return alerts
    for arm in payload["arms"]:
        state_dir = SCRIPT_DIR / arm["run_id"]
        frontier_path = state_dir / "frontier-state.json"
        index_path = WEB / "data" / "policies" / f"{arm['run_id']}.json"
        timeline_path = WEB / "data" / "timelines" / f"{arm['run_id']}.json"
        try:
            frontier = json.loads(frontier_path.read_text())
            if not _frontier_ready_for_publish(frontier):
                continue
            public_artifact = index_path if index_path.is_file() else timeline_path
            if not public_artifact.is_file():
                continue
            relative_artifact = public_artifact.relative_to(WEB).as_posix()
            deployed_hash = deployed_artifacts.get(relative_artifact)
            if deployed_hash != frontier_update.sha256_file(public_artifact):
                continue
            artifact_snapshot = (
                state_dir / "deployment-provenance" / f"{deployed_hash}.json"
            )
            frontier_update.atomic_write_text(
                artifact_snapshot, public_artifact.read_text(), mode=0o444
            )
            marker = {
                "schema_version": 2,
                "run_id": arm["run_id"],
                "batch_id": payload["batch_id"],
                "deployed_at": deploy_state.get("last_deployed_at") or utc_now(),
                "deployment_site_sha256": deploy_state["last_deployed_site_hash"],
                "public_artifact_path": relative_artifact,
                "public_artifact_sha256": deployed_hash,
                "public_artifact_snapshot_path": str(
                    artifact_snapshot.relative_to(state_dir)
                ),
                "production_alias": "https://g1-sprint.vercel.app",
            }
            local_marker = state_dir / "BATCH_SITE_DEPLOYED.json"
            try:
                existing = json.loads(local_marker.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
            if (
                existing.get("batch_id") == marker["batch_id"]
                and existing.get("public_artifact_path")
                == marker["public_artifact_path"]
                and existing.get("public_artifact_sha256")
                == marker["public_artifact_sha256"]
                and existing.get("public_artifact_snapshot_path")
                == marker["public_artifact_snapshot_path"]
                and artifact_snapshot.is_file()
                and frontier_update.sha256_file(artifact_snapshot) == deployed_hash
            ):
                continue
            temp = batch_dir(payload["batch_id"]) / f".{arm['run_id']}-site.json"
            try:
                atomic_json(temp, marker)
                _, run = sprintctl.load_run(arm["run_id"])
                sprintctl.volume_upload(
                    run,
                    artifact_snapshot,
                    (
                        f"runs/{arm['run_id']}/state/deployment-provenance/"
                        f"{deployed_hash}.json"
                    ),
                )
                sprintctl.volume_upload(
                    run,
                    temp,
                    f"runs/{arm['run_id']}/state/BATCH_SITE_DEPLOYED.json",
                )
                atomic_json(local_marker, marker)
            finally:
                temp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            alerts.append(
                {
                    "run_id": arm["run_id"],
                    "kind": "site_marker",
                    "source": type(exc).__name__,
                    "count_in_tail": "1",
                }
            )
    return alerts


def monitor_cycle(
    batch_id: str, *, deploy: bool = True, env_file: Path | None = None
) -> dict[str, Any]:
    path = batch_path(batch_id)
    with frontier_update.file_lock(path.with_suffix(".lock")):
        payload = read_batch(batch_id)
        now = dt.datetime.now(dt.timezone.utc)
        finalized = 0
        cycle_alerts: list[dict[str, str]] = []
        if payload.get("credential_status") == "active":
            credential_env = env_file or Path(payload.get("env_file", ROOT / ".env"))
            try:
                management_key = load_env(credential_env)["OPENROUTER_MANAGEMENT_KEY"]
                cycle_alerts.extend(
                    audit_openrouter_child_usage(
                        payload,
                        OpenRouterManagementClient(management_key),
                        now=now,
                    )
                )
            except (KeyError, OSError, OpenRouterManagementError, ValueError) as exc:
                cycle_alerts.append(
                    {
                        "run_id": "batch",
                        "kind": "openrouter_key_usage_audit",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
        for arm in payload["arms"]:
            run_id = arm["run_id"]
            run_path = SCRIPT_DIR / run_id / "run.json"
            if not run_path.is_file():
                if arm["status"] not in {"planned", "launch_error"}:
                    arm["status"] = "missing_run_state"
                continue
            try:
                status = live_run_monitor_status(run_id)
                if status is None:
                    status = sprintctl.monitor_once(run_id, upload=True)
                if arm.get("status") == "missing_run_state":
                    arm["status"] = "running"
                for key in (
                    "ledger",
                    "snapshot_heartbeat_ok",
                    "frontier_worker_alive",
                    "remote_error",
                    "agent_container_id",
                    "harbor_alive",
                    "stop_ack",
                ):
                    arm[key] = status.get(key)
                arm["last_monitor_at"] = utc_now()
            except Exception as exc:
                arm["monitor_error"] = f"{type(exc).__name__}: {exc}"
                cycle_alerts.append(
                    {
                        "run_id": run_id,
                        "kind": "monitor_error",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
            deadline = (
                parse_time(arm["deadline_at"]) if arm.get("deadline_at") else None
            )
            if deadline and now >= deadline and arm.get("stop_requested_at") is None:
                sprintctl.request_stop(run_id, reason="fixed_24h_batch_deadline")
                arm["stop_requested_at"] = utc_now()
                arm["status"] = "stopping"
            cycle_alerts.extend(log_alerts(run_id))

        verifier_stalls = verifier_lane_stall_alerts(payload, now=now)
        cycle_alerts.extend(verifier_stalls)
        resolve_recovered_verifier_stalls(payload, verifier_stalls)
        cycle_alerts.extend(continuous_ledger_error_alerts(payload))
        for arm in payload["arms"]:
            if int(arm.get("ledger", {}).get("error", 0) or 0) == 0:
                resolve_alerts(
                    payload,
                    run_id=arm["run_id"],
                    kind="continuous_ledger_error",
                    resolution="all accepted submissions have trusted scores",
                )

        known = {
            (item.get("run_id"), item.get("kind"), item.get("source"))
            for item in payload.get("alerts", [])
        }
        for alert in cycle_alerts:
            key = (alert.get("run_id"), alert.get("kind"), alert.get("source"))
            if key not in known:
                alert["first_seen_at"] = utc_now()
                payload.setdefault("alerts", []).append(alert)
                known.add(key)
        payload["updated_at"] = utc_now()
        write_public_batch(payload, update_current=deploy)

        performance_ready = True
        if deploy:
            deploy_state = payload.setdefault("deploy", {})
            if deployment_retry_due(deploy_state, now=now):
                try:
                    refresh_performance_snapshot(payload)
                    resolve_alerts(
                        payload,
                        run_id="batch",
                        kind="performance_export",
                        resolution="subsequent_performance_snapshot_succeeded",
                    )
                except Exception as exc:
                    performance_ready = False
                    cycle_alerts.append(
                        {
                            "run_id": "batch",
                            "kind": "performance_export",
                            "source": type(exc).__name__,
                            "count_in_tail": "1",
                        }
                    )
                try:
                    if not performance_ready:
                        raise RuntimeError(
                            "refusing website deployment with a stale performance snapshot"
                        )
                    frontier_update.deploy_if_needed(
                        deploy_state,
                        web=WEB,
                        # Once every lane is terminal, publication is the only
                        # remaining external gate. Do not make an empty or
                        # no-submission lane wait through the live-update cadence.
                        debounce_seconds=deployment_debounce_seconds(payload),
                    )
                    clear_deployment_error(deploy_state)
                    resolve_alerts(
                        payload,
                        run_id="batch",
                        kind="website_deploy",
                        resolution="subsequent_site_snapshot_succeeded",
                    )
                except Exception as exc:
                    record_deployment_error(deploy_state, exc, now=now)
                    cycle_alerts.append(
                        {
                            "run_id": "batch",
                            "kind": "website_deploy",
                            "source": type(exc).__name__,
                            "count_in_tail": "1",
                        }
                    )
        cycle_alerts.extend(mark_deployed_runs(payload))

        for arm in payload["arms"]:
            run_id = arm["run_id"]
            finalized_path = SCRIPT_DIR / run_id / "FINALIZED.json"
            finalized_current = False
            finalized_marker: dict[str, Any] = {}
            if finalized_path.is_file():
                try:
                    finalized_marker = json.loads(finalized_path.read_text())
                    _, finalized_run = sprintctl.load_run(run_id)
                    finalized_current = bool(
                        finalized_marker.get("complete") is True
                        and finalized_marker.get("integrity", {}).get("schema_version")
                        == 1
                        and finalized_marker.get("timeline_schema_version")
                        == sprintctl.UNIFIED_TIMELINE_SCHEMA_VERSION
                        and (
                            not finalized_run.get("provider_usage_ledger_required")
                            or finalized_marker.get("conditions", {}).get(
                                "provider_usage_ledger_settled"
                            )
                            is True
                        )
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    finalized_current = False
            if not finalized_current and arm_terminal(arm):
                try:
                    complete, result = sprintctl.finalize(run_id)
                    arm["finalization_conditions"] = result.get("conditions", {})
                    if complete:
                        arm["status"] = "finalized"
                        finalized_marker = result
                        finalized_current = True
                except Exception as exc:  # noqa: BLE001
                    cycle_alerts.append(
                        {
                            "run_id": run_id,
                            "kind": "finalization",
                            "source": type(exc).__name__,
                            "count_in_tail": "1",
                        }
                    )
            if finalized_current:
                integrity = finalized_marker.get("integrity", {})
                arm["integrity"] = integrity
                arm["benchmark_valid"] = bool(integrity.get("benchmark_valid") is True)
                arm["replacement_required"] = bool(
                    integrity.get("replacement_required") is True
                )
                arm["status"] = (
                    "finalized" if arm["benchmark_valid"] else "invalid_infrastructure"
                )
                if not arm["benchmark_valid"]:
                    arm["invalidated_reason"] = "invalid_infrastructure"
                    cycle_alerts.append(
                        {
                            "run_id": run_id,
                            "kind": "run_integrity_failed",
                            "source": ",".join(
                                str(reason.get("code") or "unknown")
                                for reason in integrity.get("reasons", [])
                            ),
                            "count_in_tail": "1",
                        }
                    )
                arm["finalized_at"] = arm.get("finalized_at") or utc_now()
                resolve_alerts(
                    payload,
                    run_id=run_id,
                    kind="finalization",
                    resolution="subsequent_finalization_succeeded",
                )
                finalized += 1

        known = {
            (item.get("run_id"), item.get("kind"), item.get("source"))
            for item in payload.get("alerts", [])
        }
        for alert in cycle_alerts:
            key = (alert.get("run_id"), alert.get("kind"), alert.get("source"))
            if key not in known:
                alert["first_seen_at"] = utc_now()
                payload.setdefault("alerts", []).append(alert)
                known.add(key)
        all_finalized = finalized == len(payload["arms"])
        invalid_trials = sum(
            1
            for arm in payload["arms"]
            if arm.get("status") == "invalid_infrastructure"
        )
        payload["invalid_trial_count"] = invalid_trials
        payload["replacement_required"] = invalid_trials > 0
        payload["status"] = (
            "complete_with_invalid_trials"
            if all_finalized and invalid_trials
            else "complete"
            if all_finalized
            else "running"
        )
        payload["updated_at"] = utc_now()
        write_public_batch(payload, update_current=deploy)
        if (
            deploy
            and all_finalized
            and deployment_retry_due(payload.setdefault("deploy", {}), now=now)
        ):
            try:
                refresh_performance_snapshot(payload)
                frontier_update.deploy_if_needed(
                    payload.setdefault("deploy", {}),
                    web=WEB,
                    debounce_seconds=0,
                )
            except Exception as exc:  # noqa: BLE001
                record_deployment_error(payload.setdefault("deploy", {}), exc, now=now)
                cycle_alerts.append(
                    {
                        "run_id": "batch",
                        "kind": "final_site_deploy",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
        deployed_current = not deploy or deployed_batch_current(payload)
        if all_finalized and not deployed_current:
            payload["status"] = "finalizing_site"
            payload["updated_at"] = utc_now()
            write_public_batch(payload, update_current=deploy)
        if (
            all_finalized
            and deployed_current
            and payload.get("credential_status") == "active"
        ):
            credential_env = env_file or Path(payload.get("env_file", ROOT / ".env"))
            try:
                revoke_batch_credentials(payload, credential_env)
            except (KeyError, OSError, OpenRouterManagementError, ValueError) as exc:
                payload["credential_status"] = "cleanup_error"
                cycle_alerts.append(
                    {
                        "run_id": "batch",
                        "kind": "openrouter_credential_cleanup",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
        known = {
            (item.get("run_id"), item.get("kind"), item.get("source"))
            for item in payload.get("alerts", [])
        }
        for alert in cycle_alerts:
            key = (alert.get("run_id"), alert.get("kind"), alert.get("source"))
            if key not in known:
                alert["first_seen_at"] = utc_now()
                payload.setdefault("alerts", []).append(alert)
                known.add(key)
        atomic_json(path, payload)
        return payload


def stop_batch(batch_id: str, *, env_file: Path | None = None) -> dict[str, Any]:
    # Persist every lane's stop intent first. Modal lease fencing can wait on a
    # dispatch lock, so a controller interruption must not leave later arms
    # running merely because the first arm was slow to stop.
    snapshot = read_batch(batch_id)
    for arm in snapshot["arms"]:
        if (
            arm.get("status") != "finalized"
            and (SCRIPT_DIR / arm["run_id"] / "run.json").is_file()
        ):
            sprintctl.persist_stop_request(arm["run_id"], reason="operator_batch_stop")

    path = batch_path(batch_id)
    # The monitor holds this same lock for its complete read/modify/write cycle.
    # Re-read only after acquiring it: otherwise a monitor snapshot that began
    # before this stop could later restore launched arms or active credentials.
    with frontier_update.file_lock(path.with_suffix(".lock")):
        payload = read_batch(batch_id)
        targets: list[dict[str, Any]] = []
        for arm in payload["arms"]:
            if (
                arm.get("status") != "finalized"
                and (SCRIPT_DIR / arm["run_id"] / "run.json").is_file()
            ):
                arm["stop_requested_at"] = arm.get("stop_requested_at") or utc_now()
                arm["status"] = "stopping"
                targets.append(arm)
        payload["updated_at"] = utc_now()
        atomic_json(path, payload)

        if payload.get("credential_status") == "active":
            credential_env = env_file or Path(payload.get("env_file", ROOT / ".env"))
            try:
                revoke_batch_credentials(payload, credential_env)
            except (KeyError, OSError, OpenRouterManagementError, ValueError) as exc:
                payload["credential_status"] = "cleanup_error"
                payload["credential_cleanup_errors"] = [f"{type(exc).__name__}: {exc}"]
            atomic_json(path, payload)

        for arm in targets:
            try:
                result = sprintctl.request_stop(
                    arm["run_id"], reason="operator_batch_stop"
                )
                arm["stop_dispatch_status"] = result.get("status")
                arm.pop("stop_dispatch_error", None)
            except Exception as exc:  # noqa: BLE001
                # The per-run monitor will retry from STOP_REQUESTED. Continue so
                # one provider/API failure cannot block the remaining arms.
                arm["stop_dispatch_error"] = f"{type(exc).__name__}: {exc}"
            payload["updated_at"] = utc_now()
            atomic_json(path, payload)
        return payload


def run_monitor_command(
    batch_id: str,
    *,
    deploy: bool,
    env_file: Path,
    loop: bool,
    poll_seconds: int,
) -> dict[str, Any]:
    """Run one monitor cycle or own the sole long-lived monitor loop.

    ``batch.lock`` serializes individual state transitions, but it deliberately
    does not identify the daemon responsible for future cycles.  Without a
    lifetime lease, two differently named systemd units can both survive: one
    performs work while the other waits forever on every cycle.  That hides a
    duplicate controller behind apparently correct snapshots and can make a
    manual stop wait on the wrong process.

    A one-shot monitor remains available for diagnostics.  A loop must hold
    ``monitor-owner.lock`` for its entire lifetime and a duplicate exits
    successfully, allowing the canonical systemd unit to remain the only
    authority without entering a restart storm.
    """
    if not loop:
        return monitor_cycle(batch_id, deploy=deploy, env_file=env_file)

    owner_path = batch_dir(batch_id) / "monitor-owner.lock"
    with frontier_update.file_lock(owner_path, blocking=False) as acquired:
        if not acquired:
            return {
                "schema_version": 1,
                "batch_id": batch_id,
                "status": "monitor_already_running",
                "updated_at": utc_now(),
            }
        while True:
            output = monitor_cycle(batch_id, deploy=deploy, env_file=env_file)
            print(json.dumps(public_batch(output), indent=2), flush=True)
            if output.get("status") == "complete":
                return output
            time.sleep(max(10, poll_seconds))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "launch", "monitor", "status", "stop"):
        command = sub.add_parser(name)
        command.add_argument("--batch-id", required=True)
        command.add_argument("--env-file", type=Path, default=ROOT / ".env")
        command.add_argument(
            "--modal-profile", default=os.environ.get("MODAL_PROFILE", "kevinli020508")
        )
        if name == "launch":
            command.add_argument("--confirm", action="store_true")
        if name in {"preflight", "launch"}:
            command.add_argument(
                "--coexist-with-batch",
                action="append",
                default=[],
                help="allow live Modal resources owned by this existing batch",
            )
            command.add_argument(
                "--families",
                nargs="+",
                choices=SUPPORTED_FAMILIES,
                default=list(DEFAULT_FAMILIES),
            )
            command.add_argument(
                "--trials-per-model",
                type=int,
                default=TRIALS_PER_MODEL,
                help="independent trials to launch for each selected model family",
            )
        if name == "monitor":
            command.add_argument("--loop", action="store_true")
            command.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
            command.add_argument("--no-deploy", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        output = preflight(
            batch_id=args.batch_id,
            env_file=args.env_file.resolve(),
            modal_profile=args.modal_profile,
            families=tuple(args.families),
            trials_per_model=args.trials_per_model,
            probe_training_fleet=True,
            coexist_batch_ids=tuple(args.coexist_with_batch),
        )
    elif args.command == "launch":
        if not args.confirm:
            raise SystemExit("launch requires --confirm")
        output = launch(
            args.batch_id,
            args.env_file.resolve(),
            args.modal_profile,
            families=tuple(args.families),
            trials_per_model=args.trials_per_model,
            coexist_batch_ids=tuple(args.coexist_with_batch),
        )
    elif args.command == "monitor":
        output = run_monitor_command(
            args.batch_id,
            deploy=not args.no_deploy,
            env_file=args.env_file.resolve(),
            loop=args.loop,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "stop":
        output = stop_batch(args.batch_id, env_file=args.env_file.resolve())
    else:
        output = read_batch(args.batch_id)
    print(
        json.dumps(
            output if args.command == "preflight" else public_batch(output), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
