#!/usr/bin/env python3
"""Reproducible event batch launch, monitoring, and website publishing."""

from __future__ import annotations

import argparse
import datetime as dt
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
from event_runtime.control import deepseek_pricing  # noqa: E402


UV = Path(os.environ.get("UV", "/home/ubuntu/.local/bin/uv"))
HARBOR_PYTHON = ROOT / "harbor/.venv/bin/python3"
WEB = ROOT / "web"
BATCH_ROOT = SCRIPT_DIR / "batches"
WARMUP_MANIFEST = SCRIPT_DIR / "modal-image-warmup.json"
FUNCTIONAL_CANARY_REPORT = SCRIPT_DIR / "training-gpu-canary.json"
BUDGET_CONFIG = MODULE_DIR / "budget.env"
HARBOR_REVISION = "dafb1387151e1c32702963d44fe6c3cea66cf8cb"
CODEX_VERSION = "0.147.0"
TRIALS_PER_MODEL = 3
DEFAULT_FAMILIES = ("deepseek", "luna")
SUPPORTED_FAMILIES = ("deepseek", "luna", "sol")
OPENAI_FAMILY_SPECS: dict[str, dict[str, str]] = {
    "luna": {
        "model": "openai/gpt-5.6-luna",
        "model_id": "gpt-5.6-luna",
        "preset": "@preset/sprint-gpt-5-6-luna-openai-standard",
        "preset_id": "f06bb802-6122-4e11-8e22-a3476113a3b1",
        "resolved_model": "openai/gpt-5.6-luna-20260709",
    },
    "sol": {
        "model": "openai/gpt-5.6-sol",
        "model_id": "gpt-5.6-sol",
        "preset": "@preset/sprint-gpt-5-6-sol-openai-standard",
        "preset_id": "798803cc-8d68-4249-b437-c6eb509df833",
        "resolved_model": "openai/gpt-5.6-sol-20260709",
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
PROVIDER_INFERENCE_ATTEMPTS = 3
PROVIDER_INFERENCE_RETRY_SECONDS = 1.0
PROVIDER_GENERATION_AUDIT_ATTEMPTS = 6
OPENROUTER_CREDIT_SAFETY_FACTOR = 1.05
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
    if not all(math.isfinite(value) and value >= 0 for value in (total, usage, remaining)):
        raise RuntimeError("OpenRouter credit totals were invalid")
    return {
        "total_credits_usd": total,
        "total_usage_usd": usage,
        "remaining_credit_usd": remaining,
    }


def functional_gpu_canary_ready() -> bool:
    try:
        warmup = json.loads(WARMUP_MANIFEST.read_text())
        canary = json.loads(FUNCTIONAL_CANARY_REPORT.read_text())
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
        and (canary.get("gpu_budget_mirror") or {}).get("observed_sequences")
        == [1, 2]
        and cost_proof_valid
        and canary.get("image_id") == contexts.get("agent_training", {}).get("image_id")
        and canary.get("verifier_image_id")
        == contexts.get("verifier", {}).get("image_id")
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
        "deepseek": {
            "model": "deepseek/deepseek-v4-flash",
            "wrapper": "deepseek.sh",
            "resolved_model_version": "DeepSeek-V4-Flash-0731",
        },
        **{
            family: {
                "model": spec["model"],
                "wrapper": "openai.sh",
                "resolved_model_version": spec["model_id"],
                "openrouter_preset": spec["preset"],
            }
            for family, spec in OPENAI_FAMILY_SPECS.items()
        },
    }
    selected = tuple(dict.fromkeys(families))
    unknown = sorted(set(selected) - set(specs))
    if not selected or unknown:
        raise ValueError(f"invalid model families: {unknown or list(selected)}")
    for family in selected:
        spec = specs[family]
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
                    "codex_version": CODEX_VERSION,
                    "wrapper": str(MODULE_DIR / "providers" / spec["wrapper"]),
                    "trial": trial,
                    "status": "planned",
                    **(
                        {"openrouter_preset": spec["openrouter_preset"]}
                        if "openrouter_preset" in spec
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


def read_batch(batch_id: str) -> dict[str, Any]:
    path = batch_path(batch_id)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"unknown batch: {batch_id}") from exc
    if payload.get("batch_id") != batch_id:
        raise ValueError("batch state has mismatched ID")
    return payload


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
                time.sleep(PROVIDER_INFERENCE_RETRY_SECONDS * (2**attempt))
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
                time.sleep(PROVIDER_INFERENCE_RETRY_SECONDS * (2**attempt))
                continue
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"provider inference request failed: {reason}") from exc
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("provider inference response lacked a request ID")
    generation: dict[str, Any] = {}
    if generation_audit_url:
        audit_id = generation_id or str(result["id"])
        audit_url = (
            generation_audit_url
            + "?"
            + urllib.parse.urlencode({"id": audit_id})
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
    deepseek_pricing_snapshot: dict[str, Any] | None = None
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
    }
    for family in families:
        name = required_keys[family]
        checks[f"secret_{name.lower()}"] = len(keys.get(name, "")) >= 16
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
                        "model": spec["preset"],
                        "input": "Return OK.",
                        "reasoning": {"effort": REASONING_EFFORT},
                        "max_output_tokens": 64,
                        "store": False,
                    },
                    generation_audit_url="https://openrouter.ai/api/v1/generation",
                )
                probe = provider_probes[key]
                checks[f"{key}_inference"] = (
                    probe.get("provider") == "OpenAI"
                    and probe.get("preset_id") == spec["preset_id"]
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
    if (
        "deepseek" in families
        and check_providers
        and checks["secret_openrouter_api_key"]
    ):
        try:
            deepseek_pricing_snapshot = deepseek_pricing.fetch_openrouter_snapshot(
                keys["OPENROUTER_API_KEY"]
            )
            checks["deepseek_openrouter_endpoint_contract"] = True
        except Exception as exc:  # noqa: BLE001
            checks["deepseek_openrouter_endpoint_contract"] = False
            provider_errors["deepseek_pricing"] = f"{type(exc).__name__}: {exc}"
        models = provider_models(
            "https://openrouter.ai/api/v1/models", keys["OPENROUTER_API_KEY"]
        )
        checks["deepseek_v4_flash_visible"] = (
            "deepseek/deepseek-v4-flash-0731" in models
        )
        try:
            provider_probes["deepseek_v4_flash"] = provider_inference_probe(
                "https://openrouter.ai/api/v1/responses",
                keys["OPENROUTER_API_KEY"],
                {
                    "model": ("@preset/sprint-deepseek-v4-flash-0731-official"),
                    "input": "Return OK.",
                    "reasoning": {"effort": REASONING_EFFORT},
                    "max_output_tokens": 16,
                    "store": False,
                },
                generation_audit_url="https://openrouter.ai/api/v1/generation",
            )
            deepseek_probe = provider_probes["deepseek_v4_flash"]
            expected_preset_id = (
                deepseek_pricing_snapshot.get("preset_id")
                if isinstance(deepseek_pricing_snapshot, dict)
                else None
            )
            checks["deepseek_v4_flash_inference"] = (
                deepseek_probe.get("provider") == "DeepSeek"
                and deepseek_probe.get("preset_id") == expected_preset_id
                and deepseek_probe.get("resolved_model")
                == (
                    deepseek_pricing_snapshot.get("resolved_model")
                    if isinstance(deepseek_pricing_snapshot, dict)
                    else None
                )
            )
            if not checks["deepseek_v4_flash_inference"]:
                provider_errors["deepseek_v4_flash"] = (
                    "controlled preset response did not identify DeepSeek"
                )
        except RuntimeError as exc:
            checks["deepseek_v4_flash_inference"] = False
            provider_errors["deepseek_v4_flash"] = str(exc)
    elif "deepseek" in families:
        checks["deepseek_openrouter_endpoint_contract"] = not check_providers
        checks["deepseek_v4_flash_visible"] = not check_providers
        checks["deepseek_v4_flash_inference"] = not check_providers
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
        "deepseek_pricing_snapshot": deepseek_pricing_snapshot,
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
    base_env = dict(os.environ)
    base_env.update(keys)
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
    pricing_snapshot = report.get("deepseek_pricing_snapshot")
    if "deepseek" in families:
        if not isinstance(pricing_snapshot, dict):
            raise RuntimeError("preflight did not produce a DeepSeek pricing snapshot")
        base_env["SPRINT_DEEPSEEK_PRICING_SNAPSHOT"] = json.dumps(
            pricing_snapshot, separators=(",", ":"), sort_keys=True
        )
    launched: list[dict[str, Any]] = []
    try:
        for arm in payload["arms"]:
            arm["launch_started_at"] = utc_now()
            atomic_json(batch_path(batch_id), payload)
            env = dict(base_env)
            env["RUN_ID"] = arm["run_id"]
            env["MODEL"] = arm["model"]
            if arm.get("openrouter_preset"):
                env["OPENROUTER_PRESET"] = arm["openrouter_preset"]
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
        atomic_json(batch_path(batch_id), payload)
        raise
    payload["status"] = "running"
    payload["launched_at"] = utc_now()
    atomic_json(batch_path(batch_id), payload)
    start_monitor_service(batch_id, env_file, modal_profile)
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
        for ledger in (SCRIPT_DIR / run_id / "harbor-jobs").glob(
            "*/*/artifacts/continuous/ledger.jsonl"
        ):
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
    if (state_dir / "STOP_ACK.json").is_file():
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


def public_batch(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batch_id": payload["batch_id"],
        "updated_at": payload.get("updated_at"),
        "status": payload.get("status"),
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
                )
            }
            for arm in payload["arms"]
        ],
        "alerts": payload.get("alerts", [])[-100:],
    }


def write_public_batch(payload: dict[str, Any]) -> Path:
    """Publish both the immutable batch record and the active-batch pointer."""
    public = public_batch(payload)
    public_path = WEB / "data" / "batches" / f"{payload['batch_id']}.json"
    atomic_json(public_path, public, mode=0o644)
    atomic_json(WEB / "data" / "batches" / "current.json", public, mode=0o644)
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


def deployment_debounce_seconds(payload: dict[str, Any]) -> int:
    """Publish immediately after every lane has reached a terminal state."""
    return (
        0
        if all(
            arm.get("harbor_alive") is False or arm.get("stop_ack")
            for arm in payload["arms"]
        )
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

    performance_export.build(
        f"{payload['batch_id']}-",
        WEB / "data" / "performance" / "current.json",
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


def monitor_cycle(batch_id: str, *, deploy: bool = True) -> dict[str, Any]:
    path = batch_path(batch_id)
    with frontier_update.file_lock(path.with_suffix(".lock")):
        payload = read_batch(batch_id)
        now = dt.datetime.now(dt.timezone.utc)
        finalized = 0
        cycle_alerts: list[dict[str, str]] = []
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

        cycle_alerts.extend(verifier_lane_stall_alerts(payload, now=now))
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
        write_public_batch(payload)

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
            if not finalized_path.is_file() and (
                arm.get("harbor_alive") is False or arm.get("stop_ack")
            ):
                try:
                    complete, result = sprintctl.finalize(run_id)
                    arm["finalization_conditions"] = result.get("conditions", {})
                    if complete:
                        arm["status"] = "finalized"
                except Exception as exc:  # noqa: BLE001
                    cycle_alerts.append(
                        {
                            "run_id": run_id,
                            "kind": "finalization",
                            "source": type(exc).__name__,
                            "count_in_tail": "1",
                        }
                    )
            if finalized_path.is_file():
                arm["status"] = "finalized"
                arm["finalized_at"] = arm.get("finalized_at") or utc_now()
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
        payload["status"] = "complete" if all_finalized else "running"
        payload["updated_at"] = utc_now()
        write_public_batch(payload)
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
            write_public_batch(payload)
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


def stop_batch(batch_id: str) -> dict[str, Any]:
    payload = read_batch(batch_id)
    targets: list[dict[str, Any]] = []
    # Persist every lane's stop intent first. Modal lease fencing can wait on a
    # dispatch lock, so a controller interruption must not leave later arms
    # running merely because the first arm was slow to stop.
    for arm in payload["arms"]:
        if (
            arm.get("status") != "finalized"
            and (SCRIPT_DIR / arm["run_id"] / "run.json").is_file()
        ):
            sprintctl.persist_stop_request(arm["run_id"], reason="operator_batch_stop")
            arm["stop_requested_at"] = arm.get("stop_requested_at") or utc_now()
            arm["status"] = "stopping"
            targets.append(arm)
    payload["updated_at"] = utc_now()
    atomic_json(batch_path(batch_id), payload)

    for arm in targets:
        try:
            result = sprintctl.request_stop(arm["run_id"], reason="operator_batch_stop")
            arm["stop_dispatch_status"] = result.get("status")
            arm.pop("stop_dispatch_error", None)
        except Exception as exc:  # noqa: BLE001
            # The per-run monitor will retry from STOP_REQUESTED. Continue so
            # one provider/API failure cannot block the remaining arms.
            arm["stop_dispatch_error"] = f"{type(exc).__name__}: {exc}"
        payload["updated_at"] = utc_now()
        atomic_json(batch_path(batch_id), payload)
    return payload


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
        while True:
            output = monitor_cycle(args.batch_id, deploy=not args.no_deploy)
            print(json.dumps(public_batch(output), indent=2), flush=True)
            if not args.loop or output.get("status") == "complete":
                break
            time.sleep(max(10, args.poll_seconds))
    elif args.command == "stop":
        output = stop_batch(args.batch_id)
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
