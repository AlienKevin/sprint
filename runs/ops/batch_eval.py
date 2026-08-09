#!/usr/bin/env python3
"""Reproducible Sprint batch launch, monitoring, and website publishing."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import frontier_update  # noqa: E402
import sprintctl  # noqa: E402


UV = Path(os.environ.get("UV", "/home/ubuntu/.local/bin/uv"))
HARBOR_PYTHON = ROOT / "harbor/.venv/bin/python3"
WEB = ROOT / "sprint-web"
BATCH_ROOT = SCRIPT_DIR / "batches"
WARMUP_MANIFEST = SCRIPT_DIR / "modal-image-warmup.json"
FUNCTIONAL_CANARY_REPORT = SCRIPT_DIR / "training-gpu-canary.json"
HARBOR_REVISION = "eccb31361a2ffbb8841f9e12cf1640cb0fb495a2"
CODEX_VERSION = "0.147.0"
TRIALS_PER_MODEL = 3
DEFAULT_FAMILIES = ("deepseek", "luna")
REASONING_EFFORT = "max"
RUN_HOURS: float | None = None
POLL_SECONDS = 30
# The task permits 15 minutes for one sealed verification. Leave five minutes
# for sandbox startup, result archival, and scheduler release before declaring
# the shared slot stuck; otherwise a legitimate timeout can race the watchdog.
SHARED_VERIFIER_STALL_SECONDS = 20 * 60
SHARED_VERIFIER_EVENTS = SCRIPT_DIR / "blind-verifier" / "scheduler-events.jsonl"
# The production Vercel team is on Hobby. A 20-minute rolling publication
# cadence caps publication at 72 deployments per day, leaving headroom
# below the 100/day Hobby allowance for warmups/manual releases. The final
# completed site still bypasses this delay below.
LIVE_SITE_DEPLOY_SECONDS = 20 * 60
PROVIDER_DISCOVERY_ATTEMPTS = 3
PROVIDER_DISCOVERY_RETRY_SECONDS = 1.0
PROVIDER_INFERENCE_ATTEMPTS = 3
PROVIDER_INFERENCE_RETRY_SECONDS = 1.0
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,48}$")
ALERT_PATTERNS = {
    "provider_rate_limit": re.compile(
        r"\b(?:429|rate.?limit|too many requests)\b", re.I
    ),
    "provider_quota": re.compile(
        r"\b(?:insufficient_quota|quota exceeded|billing limit|spend limit)\b", re.I
    ),
    "provider_auth": re.compile(
        r"\b(?:401|403|invalid api key|authentication failed)\b", re.I
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
        if name.strip() in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"} and value:
            values[name.strip()] = value
    return values


def functional_gpu_canary_ready() -> bool:
    try:
        warmup = json.loads(WARMUP_MANIFEST.read_text())
        canary = json.loads(FUNCTIONAL_CANARY_REPORT.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    contexts = warmup.get("contexts", {})
    return bool(
        warmup.get("completed")
        and canary.get("schema_version") == 2
        and canary.get("completed")
        and canary.get("full_path_verified")
        and canary.get("image_id")
        == contexts.get("agent_training", {}).get("image_id")
        and canary.get("verifier_image_id")
        == contexts.get("verifier", {}).get("image_id")
    )


def matrix(
    batch_id: str,
    trials_per_model: int = TRIALS_PER_MODEL,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    specs = {
        "deepseek": ("deepseek/deepseek-v4-flash", "run-deepseek.sh"),
        "luna": ("openai/gpt-5.6-luna", "run-luna.sh"),
    }
    selected = tuple(dict.fromkeys(families))
    unknown = sorted(set(selected) - set(specs))
    if not selected or unknown:
        raise ValueError(f"invalid model families: {unknown or list(selected)}")
    for family in selected:
        model, wrapper = specs[family]
        for trial in range(1, trials_per_model + 1):
            run_id = f"{batch_id}-{family}-{trial}"
            if not RUN_ID_RE.fullmatch(run_id):
                raise ValueError(f"generated run ID is invalid: {run_id}")
            arms.append(
                {
                    "run_id": run_id,
                    "family": family,
                    "model": model,
                    "resolved_model_version": (
                        "DeepSeek-V4-Flash-0731"
                        if family == "deepseek"
                        else "gpt-5.6-luna"
                    ),
                    "reasoning_effort": REASONING_EFFORT,
                    "codex_version": CODEX_VERSION,
                    "wrapper": str(ROOT / "runs" / wrapper),
                    "trial": trial,
                    "status": "planned",
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
            raise RuntimeError(
                f"provider model-list request failed: {reason}"
            ) from exc
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
    for attempt in range(PROVIDER_INFERENCE_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600 and attempt + 1 < PROVIDER_INFERENCE_ATTEMPTS:
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
            raise RuntimeError(
                f"provider inference request failed: {reason}"
            ) from exc
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("provider inference response lacked a request ID")
    usage = result.get("usage")
    return {
        "checked_at": utc_now(),
        "endpoint": url,
        "request_id": str(result["id"]),
        "response_model": str(result.get("model", "")),
        "status": str(result.get("status", result.get("object", ""))),
        "service_tier": result.get("service_tier"),
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


def vercel_project_link_ready() -> bool:
    """Accept Vercel CLI metadata while requiring the exact Sprint project."""
    try:
        frontier_update.verify_project_link(WEB)
    except RuntimeError:
        return False
    return True


def preflight(
    *,
    batch_id: str,
    env_file: Path,
    modal_profile: str,
    require_fresh: bool = True,
    check_providers: bool = True,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    provider_probes: dict[str, Any] = {}
    provider_errors: dict[str, str] = {}
    keys = load_env(env_file)
    required_keys = {
        "deepseek": "DEEPSEEK_API_KEY",
        "luna": "OPENAI_API_KEY",
    }
    for family in families:
        name = required_keys[family]
        checks[f"secret_{name.lower()}"] = len(keys.get(name, "")) >= 16
    checks["harbor_revision"] = (
        ROOT / "harbor/.sprint-upstream-commit"
    ).read_text().strip() == HARBOR_REVISION
    checks["goal_template"] = (ROOT / "runs/codex-goal-slash.j2").read_bytes() == (
        ROOT / "runs/codex-goal.j2"
    ).read_bytes()
    checks["warm_images"] = (
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "check_modal_image_warmup.py")],
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
    checks["vercel_auth"] = (
        subprocess.run(
            ["vercel", "whoami"],
            cwd=WEB,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if "luna" in families and check_providers and checks["secret_openai_api_key"]:
        models = provider_models(
            "https://api.openai.com/v1/models", keys["OPENAI_API_KEY"]
        )
        checks["openai_luna_visible"] = "gpt-5.6-luna" in models
        try:
            provider_probes["openai_luna"] = provider_inference_probe(
                "https://api.openai.com/v1/responses",
                keys["OPENAI_API_KEY"],
                {
                    "model": "gpt-5.6-luna",
                    "input": "Return OK.",
                    "reasoning": {"effort": REASONING_EFFORT},
                    "max_output_tokens": 16,
                    "store": False,
                },
            )
            checks["openai_luna_inference"] = True
        except RuntimeError as exc:
            checks["openai_luna_inference"] = False
            provider_errors["openai_luna"] = str(exc)
    elif "luna" in families:
        checks["openai_luna_visible"] = not check_providers
        checks["openai_luna_inference"] = not check_providers
    if "deepseek" in families and check_providers and checks["secret_deepseek_api_key"]:
        models = provider_models(
            "https://api.deepseek.com/models", keys["DEEPSEEK_API_KEY"]
        )
        checks["deepseek_v4_flash_visible"] = "deepseek-v4-flash" in models
        try:
            provider_probes["deepseek_v4_flash"] = provider_inference_probe(
                "https://api.deepseek.com/chat/completions",
                keys["DEEPSEEK_API_KEY"],
                {
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "Return OK."}],
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": REASONING_EFFORT,
                    "max_tokens": 16,
                    "stream": False,
                },
            )
            checks["deepseek_v4_flash_inference"] = True
        except RuntimeError as exc:
            checks["deepseek_v4_flash_inference"] = False
            provider_errors["deepseek_v4_flash"] = str(exc)
    elif "deepseek" in families:
        checks["deepseek_v4_flash_visible"] = not check_providers
        checks["deepseek_v4_flash_inference"] = not check_providers
    planned = matrix(batch_id, families=families)
    checks["unique_runs"] = len({arm["run_id"] for arm in planned}) == len(planned)
    checks["fresh_run_ids"] = not any(
        (SCRIPT_DIR / arm["run_id"]).exists() for arm in planned
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
        "env_file": str(env_file),
        "checks": checks,
        "provider_probes": provider_probes,
        "provider_errors": provider_errors,
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
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit}.service"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    run_checked(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--collect",
            "--property=Restart=on-failure",
            "--property=RestartSec=30",
            f"--setenv=PATH={service_path}",
            f"--setenv=UV={UV.resolve()}",
            f"--setenv=MODAL_PROFILE={modal_profile}",
            str(HARBOR_PYTHON),
            str(Path(__file__).resolve()),
            "monitor",
            "--batch-id",
            batch_id,
            "--env-file",
            str(env_file),
            "--modal-profile",
            modal_profile,
            "--loop",
        ]
    )


def launch(
    batch_id: str,
    env_file: Path,
    modal_profile: str,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> dict[str, Any]:
    report = preflight(
        batch_id=batch_id,
        env_file=env_file,
        modal_profile=modal_profile,
        families=families,
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
        "trials_per_model": TRIALS_PER_MODEL,
        "families": list(families),
        "run_hours": RUN_HOURS,
        "site_deploy_interval_seconds": LIVE_SITE_DEPLOY_SECONDS,
        "modal_profile": modal_profile,
        "preflight": report,
        "arms": matrix(batch_id, families=families),
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
    launched: list[dict[str, Any]] = []
    try:
        for arm in payload["arms"]:
            arm["launch_started_at"] = utc_now()
            atomic_json(batch_path(batch_id), payload)
            env = dict(base_env)
            env["RUN_ID"] = arm["run_id"]
            env["MODEL"] = arm["model"]
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


def shared_verifier_stall_alerts(
    payload: dict[str, Any], *, now: dt.datetime | None = None
) -> list[dict[str, str]]:
    """Alert when accepted blind submissions stop making scheduler progress.

    A run can legitimately enqueue a burst, and one verifier evaluation may use
    the full ten-minute task timeout.  Queue depth alone is therefore not a
    fault.  What matters is whether the shared scheduler has acquired or
    released a slot recently.  The oldest still-pending submission is also a
    progress reference so a fresh queue is not compared with stale events from
    an earlier batch.
    """
    pending = sum(
        int(arm.get("ledger", {}).get(state, 0) or 0)
        for arm in payload.get("arms", [])
        for state in ("queued", "running")
    )
    if pending == 0:
        return []

    scheduler_progress: list[dt.datetime] = []
    try:
        for raw in SHARED_VERIFIER_EVENTS.read_text().splitlines():
            try:
                event = json.loads(raw)
                if event.get("event") in {"acquired", "released"}:
                    scheduler_progress.append(parse_time(str(event["at"])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    except OSError:
        pass

    pending_submitted: list[dt.datetime] = []
    for arm in payload.get("arms", []):
        run_id = arm.get("run_id")
        if not isinstance(run_id, str):
            continue
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
                    if row.get("finished_at") is None and row.get("error") is None:
                        pending_submitted.append(parse_time(str(row["submitted_at"])))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue

    references = []
    if scheduler_progress:
        references.append(max(scheduler_progress))
    if pending_submitted:
        references.append(min(pending_submitted))
    if not references:
        return []
    current = now or dt.datetime.now(dt.timezone.utc)
    last_progress = max(references)
    age_seconds = (current - last_progress).total_seconds()
    if age_seconds <= SHARED_VERIFIER_STALL_SECONDS:
        return []
    return [
        {
            "run_id": "batch",
            "kind": "shared_verifier_stalled",
            "source": SHARED_VERIFIER_EVENTS.name,
            "count_in_tail": str(pending),
            "pending_submissions": str(pending),
            "last_progress_at": last_progress.isoformat(),
            "progress_age_seconds": str(round(age_seconds)),
        }
    ]


def live_run_monitor_status(run_id: str) -> dict[str, Any] | None:
    """Read the independent run monitor instead of duplicating its work.

    Each lane owns a long-lived ``sprintctl monitor`` process that performs
    telemetry, GPU dispatch, trace synchronization, and frontier updates.  The
    batch monitor is an observer/supervisor of those lane monitors; running a
    second full ``monitor_once`` serially here both wastes Modal API calls and
    delays later arms in the batch.
    """
    state_dir = SCRIPT_DIR / run_id
    try:
        pid = int((state_dir / "monitor.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    if not sprintctl.process_alive(pid, f"sprintctl.py monitor --run-id {run_id}"):
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
    paths = [WEB / "data" / "batches" / f"{payload['batch_id']}.json"]
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

        cycle_alerts.extend(shared_verifier_stall_alerts(payload, now=now))

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
        public_path = WEB / "data" / "batches" / f"{batch_id}.json"
        atomic_json(public_path, public_batch(payload), mode=0o644)

        if deploy:
            deploy_state = payload.setdefault("deploy", {})
            try:
                frontier_update.deploy_if_needed(
                    deploy_state,
                    web=WEB,
                    # Once every lane is terminal, publication is the only
                    # remaining external gate. Do not make an empty or
                    # no-submission lane wait through the live-update cadence.
                    debounce_seconds=deployment_debounce_seconds(payload),
                )
                resolve_alerts(
                    payload,
                    run_id="batch",
                    kind="website_deploy",
                    resolution="subsequent_site_snapshot_succeeded",
                )
            except Exception as exc:
                deploy_state["site_status"] = "error"
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
        atomic_json(public_path, public_batch(payload), mode=0o644)
        if deploy and all_finalized:
            try:
                frontier_update.deploy_if_needed(
                    payload.setdefault("deploy", {}),
                    web=WEB,
                    debounce_seconds=0,
                )
            except Exception as exc:  # noqa: BLE001
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
            atomic_json(public_path, public_batch(payload), mode=0o644)
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
            sprintctl.persist_stop_request(
                arm["run_id"], reason="operator_batch_stop"
            )
            arm["stop_requested_at"] = arm.get("stop_requested_at") or utc_now()
            arm["status"] = "stopping"
            targets.append(arm)
    payload["updated_at"] = utc_now()
    atomic_json(batch_path(batch_id), payload)

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
                "--families",
                nargs="+",
                choices=DEFAULT_FAMILIES,
                default=list(DEFAULT_FAMILIES),
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
        )
    elif args.command == "launch":
        if not args.confirm:
            raise SystemExit("launch requires --confirm")
        output = launch(
            args.batch_id,
            args.env_file.resolve(),
            args.modal_profile,
            families=tuple(args.families),
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
