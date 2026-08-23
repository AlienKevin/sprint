"""Benchmark integrity certification from trusted host-side evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


EXPECTED_TEARDOWN_MARKERS = (
    "already shut down",
    "already terminated",
    "container is not running",
    "task has already finished",
    "sandbox not found",
)

INFRA_RETRY_REASONS = {
    "activity_watchdog",
    "graceful_preemption",
    "heartbeat_stale",
    "provider_probe_unknown",
    "spawn_failed",
    "worker_lost",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _reason(
    code: str, source: str, detail: str, *, severity: str = "invalid"
) -> dict[str, str]:
    return {
        "code": code,
        "source": source,
        "detail": detail,
        "severity": severity,
    }


def build_integrity_report(
    state_dir: Path, run: dict[str, Any]
) -> dict[str, Any]:
    """Return a conservative certificate independent of archival completion."""
    reasons: list[dict[str, str]] = []
    observations: dict[str, Any] = {}

    cpu_attempts = int(run.get("cpu_launch_attempt") or 1)
    observations["cpu_launch_attempts"] = cpu_attempts
    if cpu_attempts > 1:
        reasons.append(
            _reason(
                "cpu_agent_relaunched",
                "run.json",
                f"CPU agent required {cpu_attempts} launch attempts",
            )
        )

    stop_ack = _read_json(state_dir / "STOP_ACK.json")
    stop_reason = str(stop_ack.get("reason") or "")
    observations["stop_reason"] = stop_reason or None
    if stop_reason == "budget_telemetry_unavailable":
        reasons.append(
            _reason(
                "budget_telemetry_unavailable",
                "STOP_ACK.json",
                "trusted budget telemetry failed before the normal budget stop",
            )
        )

    gpu_jobs = 0
    retried_jobs = 0
    for path in sorted((state_dir / "gpu-job-registry").glob("*.json")):
        job = _read_json(path)
        if not job:
            continue
        gpu_jobs += 1
        attempt = int(job.get("attempt") or 0)
        retry_reason = str(job.get("retry_reason") or "")
        if attempt > 1 or retry_reason in INFRA_RETRY_REASONS:
            retried_jobs += 1
            reasons.append(
                _reason(
                    "gpu_worker_retried",
                    path.name,
                    f"GPU job used attempt {attempt}"
                    + (f" after {retry_reason}" if retry_reason else ""),
                )
            )
        terminate_error = str(job.get("terminate_error") or "").strip()
        if terminate_error and not any(
            marker in terminate_error.lower() for marker in EXPECTED_TEARDOWN_MARKERS
        ):
            reasons.append(
                _reason(
                    "gpu_termination_unconfirmed",
                    path.name,
                    terminate_error[-500:],
                )
            )
        if (
            str(job.get("status") or "") == "terminated"
            and not str(job.get("termination_reason") or "").strip()
            and not any(
                marker in str(job.get("error") or "").lower()
                for marker in (
                    "agent cost budget exhausted",
                    "agent cancelled",
                    "agent canceled",
                )
            )
        ):
            reasons.append(
                _reason(
                    "gpu_termination_unattributed",
                    path.name,
                    "GPU job terminated without a host-recorded reason",
                )
            )
        provider_error = str(job.get("provider_terminal_error") or "").strip()
        if "context" in provider_error.lower() and "token" in provider_error.lower():
            reasons.append(
                _reason(
                    "provider_context_contract_failure",
                    path.name,
                    provider_error[-500:],
                )
            )
    observations["gpu_jobs"] = gpu_jobs
    observations["gpu_jobs_retried"] = retried_jobs

    requested_cancels: dict[str, str] = {}
    acknowledged_cancels: set[str] = set()
    try:
        control_lines = (state_dir / "control-events.jsonl").read_text().splitlines()
    except OSError:
        control_lines = []
    for line in control_lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_id = str(event.get("request_id") or "")
        if not request_id:
            continue
        if event.get("event") == "cancel_requested":
            requested_cancels[request_id] = str(event.get("job_id") or "unknown")
        elif event.get("event") == "cancel_acknowledged":
            acknowledged_cancels.add(request_id)
    for request_id, job_id in requested_cancels.items():
        if request_id not in acknowledged_cancels:
            reasons.append(
                _reason(
                    "gpu_cancellation_unacknowledged",
                    "control-events.jsonl",
                    f"GPU cancellation for {job_id} lacks provider acknowledgement",
                )
            )
    observations["cancel_requests"] = len(requested_cancels)
    observations["cancel_requests_acknowledged"] = len(acknowledged_cancels)

    # Collapse duplicate reason codes while retaining every affected source.
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in reasons:
        key = (item["code"], item["source"], item["detail"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    invalid = any(item["severity"] == "invalid" for item in unique)
    return {
        "schema_version": 1,
        "run_id": str(run["run_id"]),
        "status": "invalid_infrastructure" if invalid else "clean",
        "benchmark_valid": not invalid,
        "leaderboard_eligible": not invalid,
        "replacement_required": invalid,
        "reasons": unique,
        "observations": observations,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
