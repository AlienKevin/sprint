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

    execution_policy = str(run.get("cpu_execution_policy") or "")
    observations["cpu_execution_policy"] = execution_policy or None
    if execution_policy != "single_process_no_resume":
        reasons.append(
            _reason(
                "cpu_execution_policy_mismatch",
                "run.json",
                "CPU agent is not sealed to one non-resumable execution",
            )
        )
    cpu_exit_path = state_dir / "CPU_TRIAL_EXIT.json"
    cpu_exit = _read_json(cpu_exit_path)
    observations["cpu_exit"] = cpu_exit or None
    terminal_evidence = any(
        (state_dir / name).is_file()
        for name in ("FINALIZED.json", "STOP_ACK.json", "CPU_TRIAL_EXIT.json")
    )
    if terminal_evidence and not cpu_exit:
        reasons.append(
            _reason(
                "cpu_exit_record_missing",
                "CPU_TRIAL_EXIT.json",
                "terminal trial lacks its authoritative CPU process boundary",
            )
        )
    elif cpu_exit:
        if int(cpu_exit.get("attempt") or 0) != 1:
            reasons.append(
                _reason(
                    "cpu_exit_attempt_mismatch",
                    "CPU_TRIAL_EXIT.json",
                    "CPU process exit does not belong to launch attempt 1",
                )
            )
        raw_code = cpu_exit.get("raw_exit_code")
        requested = bool(cpu_exit.get("stop_requested"))
        if isinstance(raw_code, int) and raw_code != 0 and not requested:
            reasons.append(
                _reason(
                    "cpu_agent_unexpected_exit",
                    "CPU_TRIAL_EXIT.json",
                    f"authoritative CPU process exited {raw_code} without a stop request",
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

    # A persistent Codex goal may span multiple turns, but the benchmark CPU
    # process must not disappear while that goal is still active.  The trusted
    # runner records its last observed lifecycle beside Harbor's bootstrap
    # receipt.  An ordinary agent exit is valid only after a terminal goal
    # status; budget/operator stops remain authoritative regardless of status.
    agent_kind = str(run.get("agent_kind") or "")
    bootstrap_paths = sorted(
        state_dir.glob("harbor-jobs/*/*/agent/goal-bootstrap.json")
    )
    lifecycle_paths = sorted(
        state_dir.glob("harbor-jobs/*/*/agent/goal-lifecycle.json")
    )
    observations["codex_goal_bootstrap_count"] = len(bootstrap_paths)
    observations["codex_goal_lifecycle_count"] = len(lifecycle_paths)
    if lifecycle_paths:
        observations["codex_goal_lifecycle"] = _read_json(lifecycle_paths[-1]) or None
    if agent_kind == "codex" and stop_reason == "agent_exit" and bootstrap_paths:
        if len(bootstrap_paths) != 1 or len(lifecycle_paths) != 1:
            reasons.append(
                _reason(
                    "codex_goal_lifecycle_missing",
                    "harbor-jobs/*/*/agent/goal-lifecycle.json",
                    "Codex agent exited without one unambiguous persistent-goal lifecycle record",
                )
            )
        else:
            lifecycle = _read_json(lifecycle_paths[0])
            goal_status = str(lifecycle.get("goal_status") or "")
            runner_state = str(lifecycle.get("runner_state") or "")
            if (
                goal_status not in {"complete", "blocked"}
                or runner_state != "terminal"
            ):
                reasons.append(
                    _reason(
                        "codex_goal_active_at_agent_exit",
                        str(lifecycle_paths[0].relative_to(state_dir)),
                        "Codex agent exited before its persistent goal reached a terminal state",
                    )
                )

    if agent_kind == "deepseek-harness" and stop_reason == "agent_exit":
        if len(lifecycle_paths) != 1:
            reasons.append(
                _reason(
                    "deepseek_goal_lifecycle_missing",
                    "harbor-jobs/*/*/agent/goal-lifecycle.json",
                    "DeepSeek Harness exited without one native-goal lifecycle record",
                )
            )
        else:
            lifecycle = _read_json(lifecycle_paths[0])
            goal_status = str(lifecycle.get("goal_status") or "")
            runner_state = str(lifecycle.get("runner_state") or "")
            if (
                goal_status not in {"complete", "blocked"}
                or runner_state != "terminal"
            ):
                reasons.append(
                    _reason(
                        "deepseek_goal_active_at_agent_exit",
                        str(lifecycle_paths[0].relative_to(state_dir)),
                        "DeepSeek Harness exited before its native goal reached a terminal state",
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
        if str(job.get("termination_reason") or "") == "budget_telemetry_unavailable":
            reasons.append(
                _reason(
                    "gpu_budget_telemetry_unavailable",
                    path.name,
                    "GPU worker failed closed after its trusted budget feed became unavailable",
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
