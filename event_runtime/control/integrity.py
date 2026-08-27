"""Benchmark integrity certification from trusted host-side evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3


DEFAULT_MIN_BUDGET_UTILIZATION_FOR_REVIEW = 0.90


EXPECTED_TEARDOWN_MARKERS = (
    "already shut down",
    "already terminated",
    "container is not running",
    "task has already finished",
    "sandbox not found",
)

INFRA_RETRY_REASONS = {
    "activity_watchdog",
    "app_launcher_initialization_failed",
    "graceful_preemption",
    "heartbeat_stale",
    "provider_gpu_initialization_failed",
    "provider_probe_unknown",
    "spawn_failed",
    "worker_lost",
}

GPU_TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "preempted", "terminated"}
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _epoch(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _reason(
    code: str, source: str, detail: str, *, severity: str = "invalid"
) -> dict[str, str]:
    return {
        "code": code,
        "source": source,
        "detail": detail,
        "severity": severity,
    }


def submission_bridge_reasons(
    state_dir: Path, *, ledger_names: set[str] | None = None
) -> list[dict[str, str]]:
    """Return fail-closed reasons for incomplete explicit GPU submissions."""
    reasons: list[dict[str, str]] = []
    registry = state_dir / "gpu-job-registry"
    registry_jobs = (
        [_read_json(path) for path in sorted(registry.glob("*.json"))]
        if registry.is_dir()
        else []
    )
    explicit_submission_jobs = [
        job for job in registry_jobs if job.get("submission_paths")
    ]
    # STOP_ACK records whether the CPU-side bounded wait expired.  That is a
    # useful fail-closed snapshot, but it is not permanent proof of loss: the
    # trusted host bridge may finish reconciling terminal GPU jobs shortly
    # after the CPU sandbox acknowledges its stop.  Keep the timeout invalid
    # only while at least one declared submission job is still unresolved.
    unresolved_after_drain_timeout = any(
        str(job.get("status") or "") not in GPU_TERMINAL_STATUSES
        or not job.get("submission_bridge_terminal_drained_at")
        for job in explicit_submission_jobs
    )
    if (
        _read_json(state_dir / "STOP_ACK.json").get(
            "gpu_submission_drain_timed_out"
        )
        and unresolved_after_drain_timeout
    ):
        reasons.append(
            _reason(
                "gpu_submission_drain_timeout",
                "STOP_ACK.json",
                "CPU teardown timed out before final GPU submissions were forwarded",
            )
        )
    records: list[dict[str, Any]] = []
    bridge_root = state_dir / "submission-bridge"
    if bridge_root.is_dir():
        for path in sorted(bridge_root.glob("*.json")):
            record = _read_json(path)
            if not record:
                reasons.append(
                    _reason(
                        "gpu_submission_bridge_record_invalid",
                        path.name,
                        "GPU submission bridge record is missing or invalid",
                    )
                )
                continue
            records.append(record)
            if record.get("state") != "forwarded":
                reasons.append(
                    _reason(
                        "gpu_submission_forwarding_incomplete",
                        path.name,
                        str(record.get("error") or "GPU submission was not forwarded"),
                    )
                )

    forwarded_by_job: dict[str, int] = {}
    forwarded_names: list[tuple[str, str]] = []
    for record in records:
        if record.get("state") != "forwarded":
            continue
        job_id = str(record.get("gpu_job_id") or "")
        forwarded_by_job[job_id] = forwarded_by_job.get(job_id, 0) + 1
        forwarded_names.append(
            (
                str(record.get("queue_name") or ""),
                str(record.get("submission_id") or ""),
            )
        )

    if ledger_names is None:
        ledger_names = set()
        for ledger in state_dir.glob(
            "harbor-jobs/*/*/artifacts/continuous/ledger.jsonl"
        ):
            try:
                lines = ledger.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    ledger_names.add(str(row.get("name") or ""))
    for queue_name, submission_id in forwarded_names:
        if queue_name and queue_name not in ledger_names:
            reasons.append(
                _reason(
                    "gpu_submission_missing_from_harbor",
                    queue_name,
                    "Forwarded GPU submission "
                    f"{submission_id or queue_name} is absent from Harbor's ledger",
                )
            )

    if registry.is_dir():
        for path, job in zip(sorted(registry.glob("*.json")), registry_jobs):
            declared = [str(item) for item in job.get("submission_paths") or []]
            if not declared:
                continue
            # Submission results are produced by the worker's terminal drain.
            # A live job has not reached that boundary yet, so treating its
            # absent results as loss would make live integrity monitoring lie.
            status = str(job.get("status") or "")
            if status and status not in GPU_TERMINAL_STATUSES:
                continue
            job_id = str(job.get("job_id") or path.stem)
            progress = job.get("progress")
            results = (
                progress.get("submission_results")
                if isinstance(progress, dict)
                else None
            )
            result_paths = (
                [
                    str(item.get("path") or "")
                    for item in results
                    if isinstance(item, dict)
                ]
                if isinstance(results, list)
                else []
            )
            if result_paths != declared:
                reasons.append(
                    _reason(
                        "gpu_submission_results_missing",
                        path.name,
                        f"GPU job {job_id} did not finalize every explicitly "
                        "declared submission",
                    )
                )
                continue
            if any(
                not isinstance(item, dict)
                or item.get("state") not in {"staged", "rejected"}
                for item in results
            ):
                reasons.append(
                    _reason(
                        "gpu_submission_result_invalid",
                        path.name,
                        f"GPU job {job_id} has an invalid terminal submission result",
                    )
                )
                continue
            staged = sum(item.get("state") == "staged" for item in results)
            forwarded = forwarded_by_job.get(job_id, 0)
            if forwarded < staged:
                reasons.append(
                    _reason(
                        "gpu_submission_forwarding_incomplete",
                        path.name,
                        f"GPU job {job_id} staged {staged} submission(s), but "
                        f"only {forwarded} reached Harbor",
                    )
                )
    return reasons


def build_integrity_report(state_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
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
    reasons.extend(submission_bridge_reasons(state_dir))

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
        lifecycle = observations["codex_goal_lifecycle"]
        if terminal_evidence and isinstance(lifecycle, dict) and (
            str(lifecycle.get("runner_state") or "") == "invalid_infrastructure"
            or str(lifecycle.get("failure_code") or "")
        ):
            reasons.append(
                _reason(
                    "agent_goal_runner_invalid",
                    str(lifecycle_paths[-1].relative_to(state_dir)),
                    str(
                        lifecycle.get("detail")
                        or lifecycle.get("failure_code")
                        or "persistent-goal runner recorded an infrastructure failure"
                    ),
                )
            )
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
            if goal_status not in {"complete", "blocked"} or runner_state != "terminal":
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
            if goal_status not in {"complete", "blocked"} or runner_state != "terminal":
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
        if "GPU/Vulkan initialization failed" in provider_error:
            reasons.append(
                _reason(
                    "provider_gpu_initialization_failed",
                    path.name,
                    provider_error[-500:],
                )
            )
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

    requested_cancels: dict[str, tuple[str, float]] = {}
    acknowledged_cancels: set[str] = set()
    terminal_acknowledgements: list[tuple[str, float]] = []
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
            requested_cancels[request_id] = (
                str(event.get("job_id") or "unknown"),
                _epoch(event.get("recorded_at_epoch_s")),
            )
        elif event.get("event") == "cancel_acknowledged":
            acknowledged_cancels.add(request_id)
            if str(event.get("outcome") or "") in {
                "already_terminal",
                "terminated",
                "forced_terminated",
            }:
                terminal_acknowledgements.append(
                    (
                        str(event.get("job_id") or "unknown"),
                        _epoch(event.get("recorded_at_epoch_s")),
                    )
                )
    for request_id, (job_id, requested_at) in requested_cancels.items():
        # The terminal registry record is itself durable provider evidence. A
        # controller can be stopped after persisting that record but before it
        # appends the redundant acknowledgement event; do not invalidate a
        # cleanly terminated job solely because of that event-ordering gap.
        job = _read_json(state_dir / "gpu-job-registry" / f"{job_id}.json")
        registry_acknowledged = (
            str(job.get("cancel_request_id") or "") == request_id
            and str(job.get("status") or "") == "terminated"
            and str(job.get("termination_reason") or "")
            in {"agent_cancelled", "agent_cancelled_forced"}
            and not str(job.get("terminate_error") or "").strip()
        )
        if registry_acknowledged:
            acknowledged_cancels.add(request_id)
        # The agent can repeat a cancellation after the command has already
        # ended, replacing the live request file before the controller writes
        # the first acknowledgement. A later terminal acknowledgement for the
        # same job proves the earlier request cannot leave compute running. Do
        # not apply this across missing timestamps or to a request made after
        # the acknowledgement.
        superseded_acknowledged = (
            requested_at > 0
            and str(job.get("status") or "") in GPU_TERMINAL_STATUSES
            and not str(job.get("terminate_error") or "").strip()
            and any(
                ack_job_id == job_id and ack_at >= requested_at
                for ack_job_id, ack_at in terminal_acknowledgements
                if ack_at > 0
            )
        )
        if superseded_acknowledged:
            acknowledged_cancels.add(request_id)
        if request_id not in acknowledged_cancels:
            reasons.append(
                _reason(
                    "gpu_cancellation_unacknowledged",
                    "control-events.jsonl",
                    f"GPU cancellation for {job_id} lacks provider acknowledgement",
                )
            )
    observations["cancel_requests"] = len(requested_cancels)
    observations["cancel_requests_acknowledged"] = len(
        set(requested_cancels) & acknowledged_cancels
    )

    # These outcomes are suspicious, but not sufficient by themselves to call
    # a benchmark run invalid: a capable agent may intentionally withhold every
    # locally-invalid policy, and an operator may deliberately stop a run early.
    # Surface both conditions so final trial selection requires an explicit
    # trace review instead of silently treating them as ordinary clean runs.
    review_reasons: list[dict[str, str]] = []
    status = _read_json(state_dir / "status.json")
    ledger = status.get("ledger") if isinstance(status.get("ledger"), dict) else {}
    accepted_submissions = ledger.get("accepted")
    observations["accepted_submissions"] = accepted_submissions
    if terminal_evidence and accepted_submissions == 0:
        review_reasons.append(
            _reason(
                "zero_accepted_submissions",
                "status.json",
                "terminal trial has no structurally valid accepted policy submissions",
                severity="review",
            )
        )

    budget = run.get("agent_cost_budget_usd")
    cost = _read_json(state_dir / "telemetry" / "agent-cost.json")
    total = cost.get("total_usd")
    utilization: float | None = None
    try:
        if float(budget) > 0 and float(total) >= 0:
            utilization = float(total) / float(budget)
    except (TypeError, ValueError):
        pass
    observations["budget_utilization"] = utilization
    threshold = float(
        run.get(
            "integrity_min_budget_utilization",
            DEFAULT_MIN_BUDGET_UTILIZATION_FOR_REVIEW,
        )
    )
    observations["min_budget_utilization_for_review"] = threshold
    if terminal_evidence and utilization is not None and utilization < threshold:
        review_reasons.append(
            _reason(
                "low_budget_utilization",
                "telemetry/agent-cost.json",
                f"terminal trial used {utilization:.1%} of its configured budget",
                severity="review",
            )
        )

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
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run["run_id"]),
        "status": "invalid_infrastructure" if invalid else "clean",
        "benchmark_valid": not invalid,
        "leaderboard_eligible": not invalid,
        "replacement_required": invalid,
        "reasons": unique,
        "review_recommended": bool(review_reasons),
        "review_reasons": review_reasons,
        "observations": observations,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
