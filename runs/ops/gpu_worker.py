#!/usr/bin/env python3
"""Host-side GPU worker dispatch for CPU-agent durable lane runs.

Claims jobs written by in-sandbox ``sprint-gpu-train`` under
``/durable/runs/<run_id>/gpu-jobs/queue/`` and starts a preemptible A10G
Modal Sandbox that mounts the same volume. The Codex/agent sandbox stays on
CPU (gpus=0) so GPU preemption cannot kill the harness.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / "challenge" / "g1-sprint-100m-lane" / "environment"
DOCKERFILE = ENV_DIR / "Dockerfile"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ENV_DIR))
import gpu_claim  # noqa: E402
import sprintctl  # noqa: E402
from sprint_resilience import (  # noqa: E402
    Lease,
    ProbeResult,
    ProbeState,
    ProviderHandle,
    RetryPolicy,
)

WORKER_TAG_ROLE = "gpu-worker"
MAX_ACTIVE_TRAINING_JOBS_PER_RUN = 1
CLAIM_STALE_SEC = int(os.environ.get("SPRINT_GPU_CLAIM_STALE_SEC", "900"))
HEARTBEAT_TIMEOUT_SEC = int(
    os.environ.get(
        "SPRINT_GPU_HEARTBEAT_TIMEOUT_SEC",
        str(gpu_claim.DEFAULT_HEARTBEAT_TIMEOUT_SEC),
    )
)
STARTUP_GRACE_SEC = int(
    os.environ.get(
        "SPRINT_GPU_STARTUP_GRACE_SEC", str(gpu_claim.DEFAULT_STARTUP_GRACE_SEC)
    )
)
DEAD_GRACE_SEC = int(
    os.environ.get("SPRINT_GPU_DEAD_GRACE_SEC", str(gpu_claim.DEFAULT_DEAD_GRACE_SEC))
)


def jobs_prefix(run_id: str) -> str:
    return f"runs/{run_id}/gpu-jobs"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def volume_ls_json_names(run: dict[str, Any], remote_path: str) -> list[str]:
    result = sprintctl.run_command(
        sprintctl.modal_command(
            "volume", "ls", "--json", str(run["volume_name"]), remote_path
        ),
        run=run,
        check=False,
        timeout=60,
    )
    if result.returncode == 0 and result.stdout.strip().startswith("["):
        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            rows = []
        names = []
        for row in rows:
            if isinstance(row, dict):
                path = str(row.get("path") or row.get("filename") or "")
            else:
                path = str(row)
            base = Path(path.rstrip("/")).name
            if base.endswith(".json"):
                names.append(base)
        return sorted(set(names))

    # Fallback: plain / table text.
    result = sprintctl.run_command(
        sprintctl.modal_command("volume", "ls", str(run["volume_name"]), remote_path),
        run=run,
        check=False,
        timeout=60,
    )
    names = []
    for line in result.stdout.splitlines():
        for token in line.replace("│", " ").split():
            if token.endswith(".json"):
                names.append(Path(token).name)
    return sorted(set(names))


def put_json(run: dict[str, Any], remote_path: str, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp = Path(handle.name)
    try:
        os.chmod(tmp, 0o600)
        sprintctl.volume_upload(run, tmp, remote_path)
    finally:
        tmp.unlink(missing_ok=True)


def normalize_job_command(job: dict[str, Any]) -> dict[str, Any]:
    """Prefer python3 when the image has no bare ``python`` binary."""
    command = list(job.get("command") or [])
    if command and command[0] == "python":
        job = dict(job)
        job["command"] = ["python3", *command[1:]]
    return job


def load_job(run: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    """Load job state. Prefer status/ (canonical) over queue/ (enqueue snapshot).

    Queue files stay at status=pending after claim; reading them first caused
    dispatch_once to re-spawn the same job forever and never reach later jobs.
    """
    prefix = jobs_prefix(str(run["run_id"]))
    text = sprintctl.volume_get_text(run, f"{prefix}/status/{job_id}.json")
    if text is None:
        text = sprintctl.volume_get_text(run, f"{prefix}/queue/{job_id}.json")
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_remote_json(
    run: dict[str, Any], remote_path: str
) -> dict[str, Any] | None:
    text = sprintctl.volume_get_text(run, remote_path)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def attempt_path(run_id: str, job_id: str, attempt: int) -> str:
    return f"{jobs_prefix(run_id)}/attempts/{job_id}/{attempt}.json"


def heartbeat_path(run_id: str, job_id: str, attempt: int) -> str:
    return f"{jobs_prefix(run_id)}/heartbeats/{job_id}/{attempt}.json"


def load_attempt_record(
    run: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any] | None:
    attempt = int(job.get("attempt") or 0)
    if attempt <= 0:
        return None
    return load_remote_json(
        run, attempt_path(str(run["run_id"]), str(job["job_id"]), attempt)
    )


def load_heartbeat(
    run: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any] | None:
    attempt = int(job.get("attempt") or 0)
    if attempt <= 0:
        return None
    return load_remote_json(
        run, heartbeat_path(str(run["run_id"]), str(job["job_id"]), attempt)
    )


def probe_sandbox(job: dict[str, Any]) -> tuple[str, int | None, str | None]:
    """Return (alive|exited|unknown, exit_code, error)."""
    sandbox_id = str(job.get("sandbox_id") or "")
    if not sandbox_id:
        return "unknown", None, "missing_sandbox_id"
    result = ModalSandboxProvider({}).probe(
        ProviderHandle(provider=ModalSandboxProvider.name, attempt_id=sandbox_id)
    )
    return result.state.value, result.exit_code, result.error


STANDING_TAG_ROLE = "gpu-standing"


def standing_enabled(run: dict[str, Any]) -> bool:
    return bool(run.get("standing_gpu_worker"))


def _standing_state_path(run: dict[str, Any]) -> Path:
    return sprintctl.state_dir_for(str(run["run_id"])) / "gpu-standing.json"


def ensure_standing_sandbox(run: dict[str, Any]) -> dict[str, Any]:
    """Hold exactly one A10G per run for the run's lifetime.

    This is the fairness half of the design: every arm keeps its own GPU at all
    times rather than racing for an on-demand worker, so two arms cannot get
    different effective compute. The robustness half is that Modal preempts GPU
    sandboxes freely -- so when this one dies we simply create another and the
    per-job lease/checkpoint machinery resumes the interrupted job.

    Deliberately idempotent: a live sandbox is reused, a dead one is replaced.
    """
    import modal

    path = _standing_state_path(run)
    state: dict[str, Any] = {}
    if path.is_file():
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}

    sid = str(state.get("sandbox_id") or "")
    if sid:
        try:
            if modal.Sandbox.from_id(sid).poll() is None:
                return {"sandbox_id": sid, "action": "reuse"}
        except Exception:  # noqa: BLE001 - treat an unreadable sandbox as gone
            pass

    app = modal.App.lookup(
        str(run.get("training_app_name") or run["app_name"]),
        create_if_missing=True,
    )
    image = modal.Image.from_dockerfile(str(DOCKERFILE), context_dir=str(ENV_DIR))
    volume = modal.Volume.from_name(str(run["volume_name"]))
    sandbox = modal.Sandbox.create(
        "bash", "-c", "exec sleep infinity",
        app=app, image=image, gpu="A10G", cpu=8, memory=32768,
        block_network=True,
        timeout=int(run.get("sandbox_timeout_secs") or 86400),
        volumes={"/durable": volume},
        tags={
            "sprint.role": STANDING_TAG_ROLE,
            "sprint.run_id": str(run["run_id"]),
            "harbor.managed": "false",
        },
    )
    new_id = str(sandbox.object_id)
    prev = state.get("sandbox_id")
    payload = {
        "sandbox_id": new_id,
        "created_at": utc_now(),
        "replaced": prev,
        "generation": int(state.get("generation") or 0) + 1,
    }
    sprintctl.atomic_write_json(path, payload, mode=0o600)
    return {"sandbox_id": new_id, "action": "replaced" if prev else "created",
            "replaced": prev}


def exec_on_standing(run: dict[str, Any], job: dict[str, Any]) -> str:
    """Run one job inside the standing sandbox instead of a fresh one.

    The worker entrypoint, its arguments, the lease and every durable path stay
    identical to the per-job path -- only the container it runs in changes, so
    eval behaviour is unaffected.
    """
    import modal

    info = ensure_standing_sandbox(run)
    sid = info["sandbox_id"]
    sandbox = modal.Sandbox.from_id(sid)
    sandbox.exec(
        "bash", "-c",
        "python3 /opt/sprint-gpu-worker-run.py "
        + shlex.quote(str(run["run_id"])) + " "
        + shlex.quote(str(job["job_id"])) + " "
        + shlex.quote(str(int(job["attempt"]))) + " "
        + shlex.quote(str(job["lease_id"])),
    )
    return sid


def spawn_gpu_sandbox(run: dict[str, Any], job: dict[str, Any]) -> str:
    import modal

    run_id = str(run["run_id"])
    job_id = str(job["job_id"])
    attempt = int(job["attempt"])
    lease_id = str(job["lease_id"])
    app = modal.App.lookup(
        str(run.get("training_app_name") or run["app_name"]),
        create_if_missing=True,
    )
    image = modal.Image.from_dockerfile(
        str(DOCKERFILE),
        context_dir=str(ENV_DIR),
    )
    volume = modal.Volume.from_name(str(run["volume_name"]))
    timeout = int(job.get("timeout_sec") or 3600)
    timeout = max(60, min(timeout, 24 * 60 * 60))
    command = (
        "python3 /opt/sprint-gpu-worker-run.py "
        + shlex.quote(run_id)
        + " "
        + shlex.quote(job_id)
        + " "
        + shlex.quote(str(attempt))
        + " "
        + shlex.quote(lease_id)
    )
    sandbox = modal.Sandbox.create(
        "bash",
        "-c",
        command,
        app=app,
        image=image,
        gpu="A10G",
        cpu=8,
        memory=32768,
        block_network=True,
        timeout=timeout,
        volumes={"/durable": volume},
        tags={
            "sprint.role": WORKER_TAG_ROLE,
            "sprint.run_id": run_id,
            "sprint.job_id": job_id,
            "sprint.attempt": str(attempt),
            "harbor.managed": "false",
        },
    )
    return str(sandbox.object_id)


class ModalSandboxProvider:
    """Modal-specific mechanics behind the provider-neutral job contract."""

    name = "modal-sandbox"

    def __init__(self, run: dict[str, Any]) -> None:
        self.run = run

    def start(self, job: dict[str, Any], lease: Lease) -> ProviderHandle:
        if not lease.owns(job):
            raise RuntimeError("refusing to start a fenced Modal attempt")
        attempt_id = (
            exec_on_standing(self.run, job)
            if standing_enabled(self.run)
            else spawn_gpu_sandbox(self.run, job)
        )
        return ProviderHandle(provider=self.name, attempt_id=attempt_id)

    def probe(self, handle: ProviderHandle) -> ProbeResult:
        try:
            import modal

            code = modal.Sandbox.from_id(handle.attempt_id).poll()
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(
                ProbeState.UNKNOWN, error=f"{type(exc).__name__}: {exc}"
            )
        if code is None:
            return ProbeResult(ProbeState.ALIVE)
        return ProbeResult(ProbeState.EXITED, exit_code=int(code))

    def terminate(self, handle: ProviderHandle) -> str | None:
        try:
            import modal

            modal.Sandbox.from_id(handle.attempt_id).terminate()
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return None


def persist_job(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Write canonical status/ and mirror into queue/."""
    prefix = jobs_prefix(str(run["run_id"]))
    job_id = str(job["job_id"])
    put_json(run, f"{prefix}/status/{job_id}.json", job)
    put_json(run, f"{prefix}/queue/{job_id}.json", job)
    return job


def mark_dispatched(
    run: dict[str, Any], job: dict[str, Any], sandbox_id: str
) -> dict[str, Any]:
    payload = dict(job)
    now_epoch = time.time()
    payload.update(
        {
            "status": "dispatched",
            "dispatched_at": utc_now(),
            "dispatched_at_epoch_s": now_epoch,
            "sandbox_id": sandbox_id,
            "gpu_type": "A10G",
        }
    )
    return persist_job(run, payload)


def try_claim_job(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    claim_id: str,
    stale_sec: int = CLAIM_STALE_SEC,
) -> dict[str, Any] | None:
    """Persist claiming ownership before Sandbox.create. None if not ours."""
    action = gpu_claim.select_claim_action(
        job, claim_id=claim_id, stale_sec=stale_sec
    )
    if action == "skip":
        return None
    payload = gpu_claim.build_claim_payload(job, claim_id=claim_id)
    persist_job(run, payload)
    # Confirm we still own after the write (best-effort against races).
    refreshed = load_job(run, str(job["job_id"]))
    if not gpu_claim.ownership_matches(refreshed, claim_id):
        return None
    return refreshed or payload


def _timeline_event(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    phase: str,
    action: str,
    epoch_s: int | None = None,
    **detail: Any,
) -> None:
    try:
        import gpu_timeline_host

        gpu_timeline_host.host_append_event(
            run,
            phase=phase,
            action=action,
            job_id=str(job["job_id"]),
            attempt=int(job.get("attempt") or 0),
            lease_id=str(job.get("lease_id") or ""),
            epoch_s=epoch_s,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"timeline emit failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _close_attempt_timeline(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    epoch_s: int,
    reason: str,
) -> None:
    for phase in ("gpu_active", "isaac_starting", "gpu_worker_starting"):
        _timeline_event(
            run,
            job,
            phase=phase,
            action="exit",
            epoch_s=epoch_s,
            reason=reason,
            synthetic=True,
        )


def _terminate_sandbox(job: dict[str, Any]) -> str | None:
    sandbox_id = str(job.get("sandbox_id") or "")
    if not sandbox_id:
        return None
    return ModalSandboxProvider({}).terminate(
        ProviderHandle(provider=ModalSandboxProvider.name, attempt_id=sandbox_id)
    )


def schedule_retry(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    heartbeat: dict[str, Any] | None,
    exit_code: int | None,
    reason: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence the old lease, then queue the same logical job with backoff."""
    ref = time.time() if now is None else float(now)
    payload = dict(job)
    attempt = int(payload.get("attempt") or 0)
    lease_id = str(payload.get("lease_id") or "")
    close_epoch = int(gpu_claim.heartbeat_epoch(heartbeat) or ref)
    _close_attempt_timeline(run, payload, epoch_s=close_epoch, reason=reason)
    _timeline_event(
        run,
        payload,
        phase="gpu_lifecycle",
        action="instant",
        epoch_s=close_epoch,
        event="gpu_preempted" if reason in {"graceful_preemption", "worker_lost"} else "gpu_attempt_lost",
        reason=reason,
        checkpoint=(heartbeat or {}).get("checkpoint"),
        progress=(heartbeat or {}).get("progress"),
    )
    history = list(payload.get("attempt_history") or [])
    history.append(
        {
            "attempt": attempt,
            "lease_id": lease_id or None,
            "sandbox_id": payload.get("sandbox_id"),
            "status": "lost",
            "reason": reason,
            "exit_code": exit_code,
            "last_heartbeat_epoch_s": gpu_claim.heartbeat_epoch(heartbeat),
            "progress": (heartbeat or {}).get("progress"),
            "checkpoint": (heartbeat or {}).get("checkpoint"),
            "finished_at": utc_now(),
        }
    )
    payload["attempt_history"] = history
    payload["fence_epoch"] = int(payload.get("fence_epoch") or 0) + 1
    payload["fenced_lease_id"] = lease_id or None
    payload["last_sandbox_id"] = payload.get("sandbox_id")
    payload["last_exit_code"] = exit_code
    payload["last_failure_reason"] = reason
    payload["last_progress"] = (heartbeat or {}).get("progress")
    payload["last_checkpoint"] = (heartbeat or {}).get("checkpoint")
    payload.pop("sandbox_id", None)
    payload.pop("lease_id", None)
    payload.pop("claim_id", None)
    payload.pop("death_observed_at", None)
    payload.pop("death_observed_epoch_s", None)

    retry_policy = RetryPolicy.from_job(payload)
    if not retry_policy.allows_after(attempt):
        payload.update(
            {
                "status": "failed",
                "failure_reason": "max_attempts_exhausted",
                "finished_at": utc_now(),
                "finished_at_epoch_s": ref,
            }
        )
        persist_job(run, payload)
        terminate_error = _terminate_sandbox(job)
        if terminate_error:
            payload["terminate_error"] = terminate_error
            persist_job(run, payload)
        return payload

    delay = retry_policy.delay_after(attempt)
    payload.update(
        {
            "status": "retry_wait",
            "next_attempt": attempt + 1,
            "retry_count": int(payload.get("retry_count") or 0) + 1,
            "retry_not_before_epoch_s": ref + delay,
            "retry_not_before": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(ref + delay)
            ),
            "retry_reason": reason,
        }
    )
    persist_job(run, payload)
    terminate_error = _terminate_sandbox(job)
    if terminate_error:
        payload["terminate_error"] = terminate_error
        persist_job(run, payload)
    timeline_job = dict(payload)
    timeline_job["attempt"] = attempt + 1
    timeline_job["lease_id"] = ""
    _timeline_event(
        run,
        timeline_job,
        phase="gpu_queue_wait",
        action="enter",
        epoch_s=int(ref),
        reason=reason,
        retry_of_attempt=attempt,
    )
    return payload


def reconcile_job(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    now: float | None = None,
    probe_fn=probe_sandbox,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold one attempt record into canonical state and detect worker loss."""
    ref = time.time() if now is None else float(now)
    status = str(job.get("status") or "")
    if status not in gpu_claim.OWNED:
        return job, {"decision": "ignore"}

    attempt_record = load_attempt_record(run, job)
    heartbeat = load_heartbeat(run, job)
    owned_record = bool(
        attempt_record
        and int(attempt_record.get("attempt") or 0) == int(job.get("attempt") or 0)
        and str(attempt_record.get("lease_id") or "")
        == str(job.get("lease_id") or "")
    )
    if owned_record and str(attempt_record.get("status") or "") in {
        "succeeded",
        "failed",
    }:
        terminal = dict(job)
        terminal.update(
            {
                key: value
                for key, value in attempt_record.items()
                if key
                in {
                    "status",
                    "started_at",
                    "started_at_epoch_s",
                    "finished_at",
                    "finished_at_epoch_s",
                    "exit_code",
                    "error",
                    "progress",
                    "checkpoint",
                }
            }
        )
        terminal["attempt_record"] = attempt_path(
            str(run["run_id"]), str(job["job_id"]), int(job["attempt"])
        )
        persist_job(run, terminal)
        _timeline_event(
            run,
            terminal,
            phase="gpu_lifecycle",
            action="instant",
            epoch_s=int(terminal.get("finished_at_epoch_s") or ref),
            event="gpu_released",
            reason=str(terminal.get("status") or "terminal"),
        )
        return terminal, {"decision": "terminal", "status": terminal["status"]}

    if owned_record and str(attempt_record.get("status") or "") == "interrupted":
        retried = schedule_retry(
            run,
            job,
            heartbeat=heartbeat,
            exit_code=attempt_record.get("exit_code"),
            reason="graceful_preemption",
            now=ref,
        )
        return retried, {
            "decision": "retry",
            "status": retried["status"],
            "reason": "graceful_preemption",
        }

    if owned_record and str(attempt_record.get("status") or "") == "running":
        if status != "running":
            job = dict(job)
            job["status"] = "running"
            job["started_at"] = attempt_record.get("started_at")
            job["started_at_epoch_s"] = attempt_record.get("started_at_epoch_s")
            persist_job(run, job)

    probe_state, exit_code, probe_error = probe_fn(job)
    decision = gpu_claim.assess_worker_liveness(
        job,
        heartbeat,
        standing=standing_enabled(run),
        probe_state=probe_state,
        now=ref,
        heartbeat_timeout_sec=int(
            job.get("heartbeat_timeout_sec") or HEARTBEAT_TIMEOUT_SEC
        ),
        startup_grace_sec=int(job.get("startup_grace_sec") or STARTUP_GRACE_SEC),
        dead_grace_sec=int(job.get("dead_grace_sec") or DEAD_GRACE_SEC),
    )
    detail = {
        "decision": decision,
        "probe_state": probe_state,
        "exit_code": exit_code,
        "probe_error": probe_error,
        "heartbeat_epoch_s": gpu_claim.heartbeat_epoch(heartbeat),
    }
    if decision == "observe":
        observed = dict(job)
        observed["status"] = "death_observed"
        observed["death_observed_at"] = utc_now()
        observed["death_observed_epoch_s"] = ref
        observed["death_probe_state"] = probe_state
        observed["death_probe_error"] = probe_error
        persist_job(run, observed)
        return observed, detail
    if decision != "dead":
        return job, detail
    retried = schedule_retry(
        run,
        job,
        heartbeat=heartbeat,
        exit_code=exit_code,
        reason="worker_lost",
        now=ref,
    )
    return retried, detail


def list_pending_job_ids(run: dict[str, Any]) -> list[str]:
    prefix = jobs_prefix(str(run["run_id"]))
    return [
        Path(name).stem
        for name in volume_ls_json_names(run, f"{prefix}/queue")
        if name.endswith(".json")
    ]


def list_job_ids(run: dict[str, Any]) -> list[str]:
    prefix = jobs_prefix(str(run["run_id"]))
    names = set(volume_ls_json_names(run, f"{prefix}/queue"))
    names.update(volume_ls_json_names(run, f"{prefix}/status"))
    return sorted(Path(name).stem for name in names if name.endswith(".json"))


def active_training_job_ids(run: dict[str, Any]) -> list[str]:
    """Return logical jobs that currently own the run's training GPU slot."""
    active: list[str] = []
    for job_id in list_job_ids(run):
        job = load_job(run, job_id)
        if job and str(job.get("status") or "") in gpu_claim.OWNED:
            active.append(job_id)
    return active


def operator_stop_requested(state_dir: Path) -> bool:
    if (state_dir / "STOP").is_file() or (state_dir / "STOP_REQUESTED.json").is_file():
        return True
    ack = state_dir / "STOP_ACK.json"
    if not ack.is_file():
        return False
    try:
        return json.loads(ack.read_text()).get("reason") == "operator_stop"
    except (OSError, json.JSONDecodeError):
        return True


def _stop_all_locked(
    run: dict[str, Any], *, reason: str = "operator_stop"
) -> list[dict[str, Any]]:
    stopped: list[dict[str, Any]] = []
    for job_id in list_job_ids(run):
        job = load_job(run, job_id)
        if not job or str(job.get("status") or "") in gpu_claim.TERMINAL:
            continue
        heartbeat = load_heartbeat(run, job)
        payload = dict(job)
        payload["fence_epoch"] = int(payload.get("fence_epoch") or 0) + 1
        payload["fenced_lease_id"] = payload.get("lease_id")
        payload["status"] = "terminated"
        payload["termination_reason"] = reason
        payload["terminated_at"] = utc_now()
        payload["terminated_at_epoch_s"] = time.time()
        persist_job(run, payload)  # fence before issuing terminate
        _close_attempt_timeline(
            run,
            job,
            epoch_s=int(gpu_claim.heartbeat_epoch(heartbeat) or time.time()),
            reason=reason,
        )
        _timeline_event(
            run,
            job,
            phase="gpu_lifecycle",
            action="instant",
            event="gpu_released",
            reason=reason,
        )
        error = _terminate_sandbox(job)
        if error:
            payload["terminate_error"] = error
            persist_job(run, payload)
        stopped.append(payload)
    return stopped


def stop_all(
    run: dict[str, Any], *, reason: str = "operator_stop"
) -> list[dict[str, Any]]:
    state_dir = Path(str(run["state_dir"]))
    with gpu_claim.dispatch_lock(state_dir) as got_lock:
        if not got_lock:
            raise RuntimeError("dispatch lock busy while stopping GPU workers")
        return _stop_all_locked(run, reason=reason)


def terminate_job(run: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Fence and stop one GPU worker; the CPU harness is untouched."""
    state_dir = Path(str(run["state_dir"]))
    with gpu_claim.dispatch_lock(state_dir) as got_lock:
        if not got_lock:
            raise RuntimeError("dispatch lock busy while terminating GPU worker")
        job = load_job(run, job_id) or {
            "job_id": job_id,
            "run_id": run["run_id"],
        }
        if str(job.get("status") or "") in gpu_claim.TERMINAL:
            return job
        payload = dict(job)
        payload["fence_epoch"] = int(payload.get("fence_epoch") or 0) + 1
        payload["fenced_lease_id"] = payload.get("lease_id")
        payload["status"] = "terminated"
        payload["termination_reason"] = "manual_gpu_terminate"
        payload["terminated_at"] = utc_now()
        payload["terminated_at_epoch_s"] = time.time()
        persist_job(run, payload)
        error = _terminate_sandbox(job)
        if error:
            payload["terminate_error"] = error
            persist_job(run, payload)
        return payload


def _candidate_job_ids(run: dict[str, Any], *, now: float | None = None) -> list[str]:
    out: list[tuple[float, str]] = []
    for job_id in list_job_ids(run):
        job = load_job(run, job_id)
        if not job:
            continue
        if (
            gpu_claim.select_claim_action(
                job, claim_id=uuid.uuid4().hex, now=now
            )
            == "claim"
        ):
            try:
                created = float(job.get("created_at_epoch_s") or 0)
            except (TypeError, ValueError):
                created = 0.0
            out.append((created, job_id))
    return [job_id for _created, job_id in sorted(out)]


def dispatch_once(run_id: str) -> dict[str, Any]:
    """Claim (persist) then spawn at most one GPU job for the run.

    Ownership is written as ``status=claiming`` *before* ``Sandbox.create`` /
    image build so concurrent monitors cannot start duplicate workers or emit
    repeated ``gpu_worker_starting`` enters for the same claim.
    """
    state_dir, run = sprintctl.load_run(run_id)
    if not run.get("cpu_agent_gpu_worker"):
        return {
            "run_id": run_id,
            "skipped": True,
            "reason": "cpu_agent_gpu_worker unset",
        }

    actions: list[dict[str, Any]] = []
    reconciled: list[dict[str, Any]] = []
    pending: list[str] = []
    with gpu_claim.dispatch_lock(state_dir) as got_lock:
        if not got_lock:
            result = {
                "run_id": run_id,
                "pending": [],
                "actions": [],
                "skipped": True,
                "reason": "dispatch_lock_busy",
                "ts": utc_now(),
            }
            sprintctl.atomic_write_json(
                state_dir / "gpu-dispatch.json", result, mode=0o600
            )
            return result

        if operator_stop_requested(state_dir):
            stopped = _stop_all_locked(run, reason="operator_stop")
            result = {
                "run_id": run_id,
                "pending": [],
                "actions": [
                    {
                        "job_id": item.get("job_id"),
                        "status": item.get("status"),
                        "action": "stop",
                    }
                    for item in stopped
                ],
                "reconciled": [],
                "skipped": True,
                "reason": "operator_stop",
                "ts": utc_now(),
            }
            sprintctl.atomic_write_json(
                state_dir / "gpu-dispatch.json", result, mode=0o600
            )
            return result

        now = time.time()
        for job_id in list_job_ids(run):
            job = load_job(run, job_id)
            if not job or str(job.get("status") or "") not in gpu_claim.OWNED:
                continue
            updated, detail = reconcile_job(run, job, now=now)
            reconciled.append(
                {
                    "job_id": job_id,
                    "attempt": updated.get("attempt"),
                    "status": updated.get("status"),
                    **detail,
                }
            )

        pending = _candidate_job_ids(run, now=now)
        active = active_training_job_ids(run)
        for job_id in pending:
            if len(active) >= MAX_ACTIVE_TRAINING_JOBS_PER_RUN:
                break
            job = load_job(run, job_id)
            if not job:
                continue
            claim_id = uuid.uuid4().hex[:16]
            job = normalize_job_command(job)
            claimed = try_claim_job(
                run, job, claim_id=claim_id, stale_sec=CLAIM_STALE_SEC
            )
            if not claimed:
                continue
            _timeline_event(
                run,
                claimed,
                phase="gpu_worker_starting",
                action="enter",
                source="dispatch",
                claim_id=claim_id,
            )
            _timeline_event(
                run,
                claimed,
                phase="gpu_lifecycle",
                action="instant",
                event="gpu_allocated" if int(claimed.get("attempt") or 1) == 1 else "gpu_reallocated",
                retry_reason=claimed.get("retry_reason"),
            )
            try:
                lease = Lease(
                    job_id=str(claimed["job_id"]),
                    attempt=int(claimed["attempt"]),
                    lease_id=str(claimed["lease_id"]),
                    fence_epoch=int(claimed.get("fence_epoch") or 0),
                )
                handle = ModalSandboxProvider(run).start(claimed, lease)
                sandbox_id = handle.attempt_id
                # Re-confirm ownership before publishing sandbox_id.
                latest = load_job(run, job_id)
                if not gpu_claim.ownership_matches(latest, claim_id):
                    ModalSandboxProvider(run).terminate(handle)
                    actions.append(
                        {
                            "job_id": job_id,
                            "error": "lost_claim_after_spawn",
                            "sandbox_id": sandbox_id,
                        }
                    )
                    continue
                payload = mark_dispatched(run, latest or claimed, sandbox_id)
                actions.append(
                    {
                        "job_id": job_id,
                        "attempt": payload.get("attempt"),
                        "sandbox_id": sandbox_id,
                        "status": payload.get("status"),
                        "claim_id": claim_id,
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                current = load_job(run, job_id) or claimed
                error = f"{type(exc).__name__}: {exc}"
                retried = schedule_retry(
                    run,
                    current,
                    heartbeat=None,
                    exit_code=None,
                    reason="spawn_failed",
                    now=time.time(),
                )
                actions.append(
                    {
                        "job_id": job_id,
                        "attempt": current.get("attempt"),
                        "status": retried.get("status"),
                        "error": error,
                    }
                )
                break

    result = {
        "run_id": run_id,
        "pending": pending,
        "active_training_jobs": active,
        "max_active_training_jobs": MAX_ACTIVE_TRAINING_JOBS_PER_RUN,
        "actions": actions,
        "reconciled": reconciled,
        "ts": utc_now(),
    }
    if active and not actions:
        result["reason"] = "training_concurrency_limit"
    sprintctl.atomic_write_json(state_dir / "gpu-dispatch.json", result, mode=0o600)
    return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "Usage: gpu_worker.py dispatch --run-id ID\n"
            "       gpu_worker.py terminate --run-id ID --job-id ID",
            file=sys.stderr,
        )
        return 2
    action = argv[0]
    run_id = None
    job_id = None
    args = argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--run-id":
            run_id = args[i + 1]
            i += 2
        elif args[i] == "--job-id":
            job_id = args[i + 1]
            i += 2
        else:
            print(f"unknown arg: {args[i]}", file=sys.stderr)
            return 2
    if not run_id:
        print("--run-id required", file=sys.stderr)
        return 2
    os.environ.setdefault("MODAL_PROFILE", "kevinli020508")
    if action == "dispatch":
        print(json.dumps(dispatch_once(run_id), indent=2, sort_keys=True))
        return 0
    if action == "terminate":
        if not job_id:
            print("--job-id required", file=sys.stderr)
            return 2
        _, run = sprintctl.load_run(run_id)
        print(json.dumps(terminate_job(run, job_id), indent=2, sort_keys=True))
        return 0
    print(f"unknown action: {action}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
