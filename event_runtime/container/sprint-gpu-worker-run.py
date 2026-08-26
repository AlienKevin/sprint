#!/usr/bin/env python3
"""In-sandbox entrypoint for a host-dispatched GPU training job.

Starts durable GPU telemetry, emits timeline phases, runs the job command.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from pathlib import PurePosixPath

# Same directory as this script when installed at /opt/
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import sprint_gpu_timeline as timeline  # type: ignore
except ImportError:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sprint_gpu_timeline",
        Path(__file__).resolve().parent / "sprint-gpu-timeline.py",
    )
    timeline = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(timeline)

from sprint_resilience import CheckpointStore, Interruption, Lease

MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_ARTIFACT_TOTAL_BYTES = 512 * 1024 * 1024
# Modal Volume commits from the CPU sandbox are not guaranteed to become
# visible in a separately mounted GPU sandbox within 30 seconds.  The $0.10
# shutdown reserve covers more than 160 seconds of one configured A10G worker;
# keep the propagation allowance below that bound so a stale GPU view cannot
# consume the reserve while a healthy five-second CPU watchdog is still the
# primary circuit breaker.
MAX_BUDGET_SNAPSHOT_AGE_SECONDS = 120.0
RUNTIME_BUDGET_SNAPSHOT = Path("/run/sprint-budget-watchdog.json")


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def collect_output_artifacts(
    job: dict,
    *,
    run_id: str,
    job_id: str,
    attempt: int,
    durable_dir: Path = Path("/durable"),
    workspace_dir: Path = Path("/app"),
) -> tuple[list[dict], list[str]]:
    """Commit explicitly declared GPU outputs to a run-scoped durable path."""
    records: list[dict] = []
    missing: list[str] = []
    total_size = 0
    destination_root = (
        durable_dir
        / "runs"
        / run_id
        / "gpu-jobs"
        / "artifacts"
        / job_id
        / f"attempt-{attempt}"
    )
    for raw in job.get("output_paths") or []:
        declared = Path(str(raw))
        try:
            relative = declared.relative_to("/app")
        except ValueError:
            missing.append(str(declared))
            continue
        source = workspace_dir / relative
        if not relative.parts or ".." in relative.parts or not source.is_file():
            missing.append(str(declared))
            continue
        size = source.stat().st_size
        if (
            size <= 0
            or size > MAX_OUTPUT_ARTIFACT_BYTES
            or total_size + size > MAX_OUTPUT_ARTIFACT_TOTAL_BYTES
        ):
            missing.append(f"{source} (invalid size {size})")
            continue
        total_size += size
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / source.name
        tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, destination)
        digest = file_sha256(destination)
        records.append(
            {
                "source_path": str(declared),
                "path": str(destination),
                "name": source.name,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return records, missing


def submit_declared_policies(job: dict) -> list[dict]:
    """Submit only policies explicitly marked in the immutable job request."""
    results: list[dict] = []
    for raw in job.get("submission_paths") or []:
        completed = subprocess.run(
            [
                "/usr/local/bin/event",
                "archive",
                str(raw),
                "--note",
                f"explicit GPU output from job {job.get('job_id')}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(
            {
                "path": str(raw),
                "return_code": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
                "state": "staged" if completed.returncode == 0 else "rejected",
            }
        )
    return results


def safe_extract_work_archive(archive: Path, destination: Path) -> None:
    """Extract only regular files/directories rooted at ``app/``.

    GPU workspaces are agent-authored.  Never let tar links, device nodes, or
    path traversal overwrite the trusted worker/sampler outside the workspace.
    """
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise RuntimeError("GPU work archive contains too many members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != "app"
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise RuntimeError(f"unsafe GPU work archive path: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    f"unsupported GPU work archive member: {member.name!r}"
                )
            total += max(0, int(member.size))
            if total > MAX_ARCHIVE_EXPANDED_BYTES:
                raise RuntimeError("GPU work archive expands beyond the size limit")
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"unable to read archive member: {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            os.chmod(target, member.mode & 0o777)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.sync()
    except OSError:
        pass


def start_telemetry(run_id: str, job_id: str, attempt: int, lease_id: str) -> None:
    Path("/run").mkdir(parents=True, exist_ok=True)
    Path("/run/sprint-role").write_text("training-gpu\n")
    Path("/run/sprint-run-id").write_text(run_id + "\n")
    Path("/run/sprint-gpu-job-id").write_text(job_id + "\n")
    Path("/run/sprint-gpu-attempt").write_text(str(attempt) + "\n")
    os.environ["SPRINT_RUN_ID"] = run_id
    os.environ["SPRINT_GPU_JOB_ID"] = job_id
    os.environ["SPRINT_GPU_ATTEMPT"] = str(attempt)
    os.environ["SPRINT_GPU_LEASE_ID"] = lease_id
    os.environ["SPRINT_REQUESTED_CPU_CORES"] = "6"
    os.environ["SPRINT_REQUESTED_MEMORY_MIB"] = "12288"
    script = Path("/opt/sprint-telemetry.sh")
    if not script.is_file():
        print("telemetry script missing; continuing without sampler", flush=True)
        return
    env = os.environ.copy()
    env["SPRINT_TELEMETRY_INTERVAL"] = env.get("SPRINT_TELEMETRY_INTERVAL", "5")
    subprocess.run(
        [
            str(script),
            "--role",
            "training-gpu",
            "--run-id",
            run_id,
            "--job-id",
            job_id,
            "--out-dir",
            "/logs/artifacts/telemetry",
            "--durable-dir",
            "/durable",
            "--interval-seconds",
            env["SPRINT_TELEMETRY_INTERVAL"],
            "--pidfile",
            "/run/sprint-telemetry-training-gpu.pid",
            "--force",
        ],
        check=False,
        env=env,
    )


def emit(
    run_id: str,
    job_id: str,
    attempt: int,
    lease_id: str,
    phase: str,
    action: str,
    **detail,
) -> None:
    try:
        timeline.append_event(
            run_id,
            phase=phase,
            action=action,
            job_id=job_id,
            attempt=attempt,
            lease_id=lease_id,
            detail=detail,
            durable_dir="/durable",
            also_local=Path("/logs/artifacts/telemetry"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"timeline emit failed: {type(exc).__name__}: {exc}", flush=True)


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


LEASE_VISIBILITY_TIMEOUT_SEC = 120.0
LEASE_VISIBILITY_POLL_SEC = 0.25


def lease_owned(
    path: Path,
    attempt: int,
    lease_id: str,
    fence_epoch: int,
) -> bool:
    job = read_json(path)
    lease = Lease(
        job_id=str(job.get("job_id") or ""),
        attempt=attempt,
        lease_id=lease_id,
        fence_epoch=fence_epoch,
    )
    return lease.owns(job) and str(job.get("status") or "") in {
        "claiming",
        "dispatched",
        "running",
        "death_observed",
    }


def wait_for_lease(
    path: Path,
    attempt: int,
    lease_id: str,
    fence_epoch: int,
    *,
    timeout_sec: float = LEASE_VISIBILITY_TIMEOUT_SEC,
    poll_sec: float = LEASE_VISIBILITY_POLL_SEC,
) -> bool:
    """Wait for a just-published host lease to reach the mounted Volume.

    Modal can start a Sandbox from a Volume snapshot taken just before the
    host's claim record becomes visible in that mount.  Treat an older fence
    epoch as propagation lag, while rejecting a conflicting or newer fence
    immediately.  The immutable lease arguments still remain authoritative;
    this wait never adopts a lease observed from the Volume.
    """

    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        if lease_owned(path, attempt, lease_id, fence_epoch):
            return True
        job = read_json(path)
        try:
            observed_epoch = int(job.get("fence_epoch") or 0)
        except (TypeError, ValueError):
            observed_epoch = 0
        observed_lease = str(job.get("lease_id") or "")
        if observed_epoch > fence_epoch or (
            observed_epoch == fence_epoch
            and observed_lease
            and observed_lease != lease_id
        ):
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.01, poll_sec))


def latest_checkpoint(checkpoint_dir: Path, *, verify_hash: bool = True) -> str | None:
    # The recovery store is for trainer state, not exported inference policies.
    # A TorchScript policy can be perfectly valid for SCORE while being
    # impossible for the submitted process to resume. Keep older resumable
    # generations eligible when a newer policy artifact was committed by
    # mistake, and never feed a known inference-only artifact to --resume.
    non_resumable_kinds = {
        "inference_policy",
        "policy",
        "submission_policy",
        "torchscript_policy",
    }
    store = CheckpointStore(checkpoint_dir)
    for committed in store.iter_valid(verify_hash=verify_hash):
        metadata = committed.metadata if isinstance(committed.metadata, dict) else {}
        resumable = metadata.get("resumable")
        kind = str(metadata.get("kind") or metadata.get("artifact_role") or "")
        kind = kind.strip().lower().replace("-", "_")
        if resumable is False or kind in non_resumable_kinds:
            continue
        return str(committed.path)
    return None


def progress_snapshot(
    progress_file: Path,
    checkpoint_dir: Path,
    *,
    verify_checkpoint: bool = True,
) -> tuple[object, str | None]:
    progress: object = None
    if progress_file.is_file():
        try:
            progress = json.loads(progress_file.read_text())
        except (OSError, json.JSONDecodeError):
            try:
                progress = progress_file.read_text()[-1000:]
            except OSError:
                progress = None
    return progress, latest_checkpoint(checkpoint_dir, verify_hash=verify_checkpoint)


def progress_declares_completion(progress: object) -> bool:
    """Return whether the child atomically declared its useful work complete.

    Isaac/Kit teardown can hang after training has already committed every
    requested artifact. The progress file is the cooperative boundary: the
    worker terminates a lingering simulator only after ``finished`` is the
    literal JSON boolean ``true``. Required output validation still runs
    before the job can be reported as successful.
    """
    return isinstance(progress, dict) and progress.get("finished") is True


def checkpoint_resume_metadata(checkpoint: str | None) -> dict[str, str]:
    if not checkpoint:
        return {}
    manifest_path = Path(checkpoint).parent / "manifest.json"
    payload = read_json(manifest_path)
    if not payload:
        return {}
    preferred = {
        "SPRINT_GPU_RESUME_CHECKPOINT_ID": str(payload.get("checkpoint_id") or ""),
        "SPRINT_GPU_RESUME_SEQUENCE": str(payload.get("sequence") or ""),
        "SPRINT_GPU_REPLAY_CURSOR": str(payload.get("replay_cursor") or ""),
        "SPRINT_GPU_RESUME_MANIFEST": str(manifest_path),
    }
    return preferred


def build_attempt_command(
    job: dict,
    attempt: int,
    checkpoint: str | None,
    *,
    isaac_bootstrap: Path = Path("/opt/sprint-isaac-bootstrap.py"),
) -> list[str]:
    command = list(job.get("command") or [])
    if command and command[0] == "python" and shutil.which("python") is None:
        if shutil.which("python3"):
            command = ["python3", *command[1:]]
    resume_arg = str(job.get("resume_arg") or "")
    if attempt > 1:
        retry_reason = str(job.get("retry_reason") or "")
        job_kind = infer_job_kind(job)
        retry_without_checkpoint = bool(
            not checkpoint
            and (
                # No child process started, so there is no state to recover.
                retry_reason == "spawn_failed"
                # Verification/evaluation commands are intentionally
                # stateless and safe to replay from their immutable input.
                or job_kind in {"evaluate", "verify"}
                # AppLauncher failed before agent-authored code began.
                or (
                    retry_reason == "app_launcher_initialization_failed"
                    and not job.get("last_progress")
                    and not job.get("last_checkpoint")
                )
            )
        )
        if not checkpoint and not retry_without_checkpoint:
            raise RuntimeError(
                "replacement attempt requires a valid resumable training-state "
                "checkpoint; inference policies are not resumable"
            )
        if checkpoint and resume_arg:
            command.extend([resume_arg, checkpoint])
    if command and Path(command[0]).name.startswith("python"):
        script_index = 1
        while script_index < len(command) and command[script_index] in {
            "-u",
            "-B",
            "-E",
            "-s",
        }:
            script_index += 1
        if script_index < len(command):
            script = Path(command[script_index])
            try:
                source = script.read_text(errors="ignore")[:1_000_000]
            except OSError:
                source = ""
            try:
                script_in_workspace = script.resolve().is_relative_to(
                    Path(str(job.get("workdir") or "/app")).resolve()
                )
            except (OSError, RuntimeError):
                script_in_workspace = False
            if (
                isaac_bootstrap.is_file()
                and script.suffix == ".py"
                and (
                    script_in_workspace or "isaaclab" in source or "isaacsim" in source
                )
            ):
                command.insert(script_index, str(isaac_bootstrap))
    return command


def retryable_app_launcher_failure(path: Path, *, exit_code: int) -> bool:
    """Detect a provider-side AppLauncher abort before initialization completes.

    Some Modal GPU hosts make Isaac Kit terminate ``AppLauncher.__init__`` with
    ``SystemExit(0)`` after Vulkan device discovery fails. Without the sidecar
    marker this looks like a successful agent job. Normal Python exceptions,
    non-zero exits, direct ``SimulationApp`` probes, and scripts that reached a
    completed AppLauncher are deliberately excluded.
    """
    if exit_code != 0 or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    state = str(payload.get("state") or "")
    if state == "starting":
        return True
    return state == "system_exit" and payload.get("exit_code") in {None, 0}


def gpu_activity_stalled(
    samples_path: Path,
    *,
    started_epoch_s: float,
    now_epoch_s: float | None = None,
    grace_seconds: float = 300.0,
    minimum_samples: int = 12,
    active_utilization_pct: float = 5.0,
) -> bool:
    """Return true when a live GPU job has made no sampled accelerator progress.

    Isaac startup and short smoke tests can legitimately sample at zero percent,
    so the guard needs both a five-minute grace period and a useful run of
    successful ``nvidia-smi`` samples.  A job that exits inside the grace period
    is never affected.  The worker records this as a deterministic failure;
    it is not an infrastructure preemption and must not silently restart.
    """
    now = time.time() if now_epoch_s is None else now_epoch_s
    if now - started_epoch_s < grace_seconds or not samples_path.is_file():
        return False
    rows: list[dict] = []
    try:
        for line in samples_path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row.get("epoch_s") or 0) >= started_epoch_s:
                rows.append(row)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    # A job may train successfully and then hang in a later framework call.
    # Judge the most recent grace-sized window rather than letting one old
    # burst of GPU work exempt the process forever.
    recent_cutoff = max(started_epoch_s, now - grace_seconds)
    rows = [
        row
        for row in rows
        if row.get("nvidia_smi_ok") is True
        and float(row.get("epoch_s") or 0) >= recent_cutoff
    ]
    if len(rows) < minimum_samples:
        return False
    utilization: list[float] = []
    memory: list[float] = []
    for row in rows:
        for gpu in row.get("gpus") or []:
            try:
                utilization.append(float(gpu.get("util_gpu_pct") or 0))
                memory.append(float(gpu.get("mem_used_mib") or 0))
            except (TypeError, ValueError):
                continue
    return bool(
        utilization
        and memory
        and max(memory) >= 256.0
        and max(utilization) <= active_utilization_pct
    )


def progress_cursor(progress: dict | None) -> float | None:
    """Return a monotonic trainer cursor without treating timestamps as work."""
    if not isinstance(progress, dict):
        return None
    for key in (
        "completed_iteration",
        "iteration",
        "global_step",
        "step",
        "sequence",
        "cursor",
    ):
        value = progress.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def infer_job_kind(job: dict) -> str:
    """Return the watchdog contract for one GPU command."""
    explicit = str(job.get("job_kind") or "auto").strip().lower()
    if explicit in {"train", "evaluate", "verify"}:
        return explicit
    command = [str(part) for part in (job.get("command") or [])]
    searchable = " ".join(command).lower()
    script_names = {Path(part).name.lower() for part in command if part.endswith(".py")}
    if any(name.startswith(("verify", "test")) for name in script_names):
        return "verify"
    # Agent-authored evaluation programs commonly put the verb at the end
    # (``targeted_gait_eval.py``) or describe a sweep/probe instead.  Treat
    # those as evaluation jobs so the training-only utilization/progress
    # watchdogs cannot kill a simulator-heavy rollout that is making useful
    # progress while sampling below 5% GPU utilization.
    evaluation_markers = (
        "eval",
        "evaluate",
        "sweep",
        "probe",
        "inspect",
        "benchmark",
    )
    if any(
        any(
            marker in Path(name).stem.lower().replace("-", "_").split("_")
            for marker in evaluation_markers
        )
        for name in script_names
    ):
        return "evaluate"
    if "event-verifier" in searchable or "test.sh" in searchable:
        return "verify"
    return "train"


def expected_final_cursor(job: dict) -> float | None:
    """Infer the zero-based terminal training cursor from common CLI flags."""
    command = [str(part) for part in (job.get("command") or [])]
    for flag in ("--max-iterations", "--iterations", "--max-steps"):
        try:
            raw = command[command.index(flag) + 1]
            count = int(raw)
        except (ValueError, IndexError):
            continue
        if count > 0:
            return float(count - 1)
    return None


def watchdog_phase(job: dict, progress: dict | None) -> str:
    kind = infer_job_kind(job)
    if kind != "train":
        return "verifying" if kind == "verify" else "evaluating"
    cursor = progress_cursor(progress)
    final_cursor = expected_final_cursor(job)
    if cursor is not None and final_cursor is not None and cursor >= final_cursor:
        return "finalizing"
    return "training"


class TrainingProgressWatchdog:
    """Detect a live training loop that republishes one cursor forever."""

    def __init__(self, *, grace_seconds: float = 300.0) -> None:
        self.grace_seconds = float(grace_seconds)
        self.last_cursor: float | None = None
        self.last_advanced_epoch_s: float | None = None

    def observe(self, progress: dict | None, *, now_epoch_s: float) -> bool:
        cursor = progress_cursor(progress)
        if cursor is None:
            return False
        if self.last_cursor is None or cursor > self.last_cursor:
            self.last_cursor = cursor
            self.last_advanced_epoch_s = float(now_epoch_s)
            return False
        assert self.last_advanced_epoch_s is not None
        return float(now_epoch_s) - self.last_advanced_epoch_s >= self.grace_seconds


def final_attempt_outcome(
    exit_code: int,
    *,
    interrupted: bool,
    activity_watchdog_fired: bool,
    progress_watchdog_fired: bool = False,
    retryable_infrastructure_failure: bool = False,
) -> tuple[int, str]:
    """Make watchdog termination a truthful deterministic failure.

    Frameworks may catch SIGTERM during cleanup and return zero. A worker
    stopped because it never used the accelerator must not therefore become a
    successful attempt or enter the infrastructure-preemption retry path.
    """
    if interrupted or retryable_infrastructure_failure:
        return exit_code, "interrupted"
    if activity_watchdog_fired or progress_watchdog_fired:
        return (exit_code if exit_code != 0 else 1), "failed"
    return exit_code, "succeeded" if exit_code == 0 else "failed"


def heartbeat_payload(
    *,
    run_id: str,
    job_id: str,
    attempt: int,
    lease_id: str,
    status: str,
    progress: object,
    checkpoint: str | None,
    lease_seconds: int,
    job_kind: str | None = None,
    phase: str | None = None,
) -> dict:
    now = time.time()
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "job_id": job_id,
        "attempt": attempt,
        "lease_id": lease_id,
        "status": status,
        "updated_at": utc_now(),
        "updated_at_epoch_s": now,
        "lease_expires_at_epoch_s": now + lease_seconds,
        "worker_hostname": os.uname().nodename,
        "worker_pid": os.getpid(),
        "progress": progress,
        "checkpoint": checkpoint,
    }
    if job_kind:
        payload["job_kind"] = job_kind
    if phase:
        payload["phase"] = phase
    return payload


def stop_child(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def job_cancel_requested(
    run_id: str, job_id: str, durable_dir: str = "/durable"
) -> bool:
    marker = (
        Path(durable_dir)
        / "runs"
        / run_id
        / "gpu-jobs"
        / "cancel"
        / f"{job_id}.json"
    )
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == 1
        and payload.get("run_id") == run_id
        and payload.get("job_id") == job_id
        and payload.get("reason") == "agent_cancelled"
    )


def refresh_budget_stop(
    run_id: str,
    durable_dir: str = "/durable",
    *,
    now: float | None = None,
    max_snapshot_age_seconds: float = MAX_BUDGET_SNAPSHOT_AGE_SECONDS,
    runtime_snapshot: Path = RUNTIME_BUDGET_SNAPSHOT,
) -> str | None:
    """Independently verify the trusted host budget snapshot.

    The agent can write the shared durable volume, including the stop marker
    and the CPU watchdog copy, so neither is a trusted kill signal. The host
    injects its canonical merged ledger into this sandbox's private ``/run``
    filesystem. GPU work fails closed when that snapshot is absent, malformed,
    inconsistent, or stale. The durable marker remains only an audit artifact.

    Return the proven stop reason, or ``None`` while the run is within budget.
    """
    run_root = Path(durable_dir) / "runs" / run_id
    marker = run_root / "BUDGET_STOP_REQUESTED.json"

    checked_at = time.time() if now is None else float(now)
    try:
        snapshot = json.loads(runtime_snapshot.read_text())
        if snapshot.get("schema_version") != 2:
            raise ValueError("budget watchdog schema mismatch")
        if snapshot.get("run_id") != run_id:
            raise ValueError("budget watchdog identity mismatch")
        snapshot_epoch = float(snapshot["checked_at_epoch_s"])
        age = checked_at - snapshot_epoch
        if not math.isfinite(age) or age < -max_snapshot_age_seconds:
            raise ValueError("budget watchdog timestamp is invalid")
        if age > max_snapshot_age_seconds:
            raise ValueError(f"budget watchdog snapshot is stale ({age:.1f}s)")
        total = float(snapshot["total_usd"])
        threshold = float(snapshot["stop_threshold_usd"])
        if not (
            math.isfinite(total)
            and total >= 0
            and math.isfinite(threshold)
            and threshold > 0
        ):
            raise ValueError("budget watchdog totals are invalid")
        status = snapshot.get("status")
        if status == "within_budget" and total < threshold:
            return None
        if status != "stop_requested" or total < threshold:
            raise ValueError("budget watchdog status is inconsistent")
        payload = {**snapshot, "reason": "agent_cost_budget_exhausted"}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        # A snapshot older than the propagation allowance proves that the
        # independent CPU circuit breaker is no longer healthy. Stop GPU spend
        # while the configured shutdown reserve still covers this worker.
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "fail_closed",
            "reason": "budget_telemetry_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at_epoch_s": checked_at,
        }
    write_status(marker, payload)
    return str(payload["reason"])


def supervise_child(
    proc: subprocess.Popen,
    *,
    heartbeat_interval: float,
    interruption_grace_sec: float,
    on_heartbeat,
) -> tuple[int, bool]:
    """Wait for a child and forward preemption signals with checkpoint grace.

    Returns ``(exit_code, interrupted)``.  The signal handler only sets an
    event; process-group signaling and durable writes happen in normal control
    flow so they can be tested and are not constrained by signal-handler rules.
    """
    forwarded = False
    deadline: float | None = None
    last_checkpoint_heartbeat = 0.0
    with Interruption() as interruption:
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                return int(exit_code), interruption.requested
            if interruption.requested:
                if not forwarded:
                    forwarded = True
                    deadline = time.monotonic() + max(0.0, interruption_grace_sec)
                    try:
                        os.killpg(proc.pid, interruption.signum or signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                now = time.monotonic()
                if last_checkpoint_heartbeat == 0.0 or (
                    now - last_checkpoint_heartbeat >= heartbeat_interval
                ):
                    on_heartbeat("checkpointing")
                    last_checkpoint_heartbeat = now
                remaining = max(0.0, (deadline or time.monotonic()) - time.monotonic())
                if remaining <= 0:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        return int(proc.wait(timeout=2)), True
                    except subprocess.TimeoutExpired:
                        return 128 + int(interruption.signum or signal.SIGTERM), True
                interruption.wait(min(0.25, remaining))
                continue
            interruption.wait(max(0.05, heartbeat_interval))
            on_heartbeat("running")


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "Usage: sprint-gpu-worker-run.py "
            "RUN_ID JOB_ID ATTEMPT LEASE_ID FENCE_EPOCH",
            file=sys.stderr,
        )
        return 2
    run_id, job_id, raw_attempt, lease_id, raw_fence_epoch = sys.argv[1:]
    try:
        attempt = int(raw_attempt)
        fence_epoch = int(raw_fence_epoch)
    except ValueError:
        return 2
    prefix = Path("/durable") / "runs" / run_id / "gpu-jobs"
    out = prefix / "out" / job_id / f"attempt-{attempt}"
    status_path = prefix / "status" / f"{job_id}.json"
    attempt_path = prefix / "attempts" / job_id / f"{attempt}.json"
    heartbeat_path = prefix / "heartbeats" / job_id / f"{attempt}.json"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "worker.log"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> int:
            for stream in self.streams:
                stream.write(data)
                stream.flush()
            return len(data)

        def flush(self) -> None:
            for stream in self.streams:
                stream.flush()

    log_handle = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]
    if not wait_for_lease(status_path, attempt, lease_id, fence_epoch):
        print(f"lease rejected job={job_id} attempt={attempt}", flush=True)
        return 75

    job = read_json(status_path)
    checkpoint_dir = Path(
        str(job.get("checkpoint_dir") or prefix / "checkpoints" / job_id)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_file = Path(
        str(job.get("progress_file") or checkpoint_dir / "progress.json")
    )
    progress, checkpoint = progress_snapshot(progress_file, checkpoint_dir)
    lease_seconds = int(job.get("heartbeat_timeout_sec") or 45)
    heartbeat_interval = max(2, int(job.get("heartbeat_interval_sec") or 5))
    interruption_grace_sec = max(0, float(job.get("interruption_grace_sec") or 20))
    started_epoch = time.time()
    attempt_record = {
        "schema_version": 2,
        "run_id": run_id,
        "job_id": job_id,
        "attempt": attempt,
        "lease_id": lease_id,
        "status": "running",
        "started_at": utc_now(),
        "started_at_epoch_s": started_epoch,
        "worker_hostname": os.uname().nodename,
        "progress": progress,
        "checkpoint": checkpoint,
        "work_archive_sha256": job.get("work_archive_sha256"),
    }
    write_status(attempt_path, attempt_record)
    write_status(
        heartbeat_path,
        heartbeat_payload(
            run_id=run_id,
            job_id=job_id,
            attempt=attempt,
            lease_id=lease_id,
            status="running",
            progress=progress,
            checkpoint=checkpoint,
            lease_seconds=lease_seconds,
        ),
    )

    if termination_reason := refresh_budget_stop(run_id):
        attempt_record.update(
            {
                "status": "terminated",
                "termination_reason": termination_reason,
                "exit_code": 0,
                "finished_at": utc_now(),
                "finished_at_epoch_s": time.time(),
            }
        )
        write_status(attempt_path, attempt_record)
        print("budget stop already requested; skipping GPU work", flush=True)
        return 0
    if job_cancel_requested(run_id, job_id):
        attempt_record.update(
            {
                "status": "terminated",
                "termination_reason": "agent_cancelled",
                "exit_code": 0,
                "finished_at": utc_now(),
                "finished_at_epoch_s": time.time(),
            }
        )
        write_status(attempt_path, attempt_record)
        print("agent cancellation already requested; skipping GPU work", flush=True)
        return 0

    print(
        f"gpu-worker start job={job_id} attempt={attempt} "
        f"host={os.uname().nodename} ts={utc_now()}"
    )
    emit(run_id, job_id, attempt, lease_id, "gpu_queue_wait", "exit", where="worker")
    emit(
        run_id,
        job_id,
        attempt,
        lease_id,
        "gpu_worker_starting",
        "exit",
        where="worker",
    )

    smi = shutil.which("nvidia-smi")
    if not smi:
        attempt_record.update(
            {
                "status": "failed",
                "error": "nvidia-smi missing",
                "exit_code": 2,
                "finished_at": utc_now(),
                "finished_at_epoch_s": time.time(),
            }
        )
        write_status(attempt_path, attempt_record)
        return 2
    subprocess.run([smi], check=False)
    start_telemetry(run_id, job_id, attempt, lease_id)

    archive = prefix / "work" / job_id / "app.tar.gz"
    expected_archive_sha256 = str(job.get("work_archive_sha256") or "")
    try:
        if not archive.is_file() or not expected_archive_sha256:
            raise RuntimeError("missing host-pinned GPU work archive")
        actual_archive_sha256 = file_sha256(archive)
        if actual_archive_sha256 != expected_archive_sha256:
            raise RuntimeError("GPU work archive digest mismatch")
        with tempfile.TemporaryDirectory(prefix=f"sprint-{job_id}-") as raw:
            extract_root = Path(raw)
            safe_extract_work_archive(archive, extract_root)
            app_src = extract_root / "app"
            if not app_src.is_dir():
                raise RuntimeError("GPU work archive is missing app/")
            Path("/app").mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", f"{app_src}/.", "/app/"], check=True)
    except Exception as exc:  # noqa: BLE001
        finished = time.time()
        error = f"work archive rejected: {type(exc).__name__}: {exc}"
        attempt_record.update(
            {
                "status": "failed",
                "error": error,
                "exit_code": 2,
                "finished_at": utc_now(),
                "finished_at_epoch_s": finished,
            }
        )
        write_status(attempt_path, attempt_record)
        write_status(
            heartbeat_path,
            heartbeat_payload(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                lease_id=lease_id,
                status="failed",
                progress=progress,
                checkpoint=checkpoint,
                lease_seconds=lease_seconds,
            ),
        )
        print(error, flush=True)
        return 2

    try:
        command = build_attempt_command(job, attempt, checkpoint)
    except RuntimeError as exc:
        finished = time.time()
        attempt_record.update(
            {
                "status": "failed",
                "error": str(exc),
                "exit_code": 2,
                "finished_at": utc_now(),
                "finished_at_epoch_s": finished,
            }
        )
        write_status(attempt_path, attempt_record)
        write_status(
            heartbeat_path,
            heartbeat_payload(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                lease_id=lease_id,
                status="failed",
                progress=progress,
                checkpoint=checkpoint,
                lease_seconds=lease_seconds,
            ),
        )
        print(f"resume rejected: {exc}", flush=True)
        return 2
    workdir = str(job.get("workdir") or "/app")
    if not command:
        attempt_record.update(
            {
                "status": "failed",
                "error": "missing command",
                "exit_code": 2,
                "finished_at": utc_now(),
                "finished_at_epoch_s": time.time(),
            }
        )
        write_status(attempt_path, attempt_record)
        return 2

    env = os.environ.copy()
    app_launcher_state_file = Path(tempfile.gettempdir()) / (
        f"sprint-app-launcher-{job_id}-{attempt}-{lease_id}.json"
    )
    app_launcher_state_file.unlink(missing_ok=True)
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in ("/opt", "/opt/event-verifier", "/app", env.get("PYTHONPATH", ""))
        if part
    )
    env.update(
        {
            "SPRINT_RUN_ID": run_id,
            "SPRINT_GPU_JOB_ID": job_id,
            "SPRINT_GPU_ATTEMPT": str(attempt),
            "SPRINT_GPU_LEASE_ID": lease_id,
            "SPRINT_GPU_FENCE_EPOCH": str(int(job.get("fence_epoch") or 0)),
            "SPRINT_GPU_STATUS_FILE": str(status_path),
            "SPRINT_GPU_CHECKPOINT_DIR": str(checkpoint_dir),
            "SPRINT_GPU_PROGRESS_FILE": str(progress_file),
            "SPRINT_GPU_RESUME": "1" if checkpoint else "0",
            "SPRINT_GPU_RESUME_CHECKPOINT": checkpoint or "",
            "SPRINT_APP_LAUNCHER_STATE_FILE": str(app_launcher_state_file),
            "SPRINT_GPU_JOB_KIND": infer_job_kind(job),
        }
    )
    env.update(checkpoint_resume_metadata(checkpoint))
    emit(run_id, job_id, attempt, lease_id, "isaac_starting", "enter")
    time.sleep(2)
    emit(run_id, job_id, attempt, lease_id, "isaac_starting", "exit")
    emit(run_id, job_id, attempt, lease_id, "gpu_active", "enter")
    print("exec", command, "cwd", workdir, flush=True)
    interrupted = False
    activity_watchdog_fired = False
    progress_watchdog_fired = False
    completion_declared = False
    budget_watchdog_fired = False
    budget_termination_reason: str | None = None
    agent_cancel_fired = False
    progress_watchdog = TrainingProgressWatchdog()
    last_watchdog_phase: str | None = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"exec failed: {type(exc).__name__}: {exc}", flush=True)
        emit(
            run_id,
            job_id,
            attempt,
            lease_id,
            "gpu_active",
            "exit",
            error=str(exc),
        )
        exit_code = 127
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
        activity_samples = Path("/logs/artifacts/telemetry/samples.jsonl")

        def update_heartbeat(status: str) -> None:
            nonlocal progress, checkpoint, error
            nonlocal activity_watchdog_fired, progress_watchdog_fired
            nonlocal budget_watchdog_fired, budget_termination_reason
            nonlocal agent_cancel_fired
            nonlocal completion_declared
            nonlocal last_watchdog_phase
            progress, checkpoint = progress_snapshot(
                progress_file,
                checkpoint_dir,
                # Committed generations are immutable and atomically renamed.
                # Avoid re-hashing multi-GB model files on every heartbeat;
                # startup/final selection performs full validation.
                verify_checkpoint=False,
            )
            phase = watchdog_phase(
                job, progress if isinstance(progress, dict) else None
            )
            if phase != last_watchdog_phase:
                print(f"GPU job watchdog phase: {phase}", flush=True)
                last_watchdog_phase = phase
            write_status(
                heartbeat_path,
                heartbeat_payload(
                    run_id=run_id,
                    job_id=job_id,
                    attempt=attempt,
                    lease_id=lease_id,
                    status=status,
                    progress=progress,
                    checkpoint=checkpoint,
                    lease_seconds=lease_seconds,
                    job_kind=infer_job_kind(job),
                    phase=phase,
                ),
            )
            if not lease_owned(status_path, attempt, lease_id, fence_epoch):
                print("lease fenced; stopping child", flush=True)
                stop_child(proc)
                attempt_record.update(
                    {
                        "status": "fenced",
                        "finished_at": utc_now(),
                        "finished_at_epoch_s": time.time(),
                        "progress": progress,
                        "checkpoint": checkpoint,
                    }
                )
                write_status(attempt_path, attempt_record)
                raise SystemExit(75)
            if not budget_watchdog_fired:
                refreshed_stop_reason = refresh_budget_stop(run_id)
                if refreshed_stop_reason:
                    budget_watchdog_fired = True
                    budget_termination_reason = refreshed_stop_reason
                    error = budget_termination_reason.replace("_", " ")
                    print(f"{error}; stopping GPU child", flush=True)
                    stop_child(proc)
                    return
            if not agent_cancel_fired and job_cancel_requested(run_id, job_id):
                agent_cancel_fired = True
                error = "agent cancelled GPU job"
                print(f"{error}; stopping GPU child", flush=True)
                stop_child(proc)
                return
            if not completion_declared and progress_declares_completion(progress):
                completion_declared = True
                print(
                    "GPU job declared completion; ending simulator teardown",
                    flush=True,
                )
                stop_child(proc)
                return
            if phase != "training":
                return
            if not activity_watchdog_fired and gpu_activity_stalled(
                activity_samples,
                started_epoch_s=started_epoch,
            ):
                activity_watchdog_fired = True
                error = (
                    "GPU activity watchdog: no sampled accelerator utilization "
                    "above 5% during the five-minute startup/progress window"
                )
                print(error, flush=True)
                stop_child(proc)
            if not progress_watchdog_fired and progress_watchdog.observe(
                progress,
                now_epoch_s=time.time(),
            ):
                progress_watchdog_fired = True
                error = (
                    "Training progress watchdog: checkpoint/progress cursor "
                    "did not advance during the five-minute window"
                )
                print(error, flush=True)
                stop_child(proc)

        exit_code, interrupted = supervise_child(
            proc,
            heartbeat_interval=heartbeat_interval,
            interruption_grace_sec=interruption_grace_sec,
            on_heartbeat=update_heartbeat,
        )
        emit(
            run_id,
            job_id,
            attempt,
            lease_id,
            "gpu_active",
            "exit",
            exit_code=exit_code,
            interrupted=interrupted,
        )

    app_launcher_failure = retryable_app_launcher_failure(
        app_launcher_state_file,
        exit_code=exit_code,
    )
    if app_launcher_failure:
        error = (
            "Isaac AppLauncher initialization failed before the agent script "
            "started; reporting the attempt as preempted"
        )
        print(error, flush=True)
    exit_code, final_status = final_attempt_outcome(
        exit_code,
        interrupted=interrupted,
        activity_watchdog_fired=activity_watchdog_fired,
        progress_watchdog_fired=progress_watchdog_fired,
        retryable_infrastructure_failure=app_launcher_failure,
    )
    if (
        completion_declared
        and not interrupted
        and not activity_watchdog_fired
        and not progress_watchdog_fired
        and not budget_watchdog_fired
        and not agent_cancel_fired
    ):
        # The child may have required SIGKILL because Isaac/Kit teardown was
        # stuck. The durable completion marker plus the required artifact
        # checks below are the success boundary, not the destructor exit code.
        exit_code = 0
        final_status = "succeeded"
        error = None
        attempt_record["completion_source"] = "progress_file"
    if budget_watchdog_fired:
        final_status = "terminated"
        attempt_record["termination_reason"] = (
            budget_termination_reason or "budget_telemetry_unavailable"
        )
    if agent_cancel_fired:
        final_status = "terminated"
        exit_code = 0
        attempt_record["termination_reason"] = "agent_cancelled"
    progress, checkpoint = progress_snapshot(progress_file, checkpoint_dir)
    output_artifacts, missing_outputs = collect_output_artifacts(
        job,
        run_id=run_id,
        job_id=job_id,
        attempt=attempt,
    )
    if output_artifacts:
        progress_payload = dict(progress) if isinstance(progress, dict) else {}
        progress_payload["output_artifacts"] = output_artifacts
        policies = [
            item
            for item in output_artifacts
            if Path(str(item.get("path") or "")).suffix in {".pt", ".pth"}
        ]
        if policies:
            progress_payload["policy_path"] = policies[0]["path"]
        progress = progress_payload
    if missing_outputs and final_status == "succeeded":
        final_status = "failed"
        exit_code = 2
        error = "required GPU output missing or invalid: " + ", ".join(missing_outputs)
        print(error, flush=True)
    if final_status == "succeeded" and job.get("submission_paths"):
        submission_results = submit_declared_policies(job)
        progress_payload = dict(progress) if isinstance(progress, dict) else {}
        progress_payload["submission_results"] = submission_results
        progress = progress_payload
        for result in submission_results:
            print(
                f"official submission {result['state']}: {result['path']}",
                flush=True,
            )
    finished = time.time()
    attempt_record.update(
        {
            "finished_at": utc_now(),
            "finished_at_epoch_s": finished,
            "exit_code": exit_code,
            "status": final_status,
            "progress": progress,
            "checkpoint": checkpoint,
        }
    )
    if error:
        attempt_record["error"] = error
    if app_launcher_failure:
        attempt_record["retry_reason"] = "app_launcher_initialization_failed"
    write_status(attempt_path, attempt_record)
    write_status(
        heartbeat_path,
        heartbeat_payload(
            run_id=run_id,
            job_id=job_id,
            attempt=attempt,
            lease_id=lease_id,
            status=attempt_record["status"],
            progress=progress,
            checkpoint=checkpoint,
            lease_seconds=lease_seconds,
            job_kind=infer_job_kind(job),
            phase="complete" if final_status == "succeeded" else final_status,
        ),
    )
    try:
        timeline.write_summary(
            run_id,
            durable_dir="/durable",
            also_local=Path("/logs/artifacts/telemetry"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"timeline summary failed: {type(exc).__name__}: {exc}", flush=True)
    # A graceful interruption is retryable infrastructure state, independent
    # of the child's framework-specific signal exit code.
    return 75 if attempt_record["status"] == "interrupted" else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
