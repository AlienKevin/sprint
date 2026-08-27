#!/usr/bin/env python3
"""Host-side GPU worker dispatch for CPU-agent durable lane runs.

Claims jobs published by in-sandbox ``event gpu`` through the run's exact-name
dispatch index and starts a preemptible A10G Modal Sandbox that mounts the same
volume. The Codex/agent sandbox stays on CPU (gpus=0) so GPU preemption cannot
kill the harness.
"""

from __future__ import annotations

import base64
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = ROOT / "runs" / "ops"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPS_DIR))
from event_runtime.compute import claim as gpu_claim  # noqa: E402
from event_runtime.control import run as sprintctl  # noqa: E402
from event_runtime.container.sprint_resilience import (  # noqa: E402
    Lease,
    ProbeResult,
    ProbeState,
    ProviderHandle,
)

WORKER_TAG_ROLE = "gpu-worker"
MAX_ACTIVE_TRAINING_JOBS_PER_RUN = 1
STOP_DISPATCH_LOCK_TIMEOUT_SEC = 10 * 60
STOP_TERMINATE_MAX_WORKERS = 8
STOP_PROVIDER_CONFIRM_TIMEOUT_SEC = 30.0
STOP_PROVIDER_CONFIRM_POLL_SEC = 1.0
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
AGENT_GPU_MIRROR_ROOT = "/run/sprint-gpu-mirror"
AGENT_GPU_CONTROL_ROOT = "/run/sprint-gpu-control"
AGENT_GPU_CONTROL_MAX_REQUESTS = 64
GPU_AGENT_CANCEL_GRACE_SEC = 60
GPU_AGENT_CANCEL_RUNTIME_MARKER = "/run/sprint-agent-cancel.json"
AGENT_GPU_MIRROR_LOG_BYTES = 768 * 1024
AGENT_GPU_MIRROR_ARTIFACT_BYTES = 128 * 1024 * 1024
AGENT_GPU_MIRROR_ARG_BYTES = 64 * 1024
GPU_SUBMISSION_BRIDGE_ROOT = "/run/sprint-submission-bridge"
GPU_SUBMISSION_BRIDGE_MAX_REQUESTS = 8
GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES = 32 * 1024 * 1024
GPU_SUBMISSION_BRIDGE_MAX_BATCH_BYTES = 64 * 1024 * 1024
GPU_SUBMISSION_BRIDGE_MAX_DRAIN_BATCHES = 65
GPU_SUBMISSION_ID_RE = re.compile(r"^[0-9]{6}-[0-9a-f]{4}$")
GPU_SUBMISSION_DRAIN_COMPLETE = "/run/sprint-gpu-drain-complete.json"
HOST_CONTINUOUS_INCOMING = Path("artifacts/continuous/incoming")
LIVE_PROVIDER_LOG_INTERVAL_SEC = 30
LIVE_PROVIDER_LOG_TAIL_LINES = 2000
AGENT_GPU_CLI_PATH = "/opt/event_runtime/agent/gpu.py"
AGENT_COST_CLI_PATH = "/opt/event_runtime/agent/cost.py"
AGENT_COMMAND_SOURCE = ROOT / "event_runtime" / "agent"
MAX_WORK_ARCHIVE_BYTES = 256 * 1024 * 1024
WORK_ARCHIVE_TRANSFER_ATTEMPTS = 5
GPU_BUDGET_MIRROR_PATH = "/run/sprint-budget-watchdog.json"
MAX_GPU_BUDGET_MIRROR_BYTES = 1024 * 1024
# Modal's Sandbox timeout starts while the image/container is still starting.
# Keep that provider boundary separate from the agent-requested command runtime,
# which is enforced by the worker after startup.
GPU_SANDBOX_STARTUP_FINALIZATION_ALLOWANCE_SEC = 10 * 60


def submission_bridge_state_dir(run: dict[str, Any]) -> Path:
    path = Path(str(run["state_dir"])) / "submission-bridge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cpu_agent_sandbox(run: dict[str, Any]):
    """Resolve Harbor's live CPU sandbox from its task-container identity."""
    container_id = str(run.get("agent_container_id") or "")
    if not container_id.startswith("ta-"):
        raise RuntimeError("CPU agent container is unavailable")
    identity = sprintctl.exec_container(
        run,
        container_id,
        'printf "%s" "$MODAL_SANDBOX_ID"',
        check=False,
        timeout=20,
    )
    sandbox_id = (identity.stdout or "").strip()
    if identity.returncode != 0 or not sandbox_id.startswith("sb-"):
        raise RuntimeError(
            (identity.stderr or identity.stdout or "missing CPU sandbox ID")[-1000:]
        )
    return modal.Sandbox.from_id(sandbox_id)


def signal_gpu_submission_drain_complete(run: dict[str, Any]) -> None:
    """Release the CPU wrapper only after final GPU submissions are forwarded."""
    payload = base64.b64encode(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": str(run["run_id"]),
                "completed_at": utc_now(),
            },
            sort_keys=True,
        ).encode()
    ).decode("ascii")
    script = r"""
import base64, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_bytes(base64.b64decode(sys.argv[2]))
os.chmod(temporary, 0o600)
os.replace(temporary, path)
""".strip()
    process = _cpu_agent_sandbox(run).exec(
        "python3", "-c", script, GPU_SUBMISSION_DRAIN_COMPLETE, payload, timeout=30
    )
    if process.wait() != 0:
        detail = process.stderr.read() or process.stdout.read()
        raise RuntimeError(f"CPU GPU-drain handshake failed: {str(detail)[-1000:]}")


def append_control_event(
    run: dict[str, Any], event: str, *, request: dict[str, Any], **detail: Any
) -> dict[str, Any]:
    """Durably append one host-authoritative control-plane transition.

    The sandbox-local request is only a delivery mechanism. This append-only
    host record is fsynced before an action is issued, so a controller restart
    can replay any request that does not have a later acknowledgement.
    """
    state_dir = Path(str(run["state_dir"]))
    path = state_dir / "control-events.jsonl"
    lock_path = state_dir / "control-events.lock"
    with sprintctl.file_lock(lock_path):
        state_path = state_dir / "control-state.json"
        try:
            sequence = int(json.loads(state_path.read_text())["last_sequence"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            sequence = 0
        if path.is_file():
            try:
                # Verify the snapshot against the final fsynced event without
                # rescanning a multi-hour log. This closes the crash window
                # between appending the event and replacing control-state.
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 8192))
                    lines = handle.read().splitlines()
                if lines:
                    sequence = max(
                        sequence,
                        int(json.loads(lines[-1])["sequence"]),
                    )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                if sequence == 0:
                    with path.open("rb") as handle:
                        for sequence, _line in enumerate(handle, start=1):
                            pass
        row = {
            "schema_version": 1,
            "sequence": sequence + 1,
            "event": event,
            "recorded_at": utc_now(),
            "recorded_at_epoch_s": time.time(),
            "run_id": str(run["run_id"]),
            "request_id": str(request.get("request_id") or ""),
            "job_id": str(request.get("job_id") or ""),
            **detail,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        sprintctl.atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "last_sequence": row["sequence"],
                "last_event": event,
                "updated_at": row["recorded_at"],
            },
            mode=0o600,
        )
    return row


def record_heartbeat_observation(
    run: dict[str, Any], job: dict[str, Any], heartbeat: dict[str, Any] | None
) -> None:
    """Persist each newly observed worker heartbeat in the control log."""
    if not heartbeat:
        return
    epoch = float(gpu_claim.heartbeat_epoch(heartbeat) or 0)
    if epoch <= 0:
        return
    state_dir_raw = str(run.get("state_dir") or "")
    if not state_dir_raw:
        return
    state_dir = Path(state_dir_raw)
    marker = state_dir / "control-heartbeats" / f"{job['job_id']}.json"
    previous: dict[str, Any] = {}
    try:
        previous = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    if float(previous.get("heartbeat_epoch_s") or 0) >= epoch:
        return
    append_control_event(
        run,
        "gpu_heartbeat_observed",
        request={"job_id": str(job["job_id"])},
        attempt=int(job.get("attempt") or 0),
        lease_id=str(job.get("lease_id") or ""),
        heartbeat_epoch_s=epoch,
        progress=heartbeat.get("progress"),
        checkpoint=heartbeat.get("checkpoint"),
    )
    sprintctl.atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "job_id": str(job["job_id"]),
            "heartbeat_epoch_s": epoch,
            "observed_at": utc_now(),
        },
        mode=0o600,
    )


def read_live_agent_cancel_requests(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Read cancellation requests through the CPU sandbox control channel."""
    script = r"""
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1]) / "cancel"
limit = int(sys.argv[2])
job_re = re.compile(r"^[A-Za-z0-9_-]+$")
rows = []
for path in sorted(root.glob("*.json")):
    if len(rows) >= limit:
        break
    if not job_re.fullmatch(path.stem):
        continue
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(row, dict):
        rows.append(row)
print(json.dumps(rows, separators=(",", ":")))
""".strip()
    process = _cpu_agent_sandbox(run).exec(
        "python3",
        "-c",
        script,
        AGENT_GPU_CONTROL_ROOT,
        str(AGENT_GPU_CONTROL_MAX_REQUESTS),
        timeout=30,
    )
    return_code = process.wait()
    raw = process.stdout.read()
    stderr = process.stderr.read()
    if return_code != 0:
        detail = stderr or raw or "CPU cancellation outbox read failed"
        raise RuntimeError(str(detail)[-1000:])
    payload = json.loads(raw or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("CPU cancellation outbox returned a non-list payload")
    requests: list[dict[str, Any]] = []
    for request in payload:
        if not isinstance(request, dict):
            continue
        job_id = str(request.get("job_id") or "")
        request_id = str(request.get("request_id") or "")
        if (
            str(request.get("run_id") or "") != str(run["run_id"])
            or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id)
            or not re.fullmatch(r"[0-9a-f]{32}", request_id)
            or request.get("reason") != "agent_cancelled"
        ):
            continue
        requests.append(request)
    return requests


def read_durable_agent_cancel_requests(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Read cancellation requests from the exact agent-published index."""
    if not str(run.get("volume_name") or "").strip():
        return []
    requests: list[dict[str, Any]] = []
    if indexed is None:
        indexed = indexed_agent_jobs(run)
    embedded_requests = [
        detail.get("cancel_request")
        for _job_id, detail in sorted((indexed or {}).items())
        if detail.get("cancel_state") == "requested"
    ]
    for request in embedded_requests[:AGENT_GPU_CONTROL_MAX_REQUESTS]:
        if not isinstance(request, dict):
            continue
        job_id = str(request.get("job_id") or "")
        request_id = str(request.get("request_id") or "")
        if (
            str(request.get("run_id") or "") == str(run["run_id"])
            and re.fullmatch(r"[A-Za-z0-9_-]+", job_id)
            and re.fullmatch(r"[0-9a-f]{32}", request_id)
            and request.get("reason") == "agent_cancelled"
        ):
            requests.append(dict(request))
    return requests


def deliver_agent_cancel_to_gpu_worker(
    job: dict[str, Any], request: dict[str, Any]
) -> None:
    """Atomically signal a running GPU worker without killing its sandbox.

    The worker needs a brief cooperative shutdown window to stop the child,
    collect already-written declared outputs, and commit its terminal attempt
    record.  Provider termination at request receipt races that commit and can
    discard valid artifacts produced before an Isaac teardown hang.
    """
    sandbox_id = str(job.get("sandbox_id") or "")
    if not sandbox_id.startswith("sb-"):
        raise RuntimeError("GPU sandbox is unavailable for cancellation")
    payload = {
        "schema_version": 1,
        "request_id": str(request["request_id"]),
        "run_id": str(request["run_id"]),
        "job_id": str(request["job_id"]),
        "reason": "agent_cancelled",
        "requested_at": str(request.get("requested_at") or utc_now()),
        "requested_at_epoch_s": float(
            request.get("requested_at_epoch_s") or time.time()
        ),
    }
    script = r"""
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
""".strip()
    process = modal.Sandbox.from_id(sandbox_id).exec(
        "python3",
        "-c",
        script,
        GPU_AGENT_CANCEL_RUNTIME_MARKER,
        json.dumps(payload, separators=(",", ":")),
        timeout=15,
    )
    return_code = process.wait()
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    if return_code != 0:
        detail = stderr or stdout or "GPU cancellation signal failed"
        raise RuntimeError(str(detail)[-1000:])


def read_worker_submission_outbox(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    exclude_submission_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read bounded immutable archive requests from a live GPU sandbox.

    This deliberately uses the Modal sandbox control channel, not the shared
    Volume: concurrent Volume mounts are snapshots and cannot provide a live
    producer/consumer queue.
    """
    sandbox_id = str(job.get("sandbox_id") or "")
    if not sandbox_id.startswith("sb-"):
        return []
    script = r"""
import base64, gzip, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
limit = int(sys.argv[2])
max_policy = int(sys.argv[3])
max_batch = int(sys.argv[4])
excluded = set(json.loads(sys.argv[5]))
rows = []
total = 0
for receipt_path in sorted((root / "receipts").glob("*.json")):
    if len(rows) >= limit:
        break
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    submission_id = receipt_path.stem
    if submission_id in excluded:
        continue
    policy_path = root / "outbox" / f"{submission_id}.pt"
    try:
        size = policy_path.stat().st_size
    except OSError:
        continue
    if size <= 0 or size > max_policy or total + size > max_batch:
        continue
    data = policy_path.read_bytes()
    total += len(data)
    rows.append({
        "receipt": receipt,
        "policy_sha256": hashlib.sha256(data).hexdigest(),
        "policy_size_bytes": len(data),
        "policy_base64": base64.b64encode(data).decode("ascii"),
    })
sys.stdout.buffer.write(gzip.compress(json.dumps(rows, separators=(",", ":")).encode()))
""".strip()
    sandbox = modal.Sandbox.from_id(sandbox_id)
    process = sandbox.exec(
        "python3",
        "-c",
        script,
        GPU_SUBMISSION_BRIDGE_ROOT,
        str(GPU_SUBMISSION_BRIDGE_MAX_REQUESTS),
        str(GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES),
        str(GPU_SUBMISSION_BRIDGE_MAX_BATCH_BYTES),
        json.dumps(sorted(exclude_submission_ids or set())),
        text=False,
        timeout=45,
    )
    return_code = process.wait()
    raw = process.stdout.read()
    stderr = process.stderr.read()
    if return_code != 0:
        detail = stderr or raw or b"GPU submission outbox read failed"
        raise RuntimeError(detail.decode(errors="replace")[-1000:])
    if not raw:
        return []
    payload = json.loads(gzip.decompress(raw))
    if not isinstance(payload, list):
        raise RuntimeError("GPU submission outbox returned a non-list payload")
    return payload


def read_durable_submission_outbox(
    run: dict[str, Any], job: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recover GPU archive requests after the producing sandbox has exited."""
    prefix = f"runs/{run['run_id']}/submission-bridge"
    index_text = sprintctl.volume_get_text(
        run, f"{prefix}/indexes/{job['job_id']}.json", timeout_seconds=15
    )
    if not index_text:
        return []
    try:
        index = json.loads(index_text)
    except json.JSONDecodeError:
        return []
    if (
        not isinstance(index, dict)
        or index.get("schema_version") != 1
        or index.get("run_id") != str(run["run_id"])
        or index.get("gpu_job_id") != str(job["job_id"])
        or int(index.get("gpu_attempt") or 0) != int(job.get("attempt") or 0)
        or index.get("gpu_lease_id") != str(job.get("lease_id") or "")
        or not isinstance(index.get("submissions"), dict)
    ):
        raise RuntimeError("durable submission index identity/schema mismatch")
    records: list[dict[str, Any]] = []
    total_bytes = 0
    state_root = submission_bridge_state_dir(run)
    for submission_id, receipt in sorted(index["submissions"].items()):
        if not GPU_SUBMISSION_ID_RE.fullmatch(submission_id):
            continue
        if (
            not isinstance(receipt, dict)
            or receipt.get("submission_id") != submission_id
        ):
            continue
        try:
            existing = json.loads((state_root / f"{submission_id}.json").read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("state") == "forwarded":
            continue
        try:
            declared_size = int(receipt.get("policy_size_bytes") or 0)
        except (TypeError, ValueError):
            continue
        if declared_size <= 0 or declared_size > GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES:
            continue
        if total_bytes + declared_size > GPU_SUBMISSION_BRIDGE_MAX_BATCH_BYTES:
            break
        remote_policy = f"{prefix}/outbox/{submission_id}.pt"
        content = sprintctl.volume_get_bytes(
            run,
            remote_policy,
            timeout_seconds=120,
            max_bytes=GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES,
        )
        if not content:
            continue
        total_bytes += len(content)
        records.append(
            {
                "receipt": receipt,
                "policy_sha256": hashlib.sha256(content).hexdigest(),
                "policy_size_bytes": len(content),
                "policy_base64": base64.b64encode(content).decode("ascii"),
            }
        )
        if len(records) >= GPU_SUBMISSION_BRIDGE_MAX_REQUESTS:
            break
    return records


def validate_worker_submission_request(
    run: dict[str, Any], job: dict[str, Any], raw: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    receipt = raw.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("submission bridge receipt is missing")
    submission_id = str(receipt.get("submission_id") or "")
    if not GPU_SUBMISSION_ID_RE.fullmatch(submission_id):
        raise ValueError("submission bridge request id is invalid")
    expected = {
        "bridge": "host_owned_gpu_submission_v1",
        "run_id": str(run["run_id"]),
        "gpu_job_id": str(job["job_id"]),
        "gpu_attempt": int(job.get("attempt") or 0),
        "gpu_lease_id": str(job.get("lease_id") or ""),
    }
    for key, value in expected.items():
        observed = receipt.get(key)
        if key == "gpu_attempt":
            observed = int(observed or 0)
        else:
            observed = str(observed or "")
        if observed != value:
            raise ValueError(f"submission bridge {key} identity mismatch")
    try:
        content = base64.b64decode(str(raw["policy_base64"]), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("submission bridge policy encoding is invalid") from exc
    if not (0 < len(content) <= GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES):
        raise ValueError("submission bridge policy size is invalid")
    digest = hashlib.sha256(content).hexdigest()
    if (
        digest != str(raw.get("policy_sha256") or "")
        or digest != str(receipt.get("policy_sha256") or "")
        or len(content) != int(raw.get("policy_size_bytes") or 0)
        or len(content) != int(receipt.get("policy_size_bytes") or 0)
    ):
        raise ValueError("submission bridge policy integrity mismatch")
    return receipt, content


def stage_worker_policy_on_host(
    run: dict[str, Any], receipt: dict[str, Any], content: bytes
) -> Path:
    """Atomically place validated policy bytes in Harbor's trusted host queue."""
    submission_id = str(receipt.get("submission_id") or "")
    if not GPU_SUBMISSION_ID_RE.fullmatch(submission_id):
        raise ValueError("submission bridge request id is invalid")
    digest = hashlib.sha256(content).hexdigest()
    if digest != str(receipt.get("policy_sha256") or ""):
        raise ValueError("submission bridge host staging checksum mismatch")

    raw_trial = run.get("trial_path")
    trial = Path(str(raw_trial)).resolve() if raw_trial else None
    if trial is None or not trial.is_dir():
        _job, trial = sprintctl.discover_job_and_trial(
            Path(str(run["state_dir"])), run
        )
    if trial is None:
        raise RuntimeError("Harbor trial directory is unavailable for submission staging")

    incoming = trial / HOST_CONTINUOUS_INCOMING
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{submission_id}.pt"
    if target.exists():
        if target.stat().st_size != len(content) or hashlib.sha256(
            target.read_bytes()
        ).hexdigest() != digest:
            raise RuntimeError("submission bridge request id already has different bytes")
        return target

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{submission_id}.", suffix=".tmp", dir=incoming
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.stat().st_size != len(content) or hashlib.sha256(
                target.read_bytes()
            ).hexdigest() != digest:
                raise RuntimeError(
                    "submission bridge request id concurrently staged different bytes"
                )
        directory_fd = os.open(incoming, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _mirror_worker_policy_to_cpu_agent(
    run: dict[str, Any], receipt: dict[str, Any], content: bytes
) -> dict[str, Any]:
    """Best-effort compatibility mirror through the CPU agent's archive CLI."""
    envelope = gzip.compress(
        json.dumps(
            {
                "submission_id": receipt["submission_id"],
                "note": str(receipt.get("note") or ""),
                "policy_base64": base64.b64encode(content).decode("ascii"),
            },
            separators=(",", ":"),
        ).encode()
    )
    script = r"""
import base64, gzip, json, os, pathlib, subprocess, sys
payload = json.loads(gzip.decompress(sys.stdin.buffer.read(int(sys.argv[1]))))
root = pathlib.Path("/run/sprint-submission-bridge/incoming")
root.mkdir(parents=True, exist_ok=True)
submission_id = payload["submission_id"]
target = root / f"{submission_id}.pt"
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_bytes(base64.b64decode(payload["policy_base64"]))
os.chmod(temporary, 0o400)
os.replace(temporary, target)
env = os.environ.copy()
env["SPRINT_HOST_ARCHIVE_REQUEST_ID"] = submission_id
command = ["/usr/local/bin/event", "archive", str(target)]
if payload.get("note"):
    command += ["--note", payload["note"]]
try:
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    print(json.dumps({"returncode": result.returncode, "stdout": result.stdout,
                      "stderr": result.stderr}, separators=(",", ":")))
finally:
    target.unlink(missing_ok=True)
""".strip()
    sandbox = _cpu_agent_sandbox(run)
    process = sandbox.exec(
        "python3", "-c", script, str(len(envelope)), text=False, timeout=90
    )
    process.stdin.write(envelope)
    process.stdin.drain()
    return_code = process.wait()
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    if return_code != 0:
        detail = stderr or stdout or b"CPU archive bridge failed"
        raise RuntimeError(detail.decode(errors="replace")[-1000:])
    result = json.loads(stdout)
    if not isinstance(result, dict):
        raise RuntimeError("CPU archive bridge returned malformed output")
    return result


def submit_worker_policy_to_cpu_agent(
    run: dict[str, Any], receipt: dict[str, Any], content: bytes
) -> dict[str, Any]:
    """Durably submit exact bytes, then mirror them to the agent if it is alive."""
    host_path = stage_worker_policy_on_host(run, receipt, content)
    result: dict[str, Any] = {
        "returncode": 0,
        "host_queue_path": str(host_path),
    }
    try:
        mirrored = _mirror_worker_policy_to_cpu_agent(run, receipt, content)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        result.update({"stderr": error, "agent_mirror_error": error})
        return result
    result.update(
        {
            "stdout": str(mirrored.get("stdout") or "")[-2000:],
            "stderr": str(mirrored.get("stderr") or "")[-2000:],
        }
    )
    result.update(
        {
            "agent_mirror_returncode": mirrored.get("returncode"),
            "agent_mirror_stdout": str(mirrored.get("stdout") or "")[-2000:],
            "agent_mirror_stderr": str(mirrored.get("stderr") or "")[-2000:],
        }
    )
    return result


def acknowledge_worker_submission(
    job: dict[str, Any], submission_id: str, payload: dict[str, Any]
) -> None:
    sandbox_id = str(job.get("sandbox_id") or "")
    if not sandbox_id.startswith("sb-"):
        return
    encoded = base64.b64encode(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    ).decode("ascii")
    script = r"""
import base64, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
submission_id = sys.argv[2]
target = root / "acknowledgments" / f"{submission_id}.pt.json"
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_bytes(base64.b64decode(sys.argv[3]))
os.replace(temporary, target)
(root / "outbox" / f"{submission_id}.pt").unlink(missing_ok=True)
""".strip()
    process = modal.Sandbox.from_id(sandbox_id).exec(
        "python3",
        "-c",
        script,
        GPU_SUBMISSION_BRIDGE_ROOT,
        submission_id,
        encoded,
        timeout=30,
    )
    if process.wait() != 0:
        detail = process.stderr.read() or process.stdout.read()
        raise RuntimeError(str(detail)[-1000:])


def _drain_worker_submission_outbox_once(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    exclude_submission_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Idempotently forward every GPU archive request through the CPU agent."""
    if not job.get("submission_bridge_enabled"):
        return job, {"submission_bridge": "disabled"}
    payload = dict(job)
    counts = {"observed": 0, "forwarded": 0, "retry_wait": 0, "error": 0}
    try:
        requests = read_worker_submission_outbox(
            run, job, exclude_submission_ids=exclude_submission_ids
        )
    except modal.exception.NotFoundError:
        requests = []
    except Exception as exc:  # noqa: BLE001
        payload["submission_bridge_error"] = f"{type(exc).__name__}: {exc}"
        requests = []
    durable_requests: list[dict[str, Any]] = []
    if not requests:
        try:
            durable_requests = read_durable_submission_outbox(run, job)
        except Exception as exc:  # noqa: BLE001
            payload["submission_bridge_error"] = f"{type(exc).__name__}: {exc}"
    by_id: dict[str, dict[str, Any]] = {}
    for request in [*durable_requests, *requests]:
        receipt = request.get("receipt")
        if isinstance(receipt, dict) and receipt.get("submission_id"):
            by_id[str(receipt["submission_id"])] = request
    for submission_id in exclude_submission_ids or set():
        by_id.pop(submission_id, None)
    requests = list(by_id.values())

    root = submission_bridge_state_dir(run)
    for raw in requests:
        counts["observed"] += 1
        try:
            receipt, content = validate_worker_submission_request(run, job, raw)
            submission_id = str(receipt["submission_id"])
        except Exception as exc:  # noqa: BLE001
            counts["error"] += 1
            payload["submission_bridge_error"] = f"{type(exc).__name__}: {exc}"
            continue
        record_path = root / f"{submission_id}.json"
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            record = {
                "schema_version": 1,
                "run_id": run["run_id"],
                "submission_id": submission_id,
                "queue_name": f"{submission_id}.pt",
                "gpu_job_id": job["job_id"],
                "gpu_attempt": job.get("attempt"),
                "policy_sha256": receipt["policy_sha256"],
                "policy_size_bytes": receipt["policy_size_bytes"],
                "observed_at": utc_now(),
                "state": "observed",
            }
            sprintctl.atomic_write_json(record_path, record, mode=0o600)
        if record.get("policy_sha256") != receipt.get("policy_sha256"):
            counts["error"] += 1
            record.update({"state": "error", "error": "request id policy hash changed"})
            sprintctl.atomic_write_json(record_path, record, mode=0o600)
            continue
        if record.get("state") == "forwarded":
            counts["forwarded"] += 1
            try:
                acknowledge_worker_submission(
                    job,
                    submission_id,
                    {
                        "schema_version": 1,
                        "submission_id": submission_id,
                        "state": "forwarded",
                        "forwarded_at": record.get("forwarded_at"),
                    },
                )
            except Exception:  # noqa: BLE001
                pass
            continue
        try:
            result = submit_worker_policy_to_cpu_agent(run, receipt, content)
        except Exception as exc:  # noqa: BLE001
            counts["retry_wait"] += 1
            record.update(
                {
                    "state": "retry_wait",
                    "last_attempt_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            sprintctl.atomic_write_json(record_path, record, mode=0o600)
            continue
        raw_returncode = result.get("returncode") if isinstance(result, dict) else None
        if isinstance(raw_returncode, bool) or not isinstance(raw_returncode, int):
            counts["retry_wait"] += 1
            record.update(
                {
                    "state": "retry_wait",
                    "last_attempt_at": utc_now(),
                    "error": "CPU archive response omitted an integer returncode",
                    "archive_stdout": (
                        str(result.get("stdout") or "")[-2000:]
                        if isinstance(result, dict)
                        else ""
                    ),
                    "archive_stderr": (
                        str(result.get("stderr") or "")[-2000:]
                        if isinstance(result, dict)
                        else ""
                    ),
                }
            )
            sprintctl.atomic_write_json(record_path, record, mode=0o600)
            continue
        archive_returncode = raw_returncode
        record.update(
            {
                "last_attempt_at": utc_now(),
                "archive_returncode": archive_returncode,
                "archive_stdout": str(result.get("stdout") or "")[-2000:],
                "archive_stderr": str(result.get("stderr") or "")[-2000:],
            }
        )
        if archive_returncode != 0:
            counts["retry_wait"] += 1
            record["state"] = "retry_wait"
            sprintctl.atomic_write_json(record_path, record, mode=0o600)
            continue
        counts["forwarded"] += 1
        record.update({"state": "forwarded", "forwarded_at": utc_now()})
        record.pop("error", None)
        sprintctl.atomic_write_json(record_path, record, mode=0o600)
        try:
            acknowledge_worker_submission(
                job,
                submission_id,
                {
                    "schema_version": 1,
                    "submission_id": submission_id,
                    "state": "forwarded",
                    "forwarded_at": record["forwarded_at"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            record["worker_ack_error"] = f"{type(exc).__name__}: {exc}"
            sprintctl.atomic_write_json(record_path, record, mode=0o600)
    if counts["error"] or counts["retry_wait"]:
        payload["submission_bridge_error"] = (
            f"{counts['error']} invalid, {counts['retry_wait']} awaiting retry"
        )
    else:
        payload.pop("submission_bridge_error", None)
    payload["submission_bridge_last_checked_at"] = utc_now()
    payload["submission_bridge_counts"] = counts
    return payload, {
        "submission_bridge": "drained",
        "submission_ids": sorted(by_id),
        **counts,
    }


def drain_worker_submission_outbox(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Forward the bounded outbox in small pages until it is drained.

    A page is deliberately small so 15 concurrent trials cannot each allocate
    hundreds of megabytes of base64 payload at once. Successful live requests
    are acknowledged and removed before the next page is read. The durable
    recovery reader likewise skips host records already marked forwarded.
    """
    if not job.get("submission_bridge_enabled"):
        return job, {"submission_bridge": "disabled"}
    payload = dict(job)
    totals = {"observed": 0, "forwarded": 0, "retry_wait": 0, "error": 0}
    seen: set[str] = set()
    for _ in range(GPU_SUBMISSION_BRIDGE_MAX_DRAIN_BATCHES):
        payload, detail = _drain_worker_submission_outbox_once(
            run, payload, exclude_submission_ids=seen
        )
        submission_ids = {
            str(value) for value in detail.pop("submission_ids", []) if value
        }
        fresh = submission_ids - seen
        if not fresh:
            break
        seen.update(fresh)
        for key in totals:
            totals[key] += int(detail.get(key) or 0)
        if int(detail.get("error") or 0) or int(detail.get("retry_wait") or 0):
            break
    else:
        payload["submission_bridge_error"] = (
            "submission outbox exceeded the bounded drain window"
        )
        totals["error"] += 1
    if totals["error"] or totals["retry_wait"]:
        payload["submission_bridge_error"] = (
            f"{totals['error']} invalid, {totals['retry_wait']} awaiting retry"
        )
    else:
        payload.pop("submission_bridge_error", None)
    payload["submission_bridge_last_checked_at"] = utc_now()
    payload["submission_bridge_counts"] = totals
    return payload, {"submission_bridge": "drained", **totals}


def owned_terminal_attempt(
    run: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any] | None:
    """Read the immutable terminal record for this exact job lease."""
    try:
        attempt = load_attempt_record(run, job)
    except Exception:  # noqa: BLE001 - the next dispatch tick retries
        return None
    if (
        not isinstance(attempt, dict)
        or int(attempt.get("attempt") or 0) != int(job.get("attempt") or 0)
        or str(attempt.get("lease_id") or "") != str(job.get("lease_id") or "")
        or str(attempt.get("status") or "") not in gpu_claim.TERMINAL
    ):
        return None
    return attempt


def terminal_submission_bridge_complete(
    run: dict[str, Any], job: dict[str, Any], detail: dict[str, Any]
) -> bool:
    """Prove every declared terminal submission was rejected or forwarded."""
    if int(detail.get("error") or 0) or int(detail.get("retry_wait") or 0):
        return False
    declared = [str(path) for path in job.get("submission_paths") or []]
    if not declared:
        return True
    attempt = owned_terminal_attempt(run, job)
    enqueue_snapshot_recovered = bool(
        job.get("submission_enqueue_snapshot_recovered_at")
    ) and not job.get("submission_enqueue_snapshot_recovery_pending")
    if attempt is None and not enqueue_snapshot_recovered:
        return False
    progress = (
        job.get("progress") if enqueue_snapshot_recovered else attempt.get("progress")
    )
    results = progress.get("submission_results") if isinstance(progress, dict) else None
    if not isinstance(results, list) or len(results) != len(declared):
        return False
    result_paths = [
        str(item.get("path") or "")
        for item in results
        if isinstance(item, dict)
    ]
    if result_paths != declared:
        return False
    staged = sum(
        1
        for item in results
        if isinstance(item, dict) and item.get("state") == "staged"
    )
    if any(
        not isinstance(item, dict) or item.get("state") not in {"staged", "rejected"}
        for item in results
    ):
        return False
    forwarded = 0
    for path in submission_bridge_state_dir(run).glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(record, dict)
            and record.get("gpu_job_id") == job.get("job_id")
            and record.get("state") == "forwarded"
        ):
            forwarded += 1
    return forwarded >= staged


def attach_terminal_submission_evidence(
    run: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    """Persist immutable worker submission results without undoing a host fence.

    A budget/operator fence can precede the worker's terminal Volume commit.
    In that case the host-owned job must remain ``terminated``, but the later
    immutable ``submission_results`` are still authoritative evidence needed
    to prove that every explicitly declared path was rejected or forwarded.
    """
    if not job.get("submission_paths"):
        return job
    attempt = owned_terminal_attempt(run, job)
    if attempt is None:
        return job
    progress = attempt.get("progress")
    if not isinstance(progress, dict) or not isinstance(
        progress.get("submission_results"), list
    ):
        return job
    payload = dict(job)
    payload["progress"] = progress
    payload["attempt_record"] = attempt_path(
        str(run["run_id"]), str(job["job_id"]), int(job["attempt"])
    )
    return persist_job(run, payload)


def terminal_cpu_unavailable(run: dict[str, Any]) -> bool:
    state_dir_raw = str(run.get("state_dir") or "")
    if not state_dir_raw:
        return False
    state_dir = Path(state_dir_raw)
    return (state_dir / "STOP_ACK.json").is_file() or (
        state_dir / "FINALIZED.json"
    ).is_file()


def mirror_agent_job(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    log_content: bytes | None = None,
    artifact_name: str | None = None,
    artifact_content: bytes | None = None,
) -> dict[str, Any]:
    """Push canonical GPU state into the long-lived CPU container.

    Modal Volume mounts are snapshots: a long-lived CPU container does not see
    host uploads until its mount is reloaded. Reloading the whole mount can
    disrupt open agent files. A small host-owned mirror under /run gives the
    agent fresh status and a diagnostic log tail without weakening the
    no-control-plane-credentials boundary. The complete log remains on the
    durable Volume.
    """
    if terminal_cpu_unavailable(run):
        return {"agent_mirror": "terminal_cpu_unavailable"}
    container_id = str(run.get("agent_container_id") or "")
    job_id = str(job.get("job_id") or "")
    if not container_id.startswith("ta-") or not job_id:
        return {"agent_mirror": "unavailable"}

    files = {
        f"status/{job_id}.json": (
            json.dumps(job, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    log_truncated = False
    if log_content is not None:
        attempt = int(job.get("attempt") or 0)
        if attempt > 0:
            if len(log_content) > AGENT_GPU_MIRROR_LOG_BYTES:
                log_truncated = True
                marker = (
                    b"[agent mirror truncated to the final 768 KiB; "
                    b"the complete checksummed log is retained on /durable]\n"
                )
                log_content = marker + log_content[-AGENT_GPU_MIRROR_LOG_BYTES:]
            files[f"out/{job_id}/attempt-{attempt}/worker.log"] = log_content
    if artifact_name and artifact_content is not None:
        safe_name = Path(artifact_name).name
        if safe_name != artifact_name:
            return {
                "agent_mirror": "error",
                "agent_mirror_error": "invalid artifact name",
            }
        files[f"artifacts/{job_id}/{safe_name}"] = artifact_content

    cli_content = (AGENT_COMMAND_SOURCE / "gpu.py").read_bytes()
    cli_sha256 = hashlib.sha256(cli_content).hexdigest()
    envelope = {
        "files": {
            relative: base64.b64encode(content).decode("ascii")
            for relative, content in files.items()
        },
        "agent_cli": base64.b64encode(cli_content).decode("ascii"),
        "agent_cli_sha256": cli_sha256,
    }
    compressed = gzip.compress(json.dumps(envelope, separators=(",", ":")).encode())
    install = """
import base64, gzip, hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
compressed = sys.stdin.buffer.read(int(sys.argv[4])) if sys.argv[2] == "-" else base64.b64decode(sys.argv[2])
payload = json.loads(gzip.decompress(compressed))
for relative, content in payload["files"].items():
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise SystemExit("invalid mirror path")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(base64.b64decode(content))
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
cli = base64.b64decode(payload["agent_cli"])
if hashlib.sha256(cli).hexdigest() != payload["agent_cli_sha256"]:
    raise SystemExit("agent CLI checksum mismatch")
cli_target = pathlib.Path(sys.argv[3])
cli_temporary = cli_target.with_name(f".{cli_target.name}.{os.getpid()}.tmp")
cli_temporary.write_bytes(cli)
os.chmod(cli_temporary, 0o755)
os.replace(cli_temporary, cli_target)
""".strip()
    try:
        if len(compressed) <= AGENT_GPU_MIRROR_ARG_BYTES:
            encoded = base64.b64encode(compressed).decode("ascii")
            result = sprintctl.exec_container(
                run,
                container_id,
                " ".join(
                    [
                        "python3",
                        "-c",
                        shlex.quote(install),
                        shlex.quote(AGENT_GPU_MIRROR_ROOT),
                        shlex.quote(encoded),
                        shlex.quote(AGENT_GPU_CLI_PATH),
                        str(len(compressed)),
                    ]
                ),
                check=False,
                timeout=20,
            )
            return_code = result.returncode
            stdout = result.stdout or ""
            stderr = result.stderr or ""
        else:
            identity = sprintctl.exec_container(
                run,
                container_id,
                'printf "%s" "$MODAL_SANDBOX_ID"',
                check=False,
                timeout=20,
            )
            sandbox_id = (identity.stdout or "").strip()
            if identity.returncode != 0 or not sandbox_id.startswith("sb-"):
                error = (
                    identity.stderr or identity.stdout or "missing sandbox ID"
                ).strip()
                return {
                    "agent_mirror": "error",
                    "agent_mirror_error": error[-1000:],
                }
            process = modal.Sandbox.from_id(sandbox_id).exec(
                "python3",
                "-c",
                install,
                AGENT_GPU_MIRROR_ROOT,
                "-",
                AGENT_GPU_CLI_PATH,
                str(len(compressed)),
                text=False,
                timeout=30,
            )
            process.stdin.write(compressed)
            process.stdin.drain()
            return_code = process.wait()
            stdout_raw = process.stdout.read()
            stderr_raw = process.stderr.read()
            stdout = stdout_raw.decode(errors="replace")
            stderr = stderr_raw.decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {
            "agent_mirror": "error",
            "agent_mirror_error": f"{type(exc).__name__}: {exc}",
        }
    if return_code != 0:
        error = (stderr or stdout or "container exec failed").strip()
        return {
            "agent_mirror": "error",
            "agent_mirror_error": error[-1000:],
            "agent_mirror_return_code": return_code,
        }
    return {
        "agent_mirror": "updated",
        "agent_mirror_files": len(files),
        "agent_mirror_log_truncated": log_truncated,
        "agent_cli_sha256": cli_sha256,
    }


def mirror_agent_cost(run: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically publish a trusted cost snapshot and its read-only CLI.

    The CPU agent has no Modal or provider credentials.  Only the host computes
    this document, then installs it under the existing host-owned /run mirror.
    """
    state_dir_raw = str(run.get("state_dir") or "")
    if state_dir_raw:
        state_dir = Path(state_dir_raw)
        if (state_dir / "STOP_REQUESTED.json").is_file() or (
            state_dir / "STOP"
        ).is_file():
            return {"agent_cost_mirror": "agent_stopped"}
        ack = state_dir / "STOP_ACK.json"
        if ack.is_file():
            return {"agent_cost_mirror": "agent_stopped"}
    container_id = str(run.get("agent_container_id") or "")
    if not container_id.startswith("ta-"):
        return {"agent_cost_mirror": "unavailable"}
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    cli_content = (AGENT_COMMAND_SOURCE / "cost.py").read_bytes()
    cli_sha256 = hashlib.sha256(cli_content).hexdigest()
    envelope = {
        "cost": base64.b64encode(content).decode("ascii"),
        "cli": base64.b64encode(cli_content).decode("ascii"),
        "cli_sha256": cli_sha256,
    }
    compressed = gzip.compress(json.dumps(envelope, separators=(",", ":")).encode())
    if len(compressed) > AGENT_GPU_MIRROR_ARG_BYTES:
        return {
            "agent_cost_mirror": "error",
            "agent_cost_mirror_error": "cost payload exceeds trusted mirror bound",
        }
    install = """
import base64, gzip, hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(gzip.decompress(base64.b64decode(sys.argv[2])))
target = (root / "cost.json").resolve()
if root not in target.parents:
    raise SystemExit("invalid cost mirror path")
target.parent.mkdir(parents=True, exist_ok=True)
incoming = base64.b64decode(payload["cost"])
incoming_doc = json.loads(incoming)
def rank(document):
    return (
        float(document.get("checked_at_epoch_s") or 0),
        float(document.get("total_usd") or 0),
    )
stale = False
try:
    stale = rank(json.loads(target.read_text())) > rank(incoming_doc)
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    pass
if not stale:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(incoming)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
cli = base64.b64decode(payload["cli"])
if hashlib.sha256(cli).hexdigest() != payload["cli_sha256"]:
    raise SystemExit("cost CLI checksum mismatch")
cli_target = pathlib.Path(sys.argv[3])
cli_temporary = cli_target.with_name(f".{cli_target.name}.{os.getpid()}.tmp")
cli_temporary.write_bytes(cli)
os.chmod(cli_temporary, 0o755)
os.replace(cli_temporary, cli_target)
print("STALE_IGNORED" if stale else "UPDATED")
""".strip()
    encoded = base64.b64encode(compressed).decode("ascii")
    try:
        result = sprintctl.exec_container(
            run,
            container_id,
            " ".join(
                [
                    "python3",
                    "-c",
                    shlex.quote(install),
                    shlex.quote(AGENT_GPU_MIRROR_ROOT),
                    shlex.quote(encoded),
                    shlex.quote(AGENT_COST_CLI_PATH),
                ]
            ),
            check=False,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        if "Task has already finished with status" in str(exc):
            return {"agent_cost_mirror": "agent_stopped"}
        return {
            "agent_cost_mirror": "error",
            "agent_cost_mirror_error": f"{type(exc).__name__}: {exc}",
        }
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "container exec failed").strip()
        if "Task has already finished with status" in error:
            return {"agent_cost_mirror": "agent_stopped"}
        return {
            "agent_cost_mirror": "error",
            "agent_cost_mirror_error": error[-1000:],
            "agent_cost_mirror_return_code": result.returncode,
        }
    stale_ignored = "STALE_IGNORED" in (result.stdout or "")
    return {
        "agent_cost_mirror": "stale_ignored" if stale_ignored else "updated",
        "agent_cost_snapshot_bytes": len(content),
        "agent_cost_cli_sha256": cli_sha256,
        "agent_cost_as_of": payload.get("as_of"),
    }


def mirror_gpu_budget(
    run: dict[str, Any],
    payload: dict[str, Any],
    *,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inject the canonical cost snapshot into each active GPU sandbox.

    Modal Volume mounts are point-in-time views and cannot safely be reloaded
    while training processes hold files open.  The trusted host therefore
    mirrors the CPU watchdog's exact snapshot into ``/run``.  The worker fails
    closed if this heartbeat becomes stale, so loss of the controller still
    terminates GPU spend inside the configured shutdown reserve.

    Target discovery is deliberately host-local.  A claimed job is persisted
    in the controller-owned registry before ``Sandbox.create`` and gains its
    sandbox ID before the dispatch lock is released.  Consulting the remote
    agent-facing status delivery mirrors here would put the budget-critical pulse
    behind Modal Volume latency for jobs that were never allocated.  The
    dispatch startup barrier and stale-heartbeat shutdown cover the narrow
    crash window before a new sandbox ID is published locally.
    """
    run_id = str(run.get("run_id") or "")
    try:
        if payload.get("schema_version") != 2 or payload.get("run_id") != run_id:
            raise ValueError("budget snapshot identity mismatch")
        checked_at = float(payload["checked_at_epoch_s"])
        total = float(payload["total_usd"])
        threshold = float(payload["stop_threshold_usd"])
        if not (
            checked_at > 0
            and total >= 0
            and threshold > 0
            and payload.get("status") in {"within_budget", "stop_requested"}
        ):
            raise ValueError("budget snapshot fields are invalid")
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "gpu_budget_mirror": "error",
            "gpu_budget_mirror_error": f"{type(exc).__name__}: {exc}",
        }

    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(content) > MAX_GPU_BUDGET_MIRROR_BYTES:
        return {
            "gpu_budget_mirror": "error",
            "gpu_budget_mirror_error": "budget snapshot exceeds trusted mirror bound",
        }
    if jobs is None:
        jobs = [
            job
            for job_id in list_host_job_ids(run)
            if (job := load_host_job(run, job_id))
            and str(job.get("status") or "") in gpu_claim.OWNED
            and str(job.get("sandbox_id") or "").startswith("sb-")
        ]
    targets = sorted(
        {
            str(job.get("sandbox_id"))
            for job in jobs
            if str(job.get("sandbox_id") or "").startswith("sb-")
        }
    )
    if not targets:
        return {"gpu_budget_mirror": "no_active_sandbox", "sandbox_ids": []}

    encoded = base64.b64encode(content).decode("ascii")
    install = """
import base64, json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
incoming = base64.b64decode(sys.argv[2])
incoming_doc = json.loads(incoming)
def rank(document):
    return (
        float(document.get("checked_at_epoch_s") or 0),
        float(document.get("total_usd") or 0),
    )
stale = False
try:
    stale = rank(json.loads(target.read_text())) > rank(incoming_doc)
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    pass
if not stale:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(incoming)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
print("STALE_IGNORED" if stale else "UPDATED")
""".strip()
    updated: list[str] = []
    stale_ignored: list[str] = []
    finished: list[str] = []
    errors: dict[str, str] = {}
    for sandbox_id in targets:
        try:
            process = modal.Sandbox.from_id(sandbox_id).exec(
                "python3",
                "-c",
                install,
                GPU_BUDGET_MIRROR_PATH,
                encoded,
                timeout=30,
            )
            return_code = process.wait()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            if return_code != 0:
                detail = stderr or stdout or f"exit {return_code}"
                if isinstance(detail, bytes):
                    detail = detail.decode(errors="replace")
                raise RuntimeError(str(detail).strip()[-1000:])
            if "STALE_IGNORED" in str(stdout or ""):
                stale_ignored.append(sandbox_id)
            else:
                updated.append(sandbox_id)
        except modal.exception.NotFoundError:
            # A short worker can finish after the registry scan but before
            # this exec.  A terminal sandbox cannot accrue more GPU cost, so
            # the budget objective is already satisfied.  The dispatcher
            # separately reconciles the job's success/failure state.
            finished.append(sandbox_id)
        except modal.exception.ConflictError as exc:
            # Modal can expose the same terminal race as a 409 while the
            # sandbox is transitioning out of the running state.
            if "shutting down" in str(exc).lower():
                finished.append(sandbox_id)
            else:
                errors[sandbox_id] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            # The Modal client can surface the same terminal race as a plain
            # RuntimeError after process creation rather than as NotFound/409.
            # No mirror is needed once the provider reports the container as
            # stopped, and treating normal verifier teardown as a pulse outage
            # creates a false controller alert.
            message = str(exc).lower()
            terminal_exec_race = "cannot execute in container" in message and (
                "state stopped" in message or "state terminated" in message
            )
            if terminal_exec_race:
                finished.append(sandbox_id)
            else:
                errors[sandbox_id] = f"{type(exc).__name__}: {exc}"
    return {
        "gpu_budget_mirror": "updated" if not errors else "error",
        "sandbox_ids": targets,
        "updated_sandbox_ids": updated,
        "stale_ignored_sandbox_ids": stale_ignored,
        "finished_sandbox_ids": finished,
        "errors": errors,
        "snapshot_checked_at_epoch_s": checked_at,
        "snapshot_total_usd": total,
    }


def fresh_dispatch_budget_snapshot(
    state_dir: Path,
    run_id: str,
    *,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Load the canonical host ledger for a newly spawned GPU sandbox.

    The raw in-sandbox watchdog excludes host-observed training allocations and
    can therefore understate total run spend after earlier GPU work.  Dispatch
    may seed only the merged agent-cost document maintained by the independent
    pulse, and only while that document is fresh.  Otherwise the worker stays
    behind its fail-closed startup barrier until the next pulse arrives.
    """

    try:
        payload = json.loads((state_dir / "telemetry" / "agent-cost.json").read_text())
        checked_at = float(payload["checked_at_epoch_s"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    reference = time.time() if now is None else float(now)
    age = reference - checked_at
    if (
        payload.get("schema_version") != 2
        or payload.get("run_id") != run_id
        or payload.get("status") not in {"within_budget", "stop_requested"}
        or age < -5
        or age > 60
    ):
        return None
    return payload


def fetch_agent_policy_artifact(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], str | None, bytes | None, dict[str, Any]]:
    """Fetch a worker-reported policy into the trusted CPU-agent mirror.

    GPU and CPU sandboxes mount point-in-time Volume views.  The host therefore
    copies only the policy explicitly named by the worker's progress record,
    verifies its scope and size, and exposes it under /run.  The agent receives
    neither Modal credentials nor a general Volume refresh primitive.
    """
    payload = dict(job)
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        return payload, None, None, {"policy_mirror": "not_reported"}
    raw_path = str(progress.get("policy_path") or progress.get("policy") or "").strip()
    if not raw_path:
        return payload, None, None, {"policy_mirror": "not_reported"}
    policy_path = Path(raw_path)
    run_root = Path("/durable") / "runs" / str(run["run_id"])
    allowed_roots = (
        run_root / "policies",
        run_root / "candidates",
        run_root / "gpu-jobs",
    )
    relative: Path | None = None
    for expected in allowed_roots:
        try:
            relative = policy_path.relative_to(expected)
            break
        except ValueError:
            continue
    if (
        relative is None
        or not relative.parts
        or ".." in relative.parts
        or policy_path.suffix not in {".pt", ".pth"}
    ):
        return payload, None, None, {"policy_mirror": "rejected_scope"}
    remote = str(policy_path.relative_to("/durable"))
    content = sprintctl.volume_get_bytes(
        run,
        remote,
        timeout_seconds=120,
        max_bytes=AGENT_GPU_MIRROR_ARTIFACT_BYTES,
    )
    if not content:
        return payload, None, None, {"policy_mirror": "fetch_retry"}
    mirror_path = (
        f"{AGENT_GPU_MIRROR_ROOT}/artifacts/{job['job_id']}/{policy_path.name}"
    )
    payload.update(
        {
            "agent_policy_mirror_path": mirror_path,
            "agent_policy_source_path": str(policy_path),
            "agent_policy_sha256": hashlib.sha256(content).hexdigest(),
            "agent_policy_size_bytes": len(content),
        }
    )
    mirrors = dict(payload.get("agent_artifact_mirrors") or {})
    source_path = str(
        next(
            (
                item.get("source_path")
                for item in (progress.get("output_artifacts") or [])
                if isinstance(item, dict)
                and str(item.get("path") or "") == str(policy_path)
            ),
            str(Path("/app") / policy_path.name),
        )
    )
    mirrors[source_path] = {
        "mirror_path": mirror_path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    payload["agent_artifact_mirrors"] = mirrors
    return (
        payload,
        policy_path.name,
        content,
        {
            "policy_mirror": "fetched",
            "policy_size_bytes": len(content),
        },
    )


def pending_output_artifact(job: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next declared terminal output not yet mirrored to the agent."""
    progress = job.get("progress")
    if not isinstance(progress, dict):
        return None
    mirrors = job.get("agent_artifact_mirrors")
    mirrored_sources = set(mirrors) if isinstance(mirrors, dict) else set()
    for record in progress.get("output_artifacts") or []:
        if not isinstance(record, dict):
            continue
        source_path = str(record.get("source_path") or "").strip()
        durable_path = str(record.get("path") or "").strip()
        if source_path and durable_path and source_path not in mirrored_sources:
            return record
    return None


def fetch_agent_output_artifact(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], str | None, bytes | None, dict[str, Any]]:
    """Fetch one declared output into the trusted CPU-agent mirror."""
    payload = dict(job)
    record = pending_output_artifact(payload)
    if record is None:
        return payload, None, None, {"artifact_mirror": "not_reported"}
    source_path = Path(str(record.get("source_path") or ""))
    durable_path = Path(str(record.get("path") or ""))
    expected_root = (
        Path("/durable")
        / "runs"
        / str(run["run_id"])
        / "gpu-jobs"
        / "artifacts"
        / str(job["job_id"])
    )
    try:
        source_path.relative_to("/app")
        relative = durable_path.relative_to(expected_root)
    except ValueError:
        return payload, None, None, {"artifact_mirror": "rejected_scope"}
    if not relative.parts or ".." in relative.parts or not durable_path.name:
        return payload, None, None, {"artifact_mirror": "rejected_scope"}
    content = sprintctl.volume_get_bytes(
        run,
        str(durable_path.relative_to("/durable")),
        timeout_seconds=120,
        max_bytes=AGENT_GPU_MIRROR_ARTIFACT_BYTES,
    )
    if content is None:
        return payload, None, None, {"artifact_mirror": "fetch_retry"}
    expected_size = int(record.get("size_bytes") or 0)
    expected_sha = str(record.get("sha256") or "")
    actual_sha = hashlib.sha256(content).hexdigest()
    if (
        not content
        or expected_size != len(content)
        or not expected_sha
        or expected_sha != actual_sha
    ):
        return payload, None, None, {"artifact_mirror": "digest_mismatch"}
    mirror_path = (
        f"{AGENT_GPU_MIRROR_ROOT}/artifacts/{job['job_id']}/{durable_path.name}"
    )
    mirrors = dict(payload.get("agent_artifact_mirrors") or {})
    mirrors[str(source_path)] = {
        "mirror_path": mirror_path,
        "size_bytes": len(content),
        "sha256": actual_sha,
    }
    payload["agent_artifact_mirrors"] = mirrors
    if source_path.suffix in {".pt", ".pth"}:
        payload.update(
            {
                "agent_policy_mirror_path": mirror_path,
                "agent_policy_source_path": str(durable_path),
                "agent_policy_sha256": actual_sha,
                "agent_policy_size_bytes": len(content),
            }
        )
    return (
        payload,
        durable_path.name,
        content,
        {
            "artifact_mirror": "fetched",
            "artifact_source_path": str(source_path),
            "artifact_size_bytes": len(content),
        },
    )


def refresh_live_policy_mirror(
    run: dict[str, Any],
    job: dict[str, Any],
    heartbeat: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish a newly committed intermediate policy without ending training.

    A long-lived CPU sandbox has a point-in-time view of the Modal Volume, so
    it cannot see policies written later by its GPU worker.  The trusted host
    already reads the worker heartbeat on every dispatch cycle.  When that
    heartbeat atomically names a new policy, copy the bounded run-scoped bytes
    into the host-owned CPU mirror.  Repeated heartbeats for the same path are
    idempotent and avoid another Volume download.
    """
    if not isinstance(heartbeat, dict):
        return job, {"live_policy_mirror": "no_heartbeat"}
    if int(heartbeat.get("attempt") or 0) != int(job.get("attempt") or 0) or str(
        heartbeat.get("lease_id") or ""
    ) != str(job.get("lease_id") or ""):
        return job, {"live_policy_mirror": "stale_heartbeat"}
    progress = heartbeat.get("progress")
    if not isinstance(progress, dict):
        return job, {"live_policy_mirror": "not_reported"}
    raw_path = str(progress.get("policy_path") or progress.get("policy") or "").strip()
    if not raw_path:
        return job, {"live_policy_mirror": "not_reported"}
    progress_sha256 = hashlib.sha256(
        json.dumps(progress, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    payload = dict(job)
    payload["progress"] = progress
    if heartbeat.get("checkpoint"):
        payload["checkpoint"] = heartbeat["checkpoint"]
    if progress_sha256 == str(
        payload.get("agent_policy_source_progress_sha256") or ""
    ) and payload.get("agent_policy_mirror_path"):
        return payload, {"live_policy_mirror": "already_mirrored"}

    payload, artifact_name, artifact_content, detail = fetch_agent_policy_artifact(
        run, payload
    )
    if artifact_name is None or artifact_content is None:
        return payload, {"live_policy_mirror": detail.get("policy_mirror")}
    payload["agent_policy_source_progress_sha256"] = progress_sha256
    mirror_detail = mirror_agent_job(
        run,
        payload,
        artifact_name=artifact_name,
        artifact_content=artifact_content,
    )
    payload["agent_policy_mirrored_at"] = utc_now()
    persist_job(run, payload)
    return payload, {
        "live_policy_mirror": "updated",
        **detail,
        **mirror_detail,
    }


def retry_terminal_artifact_mirror(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retry a terminal artifact fetch after Modal Volume propagation lag."""
    payload, artifact_name, artifact_content, detail = fetch_agent_output_artifact(
        run, job
    )
    outcome = str(detail.get("artifact_mirror") or "")
    if artifact_name is not None and artifact_content is not None:
        payload.pop("artifact_mirror_retry_after_epoch_s", None)
        payload.pop("artifact_mirror_attempts", None)
        payload["agent_artifact_mirrored_at"] = utc_now()
        mirror_detail = mirror_agent_job(
            run,
            payload,
            artifact_name=artifact_name,
            artifact_content=artifact_content,
        )
        return payload, {**detail, **mirror_detail}
    if outcome == "fetch_retry":
        attempts = int(payload.get("artifact_mirror_attempts") or 0) + 1
        payload["artifact_mirror_attempts"] = attempts
        payload["artifact_mirror_retry_after_epoch_s"] = time.time() + min(
            15 * 60, 15 * (2 ** min(attempts - 1, 6))
        )
    else:
        payload["artifact_mirror_terminal_failure"] = outcome or "unknown"
    return payload, detail


def jobs_prefix(run_id: str) -> str:
    return f"runs/{run_id}/gpu-jobs"


def load_agent_job_index(run: dict[str, Any]) -> dict[str, Any] | None:
    """Load the agent-published exact-name GPU control-plane index."""
    if not str(run.get("volume_name") or "").strip():
        return None
    remote = f"{jobs_prefix(str(run['run_id']))}/index.json"
    raw = sprintctl.volume_get_text(run, remote, timeout_seconds=15)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("agent GPU dispatch index is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("run_id") != str(run["run_id"])
        or not isinstance(payload.get("jobs"), dict)
    ):
        raise RuntimeError("agent GPU dispatch index identity/schema mismatch")
    for job_id, detail in payload["jobs"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(job_id)) or not isinstance(
            detail, dict
        ):
            raise RuntimeError("agent GPU dispatch index contains an invalid job")
    return payload


def indexed_agent_jobs(run: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    payload = load_agent_job_index(run)
    if payload is None:
        return None
    return {str(key): dict(value) for key, value in payload["jobs"].items()}


def host_job_path(run: dict[str, Any], job_id: str) -> Path | None:
    """Return the host-owned canonical record for a logical GPU job.

    Keeping the controller's copy outside the agent sandbox makes job state
    independent from the agent-facing status mirror.
    """
    state_dir = str(run.get("state_dir") or "").strip()
    if not state_dir or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        return None
    return Path(state_dir) / "gpu-job-registry" / f"{job_id}.json"


def host_work_archive_path(run: dict[str, Any], job_id: str) -> Path | None:
    """Return the controller-owned immutable workspace for a claimed job."""
    record = host_job_path(run, job_id)
    if record is None:
        return None
    return record.parent.parent / "gpu-job-work" / job_id / "app.tar.gz"


def _validated_work_archive_remote(run: dict[str, Any], job: dict[str, Any]) -> str:
    run_id = str(run["run_id"])
    job_id = str(job["job_id"])
    expected = f"{jobs_prefix(run_id)}/work/{job_id}/app.tar.gz"
    remote = str(job.get("work_archive") or "")
    if remote != expected:
        raise RuntimeError(f"invalid work archive path for {job_id}: {remote!r}")
    return remote


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def work_archive_retry_delay(attempt: int) -> float:
    """Short bounded backoff for Modal Volume control-plane contention."""
    return float(min(8, 2 ** max(0, attempt - 1)))


def pin_work_archive(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    transfer_attempts: int = WORK_ARCHIVE_TRANSFER_ATTEMPTS,
    transfer_timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Snapshot the workspace at lease claim for identical retries.

    The queue and its Volume workspace are agent-owned until the controller
    claims them.  Once claimed, the host copy and digest become authoritative;
    every retry restores those exact bytes before allocating a replacement.
    """
    payload = dict(job)
    job_id = str(payload["job_id"])
    remote = _validated_work_archive_remote(run, payload)
    canonical = host_work_archive_path(run, job_id)
    if canonical is None:
        raise RuntimeError("run state_dir is required to pin GPU work")
    expected = str(payload.get("work_archive_sha256") or "")
    if canonical.is_file():
        size = canonical.stat().st_size
        digest = _file_sha256(canonical)
        if expected and digest != expected:
            raise RuntimeError("host-pinned GPU work archive digest mismatch")
    else:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as raw:
            downloaded = Path(raw) / "app.tar.gz"
            for attempt in range(1, transfer_attempts + 1):
                try:
                    available = sprintctl.volume_download_exact(
                        run,
                        remote,
                        downloaded,
                        timeout_seconds=transfer_timeout_seconds,
                        max_bytes=MAX_WORK_ARCHIVE_BYTES,
                    )
                except (OSError, subprocess.SubprocessError, TimeoutError):
                    available = False
                if available:
                    break
                if attempt < transfer_attempts:
                    time.sleep(work_archive_retry_delay(attempt))
            else:
                raise RuntimeError(
                    "unable to pin GPU work archive after "
                    f"{transfer_attempts} attempts: {remote}"
                )
            size = downloaded.stat().st_size
            digest = _file_sha256(downloaded)
            if expected and digest != expected:
                raise RuntimeError("GPU work archive changed after host claim")
            staged = canonical.with_name(f".{canonical.name}.{os.getpid()}.tmp")
            with downloaded.open("rb") as source, staged.open("wb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            os.chmod(staged, 0o600)
            os.replace(staged, canonical)
    submitted = str(payload.get("submitted_work_archive_sha256") or "")
    payload.update(
        {
            "work_archive_sha256": digest,
            "work_archive_size_bytes": size,
            "work_archive_pinned_at": payload.get("work_archive_pinned_at")
            or utc_now(),
            "work_archive_provenance": "host-pinned-at-lease-claim",
        }
    )
    if submitted and submitted != digest:
        payload["work_archive_changed_before_claim"] = True
    return payload


def _enqueue_snapshot_submission_id(
    run: dict[str, Any], job: dict[str, Any], path: str, digest: str
) -> str:
    """Return a stable Harbor request id for an enqueue-snapshot policy."""
    try:
        created_epoch = float(job.get("created_at_epoch_s") or 0)
    except (TypeError, ValueError):
        created_epoch = 0
    prefix = time.strftime("%H%M%S", time.gmtime(max(0, created_epoch)))
    suffix = hashlib.sha256(
        "\0".join(
            (str(run["run_id"]), str(job["job_id"]), path, digest)
        ).encode()
    ).hexdigest()[:4]
    return f"{prefix}-{suffix}"


def _submission_bytes_from_work_archive(
    archive: Path, declared_path: str
) -> tuple[bytes | None, str | None]:
    """Read one exact regular-file policy without extracting the workspace."""
    candidate = Path(declared_path)
    try:
        relative = candidate.relative_to("/app")
    except ValueError:
        return None, "submission path is outside /app"
    if not relative.parts or ".." in relative.parts:
        return None, "submission path is invalid"
    member_name = str(Path("app") / relative)
    try:
        with tarfile.open(archive, "r:gz") as handle:
            matches = [
                member for member in handle.getmembers() if member.name == member_name
            ]
            if len(matches) != 1:
                return None, "policy was not a unique file in the enqueue snapshot"
            member = matches[0]
            if not member.isfile():
                return None, "policy was not a regular file in the enqueue snapshot"
            if not (0 < member.size <= GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES):
                return None, "policy size was outside the submission limit"
            source = handle.extractfile(member)
            if source is None:
                return None, "policy bytes were unavailable in the enqueue snapshot"
            content = source.read(GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES + 1)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"invalid enqueue work archive: {exc}") from exc
    if len(content) != member.size:
        raise RuntimeError("enqueue-snapshot policy size changed while reading")
    return content, None


def recover_pending_submission_snapshots(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finish explicit submissions from their immutable enqueue snapshot.

    ``event gpu --submit-output`` snapshots /app before enqueue. If that job is
    still waiting behind another FIFO job when the budget closes, the exact
    bytes that already existed at enqueue can be admitted without starting the
    queued command. The same recovery applies when a budget/operator fence lands
    after the command exits but before the worker commits its submission
    manifest: the pinned enqueue archive still proves which bytes already
    existed when the agent explicitly requested submission. Worker-authored
    submission results always win when present. Missing files are a normal
    rejected result; transport or digest ambiguity remains fail-closed and is
    retried while the CPU supervisor waits for the submission-drain handshake.
    """
    declared = [str(path) for path in job.get("submission_paths") or []]
    payload = dict(job)
    attempt = owned_terminal_attempt(run, payload)
    attempt_progress = attempt.get("progress") if isinstance(attempt, dict) else None
    worker_results = (
        attempt_progress.get("submission_results")
        if isinstance(attempt_progress, dict)
        else None
    )
    status = str(payload.get("status") or "")
    pending_undispatched = (
        status == "pending"
        and int(payload.get("attempt") or 0) == 0
        and not payload.get("sandbox_id")
        and not payload.get("started_at")
    )
    stopped_without_worker_results = (
        status == "terminated"
        and str(payload.get("termination_reason") or "")
        in {
            "agent_cost_budget_exhausted",
            "budget_telemetry_unavailable",
            "operator_stop",
            "operator_batch_stop",
        }
        and not isinstance(worker_results, list)
        and not payload.get("submission_bridge_terminal_drained_at")
    )
    eligible = (
        bool(declared)
        and bool(payload.get("submission_bridge_enabled"))
        and (pending_undispatched or stopped_without_worker_results)
    )
    if not eligible:
        return payload, {"eligible": False, "error": 0, "retry_wait": 0}

    payload["submission_enqueue_snapshot_recovery_pending"] = True
    try:
        # The CPU supervisor's drain handshake is bounded at two minutes. Keep
        # recovery within that same window so transport trouble fails closed
        # without delaying teardown indefinitely.
        payload = pin_work_archive(
            run,
            payload,
            transfer_attempts=2,
            transfer_timeout_seconds=45,
        )
        submitted_digest = str(payload.get("submitted_work_archive_sha256") or "")
        pinned_digest = str(payload.get("work_archive_sha256") or "")
        if not submitted_digest or submitted_digest != pinned_digest:
            raise RuntimeError("enqueue work archive digest is not authoritative")
        archive = host_work_archive_path(run, str(payload["job_id"]))
        if archive is None or not archive.is_file():
            raise RuntimeError("host-pinned enqueue work archive is unavailable")

        results: list[dict[str, Any]] = []
        forwarded = 0
        rejected = 0
        for path in declared:
            content, rejection = _submission_bytes_from_work_archive(archive, path)
            if rejection is not None:
                rejected += 1
                results.append(
                    {
                        "path": path,
                        "state": "rejected",
                        "reason": rejection,
                        "source": "enqueue_snapshot_at_budget_stop",
                    }
                )
                continue
            assert content is not None
            digest = hashlib.sha256(content).hexdigest()
            submission_id = _enqueue_snapshot_submission_id(run, payload, path, digest)
            record_path = submission_bridge_state_dir(run) / f"{submission_id}.json"
            try:
                record = json.loads(record_path.read_text())
            except (OSError, json.JSONDecodeError):
                record = {}
            if isinstance(record, dict) and record.get("state") == "forwarded":
                # Older recovery code recorded all enqueue-snapshot receipts as
                # attempt zero. Repair the idempotency record from the current
                # authoritative job so later audits identify the real worker
                # attempt without forwarding the policy a second time.
                actual_attempt = int(payload.get("attempt") or 0)
                actual_lease = str(payload.get("lease_id") or "")
                if (
                    int(record.get("gpu_attempt") or 0) != actual_attempt
                    or str(record.get("gpu_lease_id") or "") != actual_lease
                ):
                    record = dict(record)
                    record["gpu_attempt"] = actual_attempt
                    record["gpu_lease_id"] = actual_lease
                    sprintctl.atomic_write_json(record_path, record, mode=0o600)
            if not isinstance(record, dict) or record.get("state") != "forwarded":
                receipt = {
                    "schema_version": 2,
                    "bridge": "host_owned_gpu_submission_v1",
                    "submission_id": submission_id,
                    "queue_name": f"{submission_id}.pt",
                    "run_id": str(run["run_id"]),
                    "gpu_job_id": str(payload["job_id"]),
                    "gpu_attempt": int(payload.get("attempt") or 0),
                    "gpu_lease_id": str(payload.get("lease_id") or ""),
                    "policy_sha256": digest,
                    "policy_size_bytes": len(content),
                    "note": str(payload.get("note") or ""),
                }
                result = submit_worker_policy_to_cpu_agent(run, receipt, content)
                raw_returncode = (
                    result.get("returncode") if isinstance(result, dict) else None
                )
                if isinstance(raw_returncode, bool) or not isinstance(
                    raw_returncode, int
                ):
                    raise RuntimeError(
                        "CPU archive response omitted an integer returncode"
                    )
                record = {
                    "schema_version": 1,
                    "run_id": str(run["run_id"]),
                    "submission_id": submission_id,
                    "queue_name": f"{submission_id}.pt",
                    "gpu_job_id": str(payload["job_id"]),
                    "gpu_attempt": int(payload.get("attempt") or 0),
                    "policy_sha256": digest,
                    "policy_size_bytes": len(content),
                    "observed_at": utc_now(),
                    "source": "enqueue_snapshot_at_budget_stop",
                    "archive_returncode": raw_returncode,
                    "archive_stdout": str(result.get("stdout") or "")[-2000:],
                    "archive_stderr": str(result.get("stderr") or "")[-2000:],
                }
                if raw_returncode == 0:
                    record.update({"state": "forwarded", "forwarded_at": utc_now()})
                    sprintctl.atomic_write_json(record_path, record, mode=0o600)
                else:
                    # A structural rejection is a complete submission result,
                    # not a half-forwarded bridge record. Keeping it out of the
                    # forwarding ledger lets integrity distinguish the two.
                    record["state"] = "rejected"

            state = "staged" if record.get("state") == "forwarded" else "rejected"
            forwarded += state == "staged"
            rejected += state == "rejected"
            result_payload = {
                "path": path,
                "state": state,
                "submission_id": submission_id,
                "policy_sha256": digest,
                "policy_size_bytes": len(content),
                "source": "enqueue_snapshot_at_budget_stop",
            }
            if state == "rejected":
                result_payload["reason"] = str(
                    record.get("archive_stderr")
                    or record.get("archive_stdout")
                    or "policy failed structural admission"
                )[-1000:]
            results.append(result_payload)

        progress = dict(payload.get("progress") or {})
        progress["submission_results"] = results
        payload.update(
            {
                "progress": progress,
                "submission_enqueue_snapshot_recovered_at": utc_now(),
                "submission_enqueue_snapshot_recovery_pending": False,
                "work_archive_provenance": "host-pinned-enqueue-snapshot-at-budget-stop",
            }
        )
        payload.pop("submission_enqueue_snapshot_recovery_error", None)
        payload = persist_job(run, payload)
        return payload, {
            "eligible": True,
            "forwarded": forwarded,
            "rejected": rejected,
            "error": 0,
            "retry_wait": 0,
        }
    except Exception as exc:  # noqa: BLE001
        payload["submission_enqueue_snapshot_recovery_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        payload = persist_job(run, payload)
        return payload, {
            "eligible": True,
            "forwarded": 0,
            "rejected": 0,
            "error": 0,
            "retry_wait": 1,
        }


def restore_pinned_work_archive(run: dict[str, Any], job: dict[str, Any]) -> None:
    """Restore the controller snapshot before every attempt allocation."""
    job_id = str(job["job_id"])
    canonical = host_work_archive_path(run, job_id)
    expected = str(job.get("work_archive_sha256") or "")
    if canonical is None or not canonical.is_file() or not expected:
        raise RuntimeError("missing host-pinned GPU work archive")
    if _file_sha256(canonical) != expected:
        raise RuntimeError("host-pinned GPU work archive digest mismatch")
    last_error: Exception | None = None
    for attempt in range(1, WORK_ARCHIVE_TRANSFER_ATTEMPTS + 1):
        try:
            sprintctl.volume_upload(
                run, canonical, _validated_work_archive_remote(run, job)
            )
            return
        except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
            last_error = exc
            if attempt < WORK_ARCHIVE_TRANSFER_ATTEMPTS:
                time.sleep(work_archive_retry_delay(attempt))
    assert last_error is not None
    raise RuntimeError(
        "unable to restore host-pinned GPU work archive after "
        f"{WORK_ARCHIVE_TRANSFER_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def load_host_job(run: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    path = host_job_path(run, job_id)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def list_host_job_ids(run: dict[str, Any]) -> list[str]:
    path = host_job_path(run, "placeholder")
    if path is None:
        return []
    return sorted(item.stem for item in path.parent.glob("*.json"))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    """Normalize Python launchers to the interpreter present in the image.

    The pip-installed Isaac Lab tree ships ``isaaclab.sh``, but its ``-p``
    branch shells out to a bare ``python`` executable.  The sealed training
    image intentionally exposes only ``python3``.  Treat that standard Isaac
    Lab spelling as an interpreter alias too so an otherwise valid agent job
    cannot burn an A10 allocation before Python starts.
    """
    command = list(job.get("command") or [])
    normalized: list[str] | None = None
    if command and command[0] == "python":
        normalized = ["python3", *command[1:]]
    elif (
        len(command) >= 3
        and Path(command[0]).name == "isaaclab.sh"
        and command[1] == "-p"
    ):
        normalized = ["python3", *command[2:]]
    if normalized is not None:
        job = dict(job)
        job["command"] = normalized
    return job


def load_job(
    run: dict[str, Any],
    job_id: str,
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Load the canonical host record or an unclaimed enqueue snapshot.

    Once claimed, the host registry wins over every agent-visible mirror, so a
    stale delivery mirror cannot respawn the job. Before claim, the job embedded
    in the exact agent-published index is authoritative.
    """
    # Once claimed, the host registry is authoritative.  Volume status and
    # queue files are agent-visible delivery mirrors and may disappear before
    # the controller has archived terminal provider output or copied the final
    # policy back into the CPU sandbox.
    local = load_host_job(run, job_id)
    if local is not None:
        return local
    if indexed is None:
        indexed = indexed_agent_jobs(run)
    if indexed is not None:
        embedded = indexed.get(job_id, {}).get("job")
        if embedded is not None:
            if (
                not isinstance(embedded, dict)
                or embedded.get("job_id") != job_id
                or embedded.get("run_id") != str(run["run_id"])
            ):
                raise RuntimeError("agent GPU dispatch index job identity mismatch")
            return dict(embedded)
    return None


def load_job_from_index_snapshot(
    run: dict[str, Any],
    job_id: str,
    indexed: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Load a job while reusing one exact-index snapshot for this tick."""
    if indexed is None:
        return load_job(run, job_id)
    return load_job(run, job_id, indexed=indexed)


def load_remote_json(run: dict[str, Any], remote_path: str) -> dict[str, Any] | None:
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


def load_heartbeat(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any] | None:
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
    image = training_image(run)
    volume = modal.Volume.from_name(str(run["volume_name"]))
    sandbox = modal.Sandbox.create(
        "bash",
        "-c",
        "while [ ! -e "
        + shlex.quote(f"/durable/runs/{run['run_id']}/BUDGET_STOP_REQUESTED.json")
        + " ]; do sleep 2; done",
        app=app,
        image=image,
        gpu="A10G",
        cpu=6,
        memory=12288,
        env={"HEADLESS": "1"},
        block_network=True,
        timeout=int(run.get("sandbox_timeout_seconds") or 86400),
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
    return {
        "sandbox_id": new_id,
        "action": "replaced" if prev else "created",
        "replaced": prev,
    }


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
        "bash",
        "-c",
        "deadline=$((SECONDS + 120)); "
        f"while [ ! -s {shlex.quote(GPU_BUDGET_MIRROR_PATH)} ]; do "
        'if [ "$SECONDS" -ge "$deadline" ]; then '
        "echo 'trusted GPU budget mirror unavailable' >&2; exit 78; fi; "
        "sleep 1; done; exec python3 /opt/sprint-gpu-worker-run.py "
        + shlex.quote(str(run["run_id"]))
        + " "
        + shlex.quote(str(job["job_id"]))
        + " "
        + shlex.quote(str(int(job["attempt"])))
        + " "
        + shlex.quote(str(job["lease_id"]))
        + " "
        + shlex.quote(str(int(job.get("fence_epoch") or 0))),
    )
    return sid


def spawn_gpu_sandbox(run: dict[str, Any], job: dict[str, Any]) -> str:
    import modal

    run_id = str(run["run_id"])
    job_id = str(job["job_id"])
    attempt = int(job["attempt"])
    lease_id = str(job["lease_id"])
    fence_epoch = int(job.get("fence_epoch") or 0)
    app = modal.App.lookup(
        str(run.get("training_app_name") or run["app_name"]),
        create_if_missing=True,
    )
    image = training_image(run)
    volume = modal.Volume.from_name(str(run["volume_name"]))
    command_timeout = int(job.get("timeout_sec") or 3600)
    command_timeout = max(60, min(command_timeout, 24 * 60 * 60))
    sandbox_timeout = (
        command_timeout + GPU_SANDBOX_STARTUP_FINALIZATION_ALLOWANCE_SEC
    )
    worker_command = (
        "python3 /opt/sprint-gpu-worker-run.py "
        + shlex.quote(run_id)
        + " "
        + shlex.quote(job_id)
        + " "
        + shlex.quote(str(attempt))
        + " "
        + shlex.quote(lease_id)
        + " "
        + shlex.quote(str(fence_epoch))
    )
    command = (
        "deadline=$((SECONDS + 120)); "
        f"while [ ! -s {shlex.quote(GPU_BUDGET_MIRROR_PATH)} ]; do "
        'if [ "$SECONDS" -ge "$deadline" ]; then '
        "echo 'trusted GPU budget mirror unavailable' >&2; exit 78; fi; "
        "sleep 1; done; exec " + worker_command
    )
    sandbox = modal.Sandbox.create(
        "bash",
        "-c",
        command,
        app=app,
        image=image,
        gpu="A10G",
        cpu=6,
        memory=12288,
        env={"HEADLESS": "1"},
        block_network=True,
        timeout=sandbox_timeout,
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


def training_image(run: dict[str, Any]):
    """Resolve the exact image exercised by the launch warm-up gate."""
    import modal

    provenance = run.get("evaluation_provenance") or {}
    image_id = provenance.get("agent_training_image_id")
    if not isinstance(image_id, str) or not re.fullmatch(r"im-[A-Za-z0-9]+", image_id):
        raise RuntimeError("run has no valid warmed agent/training Modal image ID")
    return modal.Image.from_id(image_id)


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
            return ProbeResult(ProbeState.UNKNOWN, error=f"{type(exc).__name__}: {exc}")
        if code is None:
            return ProbeResult(ProbeState.ALIVE)
        return ProbeResult(ProbeState.EXITED, exit_code=int(code))

    def terminate(self, handle: ProviderHandle) -> str | None:
        try:
            import modal

            # Submit the idempotent termination first, then confirm it with a
            # bounded poll. Modal's wait=True path can remain blocked after an
            # already-finished Sandbox disappears from the app container list;
            # an unbounded provider wait must never stall the run's budget
            # teardown state machine.
            sandbox = modal.Sandbox.from_id(handle.attempt_id)
            sandbox.terminate(wait=False)
            deadline = time.monotonic() + STOP_PROVIDER_CONFIRM_TIMEOUT_SEC
            while sandbox.poll() is None:
                if time.monotonic() >= deadline:
                    return (
                        "TimeoutError: Modal did not confirm Sandbox termination "
                        f"within {STOP_PROVIDER_CONFIRM_TIMEOUT_SEC:.0f}s"
                    )
                time.sleep(STOP_PROVIDER_CONFIRM_POLL_SEC)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            normalized = error.lower()
            if any(
                marker in normalized
                for marker in (
                    "already shut down",
                    "already terminated",
                    "container is not running",
                    "task has already finished",
                    "sandbox not found",
                )
            ):
                return None
            return error
        return None


def read_modal_sandbox_output(sandbox_id: str) -> tuple[str, str]:
    """Read complete stdout/stderr after a Modal Sandbox has exited."""
    import modal

    sandbox = modal.Sandbox.from_id(sandbox_id)
    if sandbox.poll() is None:
        raise RuntimeError(f"sandbox {sandbox_id} is still running")
    return str(sandbox.stdout.read() or ""), str(sandbox.stderr.read() or "")


def read_modal_sandbox_live_output(run: dict[str, Any], sandbox_id: str) -> str:
    """Read a bounded, non-blocking tail from a running Modal Sandbox."""
    result = sprintctl.run_command(
        sprintctl.modal_command(
            "container",
            "logs",
            sandbox_id,
            "--tail",
            str(LIVE_PROVIDER_LOG_TAIL_LINES),
        ),
        run=run,
        check=False,
        timeout=45,
    )
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "Modal logs unavailable").strip()
        raise RuntimeError(error[-1000:])
    return result.stdout or ""


def refresh_live_provider_logs(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    now: float | None = None,
    read_output=read_modal_sandbox_live_output,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror bounded provider output while a GPU attempt is still running.

    The terminal archive remains the complete, checksummed source of record.
    This tail exists so the agent can diagnose progress and failures without
    sleeping blindly until the Sandbox exits.
    """
    ref = time.time() if now is None else float(now)
    sandbox_id = str(job.get("sandbox_id") or "")
    attempt = int(job.get("attempt") or 0)
    if not sandbox_id or attempt <= 0:
        return job, {"live_provider_logs": "unavailable"}
    last_checked = float(job.get("provider_live_logs_checked_at_epoch_s") or 0)
    if ref - last_checked < LIVE_PROVIDER_LOG_INTERVAL_SEC:
        return job, {"live_provider_logs": "fresh"}

    payload = dict(job)
    payload["provider_live_logs_checked_at_epoch_s"] = ref
    try:
        text = read_output(run, sandbox_id)
    except Exception as exc:  # noqa: BLE001
        payload["provider_live_logs_error"] = f"{type(exc).__name__}: {exc}"
        return payload, {
            "live_provider_logs": "error",
            "live_provider_logs_error": payload["provider_live_logs_error"],
        }

    encoded = ("== Modal live log tail ==\n" + text).encode("utf-8", errors="replace")
    payload.pop("provider_live_logs_error", None)
    payload.update(
        {
            "provider_live_logs_mirrored_at": utc_now(),
            "provider_live_logs_size_bytes": len(encoded),
            "provider_live_logs_sha256": hashlib.sha256(encoded).hexdigest(),
            "provider_live_logs_source": "modal-sandbox-tail",
        }
    )
    mirror_detail = mirror_agent_job(run, payload, log_content=encoded)
    if mirror_detail.get("agent_mirror") != "updated":
        payload["provider_live_logs_error"] = str(
            mirror_detail.get("agent_mirror_error") or "agent mirror unavailable"
        )
        return payload, {"live_provider_logs": "mirror_error", **mirror_detail}
    return payload, {
        "live_provider_logs": "mirrored",
        "live_provider_logs_size_bytes": len(encoded),
        **mirror_detail,
    }


def provider_terminal_error(stream_text: str) -> str | None:
    """Return a definitive child failure marker hidden by a zero wrapper exit.

    Isaac/Kit can occasionally finish its outer application with status zero
    after Python emitted an unhandled exception or Kit failed before starting
    the application. Generic ``[Error]`` lines are not sufficient because
    headless Vulkan initialization emits recoverable diagnostics, but these
    semantic startup failures are terminal.
    """
    # Archived logs preserve both provider streams. Only stderr is an
    # authoritative source for these process/runtime markers: agent programs
    # may legitimately print source code or diagnostics containing the same
    # words to stdout. Modal keeps the streams separate at collection time.
    stderr_header = "== Modal stderr ==\n"
    if stderr_header in stream_text:
        stream_text = stream_text.rsplit(stderr_header, 1)[1]

    if "Traceback (most recent call last):" in stream_text:
        tail = stream_text[stream_text.rfind("Traceback (most recent call last):") :]
        final = next(
            (line.strip() for line in reversed(tail.splitlines()) if line.strip()),
            "unhandled Python exception",
        )
        return final[-1000:]
    for marker in (
        "Failed to resolve extension dependencies",
        "Failed to startup python app",
        "ModuleNotFoundError:",
        "GPU solver pipeline failed",
        "GPU Bp pipeline failed",
        "switching to software",
    ):
        if marker not in stream_text:
            continue
        line = next(
            (
                item.strip()
                for item in reversed(stream_text.splitlines())
                if marker in item
            ),
            marker,
        )
        return line[-1000:]
    return None


PROVIDER_TERMINAL_ERROR_CLASSIFIER_VERSION = 2


def provider_terminal_error_for_job(
    job: dict[str, Any], stdout: str, stderr: str
) -> str | None:
    """Classify terminal provider output using the trusted worker outcome.

    Headless Isaac routinely emits Vulkan device diagnostics and then runs a
    CUDA simulation successfully.  Missing declared outputs do not make those
    diagnostics terminal: an agent can simply write a different filename.  A
    genuine AppLauncher abort is classified by the worker's trusted sidecar,
    while archived provider streams are only used for semantic terminal errors
    such as an unhandled traceback or dependency-resolution failure.
    """
    del job, stdout
    return provider_terminal_error(stderr)


def split_archived_provider_streams(stream_text: str) -> tuple[str, str]:
    """Recover Modal's separate streams from the canonical archived envelope."""
    stdout_header = "== Modal stdout ==\n"
    stderr_header = "== Modal stderr ==\n"
    if stdout_header in stream_text and stderr_header in stream_text:
        stdout, stderr = stream_text.split(stdout_header, 1)[1].rsplit(stderr_header, 1)
        return stdout, stderr
    return "", stream_text


def apply_provider_terminal_error(
    job: dict[str, Any], terminal_error: str | None
) -> dict[str, Any]:
    payload = dict(job)
    payload["provider_terminal_error_checked_at"] = utc_now()
    payload["provider_terminal_error_classifier_version"] = (
        PROVIDER_TERMINAL_ERROR_CLASSIFIER_VERSION
    )
    payload["provider_terminal_error_detected"] = bool(terminal_error)
    if terminal_error:
        payload["provider_terminal_error"] = terminal_error
    else:
        payload.pop("provider_terminal_error", None)
    worker_error = str(payload.get("error") or "").strip()
    if worker_error and str(payload.get("status") or "") == "succeeded":
        payload["worker_reported_status"] = "succeeded"
        payload["worker_reported_exit_code"] = payload.get("exit_code")
        payload["status"] = "failed"
        if not payload.get("exit_code"):
            payload["exit_code"] = 1
        payload["failure_reason"] = "worker_reported_error"
    if terminal_error and str(payload.get("status") or "") == "succeeded":
        payload["provider_reported_status"] = "succeeded"
        payload["provider_reported_exit_code"] = payload.get("exit_code")
        payload["status"] = "failed"
        payload["exit_code"] = 1
        payload["error"] = terminal_error
        payload["failure_reason"] = "provider_stream_terminal_error"
    return payload


def audit_archived_provider_logs(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Classify logs archived before terminal-stream auditing was enabled."""
    path = str(job.get("provider_logs_path") or "")
    if not path:
        return job, {"provider_logs": "audit_unavailable"}
    text = sprintctl.volume_get_text(run, path)
    if text is None:
        return job, {"provider_logs": "audit_retry", "provider_logs_path": path}
    stdout, stderr = split_archived_provider_streams(text)
    terminal_error = provider_terminal_error_for_job(job, stdout, stderr)
    payload = apply_provider_terminal_error(job, terminal_error)
    payload, artifact_name, artifact_content, artifact_detail = (
        fetch_agent_output_artifact(run, payload)
    )
    mirror_detail = mirror_agent_job(
        run,
        payload,
        log_content=text.encode("utf-8", errors="replace"),
        artifact_name=artifact_name,
        artifact_content=artifact_content,
    )
    return payload, {
        "provider_logs": "audited",
        "provider_terminal_error": terminal_error,
        **artifact_detail,
        **mirror_detail,
    }


def archive_provider_logs(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    read_output=read_modal_sandbox_output,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy terminal provider streams into the agent-visible durable log.

    Older worker images tee Python wrapper messages but child processes inherit
    the container file descriptors directly.  Modal therefore retains their
    output while ``event gpu logs`` sees only the wrapper.  This host-side
    fallback archives both streams after exit and also preserves them for the
    final artifact bundle.  Newer images may tee the child directly; replacing
    the log with the complete provider streams is idempotent in either case.
    """
    if job.get("provider_logs_archived_at"):
        return job, {"provider_logs": "already_archived"}
    sandbox_id = str(job.get("sandbox_id") or job.get("last_sandbox_id") or "")
    attempt = int(job.get("attempt") or 0)
    job_id = str(job.get("job_id") or "")
    if not sandbox_id or attempt <= 0 or not job_id:
        return job, {"provider_logs": "unavailable"}
    try:
        stdout, stderr = read_output(sandbox_id)
        content = (
            "== Modal stdout ==\n"
            + stdout
            + ("\n" if stdout and not stdout.endswith("\n") else "")
            + "== Modal stderr ==\n"
            + stderr
            + ("\n" if stderr and not stderr.endswith("\n") else "")
        )
        encoded = content.encode("utf-8", errors="replace")
        with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as handle:
            handle.write(encoded)
            tmp = Path(handle.name)
        try:
            remote_path = (
                f"{jobs_prefix(str(run['run_id']))}/out/{job_id}/"
                f"attempt-{attempt}/worker.log"
            )
            sprintctl.volume_upload(run, tmp, remote_path)
        finally:
            tmp.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        payload = dict(job)
        attempts = int(payload.get("provider_logs_archive_attempts") or 0) + 1
        payload["provider_logs_archive_error"] = f"{type(exc).__name__}: {exc}"
        payload["provider_logs_archive_attempts"] = attempts
        payload["provider_logs_archive_retry_after_epoch_s"] = time.time() + min(
            15 * 60, 30 * (2 ** min(attempts - 1, 5))
        )
        return payload, {
            "provider_logs": "error",
            "provider_logs_error": payload["provider_logs_archive_error"],
        }
    payload = dict(job)
    payload.pop("provider_logs_archive_error", None)
    payload.pop("provider_logs_archive_retry_after_epoch_s", None)
    payload.update(
        {
            "provider_logs_archived_at": utc_now(),
            "provider_logs_path": remote_path,
            "provider_logs_sha256": hashlib.sha256(encoded).hexdigest(),
            "provider_logs_size_bytes": len(encoded),
            "provider_logs_source": "modal-sandbox-streams",
        }
    )
    terminal_error = provider_terminal_error_for_job(payload, stdout, stderr)
    payload = apply_provider_terminal_error(payload, terminal_error)
    payload, artifact_name, artifact_content, artifact_detail = (
        fetch_agent_output_artifact(run, payload)
    )
    mirror_detail = mirror_agent_job(
        run,
        payload,
        log_content=encoded,
        artifact_name=artifact_name,
        artifact_content=artifact_content,
    )
    return payload, {
        "provider_logs": "archived",
        "provider_logs_size_bytes": len(encoded),
        "provider_terminal_error": terminal_error,
        **artifact_detail,
        **mirror_detail,
    }


def persist_host_job(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Durably write only the host-authoritative GPU job record."""
    job_id = str(job["job_id"])
    local_path = host_job_path(run, job_id)
    if local_path is not None:
        previous = load_host_job(run, job_id) or {}
        keys = (
            "status",
            "attempt",
            "claim_id",
            "lease_id",
            "fence_epoch",
            "sandbox_id",
            "retry_reason",
            "termination_reason",
            "heartbeat_at_epoch_s",
            "updated_at_epoch_s",
            "finished_at_epoch_s",
        )
        before = {key: previous.get(key) for key in keys if key in previous}
        after = {key: job.get(key) for key in keys if key in job}
        if before != after:
            command = job.get("command")
            command_sha256 = (
                hashlib.sha256(
                    json.dumps(command, separators=(",", ":")).encode()
                ).hexdigest()
                if isinstance(command, list)
                else None
            )
            append_control_event(
                run,
                "gpu_job_state_persisted",
                request={"job_id": job_id},
                previous=before,
                current=after,
                command_sha256=command_sha256,
            )
        sprintctl.atomic_write_json(local_path, job, mode=0o600)
    return job


def persist_job(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Write the host canonical record, then agent-visible Volume mirrors."""
    prefix = jobs_prefix(str(run["run_id"]))
    persist_host_job(run, job)
    if terminal_cpu_unavailable(run):
        return job
    job_id = str(job["job_id"])
    put_json(run, f"{prefix}/status/{job_id}.json", job)
    mirror_agent_job(run, job)
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
    action = gpu_claim.select_claim_action(job, claim_id=claim_id, stale_sec=stale_sec)
    if action == "skip":
        return None
    payload = gpu_claim.build_claim_payload(job, claim_id=claim_id)
    persist_job(run, payload)
    # Confirm we still own after the write (best-effort against races).
    refreshed = load_job(run, str(job["job_id"]))
    if not gpu_claim.ownership_matches(refreshed, claim_id):
        return None
    return refreshed or payload


def release_abandoned_claim(
    run: dict[str, Any], job: dict[str, Any], *, now: float | None = None
) -> dict[str, Any]:
    """Fence a pre-spawn claim whose owning controller process has died.

    The orphan-sandbox audit runs before reconciliation while holding the same
    dispatch lock.  Therefore, by the time this is called, a Sandbox created in
    the narrow crash window but not durably published has already been stopped.
    Restoring the exact pre-claim state does not consume a worker attempt.
    """
    if str(job.get("status") or "") != "claiming" or job.get("sandbox_id"):
        return job
    payload = dict(job)
    restore = payload.get("claim_restore")
    if not isinstance(restore, dict) or str(restore.get("status") or "") not in {
        "pending",
        "retry_wait",
    }:
        return job
    lease_id = str(payload.get("lease_id") or "")
    recorded_at = time.time() if now is None else float(now)
    history = list(payload.get("abandoned_claim_history") or [])
    history.append(
        {
            "claim_id": payload.get("claim_id"),
            "lease_id": lease_id or None,
            "attempt": payload.get("attempt"),
            "owner_process": payload.get("claim_owner_process"),
            "abandoned_at": utc_now(),
            "abandoned_at_epoch_s": recorded_at,
        }
    )
    payload["abandoned_claim_history"] = history
    payload["fence_epoch"] = int(payload.get("fence_epoch") or 0) + 1
    payload["fenced_lease_id"] = lease_id or None
    for key in (
        "claim_id",
        "lease_id",
        "claim_owner",
        "claim_owner_process",
        "claim_restore",
        "claimed_at",
        "claimed_at_epoch_s",
        "attempt",
        "next_attempt",
        "retry_not_before",
        "retry_not_before_epoch_s",
        "retry_reason",
    ):
        payload.pop(key, None)
    payload.update(restore)
    return persist_job(run, payload)


def _timeline_event(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    phase: str,
    action: str,
    epoch_s: int | None = None,
    upload: bool = True,
    **detail: Any,
) -> None:
    try:
        from event_runtime.telemetry import timeline as gpu_timeline_host

        gpu_timeline_host.host_append_event(
            run,
            phase=phase,
            action=action,
            job_id=str(job["job_id"]),
            attempt=int(job.get("attempt") or 0),
            lease_id=str(job.get("lease_id") or ""),
            epoch_s=epoch_s,
            detail=detail,
            upload=upload,
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
    upload: bool = True,
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
            upload=upload,
        )


def _terminate_sandbox(job: dict[str, Any]) -> str | None:
    sandbox_id = str(job.get("sandbox_id") or "")
    if not sandbox_id:
        return None
    return ModalSandboxProvider({}).terminate(
        ProviderHandle(provider=ModalSandboxProvider.name, attempt_id=sandbox_id)
    )


def finalize_lost_job(
    run: dict[str, Any],
    job: dict[str, Any],
    *,
    heartbeat: dict[str, Any] | None,
    exit_code: int | None,
    reason: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence a lost worker and publish a terminal, agent-visible outcome.

    GPU jobs are immutable single attempts. The controller never reconstructs
    missing model state or silently replays agent-authored work; the agent may
    submit a new job explicitly after inspecting this terminal record.
    """
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
        event="gpu_preempted"
        if reason in {"graceful_preemption", "worker_lost"}
        else "gpu_attempt_lost",
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

    was_preempted = reason in {
        "graceful_preemption",
        "worker_lost",
        "app_launcher_initialization_failed",
    }
    payload.update(
        {
            "status": "preempted" if was_preempted else "failed",
            "failure_reason": reason,
            "retry_policy": "agent_decides_new_job",
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

    if status == "claiming" and not job.get("sandbox_id"):
        owner_state = gpu_claim.process_identity_state(job.get("claim_owner_process"))
        if owner_state == "dead":
            released = release_abandoned_claim(run, job, now=ref)
            if released is not job:
                _timeline_event(
                    run,
                    job,
                    phase="gpu_worker_starting",
                    action="exit",
                    epoch_s=int(ref),
                    reason="claim_owner_died_before_spawn",
                    synthetic=True,
                )
                _timeline_event(
                    run,
                    job,
                    phase="gpu_lifecycle",
                    action="instant",
                    epoch_s=int(ref),
                    event="gpu_claim_abandoned",
                    reason="claim_owner_died_before_spawn",
                )
                return released, {
                    "decision": "abandoned_claim_released",
                    "status": released.get("status"),
                    "fenced_lease_id": released.get("fenced_lease_id"),
                }

    attempt_record = load_attempt_record(run, job)
    heartbeat = load_heartbeat(run, job)
    record_heartbeat_observation(run, job, heartbeat)
    before_submission_bridge = job
    job, submission_bridge_detail = drain_worker_submission_outbox(run, job)
    if job != before_submission_bridge:
        persist_job(run, job)
    owned_record = bool(
        attempt_record
        and int(attempt_record.get("attempt") or 0) == int(job.get("attempt") or 0)
        and str(attempt_record.get("lease_id") or "") == str(job.get("lease_id") or "")
    )
    if owned_record and str(attempt_record.get("status") or "") in gpu_claim.TERMINAL:
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
                    "termination_reason",
                    "progress",
                    "checkpoint",
                }
            }
        )
        terminal["attempt_record"] = attempt_path(
            str(run["run_id"]), str(job["job_id"]), int(job["attempt"])
        )
        if terminal_submission_bridge_complete(
            run, terminal, submission_bridge_detail
        ):
            terminal["submission_bridge_terminal_drained_at"] = utc_now()
        terminal, log_detail = archive_provider_logs(run, terminal)
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
        return terminal, {
            "decision": "terminal",
            "status": terminal["status"],
            **submission_bridge_detail,
            **log_detail,
        }

    if owned_record and str(attempt_record.get("status") or "") == "interrupted":
        retry_reason = str(attempt_record.get("retry_reason") or "graceful_preemption")
        terminal = finalize_lost_job(
            run,
            job,
            heartbeat=heartbeat,
            exit_code=attempt_record.get("exit_code"),
            reason=retry_reason,
            now=ref,
        )
        return terminal, {
            "decision": "preempted",
            "status": terminal["status"],
            "reason": retry_reason,
            **submission_bridge_detail,
        }

    if owned_record and str(attempt_record.get("status") or "") == "running":
        if status != "running":
            job = dict(job)
            job["status"] = "running"
            job["started_at"] = attempt_record.get("started_at")
            job["started_at_epoch_s"] = attempt_record.get("started_at_epoch_s")
            persist_job(run, job)
        job, live_policy_detail = refresh_live_policy_mirror(run, job, heartbeat)
    else:
        live_policy_detail = {"live_policy_mirror": "worker_not_running"}

    probe_state, exit_code, probe_error = probe_fn(job)
    if probe_state == "exited" and not job.get("provider_exit_observed_epoch_s"):
        # Modal's provider poll is authoritative that this sandbox is no
        # longer billable even when the worker's terminal Volume commit is
        # still propagating. Persist a host-owned upper billing boundary now;
        # artifact/retry semantics continue to wait for the exact attempt
        # record below. Without this boundary the live ledger bills the worker
        # through the full Volume visibility grace, then drops by minutes when
        # the earlier exact terminal timestamp finally arrives.
        observed = dict(job)
        observed["provider_exit_observed_epoch_s"] = ref
        observed["provider_exit_code"] = exit_code
        persist_job(run, observed)
        _timeline_event(
            run,
            observed,
            phase="gpu_lifecycle",
            action="instant",
            epoch_s=int(ref),
            event="gpu_provider_exit_observed",
            exit_code=exit_code,
            lifecycle_boundary="provider_poll_exited",
        )
        job = observed
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
        **submission_bridge_detail,
        **live_policy_detail,
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
        before_live_logs = job
        job, live_log_detail = refresh_live_provider_logs(run, job, now=ref)
        if job != before_live_logs:
            persist_job(run, job)
        detail.update(live_log_detail)
        return job, detail
    terminal = finalize_lost_job(
        run,
        job,
        heartbeat=heartbeat,
        exit_code=exit_code,
        reason="worker_lost",
        now=ref,
    )
    detail["decision"] = "preempted"
    detail["status"] = terminal["status"]
    return terminal, detail


def reconcile_terminal_attempt_before_stop(
    run: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    """Commit an already-finished owned attempt before applying a stop fence."""
    job_status = str(job.get("status") or "")
    if job_status not in gpu_claim.OWNED | {"terminated"}:
        return job
    try:
        attempt_record = load_attempt_record(run, job)
    except Exception:  # noqa: BLE001 - stop must still fence on read outage
        return job
    if not attempt_record:
        return job
    if (
        int(attempt_record.get("attempt") or 0) != int(job.get("attempt") or 0)
        or str(attempt_record.get("lease_id") or "") != str(job.get("lease_id") or "")
        or str(attempt_record.get("status") or "") not in gpu_claim.TERMINAL
    ):
        return job
    if job_status == "terminated":
        try:
            terminated_at = float(
                job.get("terminated_at_epoch_s") or job.get("finished_at_epoch_s") or 0
            )
            attempt_finished_at = float(attempt_record.get("finished_at_epoch_s") or 0)
        except (TypeError, ValueError):
            return job
        # The fence won a real race if it was issued before the worker's final
        # record. Only repair the historical batch-stop overwrite case where
        # the immutable attempt had already completed.
        if (
            not terminated_at
            or not attempt_finished_at
            or terminated_at < attempt_finished_at
        ):
            return job
    payload = dict(job)
    for key in (
        "status",
        "started_at",
        "started_at_epoch_s",
        "finished_at",
        "finished_at_epoch_s",
        "exit_code",
        "error",
        "termination_reason",
        "progress",
        "checkpoint",
    ):
        if key not in attempt_record:
            continue
        if key == "progress":
            # A worker record can finish before its submission manifest reaches
            # durable storage. Stop-time recovery then writes newer host-owned
            # submission evidence into the registry. Never let the older
            # attempt record erase that evidence with a missing/null manifest.
            host_progress = payload.get("progress")
            host_results = (
                host_progress.get("submission_results")
                if isinstance(host_progress, dict)
                else None
            )
            worker_progress = attempt_record.get("progress")
            worker_results = (
                worker_progress.get("submission_results")
                if isinstance(worker_progress, dict)
                else None
            )
            if isinstance(host_results, list) and not isinstance(
                worker_results, list
            ):
                continue
        payload[key] = attempt_record[key]
    payload["attempt_record"] = attempt_path(
        str(run["run_id"]), str(job["job_id"]), int(job["attempt"])
    )
    return persist_job(run, payload)


def list_agent_cancelled_job_ids(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Return jobs explicitly hidden by an agent before host dispatch.

    The agent-facing queue is intentionally writable.  Some agents cancel a
    queued job by atomically renaming both delivery records to
    ``.cancelled-<job>.json``.  The append-only host registry must retain the
    audit record, but must not revive that job merely because its old local
    status was pending or retryable.
    """
    if not str(run.get("volume_name") or "").strip():
        return []
    if indexed is None:
        indexed = indexed_agent_jobs(run)
    return sorted(
        job_id
        for job_id, detail in (indexed or {}).items()
        if detail.get("cancel_state") == "cancelled_before_dispatch"
    )


def reconcile_agent_cancelled_jobs(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fence undispatched host records whose agent delivery was cancelled."""
    reconciled: list[dict[str, Any]] = []
    for job_id in list_agent_cancelled_job_ids(run, indexed=indexed):
        job = load_host_job(run, job_id)
        if not job:
            continue
        status = str(job.get("status") or "")
        # Never infer cancellation for an allocated worker.  A running worker
        # requires the explicit terminate path so its lease and sandbox are
        # fenced together.
        if status not in {"pending", "retry_wait", "claiming"} or job.get("sandbox_id"):
            continue
        payload = dict(job)
        payload.update(
            {
                "status": "terminated",
                "termination_reason": "agent_cancelled_before_dispatch",
                "terminated_at": utc_now(),
                "terminated_at_epoch_s": time.time(),
                "fence_epoch": int(payload.get("fence_epoch") or 0) + 1,
                "fenced_lease_id": payload.get("lease_id"),
            }
        )
        persist_job(run, payload)
        reconciled.append(
            {
                "job_id": job_id,
                "attempt": payload.get("attempt"),
                "status": "terminated",
                "decision": "agent_cancelled_before_dispatch",
            }
        )
    return reconciled


def reconcile_live_agent_cancel_requests(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Persist, deliver, and acknowledge live agent cancellation requests.

    This is called while the dispatch lock is held. The durable host event is
    written before delivery. A running worker is first signalled through its
    private live control channel so it can retain declared output artifacts;
    provider termination is only a bounded fallback. The acknowledgement is
    written only after the job is terminal. An unacknowledged request is safe
    to replay after controller restart.
    """
    state_dir = Path(str(run["state_dir"]))
    ack_dir = state_dir / "control-acks"
    ack_dir.mkdir(parents=True, exist_ok=True)
    reconciled: list[dict[str, Any]] = []
    by_request_id = {
        str(item["request_id"]): item
        for item in read_durable_agent_cancel_requests(run, indexed=indexed)
    }
    try:
        for item in read_live_agent_cancel_requests(run):
            by_request_id[str(item["request_id"])] = item
    except RuntimeError:
        # A transient exec failure cannot erase the durable fallback request.
        pass
    for request in by_request_id.values():
        request_id = str(request["request_id"])
        job_id = str(request["job_id"])
        ack_path = ack_dir / f"{request_id}.json"
        if ack_path.is_file():
            continue
        job = load_host_job(run, job_id)
        if job is None:
            append_control_event(run, "cancel_requested", request=request)
            outcome = "unknown_job"
            payload: dict[str, Any] = {
                "job_id": job_id,
                "run_id": str(run["run_id"]),
                "status": "unknown",
            }
        else:
            job = reconcile_terminal_attempt_before_stop(run, job)
            status = str(job.get("status") or "")
            if status in gpu_claim.TERMINAL:
                payload = job
                outcome = "already_terminal"
            elif not job.get("sandbox_id"):
                append_control_event(run, "cancel_requested", request=request)
                payload = _terminate_job_locked(
                    run, job_id, reason="agent_cancelled_before_dispatch"
                )
                outcome = "terminated"
            else:
                now = time.time()
                request_changed = str(job.get("cancel_request_id") or "") != request_id
                delivered_at = float(job.get("cancel_signal_delivered_at_epoch_s") or 0)
                if request_changed:
                    append_control_event(run, "cancel_requested", request=request)
                    payload = dict(job)
                    payload.update(
                        {
                            "cancel_request_id": request_id,
                            "cancel_requested_at": utc_now(),
                            "cancel_requested_at_epoch_s": now,
                        }
                    )
                    delivered_at = 0
                else:
                    payload = dict(job)
                if delivered_at <= 0:
                    try:
                        deliver_agent_cancel_to_gpu_worker(payload, request)
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        append_control_event(
                            run,
                            "cancel_delivery_failed",
                            request=request,
                            error=error,
                        )
                        payload["cancel_delivery_error"] = error
                        persist_job(run, payload)
                        reconciled.append(
                            {
                                "job_id": job_id,
                                "request_id": request_id,
                                "decision": "cancel_retry_required",
                                "error": error,
                            }
                        )
                        continue
                    delivered_at = time.time()
                    payload.update(
                        {
                            "cancel_signal_delivered_at": utc_now(),
                            "cancel_signal_delivered_at_epoch_s": delivered_at,
                            "cancel_force_after_epoch_s": (
                                delivered_at + GPU_AGENT_CANCEL_GRACE_SEC
                            ),
                        }
                    )
                    payload.pop("cancel_delivery_error", None)
                    persist_job(run, payload)
                    append_control_event(
                        run,
                        "cancel_delivered",
                        request=request,
                        sandbox_id=str(payload.get("sandbox_id") or ""),
                    )
                force_after = float(
                    payload.get("cancel_force_after_epoch_s")
                    or (delivered_at + GPU_AGENT_CANCEL_GRACE_SEC)
                )
                if time.time() < force_after:
                    reconciled.append(
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "status": payload.get("status"),
                            "decision": "agent_cancel_delivered",
                        }
                    )
                    continue
                payload = _terminate_job_locked(
                    run, job_id, reason="agent_cancelled_forced"
                )
                if payload.get("terminate_error"):
                    # Do not acknowledge. The next controller pass retries the
                    # still-persistent request after the provider recovers.
                    append_control_event(
                        run,
                        "cancel_delivery_failed",
                        request=request,
                        error=str(payload["terminate_error"]),
                    )
                    reconciled.append(
                        {
                            "job_id": job_id,
                            "request_id": request_id,
                            "decision": "cancel_retry_required",
                            "error": payload["terminate_error"],
                        }
                    )
                    continue
                outcome = "forced_terminated"
        acknowledgement = {
            "schema_version": 1,
            "request_id": request_id,
            "run_id": str(run["run_id"]),
            "job_id": job_id,
            "outcome": outcome,
            "status": payload.get("status"),
            "acknowledged_at": utc_now(),
            "acknowledged_at_epoch_s": time.time(),
        }
        sprintctl.atomic_write_json(ack_path, acknowledgement, mode=0o600)
        append_control_event(
            run,
            "cancel_acknowledged",
            request=request,
            outcome=outcome,
            status=payload.get("status"),
        )
        reconciled.append(
            {
                "job_id": job_id,
                "request_id": request_id,
                "status": payload.get("status"),
                "decision": f"agent_cancel_{outcome}",
            }
        )
    return reconciled


def list_job_ids(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    names = {f"{job_id}.json" for job_id in list_host_job_ids(run)}
    if indexed is None:
        indexed = indexed_agent_jobs(run)
    names.update(
        f"{job_id}.json"
        for job_id, detail in (indexed or {}).items()
        if detail.get("cancel_state") != "cancelled_before_dispatch"
    )
    return sorted(
        Path(name).stem
        for name in names
        if name.endswith(".json") and re.fullmatch(r"[A-Za-z0-9_-]+", Path(name).stem)
    )


def active_training_job_ids(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Return logical jobs that currently own the run's training GPU slot."""
    active: list[str] = []
    for job_id in list_job_ids(run, indexed=indexed):
        job = load_job_from_index_snapshot(run, job_id, indexed)
        if job and str(job.get("status") or "") in gpu_claim.OWNED:
            active.append(job_id)
    return active


def cleanup_orphaned_training_sandboxes(
    run: dict[str, Any],
    *,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Terminate live training sandboxes absent from every host job record.

    This is a provider-specific safety net behind the provider-neutral job
    registry.  It handles the narrow crash window where Modal accepted a
    Sandbox but the controller never durably published its ID.  The caller
    holds the per-run dispatch lock, so an unknown sandbox cannot be a
    concurrently-created legitimate worker.
    """
    app_name = str(run.get("training_app_name") or "").strip()
    if not app_name:
        return []
    expected: set[str] = set()
    for job_id in list_job_ids(run, indexed=indexed):
        job = load_job_from_index_snapshot(run, job_id, indexed)
        if not job:
            continue
        for key in ("sandbox_id", "last_sandbox_id"):
            sandbox_id = str(job.get(key) or "")
            if sandbox_id.startswith("sb-"):
                expected.add(sandbox_id)
    if standing_enabled(run):
        path = _standing_state_path(run)
        if path.is_file():
            try:
                standing_id = str(json.loads(path.read_text()).get("sandbox_id") or "")
            except (OSError, json.JSONDecodeError):
                standing_id = ""
            if standing_id.startswith("sb-"):
                expected.add(standing_id)

    try:
        import modal
    except Exception as exc:  # noqa: BLE001
        return [
            {"action": "orphan_audit_error", "error": f"{type(exc).__name__}: {exc}"}
        ]

    try:
        app = modal.App.lookup(app_name, create_if_missing=False)
        sandboxes = list(modal.Sandbox.list(app_id=str(app.app_id)))
    except modal.exception.NotFoundError:
        # The per-run training App is created lazily with the first GPU job.
        # Its absence before then means there can be no orphaned sandboxes.
        return []
    except Exception as exc:  # noqa: BLE001
        return [
            {"action": "orphan_audit_error", "error": f"{type(exc).__name__}: {exc}"}
        ]

    actions: list[dict[str, Any]] = []
    for sandbox in sandboxes:
        sandbox_id = str(sandbox.object_id)
        if sandbox_id in expected:
            continue
        error = ModalSandboxProvider(run).terminate(
            ProviderHandle(
                provider=ModalSandboxProvider.name,
                attempt_id=sandbox_id,
            )
        )
        actions.append(
            {
                "action": "orphan_terminated"
                if error is None
                else "orphan_terminate_error",
                "sandbox_id": sandbox_id,
                "error": error,
            }
        )
    return actions


def operator_stop_requested(state_dir: Path) -> bool:
    if (state_dir / "STOP").is_file() or (state_dir / "STOP_REQUESTED.json").is_file():
        return True
    ack = state_dir / "STOP_ACK.json"
    if not ack.is_file():
        return False
    try:
        return json.loads(ack.read_text()).get("reason") in {
            "operator_stop",
            "operator_batch_stop",
            "agent_cost_budget_exhausted",
            "budget_telemetry_unavailable",
        }
    except (OSError, json.JSONDecodeError):
        return True


def _stop_all_locked(
    run: dict[str, Any], *, reason: str = "operator_stop"
) -> list[dict[str, Any]]:
    stopped: list[dict[str, Any]] = []
    provider_terminations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    # The agent queue is immutable after the CPU stop signal. Read its index
    # exactly once: independently rereading the same Modal Volume snapshot for
    # every queued job makes stop latency grow by a control-plane round trip per
    # job and can delay the bounded CPU teardown by many minutes.
    indexed = indexed_agent_jobs(run)
    for job_id in list_job_ids(run, indexed=indexed):
        job = load_job(run, job_id, indexed=indexed)
        if job:
            job = reconcile_terminal_attempt_before_stop(run, job)
        if not job or str(job.get("status") or "") in gpu_claim.TERMINAL:
            continue
        heartbeat = load_heartbeat(run, job)
        payload = dict(job)
        # An undispatched explicit submission is already backed by the
        # immutable enqueue snapshot. Mark it for recovery before changing its
        # status so the slow archive transfer can happen after every lease has
        # been fenced and every provider sandbox has been stopped.
        if (
            str(payload.get("status") or "") == "pending"
            and payload.get("submission_paths")
            and payload.get("submission_bridge_enabled")
            and int(payload.get("attempt") or 0) == 0
            and not payload.get("sandbox_id")
            and not payload.get("started_at")
        ):
            payload["submission_enqueue_snapshot_recovery_pending"] = True
        payload["fence_epoch"] = int(payload.get("fence_epoch") or 0) + 1
        payload["fenced_lease_id"] = payload.get("lease_id")
        payload["status"] = "terminated"
        payload["termination_reason"] = reason
        payload["terminated_at"] = utc_now()
        payload["terminated_at_epoch_s"] = time.time()
        # Stop intent and the host registry are authoritative.  Do not make
        # teardown latency proportional to queue depth by synchronously
        # uploading a Volume status file and exec-mirroring every terminal
        # record into a CPU sandbox that has already received its stop signal.
        persist_host_job(run, payload)  # fence before issuing terminate
        _close_attempt_timeline(
            run,
            job,
            epoch_s=int(gpu_claim.heartbeat_epoch(heartbeat) or time.time()),
            reason=reason,
            upload=False,
        )
        _timeline_event(
            run,
            job,
            phase="gpu_lifecycle",
            action="instant",
            event="gpu_released",
            reason=reason,
            upload=False,
        )
        stopped.append(payload)
        provider_terminations.append((payload, job))

    # Every lease is durably fenced before any provider call starts. Remote
    # Sandbox termination can take tens of seconds even for an already-dead
    # worker, so doing it serially makes operator stop latency grow with the
    # agent's queue depth. Fan out the idempotent provider calls with a small,
    # fixed bound, then persist any errors on the controller thread.
    if provider_terminations:
        max_workers = min(STOP_TERMINATE_MAX_WORKERS, len(provider_terminations))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_terminate_sandbox, job)
                for _payload, job in provider_terminations
            ]
            for (payload, _job), future in zip(
                provider_terminations, futures, strict=True
            ):
                try:
                    error = future.result()
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                if error:
                    payload["terminate_error"] = error
                    persist_host_job(run, payload)
    return stopped


def stop_all(
    run: dict[str, Any], *, reason: str = "operator_stop"
) -> list[dict[str, Any]]:
    state_dir = Path(str(run["state_dir"]))
    # A Modal Sandbox.create call can legitimately hold this lock for longer
    # than the normal 30-second monitor budget.  The durable stop marker is
    # already present before this function is called, so wait for the in-flight
    # dispatch and then fence it instead of returning a misleading partial stop.
    with gpu_claim.dispatch_lock(
        state_dir, timeout_sec=STOP_DISPATCH_LOCK_TIMEOUT_SEC
    ) as got_lock:
        if not got_lock:
            raise RuntimeError("dispatch lock busy while stopping GPU workers")
        return _stop_all_locked(run, reason=reason)


def _terminate_job_locked(
    run: dict[str, Any], job_id: str, *, reason: str
) -> dict[str, Any]:
    """Fence and synchronously terminate one job while dispatch lock is held."""
    job = load_job(run, job_id) or {
        "job_id": job_id,
        "run_id": run["run_id"],
    }
    job = reconcile_terminal_attempt_before_stop(run, job)
    if str(job.get("status") or "") in gpu_claim.TERMINAL:
        return job
    payload = dict(job)
    payload["fence_epoch"] = int(payload.get("fence_epoch") or 0) + 1
    payload["fenced_lease_id"] = payload.get("lease_id")
    payload["status"] = "terminated"
    payload["termination_reason"] = reason
    payload["terminated_at"] = utc_now()
    payload["terminated_at_epoch_s"] = time.time()
    persist_job(run, payload)  # durable fence before provider-side termination
    _close_attempt_timeline(
        run,
        job,
        epoch_s=int(payload["terminated_at_epoch_s"]),
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
    return payload


def terminate_job(run: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Fence and stop one GPU worker; the CPU harness is untouched."""
    state_dir = Path(str(run["state_dir"]))
    with gpu_claim.dispatch_lock(state_dir) as got_lock:
        if not got_lock:
            raise RuntimeError("dispatch lock busy while terminating GPU worker")
        return _terminate_job_locked(run, job_id, reason="manual_gpu_terminate")


def _candidate_job_ids(
    run: dict[str, Any],
    *,
    now: float | None = None,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    out: list[tuple[float, str]] = []
    for job_id in list_job_ids(run, indexed=indexed):
        job = load_job_from_index_snapshot(run, job_id, indexed)
        if not job:
            continue
        if (
            gpu_claim.select_claim_action(job, claim_id=uuid.uuid4().hex, now=now)
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
    log_backfill: dict[str, Any] | None = None
    artifact_backfill: dict[str, Any] | None = None
    submission_bridge_pending_jobs: list[str] = []
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
            # The CPU supervisor waits for this handshake solely to protect
            # explicit policy submissions from teardown.  If no job ever
            # declared a submission path, release it before the slower sandbox
            # stop/reconciliation pass; otherwise a large terminal job history
            # can consume the entire drain timeout despite there being nothing
            # to forward.
            job_ids = list_job_ids(run)
            jobs_before_stop = {
                job_id: load_job(run, job_id) for job_id in job_ids
            }
            has_explicit_submissions = any(
                job and job.get("submission_paths")
                for job in jobs_before_stop.values()
            )
            # STOP_ACK is written only after the CPU wrapper either observes
            # the drain marker or exhausts its bounded wait.  A host-authored
            # acknowledgement means the CPU sandbox is already gone.  In
            # either case there is no live wrapper left to signal, although
            # explicit GPU submissions below must still be reconciled.
            drain_pre_signaled = sprintctl.terminal_stop_acknowledged(state_dir)
            drain_signal_error = None
            if not has_explicit_submissions and not drain_pre_signaled:
                try:
                    signal_gpu_submission_drain_complete(run)
                    drain_pre_signaled = True
                except Exception as exc:  # noqa: BLE001
                    drain_signal_error = f"{type(exc).__name__}: {exc}"
            stopped = _stop_all_locked(run, reason="operator_stop")
            # Provider compute is now fenced and stopped. Recover explicit
            # attempt-0 submissions from their immutable enqueue snapshots
            # only after that safety boundary; archive transfer latency must
            # never postpone budget enforcement.
            enqueue_snapshot_recoveries: dict[str, dict[str, Any]] = {}
            for job_id in job_ids:
                job = load_job(run, job_id) or jobs_before_stop.get(job_id)
                if not job:
                    continue
                recovered, detail = recover_pending_submission_snapshots(run, job)
                jobs_before_stop[job_id] = recovered
                if detail.get("eligible"):
                    enqueue_snapshot_recoveries[job_id] = detail
            bridge_pending: list[str] = []
            reconciled: list[dict[str, Any]] = []
            for job_id in job_ids:
                # Recovery may have added host-authored submission evidence
                # that is intentionally newer than the worker Volume record.
                # Prefer that exact post-recovery object; reloading here can
                # resurrect the pre-recovery terminal record and lose the
                # proof before terminal_submission_bridge_complete sees it.
                job = jobs_before_stop.get(job_id) or load_job(run, job_id)
                if not job:
                    continue
                job = reconcile_terminal_attempt_before_stop(run, job)
                job = attach_terminal_submission_evidence(run, job)
                if (
                    str(job.get("status") or "") not in gpu_claim.TERMINAL
                    or not job.get("submission_bridge_enabled")
                ):
                    continue
                bridged, detail = drain_worker_submission_outbox(run, job)
                bridged = attach_terminal_submission_evidence(run, bridged)
                if terminal_submission_bridge_complete(run, bridged, detail):
                    bridged["submission_bridge_terminal_drained_at"] = utc_now()
                else:
                    bridge_pending.append(job_id)
                if bridged != job:
                    persist_job(run, bridged)
                reconciled.append(
                    {
                        "job_id": job_id,
                        "attempt": bridged.get("attempt"),
                        "status": bridged.get("status"),
                        "decision": "stop_submission_bridge_drain",
                        **detail,
                    }
                )
            if not bridge_pending and not drain_pre_signaled:
                try:
                    signal_gpu_submission_drain_complete(run)
                    drain_signal_error = None
                except Exception as exc:  # noqa: BLE001
                    drain_signal_error = f"{type(exc).__name__}: {exc}"
                    bridge_pending.append("cpu_drain_handshake")
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
                "reconciled": reconciled,
                "submission_bridge_pending_jobs": bridge_pending,
                "enqueue_snapshot_submission_recoveries": (
                    enqueue_snapshot_recoveries
                ),
                "submission_bridge_drain_signal_error": drain_signal_error,
                "skipped": True,
                "reason": "operator_stop",
                "ts": utc_now(),
            }
            sprintctl.atomic_write_json(
                state_dir / "gpu-dispatch.json", result, mode=0o600
            )
            return result

        # One exact-file snapshot is shared by every reconciliation phase in
        # this dispatch tick.  Without this, cancellation, orphan, and queue
        # checks each re-read the same index independently.
        indexed = indexed_agent_jobs(run) or {}

        actions.extend(cleanup_orphaned_training_sandboxes(run, indexed=indexed))
        try:
            reconciled.extend(
                reconcile_live_agent_cancel_requests(run, indexed=indexed)
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            # Preserve the durable request across transient control-channel
            # failures; GPU reconciliation must continue meanwhile.
            actions.append(
                {
                    "action": "control_channel_retry",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        reconciled.extend(reconcile_agent_cancelled_jobs(run, indexed=indexed))
        now = time.time()
        for job_id in list_job_ids(run, indexed=indexed):
            job = load_job_from_index_snapshot(run, job_id, indexed)
            if not job:
                continue
            if (
                str(job.get("status") or "") in gpu_claim.TERMINAL
                and job.get("submission_bridge_enabled")
                and not job.get("submission_bridge_terminal_drained_at")
            ):
                bridged, detail = drain_worker_submission_outbox(run, job)
                bridged = attach_terminal_submission_evidence(run, bridged)
                if terminal_submission_bridge_complete(run, bridged, detail):
                    bridged["submission_bridge_terminal_drained_at"] = utc_now()
                else:
                    submission_bridge_pending_jobs.append(job_id)
                if bridged != job:
                    persist_job(run, bridged)
                job = bridged
                reconciled.append(
                    {
                        "job_id": job_id,
                        "attempt": job.get("attempt"),
                        "status": job.get("status"),
                        "decision": "terminal_submission_bridge_drain",
                        **detail,
                    }
                )
            if (
                str(job.get("status") or "") in gpu_claim.TERMINAL
                and pending_output_artifact(job) is not None
                and not job.get("artifact_mirror_terminal_failure")
                and float(job.get("artifact_mirror_retry_after_epoch_s") or 0) <= now
                and artifact_backfill is None
            ):
                artifact_backfill = job
            if (
                str(job.get("status") or "") in gpu_claim.TERMINAL
                and not job.get("provider_logs_archived_at")
                and int(job.get("attempt") or 0) > 0
                and bool(job.get("sandbox_id") or job.get("last_sandbox_id"))
                and float(job.get("provider_logs_archive_retry_after_epoch_s") or 0)
                <= now
            ):
                # Do not make new training work wait behind a historical log
                # backlog. Dispatch first, then archive at most one old attempt
                # per monitor cycle.
                if log_backfill is None:
                    log_backfill = job
                continue
            if (
                str(job.get("status") or "") in gpu_claim.TERMINAL
                and job.get("provider_logs_archived_at")
                and int(job.get("provider_terminal_error_classifier_version") or 0)
                < PROVIDER_TERMINAL_ERROR_CLASSIFIER_VERSION
            ):
                if log_backfill is None:
                    log_backfill = job
                continue
            if str(job.get("status") or "") not in gpu_claim.OWNED:
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

        pending = _candidate_job_ids(run, now=now, indexed=indexed)
        active = active_training_job_ids(run, indexed=indexed)
        for job_id in pending:
            if len(active) >= MAX_ACTIVE_TRAINING_JOBS_PER_RUN:
                break
            job = load_job_from_index_snapshot(run, job_id, indexed)
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
            try:
                claimed = pin_work_archive(run, claimed)
                persist_job(run, claimed)
                restore_pinned_work_archive(run, claimed)
                # Archive pinning/restoration is host-side preparation and
                # cannot incur Modal sandbox spend.  Record a separate
                # billing upper-bound immediately before Sandbox.create so
                # the live ledger includes provider allocation during the
                # create call without charging the potentially long archive
                # transfer that precedes it.
                _timeline_event(
                    run,
                    claimed,
                    phase="gpu_sandbox_create",
                    action="enter",
                    source="dispatch",
                    lifecycle_boundary="before_sandbox_create",
                )
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
                _timeline_event(
                    run,
                    payload,
                    phase="gpu_lifecycle",
                    action="instant",
                    event="gpu_allocated"
                    if int(payload.get("attempt") or 1) == 1
                    else "gpu_reallocated",
                    retry_reason=payload.get("retry_reason"),
                    lifecycle_boundary="sandbox_created",
                )
                budget_mirror: dict[str, Any] = {
                    "gpu_budget_mirror": "awaiting_controller_snapshot"
                }
                budget_snapshot = fresh_dispatch_budget_snapshot(state_dir, run_id)
                if budget_snapshot is None:
                    # The sandbox command remains behind its /run barrier. The
                    # independent pulse reconstructs and injects the canonical
                    # snapshot; if the controller disappears first, the barrier
                    # exits without ever starting paid model work.
                    pass
                else:
                    budget_mirror = mirror_gpu_budget(
                        run, budget_snapshot, jobs=[payload]
                    )
                    if budget_mirror.get("gpu_budget_mirror") != "updated":
                        raise RuntimeError(
                            "trusted GPU budget mirror failed after spawn: "
                            + str(
                                budget_mirror.get("gpu_budget_mirror_error")
                                or budget_mirror.get("errors")
                                or budget_mirror.get("gpu_budget_mirror")
                            )
                        )
                # STOP_REQUESTED can arrive while Sandbox.create is in flight.
                # Publish the new sandbox id first so the fencing pass can
                # terminate the exact worker, then stop it before releasing the
                # dispatch lock.  The stop caller also waits for this lock as a
                # second, independent guarantee.
                if operator_stop_requested(state_dir):
                    stopped = _stop_all_locked(run, reason="operator_stop")
                    actions.append(
                        {
                            "job_id": job_id,
                            "attempt": payload.get("attempt"),
                            "sandbox_id": sandbox_id,
                            "status": "terminated",
                            "action": "stop_after_spawn",
                            "stopped_jobs": [item.get("job_id") for item in stopped],
                        }
                    )
                    break
                actions.append(
                    {
                        "job_id": job_id,
                        "attempt": payload.get("attempt"),
                        "sandbox_id": sandbox_id,
                        "status": payload.get("status"),
                        "claim_id": claim_id,
                        "budget_mirror": budget_mirror,
                    }
                )
                break

            except Exception as exc:  # noqa: BLE001
                current = load_job(run, job_id) or claimed
                error = f"{type(exc).__name__}: {exc}"
                terminal = finalize_lost_job(
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
                        "status": terminal.get("status"),
                        "error": error,
                    }
                )
                break

        if log_backfill is not None:
            if log_backfill.get("provider_logs_archived_at"):
                archived, detail = audit_archived_provider_logs(run, log_backfill)
            else:
                archived, detail = archive_provider_logs(run, log_backfill)
            if archived != log_backfill:
                persist_job(run, archived)
            reconciled.append(
                {
                    "job_id": archived.get("job_id"),
                    "attempt": archived.get("attempt"),
                    "status": archived.get("status"),
                    "decision": "terminal_log_backfill",
                    **detail,
                }
            )

        if artifact_backfill is not None:
            mirrored, detail = retry_terminal_artifact_mirror(run, artifact_backfill)
            if mirrored != artifact_backfill:
                persist_job(run, mirrored)
            reconciled.append(
                {
                    "job_id": mirrored.get("job_id"),
                    "attempt": mirrored.get("attempt"),
                    "status": mirrored.get("status"),
                    "decision": "terminal_artifact_backfill",
                    **detail,
                }
            )

    result = {
        "run_id": run_id,
        "pending": pending,
        "active_training_jobs": active,
        "max_active_training_jobs": MAX_ACTIVE_TRAINING_JOBS_PER_RUN,
        "submission_bridge_pending_jobs": submission_bridge_pending_jobs,
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
            "Usage: python -m event_runtime.compute.worker dispatch --run-id ID\n"
            "       python -m event_runtime.compute.worker terminate "
            "--run-id ID --job-id ID",
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
