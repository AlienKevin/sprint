#!/usr/bin/env python3
"""Host-side GPU worker dispatch for CPU-agent durable lane runs.

Claims jobs written by in-sandbox ``event gpu`` under
``/durable/runs/<run_id>/gpu-jobs/queue/`` and starts a preemptible A10G
Modal Sandbox that mounts the same volume. The Codex/agent sandbox stays on
CPU (gpus=0) so GPU preemption cannot kill the harness.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
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
    RetryPolicy,
)

WORKER_TAG_ROLE = "gpu-worker"
MAX_ACTIVE_TRAINING_JOBS_PER_RUN = 1
STOP_DISPATCH_LOCK_TIMEOUT_SEC = 10 * 60
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
AGENT_GPU_MIRROR_LOG_BYTES = 768 * 1024
AGENT_GPU_MIRROR_ARTIFACT_BYTES = 32 * 1024 * 1024
AGENT_GPU_MIRROR_ARG_BYTES = 64 * 1024
LIVE_PROVIDER_LOG_INTERVAL_SEC = 30
LIVE_PROVIDER_LOG_TAIL_LINES = 2000
AGENT_GPU_CLI_PATH = "/opt/event_runtime/agent/gpu.py"
AGENT_COST_CLI_PATH = "/opt/event_runtime/agent/cost.py"
AGENT_COMMAND_SOURCE = ROOT / "event_runtime" / "agent"
MAX_WORK_ARCHIVE_BYTES = 256 * 1024 * 1024
WORK_ARCHIVE_TRANSFER_ATTEMPTS = 5
GPU_BUDGET_MIRROR_PATH = "/run/sprint-budget-watchdog.json"
MAX_GPU_BUDGET_MIRROR_BYTES = 1024 * 1024


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
    host uploads until its mount is reloaded.  Reloading the whole mount is a
    poor fit here because the agent and snapshot loop may have open files.  A
    small host-owned mirror under /run gives the agent fresh status and a
    diagnostic log tail without weakening the no-control-plane-credentials
    boundary.  The complete log remains on the durable Volume.
    """
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
            try:
                ack_reason = str(json.loads(ack.read_text()).get("reason") or "")
            except (OSError, json.JSONDecodeError):
                return {"agent_cost_mirror": "agent_stopped"}
            # A supervised provider retry replaces the previous attempt's ACK
            # with ``agent_exit`` before the next CPU sandbox starts.  That ACK
            # is an attempt boundary, not a run stop, so the fresh sandbox must
            # continue receiving the trusted cost mirror.  Every explicit stop
            # is fenced by STOP_REQUESTED; unknown ACK reasons stay fail-closed.
            if ack_reason != "agent_exit":
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
    queue/status delivery mirrors here would put the budget-critical pulse
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
            terminal_exec_race = (
                "cannot execute in container" in message
                and ("state stopped" in message or "state terminated" in message)
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
        payload = json.loads(
            (state_dir / "telemetry" / "agent-cost.json").read_text()
        )
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
    with tempfile.TemporaryDirectory() as raw:
        destination = Path(raw) / policy_path.name
        result = sprintctl.run_command(
            sprintctl.modal_command(
                "volume",
                "get",
                "--force",
                str(run["volume_name"]),
                remote,
                str(destination),
            ),
            run=run,
            check=False,
            timeout=120,
        )
        if result.returncode != 0 or not destination.is_file():
            return payload, None, None, {"policy_mirror": "fetch_retry"}
        size = destination.stat().st_size
        if size <= 0 or size > AGENT_GPU_MIRROR_ARTIFACT_BYTES:
            return (
                payload,
                None,
                None,
                {
                    "policy_mirror": "rejected_size",
                    "policy_size_bytes": size,
                },
            )
        content = destination.read_bytes()
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
    return (
        payload,
        policy_path.name,
        content,
        {
            "policy_mirror": "fetched",
            "policy_size_bytes": len(content),
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


def retry_terminal_policy_mirror(
    run: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retry a terminal artifact fetch after Modal Volume propagation lag."""
    payload, artifact_name, artifact_content, detail = fetch_agent_policy_artifact(
        run, job
    )
    outcome = str(detail.get("policy_mirror") or "")
    if artifact_name is not None and artifact_content is not None:
        payload.pop("policy_mirror_retry_after_epoch_s", None)
        payload.pop("policy_mirror_attempts", None)
        payload["agent_policy_mirrored_at"] = utc_now()
        mirror_detail = mirror_agent_job(
            run,
            payload,
            artifact_name=artifact_name,
            artifact_content=artifact_content,
        )
        return payload, {**detail, **mirror_detail}
    if outcome == "fetch_retry":
        attempts = int(payload.get("policy_mirror_attempts") or 0) + 1
        payload["policy_mirror_attempts"] = attempts
        payload["policy_mirror_retry_after_epoch_s"] = time.time() + min(
            15 * 60, 15 * (2 ** min(attempts - 1, 6))
        )
    else:
        payload["policy_mirror_terminal_failure"] = outcome or "unknown"
    return payload, detail


def jobs_prefix(run_id: str) -> str:
    return f"runs/{run_id}/gpu-jobs"


def host_job_path(run: dict[str, Any], job_id: str) -> Path | None:
    """Return the host-owned canonical record for a logical GPU job.

    The agent can legitimately remove its queue/status mirrors after ``wait``
    returns.  Keeping the controller's copy outside the agent sandbox avoids a
    race where that cleanup happens before the next host reconciliation pass.
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


def pin_work_archive(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
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
            last_error = ""
            for attempt in range(1, WORK_ARCHIVE_TRANSFER_ATTEMPTS + 1):
                downloaded.unlink(missing_ok=True)
                try:
                    result = sprintctl.run_command(
                        sprintctl.modal_command(
                            "volume",
                            "get",
                            "--force",
                            str(run["volume_name"]),
                            remote,
                            str(downloaded),
                        ),
                        run=run,
                        check=False,
                        timeout=180,
                    )
                    if result.returncode == 0 and downloaded.is_file():
                        break
                    last_error = (result.stderr or result.stdout or "").strip()[-500:]
                except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                if attempt < WORK_ARCHIVE_TRANSFER_ATTEMPTS:
                    time.sleep(work_archive_retry_delay(attempt))
            else:
                detail = f": {last_error}" if last_error else ""
                raise RuntimeError(
                    "unable to pin GPU work archive after "
                    f"{WORK_ARCHIVE_TRANSFER_ATTEMPTS} attempts: {remote}{detail}"
                )
            size = downloaded.stat().st_size
            if size <= 0 or size > MAX_WORK_ARCHIVE_BYTES:
                raise RuntimeError(f"invalid GPU work archive size: {size}")
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


def load_job(run: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    """Load the canonical host record or an unclaimed enqueue snapshot.

    Once claimed, the host registry wins over every agent-visible mirror, so a
    stale queue file cannot respawn the job.  Before claim there is no host
    record and the immutable queue document is authoritative.  Read that
    before the mutable status mirror: a slow status writer must not prevent a
    newly submitted job from entering the dispatcher.  Status remains a
    recovery fallback for legacy/status-only jobs.
    """
    # Once claimed, the host registry is authoritative.  Volume status and
    # queue files are agent-visible delivery mirrors and may disappear before
    # the controller has archived terminal provider output or copied the final
    # policy back into the CPU sandbox.
    local = load_host_job(run, job_id)
    if local is not None:
        return local
    prefix = jobs_prefix(str(run["run_id"]))
    text = sprintctl.volume_get_text(run, f"{prefix}/queue/{job_id}.json")
    if text is None:
        text = sprintctl.volume_get_text(run, f"{prefix}/status/{job_id}.json")
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


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
        "if [ \"$SECONDS\" -ge \"$deadline\" ]; then "
        "echo 'trusted GPU budget mirror unavailable' >&2; exit 78; fi; "
        "sleep 1; done; exec python3 /opt/sprint-gpu-worker-run.py "
        + shlex.quote(str(run["run_id"]))
        + " "
        + shlex.quote(str(job["job_id"]))
        + " "
        + shlex.quote(str(int(job["attempt"])))
        + " "
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
    image = training_image(run)
    volume = modal.Volume.from_name(str(run["volume_name"]))
    timeout = int(job.get("timeout_sec") or 3600)
    timeout = max(60, min(timeout, 24 * 60 * 60))
    worker_command = (
        "python3 /opt/sprint-gpu-worker-run.py "
        + shlex.quote(run_id)
        + " "
        + shlex.quote(job_id)
        + " "
        + shlex.quote(str(attempt))
        + " "
        + shlex.quote(lease_id)
    )
    command = (
        "deadline=$((SECONDS + 120)); "
        f"while [ ! -s {shlex.quote(GPU_BUDGET_MIRROR_PATH)} ]; do "
        "if [ \"$SECONDS\" -ge \"$deadline\" ]; then "
        "echo 'trusted GPU budget mirror unavailable' >&2; exit 78; fi; "
        "sleep 1; done; exec "
        + worker_command
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

            modal.Sandbox.from_id(handle.attempt_id).terminate()
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return None


def read_modal_sandbox_output(sandbox_id: str) -> tuple[str, str]:
    """Read complete stdout/stderr after a Modal Sandbox has exited."""
    import modal

    sandbox = modal.Sandbox.from_id(sandbox_id)
    if sandbox.poll() is None:
        raise RuntimeError(f"sandbox {sandbox_id} is still running")
    return str(sandbox.stdout.read() or ""), str(sandbox.stderr.read() or "")


def read_modal_sandbox_live_output(
    run: dict[str, Any], sandbox_id: str
) -> str:
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

    encoded = ("== Modal live log tail ==\n" + text).encode(
        "utf-8", errors="replace"
    )
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


def apply_provider_terminal_error(
    job: dict[str, Any], terminal_error: str | None
) -> dict[str, Any]:
    payload = dict(job)
    payload["provider_terminal_error_checked_at"] = utc_now()
    payload["provider_terminal_error_detected"] = bool(terminal_error)
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
    terminal_error = provider_terminal_error(text)
    payload = apply_provider_terminal_error(job, terminal_error)
    payload, artifact_name, artifact_content, policy_detail = (
        fetch_agent_policy_artifact(run, payload)
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
        **policy_detail,
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
    terminal_error = provider_terminal_error(content)
    payload = apply_provider_terminal_error(payload, terminal_error)
    payload, artifact_name, artifact_content, policy_detail = (
        fetch_agent_policy_artifact(run, payload)
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
        **policy_detail,
        **mirror_detail,
    }


def persist_job(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Write the host canonical record, then agent-visible Volume mirrors."""
    prefix = jobs_prefix(str(run["run_id"]))
    job_id = str(job["job_id"])
    local_path = host_job_path(run, job_id)
    if local_path is not None:
        sprintctl.atomic_write_json(local_path, job, mode=0o600)
    put_json(run, f"{prefix}/status/{job_id}.json", job)
    put_json(run, f"{prefix}/queue/{job_id}.json", job)
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
                    "progress",
                    "checkpoint",
                }
            }
        )
        terminal["attempt_record"] = attempt_path(
            str(run["run_id"]), str(job["job_id"]), int(job["attempt"])
        )
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
            **log_detail,
        }

    if owned_record and str(attempt_record.get("status") or "") == "interrupted":
        retry_reason = str(attempt_record.get("retry_reason") or "graceful_preemption")
        retried = schedule_retry(
            run,
            job,
            heartbeat=heartbeat,
            exit_code=attempt_record.get("exit_code"),
            reason=retry_reason,
            now=ref,
        )
        return retried, {
            "decision": "retry",
            "status": retried["status"],
            "reason": retry_reason,
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
    retried = schedule_retry(
        run,
        job,
        heartbeat=heartbeat,
        exit_code=exit_code,
        reason="worker_lost",
        now=ref,
    )
    return retried, detail


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
        "progress",
        "checkpoint",
    ):
        if key in attempt_record:
            payload[key] = attempt_record[key]
    payload["attempt_record"] = attempt_path(
        str(run["run_id"]), str(job["job_id"]), int(job["attempt"])
    )
    return persist_job(run, payload)


def list_pending_job_ids(run: dict[str, Any]) -> list[str]:
    prefix = jobs_prefix(str(run["run_id"]))
    return [
        Path(name).stem
        for name in volume_ls_json_names(run, f"{prefix}/queue")
        if name.endswith(".json") and re.fullmatch(r"[A-Za-z0-9_-]+", Path(name).stem)
    ]


def list_agent_cancelled_job_ids(run: dict[str, Any]) -> list[str]:
    """Return jobs explicitly hidden by an agent before host dispatch.

    The agent-facing queue is intentionally writable.  Some agents cancel a
    queued job by atomically renaming both delivery records to
    ``.cancelled-<job>.json``.  The append-only host registry must retain the
    audit record, but must not revive that job merely because its old local
    status was pending or retryable.
    """
    if not str(run.get("volume_name") or "").strip():
        return []
    prefix = jobs_prefix(str(run["run_id"]))
    cancelled: set[str] = set()
    pattern = re.compile(r"^\.cancelled-([A-Za-z0-9_-]+)\.json$")
    for remote_dir in (f"{prefix}/queue", f"{prefix}/status"):
        for name in volume_ls_json_names(run, remote_dir):
            match = pattern.fullmatch(Path(name).name)
            if match:
                cancelled.add(match.group(1))
    return sorted(cancelled)


def reconcile_agent_cancelled_jobs(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Fence undispatched host records whose agent delivery was cancelled."""
    reconciled: list[dict[str, Any]] = []
    for job_id in list_agent_cancelled_job_ids(run):
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


def list_job_ids(run: dict[str, Any]) -> list[str]:
    prefix = jobs_prefix(str(run["run_id"]))
    names = {f"{job_id}.json" for job_id in list_host_job_ids(run)}
    names.update(volume_ls_json_names(run, f"{prefix}/queue"))
    names.update(volume_ls_json_names(run, f"{prefix}/status"))
    return sorted(
        Path(name).stem
        for name in names
        if name.endswith(".json") and re.fullmatch(r"[A-Za-z0-9_-]+", Path(name).stem)
    )


def active_training_job_ids(run: dict[str, Any]) -> list[str]:
    """Return logical jobs that currently own the run's training GPU slot."""
    active: list[str] = []
    for job_id in list_job_ids(run):
        job = load_job(run, job_id)
        if job and str(job.get("status") or "") in gpu_claim.OWNED:
            active.append(job_id)
    return active


def cleanup_orphaned_training_sandboxes(run: dict[str, Any]) -> list[dict[str, Any]]:
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
    for job_id in list_job_ids(run):
        job = load_job(run, job_id)
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
    for job_id in list_job_ids(run):
        job = load_job(run, job_id)
        if job:
            job = reconcile_terminal_attempt_before_stop(run, job)
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
        job = reconcile_terminal_attempt_before_stop(run, job)
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
            gpu_claim.select_claim_action(job, claim_id=uuid.uuid4().hex, now=now)
            == "claim"
        ):
            try:
                created = float(job.get("created_at_epoch_s") or 0)
            except (TypeError, ValueError):
                created = 0.0
            if created <= 0:
                try:
                    created = datetime.fromisoformat(
                        str(job.get("created_at") or "").replace("Z", "+00:00")
                    ).astimezone(timezone.utc).timestamp()
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
    policy_backfill: dict[str, Any] | None = None
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

        actions.extend(cleanup_orphaned_training_sandboxes(run))
        reconciled.extend(reconcile_agent_cancelled_jobs(run))
        now = time.time()
        for job_id in list_job_ids(run):
            job = load_job(run, job_id)
            if not job:
                continue
            progress = job.get("progress")
            reported_policy = isinstance(progress, dict) and bool(
                str(progress.get("policy_path") or progress.get("policy") or "").strip()
            )
            if (
                str(job.get("status") or "") in gpu_claim.TERMINAL
                and reported_policy
                and not job.get("agent_policy_mirror_path")
                and not job.get("policy_mirror_terminal_failure")
                and float(job.get("policy_mirror_retry_after_epoch_s") or 0) <= now
                and policy_backfill is None
            ):
                policy_backfill = job
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
                and not job.get("provider_terminal_error_checked_at")
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

        pending = _candidate_job_ids(run, now=now)
        # A preempted logical job owns this run's single training slot through
        # its short retry backoff.  Otherwise a newly submitted job can jump
        # ahead during that window and turn transparent recovery into an
        # unbounded wait behind unrelated work from the same agent.
        retry_reservations: list[str] = []
        for job_id in list_job_ids(run):
            job = load_job(run, job_id)
            if job and str(job.get("status") or "") == "retry_wait":
                retry_reservations.append(job_id)
        if retry_reservations:
            reserved = set(retry_reservations)
            pending = [job_id for job_id in pending if job_id in reserved]
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

        if policy_backfill is not None:
            mirrored, detail = retry_terminal_policy_mirror(run, policy_backfill)
            if mirrored != policy_backfill:
                persist_job(run, mirrored)
            reconciled.append(
                {
                    "job_id": mirrored.get("job_id"),
                    "attempt": mirrored.get("attempt"),
                    "status": mirrored.get("status"),
                    "decision": "terminal_policy_backfill",
                    **detail,
                }
            )

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
    elif retry_reservations and not pending and not actions:
        result["reason"] = "retry_backoff_reserved"
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
