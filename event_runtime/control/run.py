#!/usr/bin/env python3
"""Host controller for one explicit durable Harbor lane run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.export.frontier import (  # noqa: E402
    DEPLOY_DEBOUNCE_SECONDS,
    WEB_DEFAULT,
    atomic_write_json,
    atomic_write_text,
    file_lock,
    ledger_counts,
    parse_iso,
    read_ledger,
    row_terminal,
    scan_frontier,
    sha256_file,
    utc_now,
)
from event_runtime.cost import agent as agent_cost  # noqa: E402
from event_runtime.cost import modal as modal_cost  # noqa: E402
from event_runtime.export.timeline import (  # noqa: E402
    SCHEMA_VERSION as UNIFIED_TIMELINE_SCHEMA_VERSION,
)

OPS_ROOT = ROOT / "runs" / "ops"
FRONTIER_SCRIPT = ROOT / "event_runtime/export/frontier.py"
RECONSTRUCT_CODEX_USAGE_SCRIPT = ROOT / "event_runtime/cost/model_usage.py"
UV = Path("/home/ubuntu/.local/bin/uv")
POLL_SECONDS = 30
DEFAULT_WAIT_SECONDS = 3 * 60 * 60
BUDGET_PULSE_MAX_UPSTREAM_AGE_SECONDS = 60.0
BUDGET_PULSE_MAX_CLOCK_SKEW_SECONDS = 60.0
DURABLE_TRACE_LIVE_SYNC_TIMEOUT_SECONDS = 60
DURABLE_TRACE_FINAL_SYNC_TIMEOUT_SECONDS = 300
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$")
Uploader = Callable[[Path, str], None]


def state_dir_for(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run ID")
    return OPS_ROOT / run_id


def load_run(run_id: str) -> tuple[Path, dict[str, Any]]:
    state_dir = state_dir_for(run_id)
    path = state_dir / "run.json"
    try:
        state = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"unknown run ID: {run_id}") from exc
    if state.get("run_id") != run_id:
        raise ValueError(f"{path} does not match requested run ID")
    kind = state.get("agent_kind")
    if kind not in {"claude-code", "codex"}:
        raise ValueError(f"{path} has unsupported agent_kind: {kind!r}")
    return state_dir, state


def agent_kind(run: dict[str, Any]) -> str:
    """Return the required agent kind from the current run schema."""
    kind = run.get("agent_kind")
    if kind not in {"claude-code", "codex"}:
        raise ValueError(f"run has unsupported agent_kind: {kind!r}")
    return str(kind)


def update_run_fields(state_dir: Path, **updates: Any) -> dict[str, Any]:
    """Atomically update discovered run fields without clobbering a relaunch.

    The monitor can spend tens of seconds probing Modal while the supervisor
    advances ``run.json`` to a new CPU attempt.  Rewriting the monitor's stale
    copy here would roll back the attempt number, history, and jobs root.
    Always reload under the same lock used by the launcher and apply only the
    fields this discovery pass owns.
    """
    path = state_dir / "run.json"
    with file_lock(state_dir / "run.json.lock"):
        current = json.loads(path.read_text())
        current.update(updates)
        atomic_write_json(path, current, mode=0o600)
    return current


def command_env(run: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    profile = run.get("modal_profile")
    if profile:
        env["MODAL_PROFILE"] = str(profile)
    return env


def run_command(
    command: Sequence[str],
    *,
    run: dict[str, Any] | None = None,
    check: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=command_env(run or {}),
        check=check,
        timeout=timeout,
    )


def modal_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "modal", *args]


def discover_app_id(state_dir: Path, run: dict[str, Any]) -> str | None:
    cached = run.get("app_id")
    if isinstance(cached, str) and cached.startswith("ap-"):
        return cached
    result = run_command(modal_command("app", "list", "--json"), run=run)
    apps = json.loads(result.stdout)
    exact = [
        app
        for app in apps
        if app.get("description") == run["app_name"] and app.get("state") != "stopped"
    ]
    if len(exact) != 1:
        return None
    app_id = str(exact[0]["app_id"])
    run["app_id"] = app_id
    update_run_fields(state_dir, app_id=app_id)
    return app_id


def containers_for_app(run: dict[str, Any], app_id: str) -> list[str]:
    result = run_command(
        modal_command("container", "list", "--app-id", app_id, "--json"),
        run=run,
    )
    payload = json.loads(result.stdout)
    return [
        str(item["container_id"])
        for item in payload
        if item.get("app_id") == app_id
        and isinstance(item.get("container_id"), str)
        and item["container_id"].startswith("ta-")
    ]


def exec_container(
    run: dict[str, Any],
    container_id: str,
    shell_command: str,
    *,
    check: bool = True,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    # Modal's Click parser still eats bare `-c` after CONTAINER_ID unless `--`
    # ends option parsing. Without this, host telemetry polls fail with
    # "No such option: -c" and never sample GPU/CPU.
    return run_command(
        modal_command(
            "container",
            "exec",
            "--no-pty",
            container_id,
            "--",
            "sh",
            "-c",
            shell_command,
        ),
        run=run,
        check=check,
        timeout=timeout,
    )


def is_agent_container(
    run: dict[str, Any], container_id: str, *, timeout: int = 45
) -> bool:
    try:
        result = exec_container(
            run,
            container_id,
            "test -x /opt/sprint-snapshot-loop.sh && printf SPRINT_AGENT",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and "SPRINT_AGENT" in (result.stdout or "")


def discover_agent_container(state_dir: Path, run: dict[str, Any]) -> str | None:
    app_id = discover_app_id(state_dir, run)
    if not app_id:
        return None
    containers = containers_for_app(run, app_id)
    cached = run.get("agent_container_id")
    if cached in containers and is_agent_container(run, str(cached)):
        return str(cached)
    matches = [
        container for container in containers if is_agent_container(run, container)
    ]
    if len(matches) == 1:
        run["agent_container_id"] = matches[0]
        update_run_fields(state_dir, agent_container_id=matches[0])
        return matches[0]
    return None


def volume_get_text(
    run: dict[str, Any],
    remote_path: str,
    *,
    timeout_seconds: int = 60,
) -> str | None:
    result = run_command(
        modal_command(
            "volume",
            "get",
            str(run["volume_name"]),
            remote_path,
            "-",
        ),
        run=run,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        return None
    text = result.stdout or ""
    # Modal CLI appends a success banner after writing to stdout ("-").
    marker = "\n✓ Finished downloading files to local!"
    if marker in text:
        text = text.split(marker, 1)[0]
    elif text.rstrip().endswith("✓ Finished downloading files to local!"):
        text = text.rstrip()[: -len("✓ Finished downloading files to local!")].rstrip(
            "\n"
        )
    return text


def volume_upload(run: dict[str, Any], source: Path, remote_path: str) -> None:
    run_command(
        modal_command(
            "volume",
            "put",
            "--force",
            str(run["volume_name"]),
            str(source),
            remote_path,
        ),
        run=run,
        timeout=300,
    )


def sync_durable_trace(
    state_dir: Path,
    run: dict[str, Any],
    *,
    min_interval_seconds: int = 60,
    force: bool = False,
) -> bool:
    """Best-effort download of private immutable trace chunks.

    Native Harbor traces are normally available after a trial exits. Chunks
    cover the abrupt-termination case and also make the timeline available
    while a trial is live.
    """
    stamp = state_dir / "durable-trace-sync.json"
    try:
        previous = json.loads(stamp.read_text())
    except (OSError, json.JSONDecodeError):
        previous = {}
    now = time.time()
    if (
        not force
        and now - float(previous.get("synced_at_epoch_s") or 0) < min_interval_seconds
    ):
        return bool(previous.get("ok"))
    destination = state_dir / "durable-trace"
    destination.mkdir(parents=True, exist_ok=True)
    timeout_seconds = (
        DURABLE_TRACE_FINAL_SYNC_TIMEOUT_SECONDS
        if force
        else DURABLE_TRACE_LIVE_SYNC_TIMEOUT_SECONDS
    )
    error: str | None = None
    try:
        result = run_command(
            modal_command(
                "volume",
                "get",
                "--force",
                str(run["volume_name"]),
                f"runs/{run['run_id']}/trace/raw",
                str(destination),
            ),
            run=run,
            check=False,
            timeout=timeout_seconds,
        )
        ok = result.returncode == 0
        if not ok:
            error = (result.stderr or result.stdout)[-1000:]
    except subprocess.TimeoutExpired as exc:
        # Live trace chunks are immutable observability evidence. A slow Modal
        # directory scan must not hold the controller behind the artifact
        # transport; keep the last complete snapshot and retry on a later pass.
        ok = False
        error = f"TimeoutExpired after {timeout_seconds}s: {exc}"
    payload = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "synced_at": utc_now(),
        "synced_at_epoch_s": now,
        "ok": ok,
        "error": error,
    }
    atomic_write_json(stamp, payload, mode=0o600)
    return ok


def sync_durable_api_usage(
    state_dir: Path,
    run: dict[str, Any],
    *,
    min_interval_seconds: int = 60,
    force: bool = False,
) -> bool:
    """Download the immutable OpenRouter per-request billing ledger."""
    stamp = state_dir / "provider-api-usage-sync.json"
    try:
        previous = json.loads(stamp.read_text())
    except (OSError, json.JSONDecodeError):
        previous = {}
    now = time.time()
    if (
        not force
        and now - float(previous.get("synced_at_epoch_s") or 0) < min_interval_seconds
    ):
        return bool(previous.get("ok"))
    destination = state_dir / "provider-api-usage"
    destination.mkdir(parents=True, exist_ok=True)
    result = run_command(
        modal_command(
            "volume",
            "get",
            "--force",
            str(run["volume_name"]),
            f"runs/{run['run_id']}/api-usage",
            str(destination),
        ),
        run=run,
        check=False,
        timeout=300,
    )
    payload = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "synced_at": utc_now(),
        "synced_at_epoch_s": now,
        "ok": result.returncode == 0,
        "error": (
            None if result.returncode == 0 else (result.stderr or result.stdout)[-1000:]
        ),
    }
    atomic_write_json(stamp, payload, mode=0o600)
    return result.returncode == 0


def sync_durable_telemetry(
    state_dir: Path,
    run: dict[str, Any],
    *,
    force: bool = False,
    max_age_seconds: int | None = None,
) -> bool:
    """Import authoritative in-sandbox telemetry for live and final coverage.

    Host polling is intentionally a backup.  Training workers continuously
    write their authoritative cgroup/GPU stream to the shared Volume, including
    the samples immediately preceding abrupt preemption. ``max_age_seconds``
    bounds live refresh traffic; finalization forces one last import.
    """
    out_dir = state_dir / "telemetry"
    stamp = out_dir / "durable-sync.json"
    now = time.time()
    if not force and stamp.is_file():
        try:
            previous = json.loads(stamp.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("ok") is True and previous.get("run_id") == run["run_id"]:
            if max_age_seconds is None:
                return True
            synced_at = float(previous.get("synced_at_epoch_s") or 0)
            if now - synced_at < max_age_seconds:
                return True

    prefix = f"runs/{run['run_id']}/telemetry"
    sources = {
        f"{prefix}/samples.jsonl": out_dir / "durable-samples.jsonl",
        f"{prefix}/gpu-stream/samples.jsonl": out_dir / "durable-gpu-samples.jsonl",
        f"{prefix}/gpu_timeline.jsonl": out_dir / "durable-gpu-timeline.jsonl",
    }
    captured: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    # Live telemetry is supplementary to the host poller and must not hold an
    # entire monitor cycle behind a slow Volume read. Final reconciliation is
    # allowed a longer window and retries until the required coverage is
    # present. Isolate failures per source so one busy append-only stream does
    # not prevent the other durable evidence from being imported.
    read_timeout_seconds = 180 if force else 15

    def fetch(remote: str) -> str | None:
        try:
            return volume_get_text(
                run,
                remote,
                timeout_seconds=read_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            errors[remote] = f"TimeoutExpired after {read_timeout_seconds}s: {exc}"
            return None

    for remote, local in sources.items():
        text = fetch(remote)
        if text is None:
            continue
        atomic_write_text(local, text, mode=0o600)
        captured[remote] = {
            "local": str(local.relative_to(state_dir)),
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    # The merged gpu-stream is convenient for live rendering, but Volume
    # writers can race while several short-lived workers append to it.  Each
    # worker also owns an append-only by-job stream.  Recover only jobs that
    # have lifecycle evidence but no sample in the merged stream; this keeps
    # live polling bounded while making finalization robust to merge races.
    lifecycle_jobs: set[str] = set()
    lifecycle_attempts: set[tuple[str, int]] = set()
    lifecycle_paths = (
        out_dir / "durable-gpu-timeline.jsonl",
        out_dir / "gpu_timeline.jsonl",
    )
    for lifecycle_path in lifecycle_paths:
        if not lifecycle_path.is_file():
            continue
        for raw in lifecycle_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            job_id = row.get("job_id") if isinstance(row, dict) else None
            if isinstance(job_id, str) and job_id:
                lifecycle_jobs.add(job_id)
                attempt = row.get("attempt")
                if isinstance(attempt, int) and attempt > 0:
                    lifecycle_attempts.add((job_id, attempt))
    sampled_jobs: set[str] = set()
    merged_path = out_dir / "durable-gpu-samples.jsonl"
    if merged_path.is_file():
        for raw in merged_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            job_id = row.get("job_id") if isinstance(row, dict) else None
            if isinstance(job_id, str) and job_id:
                sampled_jobs.add(job_id)
    by_job_dir = out_dir / "durable-by-job"
    by_job_targets = lifecycle_jobs if force else lifecycle_jobs - sampled_jobs
    for job_id in sorted(by_job_targets):
        remote = f"{prefix}/by-job/{job_id}/samples.jsonl"
        text = fetch(remote)
        if text is None:
            continue
        local = by_job_dir / job_id / "samples.jsonl"
        atomic_write_text(local, text, mode=0o600)
        captured[remote] = {
            "local": str(local.relative_to(state_dir)),
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    if force:
        for job_id, attempt in sorted(lifecycle_attempts):
            remote = f"runs/{run['run_id']}/gpu-jobs/attempts/{job_id}/{attempt}.json"
            text = fetch(remote)
            if text is None:
                continue
            local = out_dir / "durable-gpu-attempts" / job_id / f"{attempt}.json"
            atomic_write_text(local, text, mode=0o600)
            captured[remote] = {
                "local": str(local.relative_to(state_dir)),
                "bytes": len(text.encode()),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
    ok = bool(captured) and not errors
    atomic_write_json(
        stamp,
        {
            "schema_version": 1,
            "run_id": run["run_id"],
            "synced_at": utc_now(),
            "synced_at_epoch_s": now,
            "ok": ok,
            "sources": captured,
            "errors": errors,
        },
        mode=0o600,
    )
    return ok


def build_unified_timeline(
    state_dir: Path, run: dict[str, Any], *, upload: bool = True
) -> dict[str, Any]:
    from event_runtime.export import timeline as unified_timeline

    payload = unified_timeline.build_timeline(
        state_dir,
        web_dir=Path(str(run.get("site_dir", WEB_DEFAULT))),
        bucket_seconds=int(run.get("timeline_bucket_seconds") or 60),
    )
    if upload:
        volume_upload(
            run,
            state_dir / "telemetry" / "unified-timeline.json",
            f"runs/{run['run_id']}/telemetry/unified-timeline.json",
        )
    return payload


def reconstruct_codex_usage(state_dir: Path, run: dict[str, Any]) -> bool:
    """Materialize ATIF and cost ledgers from immutable per-attempt chunks."""
    chunks = state_dir / "durable-trace" / "raw"
    if not chunks.is_dir() or not any(
        chunks.glob("cpu-attempt-*/codex/*/chunks/*.jsonl")
    ):
        return False
    result = run_command(
        [
            str(UV),
            "run",
            "--project",
            str(run["harbor_path"]),
            "--frozen",
            "--extra",
            "modal",
            "python",
            str(RECONSTRUCT_CODEX_USAGE_SCRIPT),
            "--state-dir",
            str(state_dir),
        ],
        run=run,
        check=False,
        timeout=300,
    )
    return result.returncode == 0


def fetch_remote_json(
    state_dir: Path, run: dict[str, Any], remote_name: str, local_name: str
) -> dict[str, Any] | None:
    text = volume_get_text(run, f"runs/{run['run_id']}/{remote_name}")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    atomic_write_json(state_dir / local_name, payload, mode=0o600)
    return payload


def fetch_budget_watchdog(
    state_dir: Path, run: dict[str, Any]
) -> dict[str, Any] | None:
    """Read the live watchdog without depending on Volume control-plane sync.

    The budget watchdog writes into the agent sandbox's mounted durable path.
    Reading that path with ``container exec`` avoids transient misses when the
    artifact monitor is concurrently downloading from the same Modal Volume.
    Volume download remains the startup/restart fallback, and the last local
    copy lets a single control-plane miss preserve the pulse's freshness
    guarantee (the caller still rejects snapshots older than 60 seconds).
    """
    local_path = state_dir / "telemetry" / "budget-watchdog.json"
    remote_name = "budget/watchdog.json"
    remote_path = f"/durable/runs/{run['run_id']}/{remote_name}"
    container_id = run.get("agent_container_id")
    if isinstance(container_id, str) and container_id.startswith("ta-"):
        try:
            result = exec_container(
                run,
                container_id,
                f"cat -- {shlex.quote(remote_path)}",
                check=False,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            try:
                payload = json.loads(result.stdout or "")
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                atomic_write_json(local_path, payload, mode=0o600)
                return payload

    payload = fetch_remote_json(
        state_dir,
        run,
        remote_name,
        "telemetry/budget-watchdog.json",
    )
    if isinstance(payload, dict):
        return payload
    try:
        cached = json.loads(local_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return cached if isinstance(cached, dict) else None


def process_alive(pid: int | None, needle: str | None = None) -> bool:
    if not pid or pid <= 1:
        return False
    try:
        command = (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        )
    except OSError:
        return False
    return not needle or needle in command


def harbor_alive(run: dict[str, Any]) -> bool:
    pid_path = Path(str(run["state_dir"])) / "harbor.pid"
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return False
    return process_alive(pid, str(run["jobs_root"]))


def discover_job_and_trial(
    state_dir: Path, run: dict[str, Any]
) -> tuple[Path | None, Path | None]:
    jobs_root = Path(str(run["jobs_root"])).resolve()
    cached_job = run.get("job_path") or run.get("expected_job_path")
    job = Path(cached_job).resolve() if cached_job else None
    if not job or not job.is_dir() or jobs_root not in job.parents:
        jobs = (
            sorted(path.resolve() for path in jobs_root.iterdir() if path.is_dir())
            if jobs_root.is_dir()
            else []
        )
        if len(jobs) != 1:
            return None, None
        job = jobs[0]
        run["job_path"] = str(job)

    cached_trial = run.get("trial_path")
    trial = Path(cached_trial).resolve() if cached_trial else None
    if not trial or not trial.is_dir() or job not in trial.parents:
        trials = sorted(
            path.resolve()
            for path in job.iterdir()
            if path.is_dir() and "__" in path.name
        )
        if len(trials) != 1:
            update_run_fields(state_dir, job_path=str(job))
            return job, None
        trial = trials[0]
        run["trial_path"] = str(trial)
    updates = {"job_path": str(job)}
    if trial:
        updates["trial_path"] = str(trial)
    update_run_fields(state_dir, **updates)
    return job, trial


def persist_stop_request(
    run_id: str, *, reason: str = "operator_stop"
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Durably record stop intent without waiting on any Modal operation."""
    state_dir, run = load_run(run_id)
    kind = agent_kind(run)
    marker = state_dir / "STOP_REQUESTED.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
    else:
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "agent_kind": kind,
            "requested_at": utc_now(),
            "container_id": run.get("agent_container_id"),
            "method": "create /run/sprint-stop",
            "reason": reason,
        }
        atomic_write_json(marker, payload, mode=0o600)
    return state_dir, run, payload


def request_stop(run_id: str, *, reason: str = "operator_stop") -> dict[str, Any]:
    state_dir, run, payload = persist_stop_request(run_id, reason=reason)
    kind = agent_kind(run)
    marker = state_dir / "STOP_REQUESTED.json"

    # Signal the CPU harness before waiting for the GPU dispatch lock. A
    # Sandbox.create call can hold that lock for minutes; waiting for it first
    # would let API and CPU spend continue past the durable budget marker.
    agent_stop_error = None
    try:
        container = discover_agent_container(state_dir, run)
        if container:
            exec_container(
                run,
                container,
                "umask 077; printf '%s\\n' "
                + shlex.quote(str(payload.get("reason") or reason))
                + " > /run/sprint-stop; chmod 0600 /run/sprint-stop",
            )
            if payload.get("container_id") != container:
                payload["container_id"] = container
                atomic_write_json(marker, payload, mode=0o600)
    except Exception as exc:  # noqa: BLE001
        agent_stop_error = f"{type(exc).__name__}: {exc}"

    # Fence GPU leases after the CPU has received its stop signal. The monitor
    # sees STOP_REQUESTED and repeats this idempotently if this call is cut off.
    gpu_stopped: list[str] = []
    gpu_stop_error = None
    try:
        from event_runtime.compute import worker as gpu_worker

        gpu_stopped = [
            str(item.get("job_id")) for item in gpu_worker.stop_all(run, reason=reason)
        ]
    except Exception as exc:  # noqa: BLE001
        gpu_stop_error = f"{type(exc).__name__}: {exc}"

    try:
        ack = fetch_remote_json(state_dir, run, "STOP_ACK", "STOP_ACK.json")
    except Exception:  # noqa: BLE001
        ack = None
    if ack and str(ack.get("reason") or "") == "operator_stop":
        return {
            "status": "acknowledged",
            "agent_kind": kind,
            "ack": ack,
            "agent_stop_error": agent_stop_error,
            "gpu_workers_stopped": gpu_stopped,
            "gpu_stop_error": gpu_stop_error,
        }
    return {
        "status": "requested",
        "agent_kind": kind,
        **payload,
        "agent_stop_error": agent_stop_error,
        "gpu_workers_stopped": gpu_stopped,
        "gpu_stop_error": gpu_stop_error,
    }


def enforce_agent_cost_budget(
    run_id: str,
    state_dir: Path,
    run: dict[str, Any],
    cost_payload: dict[str, Any],
) -> bool:
    """Request a durable stop once a complete snapshot reaches its stop threshold."""
    budget = run.get("agent_cost_budget_usd")
    if budget is None:
        return False
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
        or float(budget) <= 0
    ):
        raise ValueError(f"invalid agent_cost_budget_usd: {budget!r}")
    if (state_dir / "STOP_REQUESTED.json").is_file():
        return False
    threshold = cost_payload.get("stop_threshold_usd", budget)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0
        or float(threshold) > float(budget)
    ):
        raise ValueError(f"invalid stop_threshold_usd: {threshold!r}")
    total = cost_payload.get("total_usd")
    if (
        cost_payload.get("status")
        not in {"complete", "within_budget", "stop_requested"}
        or isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isfinite(float(total))
        or float(total) < float(threshold)
    ):
        return False
    request_stop(run_id, reason="agent_cost_budget_exhausted")
    return True


def immutable_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    tmp = Path(raw)
    try:
        shutil.copy2(source, tmp)
        os.chmod(tmp, 0o444)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def history_sources(job: Path, trial: Path) -> list[Path]:
    sources = [
        job / "job.log",
        job / "result.json",
        trial / "trial.log",
        trial / "config.json",
        trial / "result.json",
        trial / "artifacts" / "manifest.json",
        trial / "artifacts" / "continuous" / "ledger.jsonl",
    ]
    sources.extend(sorted(job.glob("*.lock")))
    sources.extend(sorted(trial.glob("*.lock")))
    sources.extend(sorted((trial / "verifier").glob("*")))
    return [path for path in sources if path.is_file()]


def snapshot_host_history(
    state_dir: Path,
    run: dict[str, Any],
    job: Path,
    trial: Path,
    *,
    upload: bool = True,
) -> list[Path]:
    index_path = state_dir / "history-index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        index = {"schema_version": 1, "files": {}}
    written: list[Path] = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for source in history_sources(job, trial):
        digest = sha256_file(source)
        key = str(source)
        if index["files"].get(key) == digest:
            continue
        relative = (
            f"job/{source.relative_to(job)}"
            if job in source.parents
            else f"trial/{source.relative_to(trial)}"
        )
        safe = relative.replace("/", "__")
        destination = state_dir / "history" / f"{stamp}-{digest[:12]}-{safe}"
        immutable_copy(source, destination)
        index["files"][key] = digest
        written.append(destination)
        if upload:
            volume_upload(
                run,
                destination,
                f"runs/{run['run_id']}/host-history/{destination.name}",
            )
    index["updated_at"] = utc_now()
    atomic_write_json(index_path, index, mode=0o600)
    return written


def tar_attempt_atomic(
    attempt: Path, archive_dir: Path, *, refresh: bool = False
) -> tuple[Path, str, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(archive_dir.glob(f"{attempt.name}.*.tar"))
    if existing and not refresh:
        archive = existing[0]
        digest = sha256_file(archive)
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        if checksum.exists() and checksum.read_text().split()[0] == digest:
            return archive, digest, checksum

    fd, raw = tempfile.mkstemp(
        prefix=f".{attempt.name}.", suffix=".tar.tmp", dir=archive_dir
    )
    os.close(fd)
    tmp = Path(raw)
    try:
        with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as archive_handle:
            archive_handle.add(attempt, arcname=attempt.name, recursive=True)
        digest = sha256_file(tmp)
        final = archive_dir / f"{attempt.name}.{digest[:16]}.tar"
        os.chmod(tmp, 0o444)
        os.replace(tmp, final)
        checksum = final.with_suffix(final.suffix + ".sha256")
        atomic_write_text(checksum, f"{digest}  {final.name}\n", mode=0o444)
        return final, digest, checksum
    finally:
        tmp.unlink(missing_ok=True)


def archive_completed_attempts(
    state_dir: Path,
    run: dict[str, Any],
    trial: Path,
    *,
    upload: bool = True,
) -> dict[str, Any]:
    ledger = read_ledger(trial / "artifacts" / "continuous" / "ledger.jsonl")
    manifest_path = state_dir / "archive-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        manifest = {"schema_version": 1, "run_id": run["run_id"], "attempts": {}}
    attempts_root = trial / "artifacts" / "continuous" / "attempts"
    for row in ledger.rows:
        if not row_terminal(row):
            continue
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        matches = sorted(attempts_root.glob(f"{index:04d}-*"))
        if len(matches) != 1 or not matches[0].is_dir():
            continue
        attempt = matches[0]
        row_digest = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        previous = manifest["attempts"].get(attempt.name)
        if previous and previous.get("ledger_row_sha256") == row_digest:
            continue
        archive, digest, checksum = tar_attempt_atomic(
            attempt, state_dir / "archives", refresh=previous is not None
        )
        record = {
            "index": index,
            "attempt": attempt.name,
            "archive": str(archive),
            "sha256": digest,
            "bytes": archive.stat().st_size,
            "ledger_row_sha256": row_digest,
            "archived_at": utc_now(),
            "uploaded": False,
        }
        if upload:
            volume_upload(
                run,
                archive,
                f"runs/{run['run_id']}/host-archives/{archive.name}",
            )
            volume_upload(
                run,
                checksum,
                f"runs/{run['run_id']}/host-archives/{checksum.name}",
            )
            record["uploaded"] = True
        manifest["attempts"][attempt.name] = record
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest, mode=0o600)
    manifest["ledger_errors"] = ledger.errors
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest, mode=0o600)
    return manifest


def verify_attempt_archives(path: Path) -> dict[str, Any]:
    checksums = sorted(path.rglob("*.tar.sha256"))
    errors: list[str] = []
    verified = 0
    for checksum in checksums:
        parts = checksum.read_text().split()
        if len(parts) < 2:
            errors.append(f"malformed checksum: {checksum}")
            continue
        archive = checksum.parent / parts[1]
        if not archive.is_file():
            errors.append(f"missing archive: {archive}")
            continue
        actual = sha256_file(archive)
        if actual != parts[0]:
            errors.append(f"checksum mismatch: {archive}")
            continue
        verified += 1
    return {"verified": verified, "errors": errors, "valid": not errors}


def sync_frontier_artifacts(
    state_dir: Path, run: dict[str, Any], *, upload: bool = True
) -> list[Path]:
    index_path = state_dir / "frontier-upload-index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        index = {"schema_version": 1, "files": {}}
    files = sorted(
        path for path in (state_dir / "captures").glob("*") if path.is_file()
    )
    try:
        frontier = json.loads((state_dir / "frontier-state.json").read_text())
    except (OSError, json.JSONDecodeError):
        frontier = {}
    site_dir = Path(str(run.get("site_dir", WEB_DEFAULT))).resolve()
    for capture in frontier.get("captures", {}).values():
        for key in ("web_capture", "web_html"):
            relative = capture.get(key)
            if not isinstance(relative, str):
                continue
            candidate = (site_dir / relative).resolve()
            if site_dir in candidate.parents and candidate.is_file():
                files.append(candidate)
    files = sorted(set(files))
    uploaded: list[Path] = []
    for path in files:
        digest = sha256_file(path)
        if index["files"].get(str(path)) == digest:
            continue
        if upload:
            volume_upload(
                run,
                path,
                f"runs/{run['run_id']}/frontier-artifacts/{path.name}",
            )
        index["files"][str(path)] = digest
        uploaded.append(path)
    index["updated_at"] = utc_now()
    atomic_write_json(index_path, index, mode=0o600)
    return uploaded


def worker_alive(state_dir: Path) -> bool:
    try:
        pid = int((state_dir / "frontier-worker.pid").read_text().strip())
    except (OSError, ValueError):
        return False
    return process_alive(pid, "event_runtime/export/frontier.py")


def budget_pulse_alive(state_dir: Path) -> bool:
    try:
        pid = int((state_dir / "budget-pulse.pid").read_text().strip())
    except (OSError, ValueError):
        return False
    return process_alive(pid, "budget-pulse")


def gpu_dispatch_loop_alive(state_dir: Path) -> bool:
    try:
        pid = int((state_dir / "gpu-dispatch-loop.pid").read_text().strip())
    except (OSError, ValueError):
        return False
    return process_alive(pid, "gpu-dispatch-loop")


def refresh_agent_cost_snapshot(
    run_id: str,
    state_dir: Path,
    run: dict[str, Any],
    timeline: dict[str, Any],
) -> dict[str, Any]:
    """Build and persist one serialized artifact-monitor cost snapshot.

    The artifact monitor and the independent budget pulse are concurrent
    writers. Local serialization keeps the ledger monotonic. While the
    dedicated pulse is alive it alone owns remote propagation: an artifact
    timeline can lag the live watchdog and must never overwrite the fresher
    agent/GPU heartbeat merely because it finished later.
    """
    with file_lock(state_dir / "telemetry" / "agent-cost.lock"):
        cost_payload = agent_cost.build_snapshot(timeline, state_dir=state_dir)
        cost_path = state_dir / "telemetry" / "agent-cost.json"
        atomic_write_json(cost_path, cost_payload, mode=0o600)
        enforce_agent_cost_budget(run_id, state_dir, run, cost_payload)

    if budget_pulse_alive(state_dir):
        delegated = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "agent_cost_mirror": "delegated_to_budget_pulse",
        }
        atomic_write_json(
            state_dir / "telemetry" / "agent-cost-mirror.json",
            delegated,
            mode=0o600,
        )
        atomic_write_json(
            state_dir / "telemetry" / "gpu-budget-mirror.json",
            {
                "schema_version": 1,
                "updated_at": utc_now(),
                "gpu_budget_mirror": "delegated_to_budget_pulse",
            },
            mode=0o600,
        )
        return cost_payload

    from event_runtime.compute import worker as gpu_worker

    mirror = gpu_worker.mirror_agent_cost(run, cost_payload)
    atomic_write_json(
        state_dir / "telemetry" / "agent-cost-mirror.json",
        {"schema_version": 1, "updated_at": utc_now(), **mirror},
        mode=0o600,
    )
    if mirror.get("agent_cost_mirror") == "error":
        raise RuntimeError(
            "agent cost mirror failed: "
            + str(mirror.get("agent_cost_mirror_error") or "unknown")
        )
    gpu_budget_mirror = gpu_worker.mirror_gpu_budget(run, cost_payload)
    atomic_write_json(
        state_dir / "telemetry" / "gpu-budget-mirror.json",
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            **gpu_budget_mirror,
        },
        mode=0o600,
    )
    if gpu_budget_mirror.get("gpu_budget_mirror") == "error":
        raise RuntimeError(
            "GPU budget mirror failed: "
            + str(
                gpu_budget_mirror.get("gpu_budget_mirror_error")
                or gpu_budget_mirror.get("errors")
                or "unknown"
            )
        )
    return cost_payload


def maybe_start_frontier_worker(
    state_dir: Path, run: dict[str, Any], job: Path, trial: Path
) -> int | None:
    if worker_alive(state_dir):
        return None
    frontier_path = state_dir / "frontier-state.json"
    try:
        state = json.loads(frontier_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    queued = any(
        item.get("status") == "queued" for item in state.get("capture_queue", [])
    )
    due_deploy = False
    if state.get("pending_site_hash"):
        first = parse_iso(state.get("site_change_first_seen_at")) or time.time()
        due_deploy = time.time() - first >= int(
            run.get("deploy_debounce_seconds", DEPLOY_DEBOUNCE_SECONDS)
        )
    if not queued and not due_deploy:
        return None
    log = (state_dir / "frontier-worker.log").open("a")
    command = [
        sys.executable,
        str(FRONTIER_SCRIPT),
        "worker",
        "--job",
        str(job),
        "--trial",
        str(trial),
        "--state",
        str(frontier_path),
        "--web",
        str(run.get("site_dir", WEB_DEFAULT)),
    ]
    if not run.get("batch_id"):
        command.extend(
            [
                "--deploy",
                "--debounce-seconds",
                str(run.get("deploy_debounce_seconds", DEPLOY_DEBOUNCE_SECONDS)),
            ]
        )
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=command_env(run),
        start_new_session=True,
        close_fds=True,
    )
    log.close()
    atomic_write_text(state_dir / "frontier-worker.pid", f"{process.pid}\n", 0o600)
    return process.pid


def status_snapshot(
    state_dir: Path,
    run: dict[str, Any],
    *,
    include_remote: bool = True,
) -> dict[str, Any]:
    job, trial = discover_job_and_trial(state_dir, run)
    ledger_read = (
        read_ledger(trial / "artifacts" / "continuous" / "ledger.jsonl")
        if trial
        else None
    )
    heartbeat = None
    ack = None
    container = None
    app_id = run.get("app_id")
    remote_error = None
    if include_remote:
        try:
            heartbeat = fetch_remote_json(
                state_dir,
                run,
                "snapshot/heartbeat.json",
                "snapshot-heartbeat.json",
            )
            ack = fetch_remote_json(state_dir, run, "STOP_ACK", "STOP_ACK.json")
            container = discover_agent_container(state_dir, run)
            app_id = run.get("app_id")
        except Exception as exc:  # noqa: BLE001
            remote_error = str(exc)
    else:
        for source, name in (
            (state_dir / "snapshot-heartbeat.json", "heartbeat"),
            (state_dir / "STOP_ACK.json", "ack"),
        ):
            try:
                value = json.loads(source.read_text())
            except (OSError, json.JSONDecodeError):
                value = None
            if name == "heartbeat":
                heartbeat = value
            else:
                ack = value

    stale = None
    if heartbeat:
        updated = parse_iso(heartbeat.get("updated_at"))
        stale = None if updated is None else max(0, int(time.time() - updated))
    kind = agent_kind(run)
    heartbeat_kind = heartbeat.get("agent_kind") if heartbeat else None
    payload = {
        "schema_version": 2,
        "run_id": run["run_id"],
        "agent_kind": kind,
        "updated_at": utc_now(),
        "harbor_alive": harbor_alive(run),
        "app_id": app_id,
        "app_name": run["app_name"],
        "agent_container_id": container or run.get("agent_container_id"),
        "volume_name": run["volume_name"],
        "job": str(job) if job else None,
        "trial": str(trial) if trial else None,
        "ledger": ledger_counts(ledger_read) if ledger_read else {},
        "ledger_errors": ledger_read.errors if ledger_read else [],
        "snapshot_heartbeat": heartbeat,
        "snapshot_agent_kind_matches": heartbeat_kind in {None, kind},
        "snapshot_heartbeat_age_seconds": stale,
        "snapshot_heartbeat_ok": bool(
            heartbeat and stale is not None and stale <= 15 * 60
        ),
        "stop_ack": ack,
        "frontier_worker_alive": worker_alive(state_dir),
        "remote_error": remote_error,
    }
    atomic_write_json(state_dir / "status.json", payload, mode=0o600)
    return payload


def monitor_once(
    run_id: str,
    *,
    upload: bool = True,
    include_remote: bool = True,
    launch_worker: bool = True,
) -> dict[str, Any]:
    state_dir, run = load_run(run_id)
    Path("/data/.keepalive").touch()
    # The in-sandbox watchdog writes this marker directly to the shared Modal
    # Volume. Import it before GPU dispatch so controller recovery cannot start
    # fresh work after a cloud-side budget stop.
    stop_marker = state_dir / "STOP_REQUESTED.json"
    if not stop_marker.is_file() and (run.get("budget_enforcement") or {}).get(
        "in_sandbox_watchdog"
    ):
        try:
            cloud_stop = fetch_remote_json(
                state_dir,
                run,
                "BUDGET_STOP_REQUESTED.json",
                "BUDGET_STOP_REQUESTED.remote.json",
            )
            if cloud_stop:
                persist_stop_request(
                    run_id,
                    reason=str(
                        cloud_stop.get("reason") or "agent_cost_budget_exhausted"
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            record_controller_error(run_id, exc)
    # A stop may be requested while Modal is still resolving the image and no
    # runtime sandbox exists. Keep reapplying the durable request until the
    # actual agent acknowledges it; never mistake an image-build container for
    # the CPU agent or let a post-build sandbox escape an earlier stop.
    if stop_marker.is_file() and not terminal_stop_acknowledged(state_dir):
        try:
            stop_payload = json.loads(stop_marker.read_text())
            request_stop(
                run_id,
                reason=str(stop_payload.get("reason") or "operator_stop"),
            )
            state_dir, run = load_run(run_id)
        except Exception as exc:  # noqa: BLE001
            record_controller_error(run_id, exc)
    # CPU-agent runs: claim event-gpu queue jobs and spawn A10G workers.
    # Dispatch before telemetry: Modal Volume scans/uploads are intentionally
    # best-effort and can take close to their one-minute timeout.  A dead lease
    # must be fenced/retried without waiting behind observability I/O.
    if run.get("cpu_agent_gpu_worker") and not gpu_dispatch_loop_alive(state_dir):
        try:
            from event_runtime.compute import worker as gpu_worker

            gpu_worker.dispatch_once(run_id)
        except Exception as exc:  # noqa: BLE001
            record_controller_error(run_id, exc)
    # Host-side GPU/CPU telemetry backup (agent + best-effort verifiers).
    # In-sandbox sidecar is primary; this persists even if the mount lags.
    try:
        from event_runtime.telemetry import host as telemetry_host

        telemetry_host.poll_once(run_id)
    except Exception as exc:  # noqa: BLE001
        record_controller_error(run_id, exc)
    job, trial = discover_job_and_trial(state_dir, run)
    if job and trial:
        snapshot_host_history(state_dir, run, job, trial, upload=upload)
        archive_completed_attempts(state_dir, run, trial, upload=upload)
        if run.get("unified_timeline_required"):
            try:
                sync_durable_telemetry(
                    state_dir,
                    run,
                    max_age_seconds=5 * 60,
                )
                fetch_remote_json(
                    state_dir,
                    run,
                    "budget/watchdog.json",
                    "telemetry/budget-watchdog.json",
                )
                if upload:
                    sync_durable_trace(state_dir, run)
                if run.get("usage_audit_required"):
                    sync_durable_api_usage(state_dir, run)
                    reconstruct_codex_usage(state_dir, run)
                timeline = build_unified_timeline(state_dir, run, upload=upload)
                refresh_agent_cost_snapshot(run_id, state_dir, run, timeline)
            except Exception as exc:  # noqa: BLE001
                record_controller_error(run_id, exc)
        frontier_path = state_dir / "frontier-state.json"
        if not worker_alive(state_dir):
            sync_frontier_artifacts(state_dir, run, upload=upload)
            scan_frontier(
                job=job,
                trial=trial,
                state_path=frontier_path,
                web=Path(str(run.get("site_dir", WEB_DEFAULT))),
            )
            if upload:
                volume_upload(
                    run,
                    frontier_path,
                    f"runs/{run_id}/state/frontier-state.json",
                )
            if launch_worker:
                maybe_start_frontier_worker(state_dir, run, job, trial)
    return status_snapshot(state_dir, run, include_remote=include_remote)


def unified_timeline_ready(payload: dict[str, Any], run_id: str) -> bool:
    """Accept only the current role-separated timeline contract."""
    return bool(
        payload.get("schema_version") == UNIFIED_TIMELINE_SCHEMA_VERSION
        and payload.get("coverage", {}).get("ready")
        and payload.get("run", {}).get("run_id") == run_id
    )


def modal_billing_ready(state_dir: Path, run_id: str) -> tuple[bool, list[str]]:
    path = state_dir / "telemetry" / "modal-cost.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, ["missing or invalid Modal cost artifact"]
    details: list[str] = []
    if payload.get("schema_version") != modal_cost.SCHEMA_VERSION:
        details.append("Modal cost artifact schema is stale")
    if payload.get("run_id") != run_id:
        details.append("Modal cost artifact run ID mismatch")
    provider_compute_complete = bool(
        payload.get("provider_complete") is True
        or payload.get("provider_compute_complete") is True
    )
    if not provider_compute_complete:
        details.append(
            "Modal provider compute billing is not complete: "
            + str(payload.get("pending_reason") or payload.get("error") or "pending")
        )
    if payload.get("provider_cost_precredits_usd") is None:
        details.append("Modal provider billing total is missing")
    for key in ("by_role_usd", "by_category_usd", "by_role_category_usd"):
        if not isinstance(payload.get(key), dict):
            details.append(f"Modal provider billing {key} is missing")
    volume_storage = payload.get("volume_storage")
    volume_unavailable_but_excluded = bool(
        provider_compute_complete
        and payload.get("pending_reason")
        == "provider_volume_storage_snapshot_unavailable"
    )
    if (
        not isinstance(volume_storage, dict)
        or volume_storage.get("status") != "captured"
    ) and not volume_unavailable_but_excluded:
        details.append("Modal Volume storage snapshot is missing")
    return not details, details


def usage_audit_ready(trial: Path, run: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate raw JSONL -> usage audit -> ATIF cost reconstruction."""
    details: list[str] = []
    audit_path = trial / "agent" / "usage-audit.json"
    trajectory_path = trial / "agent" / "trajectory.json"
    result_path = trial / "result.json"
    try:
        audit = json.loads(audit_path.read_text())
        trajectory = json.loads(trajectory_path.read_text())
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"usage audit artifact missing or invalid: {exc}"]

    if audit.get("schema_version") != 1:
        details.append("usage audit schema_version is not 1")
    if audit.get("cost_reconstruction_complete") is not True:
        details.append("usage audit cost reconstruction is incomplete")
    requests = audit.get("requests")
    if not isinstance(requests, list) or not requests:
        details.append("usage audit has no requests")
        requests = []
    if audit.get("request_count") != len(requests):
        details.append("usage audit request_count mismatch")
    if audit.get("reconciliation_mismatches") != {}:
        details.append("usage audit cumulative counters do not reconcile")

    expected_model = str(run.get("model") or "").split("/", 1)[-1]
    expected_service_tier = (
        "default"
        if expected_model in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
        else None
    )
    for index, request in enumerate(requests, start=1):
        if not isinstance(request, dict):
            details.append(f"usage audit request {index} is not an object")
            continue
        if request.get("model") != expected_model:
            details.append(f"usage audit request {index} model mismatch")
        if request.get("reasoning_effort") != run.get("reasoning_effort"):
            details.append(f"usage audit request {index} reasoning effort mismatch")
        if request.get("service_tier") != expected_service_tier:
            details.append(f"usage audit request {index} service tier mismatch")
        if request.get("cost_reconstruction_status") != "complete":
            details.append(f"usage audit request {index} cost is incomplete")

    snapshots = audit.get("pricing_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        details.append("usage audit must contain pricing snapshot provenance")
    else:
        valid_snapshots = [row for row in snapshots if isinstance(row, dict)]
        if len(valid_snapshots) != len(snapshots):
            details.append("usage audit pricing snapshot is malformed")
        for snapshot in valid_snapshots:
            if snapshot.get("model") != expected_model:
                details.append("usage audit pricing snapshot model mismatch")
            if not snapshot.get("captured_at") or not snapshot.get("source_url"):
                details.append("usage audit pricing snapshot provenance is incomplete")
        snapshot = valid_snapshots[0] if valid_snapshots else {}
        resolved = run.get("resolved_model_version")
        if (
            expected_model == "deepseek-v4-flash"
            and snapshot.get("model_version") != resolved
        ):
            details.append("usage audit DeepSeek model version mismatch")

    calculated = audit.get("calculated_api_usage_usd")
    selected = audit.get("selected_total_cost_usd")
    if not isinstance(calculated, (int, float)) or isinstance(calculated, bool):
        details.append("usage audit calculated cost is missing")
    elif not isinstance(selected, (int, float)) or isinstance(selected, bool):
        details.append("usage audit selected cost is missing")
    elif abs(float(calculated) - float(selected)) > 1e-12:
        details.append("usage audit selected cost differs from calculated cost")

    final_cost = (trajectory.get("final_metrics") or {}).get("total_cost_usd")
    if isinstance(calculated, (int, float)) and not isinstance(calculated, bool):
        if not isinstance(final_cost, (int, float)) or isinstance(final_cost, bool):
            details.append("ATIF total cost is missing")
        elif abs(float(calculated) - float(final_cost)) > 1e-12:
            details.append("ATIF total cost differs from usage audit")

        result_cost = (result.get("agent_result") or {}).get("cost_usd")
        if not isinstance(result_cost, (int, float)) or isinstance(result_cost, bool):
            details.append("Harbor result cost is missing")
        elif abs(float(calculated) - float(result_cost)) > 1e-12:
            details.append("Harbor result cost differs from usage audit")

    provenance = audit.get("provenance") or {}
    agent_dir = (trial / "agent").resolve()

    def signed_snapshot(path_key: str, hash_key: str, label: str) -> None:
        relative = provenance.get(path_key)
        if not isinstance(relative, str) or not relative:
            details.append(f"usage audit {label} provenance path is missing")
            return
        candidate = (agent_dir / relative).resolve()
        try:
            candidate.relative_to(agent_dir)
        except ValueError:
            details.append(f"usage audit {label} provenance path escapes agent dir")
            return
        if not candidate.is_file() or provenance.get(hash_key) != sha256_file(
            candidate
        ):
            details.append(f"usage audit {label} snapshot checksum mismatch")

    signed_snapshot("trajectory_path", "trajectory_sha256", "trajectory")
    signed_snapshot("source_session_path", "source_session_sha256", "source session")
    return not details, details


def run_usage_audit_ready(
    state_dir: Path, run: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate accounting across every CPU attempt, including killed attempts."""
    details: list[str] = []
    path = state_dir / "usage" / "run-usage-audit.json"
    try:
        audit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"run usage audit artifact missing or invalid: {exc}"]
    expected_model = str(run.get("model") or "")
    if audit.get("schema_version") not in {1, 2}:
        details.append("run usage audit schema mismatch")
    if audit.get("run_id") != run.get("run_id"):
        details.append("run usage audit run_id mismatch")
    if audit.get("model") != expected_model:
        details.append("run usage audit model mismatch")
    if audit.get("reasoning_effort") != run.get("reasoning_effort"):
        details.append("run usage audit reasoning effort mismatch")
    if audit.get("resolved_model_version") != run.get("resolved_model_version"):
        details.append("run usage audit resolved model version mismatch")
    if audit.get("attempt_coverage_complete") is not True:
        details.append("run usage audit does not cover every CPU attempt")
    if audit.get("cost_reconstruction_complete") is not True:
        details.append("run usage audit cost reconstruction is incomplete")
    requests = audit.get("requests")
    if not isinstance(requests, list):
        details.append("run usage audit requests are missing")
        requests = []
    if audit.get("request_count") != len(requests):
        details.append("run usage audit request count mismatch")
    if not requests and audit.get("zero_request_reason") != (
        "no completed model request was present in any captured CPU attempt"
    ):
        details.append("run usage audit has no attested zero-request reason")
    if requests and audit.get("zero_request_reason") is not None:
        details.append("run usage audit has a contradictory zero-request reason")
    calculated = audit.get("calculated_api_usage_usd")
    request_cost = sum(
        float(request.get("calculated_cost_usd") or 0)
        for request in requests
        if isinstance(request, dict)
    )
    if not isinstance(calculated, (int, float)) or isinstance(calculated, bool):
        details.append("run usage audit calculated cost is missing")
    elif abs(float(calculated) - request_cost) > 1e-12:
        details.append("run usage audit request costs do not sum to total")
    expected_short_model = expected_model.split("/", 1)[-1]
    openrouter_list_price = (run.get("budget_enforcement") or {}).get(
        "api_budget_cost_basis"
    ) == "openrouter_list_price_before_endpoint_discount"
    if openrouter_list_price and audit.get("calculated_api_usage_cost_basis") != (
        "openrouter_list_price_before_endpoint_discount"
    ):
        details.append(
            "run usage audit does not use undiscounted OpenRouter list price"
        )
    provider_billing_reconciled = bool(
        openrouter_list_price
        and audit.get("provider_billing_reconciled") is True
    )
    for index, request in enumerate(requests, start=1):
        if not isinstance(request, dict):
            details.append(f"run usage request {index} is not an object")
            continue
        if request.get("model") != expected_short_model:
            details.append(f"run usage request {index} model mismatch")
        if request.get("reasoning_effort") != run.get("reasoning_effort"):
            details.append(f"run usage request {index} reasoning effort mismatch")
        if request.get("cost_reconstruction_status") != "complete":
            details.append(f"run usage request {index} cost is incomplete")
        if not request.get("usage_reported_at"):
            details.append(f"run usage request {index} has no timestamp")
        if (run.get("budget_enforcement") or {}).get("api_cost_source") == (
            "openrouter_reported_per_request"
        ):
            if not isinstance(request.get("provider_reported_cost_usd"), (int, float)):
                details.append(f"run usage request {index} lacks OpenRouter cost")
            if not request.get("openrouter_generation_id"):
                details.append(f"run usage request {index} lacks generation ID")
            if openrouter_list_price:
                benchmark_cost = request.get("calculated_cost_usd")
                provider_cost = request.get("provider_reported_cost_usd")
                if not (
                    isinstance(benchmark_cost, (int, float))
                    and not isinstance(benchmark_cost, bool)
                    and isinstance(provider_cost, (int, float))
                    and not isinstance(provider_cost, bool)
                    and float(benchmark_cost) >= float(provider_cost)
                ):
                    details.append(
                        f"run usage request {index} has invalid undiscounted cost"
                    )
                if not isinstance(request.get("promotion_snapshot"), dict):
                    details.append(
                        f"run usage request {index} lacks promotion snapshot"
                    )
    for source in audit.get("source_sessions") or []:
        if not isinstance(source, dict):
            details.append("run usage source session is malformed")
            continue
        # Codex session logs contain token usage, but OpenRouter owns the exact
        # per-request charge. The run-level audit binds every signed session
        # request to that durable provider ledger above, so session-local cost
        # completeness is neither expected nor authoritative on this path.
        if (
            source.get("cost_reconstruction_complete") is not True
            and not provider_billing_reconciled
        ):
            details.append("run usage source session cost is incomplete")
        chunks = source.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            details.append("run usage source session has no raw chunks")
            chunks = []
        for chunk in chunks:
            relative = chunk.get("path") if isinstance(chunk, dict) else None
            chunk_path = state_dir / str(relative)
            if (
                not relative
                or not chunk_path.is_file()
                or chunk.get("sha256") != sha256_file(chunk_path)
            ):
                details.append("run usage source chunk checksum mismatch")
                break
        trajectory_relative = source.get("trajectory_path")
        trajectory = state_dir / str(trajectory_relative)
        if (
            not trajectory_relative
            or not trajectory.is_file()
            or source.get("trajectory_sha256") != sha256_file(trajectory)
        ):
            details.append("run usage source trajectory checksum mismatch")
    return not details, details


def final_policy_frozen_ready(trial: Path) -> bool:
    """Verify that Harbor collected a non-empty final policy after agent exit."""
    final_policy = trial / "artifacts" / "app" / "submission" / "policy.pt"
    try:
        return final_policy.is_file() and final_policy.stat().st_size > 0
    except OSError:
        return False


def batch_site_deployed_ready(state_dir: Path, run: dict[str, Any]) -> bool:
    """Require proof that this run's latest public artifact reached production."""
    batch_id = run.get("batch_id")
    if not batch_id:
        return True
    marker_path = state_dir / "BATCH_SITE_DEPLOYED.json"
    allowed = {
        f"data/policies/{run['run_id']}.json",
        f"data/timelines/{run['run_id']}.json",
    }
    try:
        marker = json.loads(marker_path.read_text())
        relative = marker.get("public_artifact_path")
        if relative not in allowed:
            return False
        snapshot_relative = marker.get("public_artifact_snapshot_path")
        if not isinstance(snapshot_relative, str) or not snapshot_relative.startswith(
            "deployment-provenance/"
        ):
            return False
        snapshot = (state_dir / snapshot_relative).resolve()
        snapshot.relative_to(state_dir.resolve())
        artifact_hash = marker.get("public_artifact_sha256")
        return bool(
            marker.get("schema_version") == 2
            and marker.get("run_id") == run.get("run_id")
            and marker.get("batch_id") == batch_id
            and marker.get("production_alias") == "https://g1-sprint.vercel.app"
            and isinstance(artifact_hash, str)
            and len(artifact_hash) == 64
            and snapshot.is_file()
            and artifact_hash == sha256_file(snapshot)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def final_conditions(
    state_dir: Path, run: dict[str, Any]
) -> tuple[bool, dict[str, bool], list[str]]:
    job, trial = discover_job_and_trial(state_dir, run)
    evaluation_result_policy = run.get("evaluation_result_policy")
    archival_submissions = (
        evaluation_result_policy == "all_blind_archival_submissions"
    )
    all_submissions = evaluation_result_policy in {
        "all_blind_submissions",
        "all_blind_archival_submissions",
        # Retained while the stopped 2026-08-08 batch finishes draining.
        "all_blind_submissions_by_deadline",
    }
    stop_was_requested = (state_dir / "STOP_REQUESTED.json").is_file()
    conditions: dict[str, bool] = {
        # Natural completion has no controller stop to acknowledge. A run that
        # did receive a stop request must prove either that the CPU sandbox
        # observed it or that Harbor sealed both terminal result records. The
        # latter covers the narrow race where an already-exiting agent finishes
        # before it can acknowledge a late fail-closed telemetry stop.
        "stop_ack": (
            not stop_was_requested
            or terminal_stop_acknowledged(state_dir)
            or run_results_finished(state_dir, run)
        ),
        "job_found": job is not None,
        "trial_found": trial is not None,
        "ledger_parseable": False,
        "ledger_terminal": False,
        "attempt_archives": False,
        "artifact_manifest": False,
        "finished_at": False,
        "harbor_exited": not harbor_alive(run),
        "site_current": False,
    }
    if all_submissions:
        conditions["continuous_result_set"] = False
    else:
        # Retained while already-running frozen-final trials finish.
        conditions["final_verifier"] = False
    if run.get("unified_timeline_required"):
        conditions["unified_timeline_ready"] = False
    if run.get("usage_audit_required"):
        conditions["usage_audit_complete"] = False
    if run.get("primary_score_policy") == "frozen_final_artifact":
        conditions["final_policy_frozen"] = False
    if run.get("batch_id"):
        conditions["batch_site_deployed"] = False
    details: list[str] = []
    if not job or not trial:
        return False, conditions, ["job or trial path is not available"]

    ledger_path = trial / "artifacts" / "continuous" / "ledger.jsonl"
    ledger = read_ledger(ledger_path)
    conditions["ledger_parseable"] = not ledger.errors
    conditions["ledger_terminal"] = all(row_terminal(row) for row in ledger.rows)
    details.extend(ledger.errors)

    if run.get("primary_score_policy") == "frozen_final_artifact":
        conditions["final_policy_frozen"] = final_policy_frozen_ready(trial)

    manifest_path = trial / "artifacts" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = manifest if isinstance(manifest, list) else []
        if all_submissions:
            conditions["artifact_manifest"] = bool(
                isinstance(manifest, list)
                and (
                    archival_submissions
                    or all(
                        isinstance(entry, dict) and entry.get("status") != "failed"
                        for entry in entries
                    )
                )
            )
        else:
            policy_entries = [
                entry
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("destination") == "artifacts/app/submission/policy.pt"
            ]
            conditions["artifact_manifest"] = bool(
                entries
                and all(
                    isinstance(entry, dict) and entry.get("status") != "failed"
                    for entry in entries
                )
                and len(policy_entries) == 1
                and policy_entries[0].get("status") == "ok"
            )
    except (OSError, json.JSONDecodeError):
        pass

    try:
        trial_result = json.loads((trial / "result.json").read_text())
        job_result = json.loads((job / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        trial_result, job_result = {}, {}
    if all_submissions:
        summary = trial_result.get("continuous_verification")
        conditions["continuous_result_set"] = bool(
            isinstance(summary, dict)
            and isinstance(summary.get("submissions"), list)
            and conditions["ledger_parseable"]
            and conditions["ledger_terminal"]
        )
    else:
        verifier_file = any(
            path.is_file()
            for path in (
                trial / "verifier" / "reward.json",
                trial / "verifier" / "reward.txt",
            )
        )
        conditions["final_verifier"] = bool(
            verifier_file
            and trial_result.get("verifier_result") is not None
            and (trial_result.get("verifier") or {}).get("finished_at")
        )
    conditions["finished_at"] = bool(
        trial_result.get("finished_at") and job_result.get("finished_at")
    )

    try:
        archives = json.loads((state_dir / "archive-manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        archives = {"attempts": {}}
    archived_indices = {
        int(record["index"])
        for record in archives.get("attempts", {}).values()
        if record.get("sha256")
    }
    required_indices: set[int] = set()
    attempts_root = trial / "artifacts" / "continuous" / "attempts"
    for row in ledger.rows:
        if not row_terminal(row):
            continue
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(attempts_root.glob(f"{index:04d}-*")):
            required_indices.add(index)
    conditions["attempt_archives"] = required_indices <= archived_indices

    frontier_path = state_dir / "frontier-state.json"
    try:
        frontier = json.loads(frontier_path.read_text())
    except (OSError, json.JSONDecodeError):
        frontier = {}
    pending = any(
        item.get("status") in {"queued", "running"}
        for item in frontier.get("capture_queue", [])
    )
    captures = frontier.get("captures", {})
    frontier_candidates = frontier.get("frontier_candidates", [])
    active_captured = all(
        candidate.get("policy_hash")
        and captures.get(candidate["policy_hash"], {}).get("valid")
        for candidate in frontier_candidates
    )
    conditions["site_current"] = bool(
        frontier.get("ledger_hash") == ledger.digest
        and not pending
        and not frontier.get("pending_site_hash")
        and frontier.get("site_status") in {"noop", "deployed"}
        and active_captured
        and not worker_alive(state_dir)
    )
    if run.get("batch_id"):
        conditions["batch_site_deployed"] = batch_site_deployed_ready(state_dir, run)
    if run.get("unified_timeline_required"):
        timeline_path = state_dir / "telemetry" / "unified-timeline.json"
        try:
            timeline = json.loads(timeline_path.read_text())
            conditions["unified_timeline_ready"] = unified_timeline_ready(
                timeline, run["run_id"]
            )
        except (OSError, json.JSONDecodeError):
            pass
    if run.get("usage_audit_required"):
        run_audit_ready, run_audit_details = run_usage_audit_ready(state_dir, run)
        try:
            run_audit = json.loads(
                (state_dir / "usage" / "run-usage-audit.json").read_text()
            )
        except (OSError, json.JSONDecodeError):
            run_audit = {}
        # The archival policy intentionally survives CPU-container teardown.
        # Its all-attempt host reconstruction is authoritative because it
        # attests every raw chunk, request, cost, and reconstructed ATIF
        # checksum. The legacy policies still require the independent Harbor
        # audit -> ATIF -> result reconciliation, except for attested
        # zero-request provider rejections.
        if run_audit_ready and (
            archival_submissions or run_audit.get("request_count") == 0
        ):
            trial_audit_ready, trial_audit_details = True, []
        else:
            trial_audit_ready, trial_audit_details = usage_audit_ready(trial, run)
        conditions["usage_audit_complete"] = trial_audit_ready and run_audit_ready
        details.extend(trial_audit_details)
        details.extend(run_audit_details)
    if run.get("modal_billing_required"):
        modal_ready, modal_details = modal_billing_ready(state_dir, run["run_id"])
        conditions["modal_billing_complete"] = modal_ready
        details.extend(modal_details)
    for name, value in conditions.items():
        if not value:
            details.append(name)
    return all(conditions.values()), conditions, details


def finalize(
    run_id: str,
    *,
    upload: bool = True,
    include_remote: bool = True,
) -> tuple[bool, dict[str, Any]]:
    state_dir, run = load_run(run_id)
    final_path = state_dir / "FINALIZED.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        if (
            existing.get("complete") is True
            and existing.get("timeline_schema_version")
            == UNIFIED_TIMELINE_SCHEMA_VERSION
        ):
            return True, existing
    monitor_once(
        run_id,
        upload=upload,
        include_remote=include_remote,
        launch_worker=False,
    )
    if terminal_stop_acknowledged(state_dir):
        sync_durable_telemetry(state_dir, run, force=True)
    if run.get("usage_audit_required"):
        sync_durable_trace(state_dir, run, force=True)
        sync_durable_api_usage(state_dir, run, force=True)
        reconstruct_codex_usage(state_dir, run)
        if run.get("unified_timeline_required"):
            build_unified_timeline(state_dir, run, upload=upload)
    if run.get("modal_billing_required"):
        # Do not declare a provider report complete while accepted verifier
        # work can still extend the run-owned allocation window. All other
        # finalization conditions must be green first. FINALIZED itself is a
        # host-side timestamp and is deliberately excluded from run bounds.
        _, preliminary_conditions, _ = final_conditions(state_dir, run)
        non_billing_ready = all(
            value
            for name, value in preliminary_conditions.items()
            if name != "modal_billing_complete"
        )
        if non_billing_ready:
            billing = modal_cost.collect_provider_billing(state_dir)
            billing_path = state_dir / "telemetry" / "modal-cost.json"
            if billing.get("provider_complete") is True and upload:
                # The reconciled report must win over any older pending copy
                # on the durable Volume before FINALIZED can be written.
                volume_upload(
                    run,
                    billing_path,
                    f"runs/{run_id}/telemetry/modal-cost.json",
                )
            if run.get("unified_timeline_required"):
                build_unified_timeline(state_dir, run, upload=upload)
    complete, conditions, details = final_conditions(state_dir, run)
    payload = {
        "schema_version": 1,
        "timeline_schema_version": UNIFIED_TIMELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "agent_kind": agent_kind(run),
        "complete": complete,
        "conditions": conditions,
        "details": details,
        "checked_at": utc_now(),
    }
    if not complete:
        return False, payload
    payload["finalized_at"] = utc_now()
    try:
        payload["archives"] = json.loads(
            (state_dir / "archive-manifest.json").read_text()
        )
    except (OSError, json.JSONDecodeError):
        payload["archives"] = {}
    atomic_write_json(final_path, payload, mode=0o444)
    if upload:
        volume_upload(
            run,
            final_path,
            f"runs/{run_id}/state/FINALIZED.json",
        )
    return True, payload


def record_controller_error(run_id: str, exc: BaseException) -> None:
    state_dir = state_dir_for(run_id)
    path = state_dir / "controller-errors.jsonl"
    row = {
        "at": utc_now(),
        "type": type(exc).__name__,
        "message": str(exc),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def wait_for_run(run_id: str, timeout_seconds: int, poll_seconds: int) -> int:
    deadline = time.time() + timeout_seconds
    state_dir, run = load_run(run_id)
    kind = agent_kind(run)
    with file_lock(state_dir / "monitor.lock", blocking=False) as owns_monitor:
        while True:
            try:
                if owns_monitor:
                    monitor_once(run_id)
                    complete, payload = finalize(run_id)
                else:
                    final_path = state_dir / "FINALIZED.json"
                    complete = final_path.is_file()
                    source = final_path if complete else state_dir / "status.json"
                    payload = (
                        json.loads(source.read_text())
                        if source.exists()
                        else {
                            "run_id": run_id,
                            "agent_kind": kind,
                            "complete": False,
                            "checked_at": utc_now(),
                            "status": "waiting for host monitor",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                record_controller_error(run_id, exc)
                complete = False
                payload = {
                    "run_id": run_id,
                    "agent_kind": kind,
                    "complete": False,
                    "checked_at": utc_now(),
                    "transient_error": str(exc),
                }
            print(json.dumps(payload, sort_keys=True), flush=True)
            if complete:
                return 0
            if time.time() >= deadline:
                return 2
            time.sleep(poll_seconds)


def monitor_loop(run_id: str, poll_seconds: int) -> int:
    state_dir, run = load_run(run_id)
    kind = agent_kind(run)
    with file_lock(state_dir / "monitor.lock", blocking=False) as acquired:
        if not acquired:
            print(f"monitor already running for {run_id}", file=sys.stderr)
            return 2
        pid_path = state_dir / "monitor.pid"
        own_pid = os.getpid()
        atomic_write_text(pid_path, f"{own_pid}\n", 0o600)
        try:
            while True:
                try:
                    status = monitor_once(run_id)
                    complete, _ = finalize(run_id)
                except Exception as exc:  # noqa: BLE001
                    record_controller_error(run_id, exc)
                    status = {
                        "run_id": run_id,
                        "agent_kind": kind,
                        "updated_at": utc_now(),
                        "transient_error": str(exc),
                    }
                    complete = False
                print(json.dumps(status, sort_keys=True), flush=True)
                if complete:
                    return 0
                time.sleep(poll_seconds)
        finally:
            # Do not erase a replacement monitor's registration if systemd or
            # an operator started it while this process was unwinding.
            try:
                registered = int(pid_path.read_text().strip())
            except (OSError, ValueError):
                registered = None
            if registered == own_pid:
                pid_path.unlink(missing_ok=True)


def _budget_watchdog_age(ref: float, checked_at: float) -> float:
    """Validate freshness while allowing bounded cross-sandbox clock skew."""
    age = ref - checked_at
    if (
        not math.isfinite(age)
        or age < -BUDGET_PULSE_MAX_CLOCK_SKEW_SECONDS
        or age > BUDGET_PULSE_MAX_UPSTREAM_AGE_SECONDS
    ):
        raise RuntimeError(f"budget pulse watchdog snapshot is stale ({age:.1f}s)")
    return age


def _supervisor_lock_held(state_dir: Path) -> bool:
    """Return whether the lane supervisor still owns its retry lifecycle."""
    lock_path = state_dir / "supervise.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fd)


def _supervised_retry_gap_allows_host_pulse(
    state_dir: Path, run: dict[str, Any], canonical: dict[str, Any]
) -> bool:
    """Allow host accounting while no billable CPU attempt is alive.

    A provider-rate-limit exit can leave the lane supervisor in exponential
    backoff while an already-dispatched GPU job keeps accruing cost.  There is
    no sandbox watchdog during that gap, so requiring its timestamp to remain
    fresh eventually starves the GPU mirror and makes the worker fail closed.

    The host pulse is authoritative during this narrow state only when the
    last sandbox acknowledged a recoverable ``agent_exit``, the supervisor
    still owns its lock, Harbor is not alive, and the last trusted snapshot
    proves there was no API request in flight.  Completed OpenRouter charges
    are immutable in the durable ledger and CPU/GPU allocation is tracked by
    host lifecycle events, so advancing the host mirror here cannot hide new
    spend.  As soon as the next Harbor attempt is alive, freshness is required
    again.
    """
    if terminal_stop_acknowledged(state_dir):
        return False
    ack_path = state_dir / "STOP_ACK.json"
    try:
        ack_reason = str(json.loads(ack_path.read_text()).get("reason") or "")
    except (OSError, json.JSONDecodeError):
        return False
    if ack_reason != "agent_exit":
        return False
    try:
        if harbor_alive(run):
            return False
    except (KeyError, OSError, ValueError):
        return False

    model_api = (canonical.get("components") or {}).get("model_api") or {}
    pending = model_api.get("pending_request_count")
    if isinstance(pending, bool) or not isinstance(pending, int) or pending != 0:
        return False

    return _supervisor_lock_held(state_dir)


def _budget_pulse_once_unlocked(
    run_id: str, *, now: float | None = None
) -> dict[str, Any]:
    """Refresh the trusted budget mirror without waiting for artifact sync."""
    state_dir, run = load_run(run_id)
    ref = time.time() if now is None else float(now)
    canonical = fetch_budget_watchdog(state_dir, run)
    if not isinstance(canonical, dict) or canonical.get("schema_version") != 2:
        raise RuntimeError("budget pulse has no valid in-sandbox watchdog snapshot")
    if canonical.get("run_id") != run_id:
        raise RuntimeError("budget pulse watchdog run ID mismatch")
    checked_at = canonical.get("checked_at_epoch_s")
    if not isinstance(checked_at, (int, float)) or isinstance(checked_at, bool):
        raise RuntimeError("budget pulse watchdog timestamp is missing")
    pulse_source = "in_sandbox_watchdog"
    try:
        upstream_age = _budget_watchdog_age(ref, float(checked_at))
    except RuntimeError:
        if not _supervised_retry_gap_allows_host_pulse(state_dir, run, canonical):
            raise
        upstream_age = ref - float(checked_at)
        pulse_source = "host_supervised_retry_gap"

    # This is local-only and fast: provider charges come from the freshly
    # fetched watchdog, while host GPU lifecycle events are already persisted
    # locally by the dispatcher.  No history archives or Volume uploads block
    # this path.
    timeline = build_unified_timeline(state_dir, run, upload=False)
    payload = agent_cost.build_snapshot(timeline, state_dir=state_dir)
    # ``build_snapshot`` normally inherits the unified timeline cutoff.  The
    # artifact monitor can legitimately lag while it archives/restores large
    # histories, so that cutoff is not a freshness signal for the independent
    # budget pulse.  Stamp the merged document with this pulse's fresh trusted
    # watchdog read; otherwise a GPU worker can receive new totals carrying an
    # old timestamp and fail closed after 120 seconds despite healthy updates.
    snapshot_epoch = max(ref, float(checked_at))
    payload["checked_at_epoch_s"] = snapshot_epoch
    payload["as_of_epoch_ms"] = round(snapshot_epoch * 1000)
    payload["as_of"] = dt.datetime.fromtimestamp(
        snapshot_epoch, tz=dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["budget_pulse"] = {
        "checked_at": utc_now(),
        "upstream_watchdog_age_seconds": round(upstream_age, 3),
        "interval_seconds": 15,
        "source": pulse_source,
    }
    atomic_write_json(
        state_dir / "telemetry" / "agent-cost.json", payload, mode=0o600
    )

    from event_runtime.compute import worker as gpu_worker

    gpu_mirror = gpu_worker.mirror_gpu_budget(run, payload)
    atomic_write_json(
        state_dir / "telemetry" / "gpu-budget-mirror.json",
        {"schema_version": 1, "updated_at": utc_now(), **gpu_mirror},
        mode=0o600,
    )
    if gpu_mirror.get("gpu_budget_mirror") == "error":
        raise RuntimeError(
            "GPU budget pulse mirror failed: "
            + str(gpu_mirror.get("errors") or "unknown")
        )

    agent_mirror = gpu_worker.mirror_agent_cost(run, payload)
    atomic_write_json(
        state_dir / "telemetry" / "agent-cost-mirror.json",
        {"schema_version": 1, "updated_at": utc_now(), **agent_mirror},
        mode=0o600,
    )
    if agent_mirror.get("agent_cost_mirror") == "error":
        raise RuntimeError(
            "agent cost pulse mirror failed: "
            + str(agent_mirror.get("agent_cost_mirror_error") or "unknown")
        )
    enforce_agent_cost_budget(run_id, state_dir, run, payload)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "updated_at": utc_now(),
        "total_usd": payload.get("total_usd"),
        "budget_remaining_usd": payload.get("budget_remaining_usd"),
        "status": payload.get("status"),
        "upstream_watchdog_age_seconds": round(upstream_age, 3),
        "source": pulse_source,
        "gpu_mirror": gpu_mirror.get("gpu_budget_mirror"),
        "agent_mirror": agent_mirror.get("agent_cost_mirror"),
    }
    atomic_write_json(
        state_dir / "telemetry" / "budget-pulse.json", result, mode=0o600
    )
    return result


def budget_pulse_once(run_id: str, *, now: float | None = None) -> dict[str, Any]:
    """Refresh and propagate one cost snapshot without concurrent writers."""
    state_dir, _run = load_run(run_id)
    lock_path = state_dir / "telemetry" / "agent-cost.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(lock_path):
        return _budget_pulse_once_unlocked(run_id, now=now)


def run_results_finished(state_dir: Path, run: dict[str, Any]) -> bool:
    """Return whether Harbor has durably sealed both run result records.

    A normally completing agent does not write ``STOP_ACK.json`` because no
    controller stop was requested. Its watchdog stops with the sandbox, so a
    pulse process that only watches STOP_ACK will otherwise report an
    indefinitely stale upstream snapshot while finalization is in progress.
    The host-owned job and trial results are the durable natural-completion
    boundary.
    """
    job, trial = discover_job_and_trial(state_dir, run)
    if job is None or trial is None:
        return False
    for path in (job / "result.json", trial / "result.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if not payload.get("finished_at"):
            return False
    return True


def terminal_stop_acknowledged(state_dir: Path) -> bool:
    """Return whether STOP_ACK closes the run rather than one CPU attempt."""
    path = state_dir / "STOP_ACK.json"
    if not path.is_file():
        return False
    try:
        reason = str(json.loads(path.read_text()).get("reason") or "")
    except (OSError, json.JSONDecodeError):
        # An unreadable acknowledgement cannot safely authorize more work.
        return True
    # Supervised provider retries write this when the current CPU sandbox
    # exits.  The next attempt must keep its budget pulse and GPU dispatcher.
    return reason != "agent_exit"


def run_services_should_exit(state_dir: Path, run: dict[str, Any]) -> bool:
    """Keep retry-owned safety services alive across Harbor attempt results."""
    if (state_dir / "FINALIZED.json").is_file() or terminal_stop_acknowledged(
        state_dir
    ):
        return True
    # Each supervised CPU attempt writes finished Harbor results, including
    # transient provider failures.  Those files are not a run boundary while
    # the supervisor still owns the lane and may relaunch the next attempt.
    if _supervisor_lock_held(state_dir):
        return False
    return run_results_finished(state_dir, run)


def budget_pulse_loop(run_id: str, poll_seconds: int) -> int:
    state_dir, run = load_run(run_id)
    with file_lock(state_dir / "budget-pulse.lock", blocking=False) as acquired:
        if not acquired:
            print(f"budget pulse already running for {run_id}", file=sys.stderr)
            return 2
        pid_path = state_dir / "budget-pulse.pid"
        own_pid = os.getpid()
        atomic_write_text(pid_path, f"{own_pid}\n", 0o600)
        try:
            while not run_services_should_exit(state_dir, run):
                started = time.monotonic()
                try:
                    payload = budget_pulse_once(run_id)
                except Exception as exc:  # noqa: BLE001
                    record_controller_error(run_id, exc)
                    payload = {
                        "run_id": run_id,
                        "updated_at": utc_now(),
                        "status": "pulse_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                print(json.dumps(payload, sort_keys=True), flush=True)
                elapsed = time.monotonic() - started
                time.sleep(max(1.0, float(poll_seconds) - elapsed))
            return 0
        finally:
            try:
                registered = int(pid_path.read_text().strip())
            except (OSError, ValueError):
                registered = None
            if registered == own_pid:
                pid_path.unlink(missing_ok=True)


def gpu_dispatch_loop(run_id: str, poll_seconds: int) -> int:
    """Reconcile and dispatch GPU jobs independently of observability I/O."""

    state_dir, run = load_run(run_id)
    if not run.get("cpu_agent_gpu_worker"):
        return 0
    with file_lock(state_dir / "gpu-dispatch-loop.lock", blocking=False) as acquired:
        if not acquired:
            print(f"GPU dispatch loop already running for {run_id}", file=sys.stderr)
            return 2
        pid_path = state_dir / "gpu-dispatch-loop.pid"
        own_pid = os.getpid()
        atomic_write_text(pid_path, f"{own_pid}\n", 0o600)
        try:
            while not run_services_should_exit(state_dir, run):
                started = time.monotonic()
                try:
                    from event_runtime.compute import worker as gpu_worker

                    payload = gpu_worker.dispatch_once(run_id)
                except Exception as exc:  # noqa: BLE001
                    record_controller_error(run_id, exc)
                    payload = {
                        "run_id": run_id,
                        "updated_at": utc_now(),
                        "status": "gpu_dispatch_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                print(json.dumps(payload, sort_keys=True), flush=True)
                elapsed = time.monotonic() - started
                time.sleep(max(1.0, float(poll_seconds) - elapsed))
            return 0
        finally:
            try:
                registered = int(pid_path.read_text().strip())
            except (OSError, ValueError):
                registered = None
            if registered == own_pid:
                pid_path.unlink(missing_ok=True)


def download_run_volume(
    state_dir: Path, run: dict[str, Any], destination: Path
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    run_command(
        modal_command(
            "volume",
            "get",
            "--force",
            str(run["volume_name"]),
            f"runs/{run['run_id']}",
            str(destination),
        ),
        run=run,
        timeout=1800,
    )
    candidates = [
        path.parent for path in destination.rglob("restic/config") if path.is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError("downloaded run does not contain one restic repository")
    return candidates[0].parent


def restic_command(
    run_root: Path, *args: str, timeout: int = 7200
) -> subprocess.CompletedProcess[str]:
    repo = run_root / "restic"
    password = run_root / "secrets" / "restic-password"
    if not repo.is_dir() or not password.is_file():
        raise RuntimeError("restic repository or password file is missing")
    command = [
        "run-heavy",
        "restic",
        "-r",
        str(repo),
        "--password-file",
        str(password),
        *args,
    ]
    return run_command(command, timeout=timeout)


def check_recovery(run_id: str, cache: Path | None = None) -> dict[str, Any]:
    state_dir, run = load_run(run_id)
    if cache is None:
        cache = (
            state_dir
            / "recovery"
            / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        run_root = download_run_volume(state_dir, run, cache)
    else:
        matches = [
            path.parent.parent
            for path in cache.rglob("restic/config")
            if path.is_file()
        ]
        if len(matches) != 1:
            raise RuntimeError("cache does not contain one downloaded run")
        run_root = matches[0]
    restic = restic_command(run_root, "check")
    archives = verify_attempt_archives(run_root / "host-archives")
    return {
        "run_id": run_id,
        "agent_kind": agent_kind(run),
        "run_root": str(run_root),
        "restic_check": restic.stdout.strip(),
        "archives": archives,
        "valid": restic.returncode == 0 and archives["valid"],
    }


def recover_run(
    run_id: str,
    destination: Path,
    *,
    snapshot: str = "latest",
    force: bool = False,
) -> dict[str, Any]:
    state_dir, run = load_run(run_id)
    if destination.exists() and any(destination.iterdir()) and not force:
        raise RuntimeError("restore destination is not empty; pass --force to use it")
    destination.mkdir(parents=True, exist_ok=True)
    cache = (
        state_dir
        / "recovery"
        / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = download_run_volume(state_dir, run, cache)
    check = restic_command(run_root, "check")
    restic_command(run_root, "restore", snapshot, "--target", str(destination))
    archives = verify_attempt_archives(run_root / "host-archives")
    payload = {
        "run_id": run_id,
        "agent_kind": agent_kind(run),
        "restored_at": utc_now(),
        "snapshot": snapshot,
        "destination": str(destination.resolve()),
        "volume_cache": str(run_root),
        "restic_check": check.stdout.strip(),
        "attempt_archives": archives,
    }
    atomic_write_json(state_dir / "last-recovery.json", payload, mode=0o600)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "status",
        "stop",
        "finalize",
        "monitor",
        "budget-pulse",
        "gpu-dispatch-loop",
        "wait",
        "check",
        "recover",
        "gpu-dispatch",
        "gpu-terminate",
        "modal-cost",
    ):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        if name in {"monitor", "budget-pulse", "gpu-dispatch-loop", "wait"}:
            command.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
        if name == "wait":
            command.add_argument(
                "--timeout-seconds", type=int, default=DEFAULT_WAIT_SECONDS
            )
        if name == "status":
            command.add_argument("--offline", action="store_true")
        if name == "check":
            command.add_argument("--cache", type=Path)
        if name == "recover":
            command.add_argument("--destination", type=Path, required=True)
            command.add_argument("--snapshot", default="latest")
            command.add_argument("--force", action="store_true")
        if name == "gpu-terminate":
            command.add_argument("--job-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "status":
            state_dir, run = load_run(args.run_id)
            payload = status_snapshot(state_dir, run, include_remote=not args.offline)
        elif args.command == "stop":
            payload = request_stop(args.run_id)
        elif args.command == "monitor":
            return monitor_loop(args.run_id, args.poll_seconds)
        elif args.command == "budget-pulse":
            return budget_pulse_loop(args.run_id, args.poll_seconds)
        elif args.command == "gpu-dispatch-loop":
            return gpu_dispatch_loop(args.run_id, args.poll_seconds)
        elif args.command == "wait":
            return wait_for_run(args.run_id, args.timeout_seconds, args.poll_seconds)
        elif args.command == "finalize":
            complete, payload = finalize(args.run_id)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if complete else 2
        elif args.command == "check":
            payload = check_recovery(args.run_id, args.cache)
        elif args.command == "gpu-dispatch":
            from event_runtime.compute import worker as gpu_worker

            payload = gpu_worker.dispatch_once(args.run_id)
        elif args.command == "gpu-terminate":
            from event_runtime.compute import worker as gpu_worker

            _, run = load_run(args.run_id)
            payload = gpu_worker.terminate_job(run, args.job_id)
        elif args.command == "modal-cost":
            state_dir, _run = load_run(args.run_id)
            payload = modal_cost.collect_provider_billing(state_dir)
        else:
            payload = recover_run(
                args.run_id,
                args.destination,
                snapshot=args.snapshot,
                force=args.force,
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
