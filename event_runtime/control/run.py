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
from event_runtime.control import integrity as run_integrity  # noqa: E402
from event_runtime.export.timeline import (  # noqa: E402
    SCHEMA_VERSION as UNIFIED_TIMELINE_SCHEMA_VERSION,
)

OPS_ROOT = ROOT / "runs" / "ops"
FRONTIER_SCRIPT = ROOT / "event_runtime/export/frontier.py"
RECONSTRUCT_MODEL_USAGE_SCRIPT = ROOT / "event_runtime/cost/model_usage.py"
UV = Path("/home/ubuntu/.local/bin/uv")
POLL_SECONDS = 30
DEFAULT_WAIT_SECONDS = 3 * 60 * 60
MODAL_VOLUME_LIST_MIN_GAP_SECONDS = 1.0
MODAL_VOLUME_LIST_MAX_ATTEMPTS = 5
MODAL_VOLUME_LIST_BACKOFF_MAX_SECONDS = 30.0
MODAL_VOLUME_LIST_LOCK = OPS_ROOT / ".modal-volume-list.lock"
MODAL_VOLUME_LIST_STATE = OPS_ROOT / ".modal-volume-list-state.json"
BUDGET_PULSE_MAX_UPSTREAM_AGE_SECONDS = 60.0
BUDGET_PULSE_MAX_CLOCK_SKEW_SECONDS = 60.0
# A fresh Modal sandbox starts the ledger proxy/watchdog before Codex, but the
# first host pulse can race the watchdog's first atomic snapshot.  Represent
# that bounded interval explicitly; after it expires the same absence is a
# fail-closed infrastructure error.
BUDGET_PULSE_STARTUP_GRACE_SECONDS = 60.0
DURABLE_TRACE_LIVE_SYNC_TIMEOUT_SECONDS = 60
DURABLE_TRACE_FINAL_SYNC_TIMEOUT_SECONDS = 300
AGENT_STOP_GRACE_SECONDS = 15.0
AGENT_STOP_FORCE_WAIT_SECONDS = 30.0
AGENT_STOP_POLL_SECONDS = 1.0
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
    if kind not in {"codex", "deepseek-harness"}:
        raise ValueError(f"{path} has unsupported agent_kind: {kind!r}")
    return state_dir, state


def agent_kind(run: dict[str, Any]) -> str:
    """Return the required agent kind from the current run schema."""
    kind = run.get("agent_kind")
    if kind not in {"codex", "deepseek-harness"}:
        raise ValueError(f"run has unsupported agent_kind: {kind!r}")
    return str(kind)


def update_run_fields(state_dir: Path, **updates: Any) -> dict[str, Any]:
    """Atomically update the authoritative single-attempt run record."""
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
    argv = list(command)
    if _modal_volume_list_operation(argv):
        return run_modal_volume_list_command(
            argv,
            run=run,
            check=check,
            timeout=timeout,
        )
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=command_env(run or {}),
        check=check,
        timeout=timeout,
    )


def _modal_volume_list_operation(command: Sequence[str]) -> str | None:
    """Return the Modal operation when it consumes VolumeListFiles quota."""
    argv = [str(item) for item in command]
    for index in range(len(argv) - 2):
        if argv[index : index + 2] == ["modal", "volume"]:
            operation = argv[index + 2]
            return operation if operation in {"get", "ls"} else None
    return None


def _modal_volume_rate_limited(result: subprocess.CompletedProcess[str]) -> bool:
    detail = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return "volumelistfiles rate limit exceeded" in detail or (
        "rate limit" in detail and "volume" in detail
    )


def _modal_volume_list_state() -> dict[str, Any]:
    try:
        payload = json.loads(MODAL_VOLUME_LIST_STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def run_modal_volume_list_command(
    command: Sequence[str],
    *,
    run: dict[str, Any] | None,
    check: bool,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Serialize/persistently pace list-based Modal Volume operations.

    Six independent lane services share one Modal account-wide
    ``VolumeListFiles`` quota.  A host lock prevents synchronized polling
    bursts, while the persisted cooldown makes every process honor a rate-limit
    response observed by any other process.  Exact-file reads bypass this path.
    """
    argv = list(command)
    operation = _modal_volume_list_operation(argv) or "unknown"
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, MODAL_VOLUME_LIST_MAX_ATTEMPTS + 1):
        with file_lock(MODAL_VOLUME_LIST_LOCK):
            state = _modal_volume_list_state()
            now = time.time()
            not_before = max(
                float(state.get("last_finished_at_epoch_s") or 0)
                + MODAL_VOLUME_LIST_MIN_GAP_SECONDS,
                float(state.get("not_before_epoch_s") or 0),
            )
            if not_before > now:
                time.sleep(not_before - now)
            started_at = time.time()
            result = subprocess.run(
                argv,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=command_env(run or {}),
                check=False,
                timeout=timeout,
            )
            finished_at = time.time()
            limited = result.returncode != 0 and _modal_volume_rate_limited(result)
            cooldown = (
                min(
                    MODAL_VOLUME_LIST_BACKOFF_MAX_SECONDS,
                    float(2 ** max(0, attempt)),
                )
                if limited
                else 0.0
            )
            atomic_write_json(
                MODAL_VOLUME_LIST_STATE,
                {
                    "schema_version": 1,
                    "operation": operation,
                    "run_id": (run or {}).get("run_id"),
                    "attempt": attempt,
                    "started_at_epoch_s": started_at,
                    "last_finished_at_epoch_s": finished_at,
                    "not_before_epoch_s": finished_at + cooldown,
                    "rate_limited": limited,
                    "returncode": result.returncode,
                },
                mode=0o600,
            )
        if not limited:
            break
    assert result is not None
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            argv,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


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
            "test -x /opt/sprint-agent-supervisor.sh && printf SPRINT_AGENT",
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
    # The stock ``modal volume get`` CLI calls VolumeListFiles even for an
    # exact filename. Use the exact-file RPC helper so budget/heartbeat/index
    # polling cannot starve the shared directory-list control plane.
    result = run_command(
        [
            sys.executable,
            "-m",
            "event_runtime.control.volume_read",
            str(run["volume_name"]),
            remote_path,
        ],
        run=run,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        return None
    return result.stdout or ""


def volume_get_bytes(
    run: dict[str, Any],
    remote_path: str,
    *,
    timeout_seconds: int = 60,
    max_bytes: int,
) -> bytes | None:
    """Read one bounded binary file without consuming directory-list quota."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "event_runtime.control.volume_read",
            str(run["volume_name"]),
            remote_path,
            "--max-bytes",
            str(max_bytes),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=command_env(run),
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        return None
    return bytes(result.stdout)


def volume_download_exact(
    run: dict[str, Any],
    remote_path: str,
    destination: Path,
    *,
    timeout_seconds: int = 60,
    max_bytes: int,
) -> bool:
    """Stream one bounded binary file to disk without a directory listing."""
    destination.unlink(missing_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "event_runtime.control.volume_read",
            str(run["volume_name"]),
            remote_path,
            "--max-bytes",
            str(max_bytes),
            "--output",
            str(destination),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=command_env(run),
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        return False
    return destination.is_file()


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
    error: str | None = None
    if force:
        # Final reconciliation needs every immutable request record. This is a
        # deliberately rare recursive directory download and therefore passes
        # through the host-wide VolumeListFiles coordinator.
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
        ok = result.returncode == 0
        if not ok:
            error = (result.stderr or result.stdout)[-1000:]
        mode = "full-final"
    else:
        # Live cost/token rendering only needs the cumulative proxy summary.
        # Pulling the whole requests directory once per minute caused a
        # recursive VolumeListFiles scan in every lane.
        remote = f"runs/{run['run_id']}/api-usage/summary.json"
        try:
            text = volume_get_text(run, remote, timeout_seconds=30)
        except subprocess.TimeoutExpired as exc:
            text = None
            error = f"TimeoutExpired after 30s: {exc}"
        ok = text is not None
        if ok:
            local = destination / "api-usage" / "summary.json"
            atomic_write_text(local, text, mode=0o600)
        elif error is None:
            error = "exact provider summary is not available yet"
        mode = "summary-live"
    payload = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "synced_at": utc_now(),
        "synced_at_epoch_s": now,
        "mode": mode,
        "ok": ok,
        "error": error,
    }
    atomic_write_json(stamp, payload, mode=0o600)
    return ok


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


def reconstruct_model_usage(state_dir: Path, run: dict[str, Any]) -> bool:
    """Materialize ATIF and cost ledgers from immutable per-attempt chunks."""
    chunks = state_dir / "durable-trace" / "raw"
    if not chunks.is_dir() or not any(
        chunks.glob(pattern)
        for pattern in (
            "cpu-attempt-*/codex/*/chunks/*.jsonl",
            "cpu-attempt-*/deepseek-harness/*/chunks/*.jsonl",
        )
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
            str(RECONSTRUCT_MODEL_USAGE_SCRIPT),
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


def agent_container_running(run: dict[str, Any], container_id: str) -> bool:
    """Query Modal authoritatively instead of treating signal delivery as exit."""
    app_id = run.get("app_id")
    if not isinstance(app_id, str) or not app_id.startswith("ap-"):
        return False
    return container_id in containers_for_app(run, app_id)


def terminate_agent_container(run: dict[str, Any], container_id: str) -> None:
    """Force-close an agent sandbox after its bounded graceful-stop window."""
    run_command(
        modal_command("container", "stop", "--yes", container_id),
        run=run,
        timeout=60,
    )


def persist_host_stop_ack(
    state_dir: Path,
    run: dict[str, Any],
    payload: dict[str, Any],
    *,
    forced: bool,
) -> dict[str, Any]:
    """Seal host-observed CPU termination after Modal no longer lists it."""
    ack = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "reason": str(payload.get("reason") or "operator_stop"),
        "requested_at": payload.get("requested_at"),
        "acknowledged_at": utc_now(),
        "source": "host_modal_container_audit",
        "container_id": payload.get("container_id"),
        "forced": forced,
    }
    atomic_write_json(state_dir / "STOP_ACK.json", ack, mode=0o600)
    return ack


def request_stop(
    run_id: str,
    *,
    reason: str = "operator_stop",
    wait_for_termination: bool = False,
) -> dict[str, Any]:
    state_dir, run, payload = persist_stop_request(run_id, reason=reason)
    kind = agent_kind(run)
    marker = state_dir / "STOP_REQUESTED.json"

    # Signal the CPU harness before waiting for the GPU dispatch lock. A
    # Sandbox.create call can hold that lock for minutes; waiting for it first
    # would let API and CPU spend continue past the durable budget marker.
    agent_stop_error = None
    container: str | None = None
    container_discovery_succeeded = False
    try:
        container = discover_agent_container(state_dir, run)
        container_discovery_succeeded = True
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
    expected_reason = str(payload.get("reason") or reason)
    if ack and str(ack.get("reason") or "") == expected_reason:
        return {
            "status": "acknowledged",
            "agent_kind": kind,
            "ack": ack,
            "agent_stop_error": agent_stop_error,
            "gpu_workers_stopped": gpu_stopped,
            "gpu_stop_error": gpu_stop_error,
        }

    requested_epoch = parse_iso(str(payload.get("requested_at") or ""))
    request_age = (
        0.0 if requested_epoch is None else max(0.0, time.time() - requested_epoch)
    )
    should_wait = wait_for_termination or request_age >= AGENT_STOP_GRACE_SECONDS
    if should_wait and container_discovery_succeeded:
        if container is None:
            ack = persist_host_stop_ack(state_dir, run, payload, forced=False)
            return {
                "status": "acknowledged",
                "agent_kind": kind,
                "ack": ack,
                "agent_stop_error": agent_stop_error,
                "gpu_workers_stopped": gpu_stopped,
                "gpu_stop_error": gpu_stop_error,
            }

        grace_deadline = time.monotonic() + max(
            0.0, AGENT_STOP_GRACE_SECONDS - request_age
        )
        while time.monotonic() < grace_deadline:
            if not agent_container_running(run, container):
                ack = persist_host_stop_ack(state_dir, run, payload, forced=False)
                return {
                    "status": "acknowledged",
                    "agent_kind": kind,
                    "ack": ack,
                    "agent_stop_error": agent_stop_error,
                    "gpu_workers_stopped": gpu_stopped,
                    "gpu_stop_error": gpu_stop_error,
                }
            time.sleep(AGENT_STOP_POLL_SECONDS)

        forced_error = None
        try:
            terminate_agent_container(run, container)
            force_deadline = time.monotonic() + AGENT_STOP_FORCE_WAIT_SECONDS
            while agent_container_running(run, container):
                if time.monotonic() >= force_deadline:
                    raise TimeoutError(
                        f"Modal container {container} remained active after force-stop"
                    )
                time.sleep(AGENT_STOP_POLL_SECONDS)
            ack = persist_host_stop_ack(state_dir, run, payload, forced=True)
            return {
                "status": "acknowledged",
                "agent_kind": kind,
                "ack": ack,
                "agent_stop_error": agent_stop_error,
                "gpu_workers_stopped": gpu_stopped,
                "gpu_stop_error": gpu_stop_error,
            }
        except Exception as exc:  # noqa: BLE001
            forced_error = f"{type(exc).__name__}: {exc}"
            # Modal may finish graceful termination between the last poll and
            # the force-stop RPC. Its CLI then reports "not found" even though
            # the required postcondition is already true. Re-audit instead of
            # converting that benign race into a false teardown failure.
            try:
                if not agent_container_running(run, container):
                    ack = persist_host_stop_ack(state_dir, run, payload, forced=True)
                    return {
                        "status": "acknowledged",
                        "agent_kind": kind,
                        "ack": ack,
                        "agent_stop_error": agent_stop_error,
                        "forced_stop_warning": forced_error,
                        "gpu_workers_stopped": gpu_stopped,
                        "gpu_stop_error": gpu_stop_error,
                    }
            except Exception as audit_exc:  # noqa: BLE001
                forced_error += (
                    "; postcondition audit failed: "
                    f"{type(audit_exc).__name__}: {audit_exc}"
                )
        return {
            "status": "teardown_failed",
            "agent_kind": kind,
            **payload,
            "agent_stop_error": agent_stop_error,
            "forced_stop_error": forced_error,
            "gpu_workers_stopped": gpu_stopped,
            "gpu_stop_error": gpu_stop_error,
        }

    if wait_for_termination and not container_discovery_succeeded:
        return {
            "status": "teardown_failed",
            "agent_kind": kind,
            **payload,
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
    dedicated pulse is alive it alone owns CPU-agent propagation; the separate
    GPU pulse consumes that same snapshot. An artifact timeline can lag the
    live watchdog and must never overwrite either fresher heartbeat merely
    because it finished later.
    """
    with file_lock(state_dir / "telemetry" / "agent-cost.lock"):
        cost_payload = agent_cost.build_snapshot(timeline, state_dir=state_dir)
        cost_path = state_dir / "telemetry" / "agent-cost.json"
        atomic_write_json(cost_path, cost_payload, mode=0o600)
        enforce_agent_cost_budget(run_id, state_dir, run, cost_payload)

    # Once the run boundary is durable there is no live consumer to update.
    # Replaying an expired sandbox's last mirror during archival is both
    # unnecessary and unsafe: a previous attempt may legitimately have a
    # different snapshot identity. Keep the final host ledger, but never let
    # that stale live channel abort finalization.
    if run_services_should_exit(state_dir, run):
        terminal = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "agent_cost_mirror": "terminal_snapshot_not_mirrored",
        }
        atomic_write_json(
            state_dir / "telemetry" / "agent-cost-mirror.json",
            terminal,
            mode=0o600,
        )
        atomic_write_json(
            state_dir / "telemetry" / "gpu-budget-mirror.json",
            {
                "schema_version": 1,
                "updated_at": terminal["updated_at"],
                "gpu_budget_mirror": "terminal_snapshot_not_mirrored",
            },
            mode=0o600,
        )
        return cost_payload

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
        # The independent GPU pulse owns its status file. Do not overwrite its
        # sandbox IDs or errors with an artifact-monitor delegation marker.
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
                "supervisor/heartbeat.json",
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
) -> dict[str, Any]:
    state_dir, run = load_run(run_id)
    Path("/data/.keepalive").touch()
    # Only the host's independently reconstructed ledger can create a trusted
    # STOP_REQUESTED marker. The agent can write its shared Modal Volume, so a
    # cloud-side BUDGET_STOP_REQUESTED file is audit data rather than authority.
    stop_marker = state_dir / "STOP_REQUESTED.json"
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
                if run.get("provider_usage_ledger_required"):
                    sync_durable_api_usage(state_dir, run)
                if run.get("usage_audit_required"):
                    reconstruct_model_usage(state_dir, run)
                timeline = build_unified_timeline(state_dir, run, upload=upload)
                refresh_agent_cost_snapshot(run_id, state_dir, run, timeline)
            except Exception as exc:  # noqa: BLE001
                record_controller_error(run_id, exc)
    return status_snapshot(state_dir, run, include_remote=include_remote)


def refresh_public_projection(
    run_id: str, *, upload: bool = True, launch_worker: bool = True
) -> dict[str, Any]:
    """Refresh replay/site artifacts outside the experiment monitor.

    This observer path may be slow or fail indefinitely without affecting GPU
    dispatch, telemetry, budget enforcement, stop handling, or finalization.
    """

    state_dir, run = load_run(run_id)
    job, trial = discover_job_and_trial(state_dir, run)
    if not job or not trial:
        return {"run_id": run_id, "status": "waiting_for_harbor_artifacts"}
    frontier_path = state_dir / "frontier-state.json"
    if not worker_alive(state_dir):
        sync_frontier_artifacts(state_dir, run, upload=upload)
        state = scan_frontier(
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
        return state
    try:
        state = json.loads(frontier_path.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    return state


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
    """Validate accounting across the complete authoritative CPU execution."""
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
    expected_openrouter_cost_basis = (run.get("budget_enforcement") or {}).get(
        "api_budget_cost_basis"
    )
    openrouter_list_price = expected_openrouter_cost_basis in {
        "openrouter_list_price_before_endpoint_discount",
        "openrouter_list_price_with_deepseek_peak_floor",
        (
            "openai_sol_official_promotional_list_price_after_"
            "openrouter_discount_reversal"
        ),
    }
    if (
        openrouter_list_price
        and audit.get("calculated_api_usage_cost_basis")
        != expected_openrouter_cost_basis
    ):
        details.append(
            "run usage audit does not use the configured OpenRouter benchmark cost"
        )
    provider_billing_reconciled = bool(
        openrouter_list_price and audit.get("provider_billing_reconciled") is True
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
                        f"run usage request {index} has invalid benchmark cost"
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


def provider_usage_ledger_settled(
    state_dir: Path, run: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Require the provider's durable billing ledger to have no open request."""
    if not run.get("provider_usage_ledger_required"):
        return True, []
    root = state_dir / "provider-api-usage"
    summaries = sorted(root.glob("**/summary.json")) if root.is_dir() else []
    if not summaries:
        return False, ["provider usage summary is missing"]
    details: list[str] = []
    for path in summaries:
        try:
            summary = json.loads(path.read_text())
            valid = (
                summary.get("schema_version") == 3
                and summary.get("run_id") == run.get("run_id")
                and int(summary.get("pending_request_count") or 0) == 0
                and int(summary.get("in_flight_request_count") or 0) == 0
                and int(summary.get("cost_recovery_required_count") or 0) == 0
                and not (summary.get("in_flight_request_ids") or [])
                and not (summary.get("cost_recovery_required_request_ids") or [])
                and isinstance(
                    summary.get("provider_billed_model_api_usd"), (int, float)
                )
                and not isinstance(summary.get("provider_billed_model_api_usd"), bool)
                and math.isfinite(float(summary.get("provider_billed_model_api_usd")))
                and float(summary.get("provider_billed_model_api_usd")) >= 0
                and isinstance(summary.get("model_api_usd"), (int, float))
                and not isinstance(summary.get("model_api_usd"), bool)
                and math.isfinite(float(summary.get("model_api_usd")))
                and float(summary.get("model_api_usd"))
                >= float(summary.get("provider_billed_model_api_usd"))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            details.append(f"provider usage ledger is not settled: {path.name}")
    return not details, details


def final_conditions(
    state_dir: Path, run: dict[str, Any]
) -> tuple[bool, dict[str, bool], list[str]]:
    job, trial = discover_job_and_trial(state_dir, run)
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
        "submission_bridge_drained": False,
        "attempt_archives": False,
        "artifact_manifest": False,
        "finished_at": False,
        "harbor_exited": not harbor_alive(run),
        # The non-restarting host wrapper owns the one authoritative CPU
        # process boundary. Harbor can seal its result files a few moments
        # before that wrapper returns, so finalization must wait for the exit
        # record instead of certifying a still-unwinding process.
        "cpu_process_exited": (state_dir / "CPU_TRIAL_EXIT.json").is_file(),
    }
    conditions["continuous_result_set"] = False
    if run.get("unified_timeline_required"):
        conditions["unified_timeline_ready"] = False
    if run.get("usage_audit_required"):
        conditions["usage_audit_complete"] = False
    details: list[str] = []
    if run.get("evaluation_result_policy") != "all_blind_archival_submissions":
        details.append("run does not use the current archival scoring policy")
    if not job or not trial:
        return False, conditions, ["job or trial path is not available"]

    ledger_path = trial / "artifacts" / "continuous" / "ledger.jsonl"
    ledger = read_ledger(ledger_path)
    conditions["ledger_parseable"] = not ledger.errors
    conditions["ledger_terminal"] = all(row_terminal(row) for row in ledger.rows)
    details.extend(ledger.errors)

    bridge_root = state_dir / "submission-bridge"
    bridge_records: list[dict[str, Any]] = []
    bridge_errors: list[str] = []
    if bridge_root.is_dir():
        for path in sorted(bridge_root.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                bridge_errors.append(
                    f"invalid submission bridge record {path.name}: {exc}"
                )
                continue
            if not isinstance(record, dict):
                bridge_errors.append(
                    f"invalid submission bridge record {path.name}: not an object"
                )
                continue
            bridge_records.append(record)
    ledger_names = {str(row.get("name") or "") for row in ledger.rows}
    missing_bridge_names = [
        str(record.get("queue_name") or "")
        for record in bridge_records
        if record.get("state") == "forwarded"
        and str(record.get("queue_name") or "") not in ledger_names
    ]
    nonforwarded_bridge = [
        str(record.get("submission_id") or "unknown")
        for record in bridge_records
        if record.get("state") != "forwarded"
    ]
    conditions["submission_bridge_drained"] = not (
        bridge_errors or missing_bridge_names or nonforwarded_bridge
    )
    details.extend(bridge_errors)
    if missing_bridge_names:
        details.append(
            "GPU submission(s) forwarded but absent from Harbor ledger: "
            + ", ".join(missing_bridge_names)
        )
    if nonforwarded_bridge:
        details.append(
            "GPU submission bridge request(s) not forwarded: "
            + ", ".join(nonforwarded_bridge)
        )

    manifest_path = trial / "artifacts" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        conditions["artifact_manifest"] = isinstance(manifest, list)
    except (OSError, json.JSONDecodeError):
        pass

    try:
        trial_result = json.loads((trial / "result.json").read_text())
        job_result = json.loads((job / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        trial_result, job_result = {}, {}
    summary = trial_result.get("continuous_verification")
    conditions["continuous_result_set"] = bool(
        isinstance(summary, dict)
        and isinstance(summary.get("submissions"), list)
        and conditions["ledger_parseable"]
        and conditions["ledger_terminal"]
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
        # The archival policy intentionally survives CPU-container teardown.
        # Its all-attempt host reconstruction is authoritative because it
        # attests every raw chunk, request, cost, and reconstructed ATIF
        # checksum.
        if run_audit_ready:
            trial_audit_ready, trial_audit_details = True, []
        else:
            trial_audit_ready, trial_audit_details = usage_audit_ready(trial, run)
        conditions["usage_audit_complete"] = trial_audit_ready and run_audit_ready
        details.extend(trial_audit_details)
        details.extend(run_audit_details)
    if run.get("provider_usage_ledger_required"):
        provider_settled, provider_details = provider_usage_ledger_settled(
            state_dir, run
        )
        conditions["provider_usage_ledger_settled"] = provider_settled
        details.extend(provider_details)
    if run.get("modal_billing_required"):
        modal_ready, modal_details = modal_billing_ready(state_dir, run["run_id"])
        conditions["modal_billing_complete"] = modal_ready
        details.extend(modal_details)
    for name, value in conditions.items():
        if not value:
            details.append(name)
    return all(conditions.values()), conditions, details


def _finalize_owned(
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
            and existing.get("integrity", {}).get("schema_version")
            == run_integrity.SCHEMA_VERSION
            and existing.get("timeline_schema_version")
            == UNIFIED_TIMELINE_SCHEMA_VERSION
            and (
                not run.get("provider_usage_ledger_required")
                or existing.get("conditions", {}).get("provider_usage_ledger_settled")
                is True
            )
        ):
            return True, existing
    terminal_before_refresh = run_services_should_exit(state_dir, run)
    monitor_once(
        run_id,
        upload=upload,
        include_remote=include_remote,
    )
    # Final reconciliation is intentionally expensive: it recursively imports
    # the immutable trace and every provider request record.  It must never run
    # on each live monitor tick.  The previous behavior doubled monitor work
    # and forced six lanes through at least twelve recursive VolumeListFiles
    # scans per minute before any GPU queue polling was counted.
    if not (terminal_before_refresh or run_services_should_exit(state_dir, run)):
        return False, {
            "schema_version": 1,
            "timeline_schema_version": UNIFIED_TIMELINE_SCHEMA_VERSION,
            "run_id": run_id,
            "agent_kind": agent_kind(run),
            "complete": False,
            "conditions": {"run_terminal": False},
            "details": ["run is still active; final reconciliation deferred"],
            "checked_at": utc_now(),
        }
    # Natural completion and an explicit budget/operator stop both close the
    # allocation window. Recover per-job streams in either case; otherwise a
    # naturally completing lane can finalize from a stale five-minute live
    # telemetry snapshot.
    if run.get("unified_timeline_required"):
        sync_durable_telemetry(state_dir, run, force=True)
    provider_usage_required = bool(run.get("provider_usage_ledger_required"))
    if provider_usage_required:
        sync_durable_api_usage(state_dir, run, force=True)
    if run.get("usage_audit_required"):
        sync_durable_trace(state_dir, run, force=True)
        reconstruct_model_usage(state_dir, run)
    if (provider_usage_required or run.get("usage_audit_required")) and run.get(
        "unified_timeline_required"
    ):
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
    integrity = run_integrity.build_integrity_report(state_dir, run)
    payload["integrity"] = integrity
    payload["benchmark_valid"] = integrity["benchmark_valid"]
    atomic_write_json(state_dir / "INTEGRITY.json", integrity, mode=0o444)
    if integrity["replacement_required"]:
        atomic_write_json(
            state_dir / "REPLACEMENT_REQUIRED.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "reason": "invalid_infrastructure",
                "integrity_reasons": integrity["reasons"],
                "created_at": utc_now(),
            },
            mode=0o444,
        )
    else:
        # A newer integrity schema may clear a false-positive from an older
        # certification pass. Do not leave that obsolete derived marker behind.
        (state_dir / "REPLACEMENT_REQUIRED.json").unlink(missing_ok=True)
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
            state_dir / "INTEGRITY.json",
            f"runs/{run_id}/state/INTEGRITY.json",
        )
        volume_upload(
            run,
            final_path,
            f"runs/{run_id}/state/FINALIZED.json",
        )
    return True, payload


def finalize(
    run_id: str,
    *,
    upload: bool = True,
    include_remote: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Finalize one run under a nonblocking, process-wide ownership lease.

    Every run has an independent monitor, while its batch monitor is also able
    to recover finalization if that lane monitor dies.  Without a dedicated
    lease both processes can recursively download the same durable Volume and
    compete with every other lane for the account-wide VolumeListFiles lock.
    The observer must return promptly when another healthy finalizer owns the
    work; it can consume ``FINALIZED.json`` on its next cycle.
    """

    state_dir, run = load_run(run_id)
    with file_lock(state_dir / "finalize.lock", blocking=False) as acquired:
        if acquired:
            return _finalize_owned(
                run_id,
                upload=upload,
                include_remote=include_remote,
            )

    final_path = state_dir / "FINALIZED.json"
    try:
        payload = json.loads(final_path.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {
            "schema_version": 1,
            "timeline_schema_version": UNIFIED_TIMELINE_SCHEMA_VERSION,
            "run_id": run_id,
            "agent_kind": agent_kind(run),
            "complete": False,
            "conditions": {"finalization_lease_available": False},
            "details": ["another host process is finalizing this run"],
            "checked_at": utc_now(),
        }
    return bool(payload.get("complete") is True), payload


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
                    complete, finalization = finalize(run_id)
                    status_path = state_dir / "status.json"
                    if status_path.is_file():
                        try:
                            status = json.loads(status_path.read_text())
                        except (OSError, json.JSONDecodeError):
                            status = finalization
                    else:
                        status = finalization
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


def _budget_pulse_once_unlocked(
    run_id: str, *, now: float | None = None
) -> dict[str, Any]:
    """Refresh the trusted budget mirror without waiting for artifact sync."""
    state_dir, run = load_run(run_id)
    ref = time.time() if now is None else float(now)
    canonical = fetch_budget_watchdog(state_dir, run)
    if not isinstance(canonical, dict) or canonical.get("schema_version") != 2:
        created_at = parse_iso(run.get("created_at"))
        startup_age = None if created_at is None else max(0.0, ref - created_at)
        if (
            startup_age is not None
            and startup_age <= BUDGET_PULSE_STARTUP_GRACE_SECONDS
        ):
            result = {
                "schema_version": 1,
                "run_id": run_id,
                "updated_at": utc_now(),
                "status": "watchdog_starting",
                "startup_age_seconds": round(startup_age, 3),
                "startup_grace_seconds": BUDGET_PULSE_STARTUP_GRACE_SECONDS,
                "source": "host_watchdog_startup_gate",
                "gpu_mirror": "not_started",
                "agent_mirror": "not_started",
            }
            atomic_write_json(
                state_dir / "telemetry" / "budget-pulse.json",
                result,
                mode=0o600,
            )
            return result
        raise RuntimeError("budget pulse has no valid in-sandbox watchdog snapshot")
    if canonical.get("run_id") != run_id:
        raise RuntimeError("budget pulse watchdog run ID mismatch")
    checked_at = canonical.get("checked_at_epoch_s")
    if not isinstance(checked_at, (int, float)) or isinstance(checked_at, bool):
        raise RuntimeError("budget pulse watchdog timestamp is missing")
    pulse_source = "in_sandbox_watchdog"
    upstream_age = _budget_watchdog_age(ref, float(checked_at))

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
    atomic_write_json(state_dir / "telemetry" / "agent-cost.json", payload, mode=0o600)

    # STOP_ACK can arrive while this pulse is rebuilding the local snapshot.
    # Preserve the final accounting, but never race Modal teardown by
    # contacting a sandbox after terminal state is durable.
    if budget_safety_should_exit(state_dir, run):
        mirror_status = "terminal_snapshot_not_mirrored"
        atomic_write_json(
            state_dir / "telemetry" / "gpu-budget-mirror.json",
            {
                "schema_version": 1,
                "updated_at": utc_now(),
                "gpu_budget_mirror": mirror_status,
            },
            mode=0o600,
        )
        atomic_write_json(
            state_dir / "telemetry" / "agent-cost-mirror.json",
            {
                "schema_version": 1,
                "updated_at": utc_now(),
                "agent_cost_mirror": mirror_status,
            },
            mode=0o600,
        )
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "updated_at": utc_now(),
            "total_usd": payload.get("total_usd"),
            "budget_remaining_usd": payload.get("budget_remaining_usd"),
            "status": payload.get("status"),
            "upstream_watchdog_age_seconds": round(upstream_age, 3),
            "source": pulse_source,
            "gpu_mirror": mirror_status,
            "agent_mirror": mirror_status,
        }
        atomic_write_json(
            state_dir / "telemetry" / "budget-pulse.json", result, mode=0o600
        )
        return result

    from event_runtime.compute import worker as gpu_worker

    # Refresh the CPU agent first.  A Modal exec into a newly starting or
    # stopping GPU sandbox can block for most of its control-plane timeout.  If
    # GPU mirroring runs first, that delay can age the CPU watchdog's host cost
    # mirror past its safety window even though this pulse is healthy.  The CPU
    # agent is the authority that launches model requests and GPU jobs, so its
    # circuit breaker must receive each pulse before slower downstream mirrors.
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

    # GPU control-plane calls can wait for a newly allocated sandbox to become
    # executable. They run in a separate restartable service so that delay can
    # never starve the CPU/API authority's budget heartbeat.
    gpu_mirror = {"gpu_budget_mirror": "delegated_to_gpu_budget_pulse"}

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
    atomic_write_json(state_dir / "telemetry" / "budget-pulse.json", result, mode=0o600)
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
    """Return whether the authoritative CPU execution acknowledged a stop."""
    path = state_dir / "STOP_ACK.json"
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # An unreadable acknowledgement cannot safely authorize more work.
        return True
    return True


def run_services_should_exit(state_dir: Path, run: dict[str, Any]) -> bool:
    """Stop safety services after the one CPU execution reaches a boundary."""
    if (state_dir / "FINALIZED.json").is_file() or terminal_stop_acknowledged(
        state_dir
    ):
        return True
    try:
        return run_results_finished(state_dir, run)
    except KeyError:
        # An incomplete terminal record cannot authorize additional work.
        return True


def budget_safety_should_exit(state_dir: Path, run: dict[str, Any]) -> bool:
    """Keep the independent budget feed alive through a CPU self-stop race.

    The in-sandbox watchdog can acknowledge a budget stop before the host has
    observed it and fenced an already-running GPU worker.  STOP_ACK alone is
    therefore not a safe boundary for the host budget pulse: ending the pulse
    at that instant makes the GPU's private trusted snapshot go stale and
    converts a normal budget stop into a fail-closed infrastructure stop.

    A host STOP_REQUESTED marker proves the controller has entered the
    synchronous CPU/GPU teardown path. Natural completion and FINALIZED remain
    terminal without such a marker.
    """
    if (state_dir / "FINALIZED.json").is_file():
        return True
    ack_path = state_dir / "STOP_ACK.json"
    if ack_path.is_file():
        try:
            ack_reason = str(json.loads(ack_path.read_text()).get("reason") or "")
        except (OSError, json.JSONDecodeError):
            return True
        if ack_reason == "agent_exit" and gpu_dispatch_loop_alive(state_dir):
            return False
        if ack_reason != "agent_cost_budget_exhausted":
            return True
        if (state_dir / "STOP_REQUESTED.json").is_file():
            return True
    try:
        return run_results_finished(state_dir, run)
    except KeyError:
        return True


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
            while not budget_safety_should_exit(state_dir, run):
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


def gpu_budget_pulse_once(run_id: str) -> dict[str, Any]:
    """Mirror the latest trusted host ledger into active GPU sandboxes."""
    state_dir, run = load_run(run_id)
    try:
        payload = json.loads((state_dir / "telemetry" / "agent-cost.json").read_text())
    except FileNotFoundError:
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "updated_at": utc_now(),
            "gpu_budget_mirror": "not_started",
            "status": "watchdog_starting",
        }
        atomic_write_json(
            state_dir / "telemetry" / "gpu-budget-mirror.json", result, mode=0o600
        )
        return result
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GPU budget pulse has no valid host snapshot") from exc

    from event_runtime.compute import worker as gpu_worker

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "updated_at": utc_now(),
        **gpu_worker.mirror_gpu_budget(run, payload),
    }
    atomic_write_json(
        state_dir / "telemetry" / "gpu-budget-mirror.json", result, mode=0o600
    )
    if result.get("gpu_budget_mirror") == "error":
        detail = (
            result.get("errors")
            or result.get("gpu_budget_mirror_error")
            or "unknown"
        )
        raise RuntimeError(f"GPU budget pulse mirror failed: {detail}")
    return result


def gpu_budget_pulse_loop(run_id: str, poll_seconds: int) -> int:
    """Refresh GPU watchdogs independently from CPU/API enforcement."""
    state_dir, run = load_run(run_id)
    lock_path = state_dir / "gpu-budget-pulse.lock"
    with file_lock(lock_path, blocking=False) as acquired:
        if not acquired:
            print(f"GPU budget pulse already running for {run_id}", file=sys.stderr)
            return 2
        pid_path = state_dir / "gpu-budget-pulse.pid"
        own_pid = os.getpid()
        atomic_write_text(pid_path, f"{own_pid}\n", 0o600)
        try:
            while not budget_safety_should_exit(state_dir, run):
                started = time.monotonic()
                try:
                    payload = gpu_budget_pulse_once(run_id)
                except Exception as exc:  # noqa: BLE001
                    record_controller_error(run_id, exc)
                    payload = {
                        "run_id": run_id,
                        "updated_at": utc_now(),
                        "status": "gpu_pulse_error",
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
            while True:
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
                if (
                    run_services_should_exit(state_dir, run)
                    and not payload.get("active_training_jobs")
                    and not payload.get("pending")
                    and not payload.get("submission_bridge_pending_jobs")
                ):
                    return 0
                elapsed = time.monotonic() - started
                time.sleep(max(1.0, float(poll_seconds) - elapsed))
        finally:
            try:
                registered = int(pid_path.read_text().strip())
            except (OSError, ValueError):
                registered = None
            if registered == own_pid:
                pid_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "status",
        "stop",
        "finalize",
        "monitor",
        "budget-pulse",
        "gpu-budget-pulse",
        "gpu-dispatch-loop",
        "wait",
        "gpu-dispatch",
        "gpu-terminate",
        "modal-cost",
    ):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        if name in {
            "monitor",
            "budget-pulse",
            "gpu-budget-pulse",
            "gpu-dispatch-loop",
            "wait",
        }:
            command.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
        if name == "wait":
            command.add_argument(
                "--timeout-seconds", type=int, default=DEFAULT_WAIT_SECONDS
            )
        if name == "status":
            command.add_argument("--offline", action="store_true")
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
        elif args.command == "gpu-budget-pulse":
            return gpu_budget_pulse_loop(args.run_id, args.poll_seconds)
        elif args.command == "gpu-dispatch-loop":
            return gpu_dispatch_loop(args.run_id, args.poll_seconds)
        elif args.command == "wait":
            return wait_for_run(args.run_id, args.timeout_seconds, args.poll_seconds)
        elif args.command == "finalize":
            complete, payload = finalize(args.run_id)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if complete else 2
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
            raise AssertionError(f"unhandled command: {args.command}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
