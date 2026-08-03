#!/usr/bin/env python3
"""Host controller for one explicit durable Harbor lane run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frontier_update import (  # noqa: E402
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

ROOT = Path("/data/qwop-bench")
OPS_ROOT = ROOT / "runs" / "ops"
FRONTIER_SCRIPT = SCRIPT_DIR / "frontier_update.py"
POLL_SECONDS = 30
DEFAULT_WAIT_SECONDS = 3 * 60 * 60
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
    kind = state.get("agent_kind", "claude-code")
    if kind not in {"claude-code", "codex"}:
        raise ValueError(f"{path} has unsupported agent_kind: {kind!r}")
    return state_dir, state


def agent_kind(run: dict[str, Any]) -> str:
    """Return the explicit kind, with a compatibility default for schema v1."""
    return str(run.get("agent_kind", "claude-code"))


def save_run(state_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(state_dir / "run.json", state, mode=0o600)


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
        if app.get("description") == run["app_name"]
        and app.get("state") != "stopped"
    ]
    if len(exact) != 1:
        return None
    app_id = str(exact[0]["app_id"])
    run["app_id"] = app_id
    save_run(state_dir, run)
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
        timeout=60,
    )


def is_agent_container(run: dict[str, Any], container_id: str) -> bool:
    result = exec_container(
        run,
        container_id,
        "test -x /opt/qwop-snapshot-loop.sh && printf QWOP_AGENT",
        check=False,
    )
    return result.returncode == 0 and "QWOP_AGENT" in result.stdout


def discover_agent_container(state_dir: Path, run: dict[str, Any]) -> str | None:
    app_id = discover_app_id(state_dir, run)
    if not app_id:
        return None
    containers = containers_for_app(run, app_id)
    cached = run.get("agent_container_id")
    if cached in containers and is_agent_container(run, str(cached)):
        return str(cached)
    matches = [container for container in containers if is_agent_container(run, container)]
    if len(matches) != 1:
        return None
    run["agent_container_id"] = matches[0]
    save_run(state_dir, run)
    return matches[0]


def volume_get_text(run: dict[str, Any], remote_path: str) -> str | None:
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
        timeout=60,
    )
    return result.stdout if result.returncode == 0 else None


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
            save_run(state_dir, run)
            return job, None
        trial = trials[0]
        run["trial_path"] = str(trial)
    save_run(state_dir, run)
    return job, trial


def request_stop(run_id: str) -> dict[str, Any]:
    state_dir, run = load_run(run_id)
    kind = agent_kind(run)
    ack = fetch_remote_json(state_dir, run, "STOP_ACK", "STOP_ACK.json")
    marker = state_dir / "STOP_REQUESTED.json"
    if ack:
        return {"status": "acknowledged", "agent_kind": kind, "ack": ack}
    container = discover_agent_container(state_dir, run)
    if not container:
        if marker.exists():
            return {
                "status": "requested",
                "agent_kind": kind,
                "container_id": run.get("agent_container_id"),
            }
        raise RuntimeError(
            "the exact agent container is unavailable and STOP_ACK is missing"
        )
    exec_container(
        run,
        container,
        "umask 077; : > /run/qwop-stop; chmod 0600 /run/qwop-stop",
    )
    if marker.exists():
        payload = json.loads(marker.read_text())
    else:
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "agent_kind": kind,
            "requested_at": utc_now(),
            "container_id": container,
            "method": "create /run/qwop-stop",
        }
        atomic_write_json(marker, payload, mode=0o600)
    return {"status": "requested", "agent_kind": kind, **payload}


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


def tar_attempt_atomic(attempt: Path, archive_dir: Path) -> tuple[Path, str, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(archive_dir.glob(f"{attempt.name}.*.tar"))
    if existing:
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
        if attempt.name in manifest["attempts"]:
            continue
        archive, digest, checksum = tar_attempt_atomic(
            attempt, state_dir / "archives"
        )
        record = {
            "index": index,
            "attempt": attempt.name,
            "archive": str(archive),
            "sha256": digest,
            "bytes": archive.stat().st_size,
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
        path
        for path in (state_dir / "captures").glob("*")
        if path.is_file()
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
    return process_alive(pid, "frontier_update.py")


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
    process = subprocess.Popen(
        [
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
            "--deploy",
            "--debounce-seconds",
            str(run.get("deploy_debounce_seconds", DEPLOY_DEBOUNCE_SECONDS)),
        ],
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
    # Host-side GPU/CPU telemetry backup (agent + best-effort verifiers).
    # In-sandbox sidecar is primary; this persists even if the mount lags.
    try:
        import telemetry_host

        telemetry_host.poll_once(run_id)
    except Exception as exc:  # noqa: BLE001
        record_controller_error(run_id, exc)
    job, trial = discover_job_and_trial(state_dir, run)
    if job and trial:
        snapshot_host_history(state_dir, run, job, trial, upload=upload)
        archive_completed_attempts(state_dir, run, trial, upload=upload)
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


def final_conditions(
    state_dir: Path, run: dict[str, Any]
) -> tuple[bool, dict[str, bool], list[str]]:
    job, trial = discover_job_and_trial(state_dir, run)
    conditions: dict[str, bool] = {
        "stop_ack": (state_dir / "STOP_ACK.json").is_file(),
        "job_found": job is not None,
        "trial_found": trial is not None,
        "ledger_parseable": False,
        "ledger_terminal": False,
        "attempt_archives": False,
        "artifact_manifest": False,
        "final_verifier": False,
        "finished_at": False,
        "harbor_exited": not harbor_alive(run),
        "site_current": False,
    }
    details: list[str] = []
    if not job or not trial:
        return False, conditions, ["job or trial path is not available"]

    ledger_path = trial / "artifacts" / "continuous" / "ledger.jsonl"
    ledger = read_ledger(ledger_path)
    conditions["ledger_parseable"] = not ledger.errors
    conditions["ledger_terminal"] = all(row_terminal(row) for row in ledger.rows)
    details.extend(ledger.errors)

    manifest_path = trial / "artifacts" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = manifest if isinstance(manifest, list) else []
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
        return True, json.loads(final_path.read_text())
    monitor_once(
        run_id,
        upload=upload,
        include_remote=include_remote,
        launch_worker=False,
    )
    complete, conditions, details = final_conditions(state_dir, run)
    payload = {
        "schema_version": 1,
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
                    payload = json.loads(source.read_text()) if source.exists() else {
                        "run_id": run_id,
                        "agent_kind": kind,
                        "complete": False,
                        "checked_at": utc_now(),
                        "status": "waiting for host monitor",
                    }
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
        cache = state_dir / "recovery" / dt.datetime.now(
            dt.timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
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
    cache = state_dir / "recovery" / dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
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


HUB_TRACK = ROOT / "runs" / "hub_track_upload.py"


def hub_upload(*, scrub_only: bool = False, force_rescrub: bool = False) -> int:
    """Default Hub path: scrub → leak-gate → upload (or scrub-only)."""
    if not HUB_TRACK.is_file():
        raise RuntimeError(f"missing {HUB_TRACK}")
    cmd = [sys.executable, str(HUB_TRACK), "--once"]
    if scrub_only:
        cmd.append("--scrub-only")
    if force_rescrub:
        cmd.append("--force-rescrub")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "status",
        "stop",
        "finalize",
        "monitor",
        "wait",
        "check",
        "recover",
    ):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        if name in {"monitor", "wait"}:
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
    hub = sub.add_parser(
        "hub-upload",
        help="Scrub job dirs, leak-gate, then private Hub upload (default path).",
    )
    hub.add_argument(
        "--scrub-only",
        action="store_true",
        help="Scrub + leak-gate only; do not call harbor upload",
    )
    hub.add_argument("--force-rescrub", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "hub-upload":
            return hub_upload(
                scrub_only=args.scrub_only, force_rescrub=args.force_rescrub
            )
        if args.command == "status":
            state_dir, run = load_run(args.run_id)
            payload = status_snapshot(
                state_dir, run, include_remote=not args.offline
            )
        elif args.command == "stop":
            payload = request_stop(args.run_id)
        elif args.command == "monitor":
            return monitor_loop(args.run_id, args.poll_seconds)
        elif args.command == "wait":
            return wait_for_run(
                args.run_id, args.timeout_seconds, args.poll_seconds
            )
        elif args.command == "finalize":
            complete, payload = finalize(args.run_id)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if complete else 2
        elif args.command == "check":
            payload = check_recovery(args.run_id, args.cache)
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
