#!/usr/bin/env python3
"""Enqueue Isaac/GPU training for a host-dispatched Modal A10G worker.

The agent sandbox is CPU-only so Modal GPU preemption cannot kill Codex.
Training runs on a separate GPU worker that mounts the same /durable volume.

Usage:
  event gpu [options] -- CMD...
  event gpu status [JOB_ID]
  event gpu wait [JOB_ID] [--timeout SEC]
  event gpu logs [JOB_ID]

Jobs land under $SPRINT_GPU_JOBS_ROOT (default
/durable/runs/$SPRINT_RUN_ID/gpu-jobs/). The host monitor claims pending jobs and
starts an A10G sandbox with the same image + volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/opt")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sprint_resilience import CheckpointStore, CompletionJournal

AGENT_MIRROR_ROOT = Path("/run/sprint-gpu-mirror")
AGENT_WORKSPACE_ROOT = Path("/app")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_run_file(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def jobs_root() -> Path:
    explicit = os.environ.get("SPRINT_GPU_JOBS_ROOT", "").strip()
    if not explicit:
        explicit = _read_run_file("/run/sprint-gpu-jobs-root")
    if explicit:
        return Path(explicit)
    run_id = os.environ.get("SPRINT_RUN_ID", "").strip() or _read_run_file(
        "/run/sprint-run-id"
    )
    if not run_id:
        raise SystemExit(
            "SPRINT_RUN_ID or SPRINT_GPU_JOBS_ROOT must be set "
            "(durable keepalive writes /run/sprint-run-id)"
        )
    return Path("/durable") / "runs" / run_id / "gpu-jobs"


def run_id_from_env(root: Path) -> str:
    env_id = os.environ.get("SPRINT_RUN_ID", "").strip() or _read_run_file(
        "/run/sprint-run-id"
    )
    if env_id:
        return env_id
    # .../runs/<run_id>/gpu-jobs
    return root.parent.name


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def flush_durable(root: Path) -> None:
    """Best-effort volume flush (v2 sync; v1 relies on background commits)."""
    durable = Path("/durable")
    if durable.is_dir():
        try:
            os.sync()
        except OSError:
            pass
        # Volumes v2 accept sync on the mount; ignore failures on v1.
        shutil.which("sync") and os.system(f"sync {durable} >/dev/null 2>&1")
    # Touch a host-visible marker so dispatchers polling the volume wake sooner.
    marker = root / ".flush"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(utc_now() + "\n")
    except OSError:
        pass


def latest_job_id(root: Path) -> str | None:
    files = []
    for status_dir in (root / "status", AGENT_MIRROR_ROOT / "status"):
        if status_dir.is_dir():
            files.extend(status_dir.glob("*.json"))
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return files[-1].stem if files else None


def read_status(root: Path, job_id: str) -> dict:
    paths = (
        AGENT_MIRROR_ROOT / "status" / f"{job_id}.json",
        root / "status" / f"{job_id}.json",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise SystemExit(f"unknown job: {job_id}")
    return json.loads(path.read_text())


def pack_sync_dirs(archive_path: Path, sync_dirs: list[Path]) -> list[str]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    with tarfile.open(archive_path, "w:gz") as tar:
        for directory in sync_dirs:
            directory = directory.resolve()
            if not directory.is_dir():
                raise SystemExit(f"sync path is not a directory: {directory}")
            workspace = AGENT_WORKSPACE_ROOT.resolve()
            try:
                relative = directory.relative_to(workspace)
            except ValueError:
                # Agents commonly stage a complete workspace in a temporary
                # directory. A staged root containing train/ replaces /app;
                # a supplemental external directory is mounted by basename.
                arcname = (
                    "app"
                    if directory.name == "app" or (directory / "train").is_dir()
                    else str(Path("app") / directory.name)
                )
            else:
                arcname = (
                    "app" if relative == Path(".") else str(Path("app") / relative)
                )
            tar.add(directory, arcname=arcname, filter=_tar_filter)
            included.append(str(directory))
    return included


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = tarinfo.name
    # The worker extractor deliberately accepts only regular files and
    # directories. Match that contract at submission time so agent-visible
    # convenience symlinks (for example /app/verifier) cannot poison an
    # otherwise valid workspace archive.
    if not (tarinfo.isfile() or tarinfo.isdir()):
        return None
    skip_parts = {
        "__pycache__",
        ".git",
        ".cache",
        "outputs",
        "logs",
        "wandb",
        "isaac-sim",
        "kit",
    }
    parts = Path(name).parts
    if any(part in skip_parts for part in parts):
        return None
    if name.endswith((".pt", ".pth", ".ckpt")) and tarinfo.size > 80_000_000:
        # Large checkpoints should already live on /durable; skip huge copies.
        return None
    return tarinfo


def cmd_submit(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("missing command after --")
    root = jobs_root()
    run_id = run_id_from_env(root)
    job_id = uuid.uuid4().hex[:12]
    created = utc_now()
    sync_dirs = [Path(p) for p in (args.sync or ["/app"])]
    work_rel = f"runs/{run_id}/gpu-jobs/work/{job_id}/app.tar.gz"
    archive = Path("/durable") / work_rel
    included = pack_sync_dirs(archive, sync_dirs)
    archive_size = archive.stat().st_size
    with archive.open("rb") as handle:
        submitted_archive_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint_dir = Path(args.checkpoint_dir or root / "checkpoints" / job_id)
    progress_file = Path(args.progress_file or checkpoint_dir / "progress.json")

    job = {
        "schema_version": 3,
        "job_id": job_id,
        "run_id": run_id,
        "created_at": created,
        "command": list(args.command),
        "workdir": args.workdir,
        "timeout_sec": int(args.timeout),
        "note": args.note or "",
        "sync_dirs": included,
        "work_archive": work_rel,
        "submitted_work_archive_sha256": submitted_archive_sha256,
        "submitted_work_archive_size_bytes": archive_size,
        "status": "pending",
        "attempt": 0,
        "max_attempts": int(args.max_attempts),
        "retry_backoff_sec": float(args.retry_backoff),
        "retry_backoff_max_sec": float(args.retry_backoff_max),
        "heartbeat_interval_sec": int(args.heartbeat_interval),
        "heartbeat_timeout_sec": int(args.heartbeat_timeout),
        "interruption_grace_sec": float(args.interruption_grace),
        "checkpoint_dir": str(checkpoint_dir),
        "progress_file": str(progress_file),
        "checkpoint_protocol": "sprint-v1",
        "resume_arg": args.resume_arg or "",
        "gpu_type": "A10G",
    }
    # Queue entry (claimed by host dispatcher) + status mirror for the agent.
    atomic_write_json(root / "queue" / f"{job_id}.json", job)
    atomic_write_json(root / "status" / f"{job_id}.json", job)
    # Timeline: waiting for host to allocate a GPU worker.
    try:
        sys.path.insert(0, "/opt")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sprint_gpu_timeline", "/opt/sprint-gpu-timeline.py"
        )
        if spec and spec.loader:
            tl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tl)
            tl.append_event(
                run_id,
                phase="gpu_queue_wait",
                action="enter",
                job_id=job_id,
                attempt=1,
                detail={"source": "event gpu"},
                durable_dir="/durable",
                also_local=Path("/logs/artifacts/telemetry"),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"timeline emit skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    flush_durable(root)
    print(job_id)
    print(f"queued gpu job {job_id} timeout={args.timeout}s", file=sys.stderr)
    print(f"status: {root / 'status' / (job_id + '.json')}", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = jobs_root()
    job_id = args.job_id or latest_job_id(root)
    if not job_id:
        raise SystemExit("no gpu jobs yet")
    payload = read_status(root, job_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    root = jobs_root()
    job_id = args.job_id or latest_job_id(root)
    if not job_id:
        raise SystemExit("no gpu jobs yet")
    deadline = time.time() + int(args.timeout)
    while True:
        payload = read_status(root, job_id)
        status = str(payload.get("status", ""))
        if status in {"succeeded", "failed", "terminated"}:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if status == "succeeded" else 1
        if time.time() >= deadline:
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise SystemExit(f"timeout waiting for job {job_id} (status={status})")
        time.sleep(5)


def cmd_logs(args: argparse.Namespace) -> int:
    root = jobs_root()
    job_id = args.job_id or latest_job_id(root)
    if not job_id:
        raise SystemExit("no gpu jobs yet")
    log_root = root / "out" / job_id
    logs_by_attempt = {
        path.parent.name: path for path in log_root.glob("attempt-*/worker.log")
    }
    mirror_log_root = AGENT_MIRROR_ROOT / "out" / job_id
    logs_by_attempt.update(
        {
            path.parent.name: path
            for path in mirror_log_root.glob("attempt-*/worker.log")
        }
    )
    logs = [logs_by_attempt[key] for key in sorted(logs_by_attempt)]
    if not logs:
        raise SystemExit(f"no logs yet for {job_id}: {log_root}")
    for path in logs:
        sys.stdout.write(f"== {path.parent.name} ==\n")
        sys.stdout.write(path.read_text())
    return 0


def checkpoint_store(args: argparse.Namespace) -> CheckpointStore:
    return CheckpointStore(args.root) if args.root else CheckpointStore.from_env()


def cmd_checkpoint(args: argparse.Namespace) -> int:
    store = checkpoint_store(args)
    if args.checkpoint_action == "save":
        metadata = json.loads(args.metadata) if args.metadata else {}
        if not isinstance(metadata, dict):
            raise SystemExit("--metadata must decode to a JSON object")
        metadata.setdefault("kind", "training_state")
        kind = str(metadata.get("kind") or "").strip().lower().replace("-", "_")
        if (
            kind
            in {
                "inference_policy",
                "policy",
                "submission_policy",
                "torchscript_policy",
            }
            or metadata.get("resumable") is False
        ):
            raise SystemExit(
                "checkpoint save requires resumable trainer state; keep "
                "TorchScript/submission policies outside the recovery store"
            )
        checkpoint = store.commit(
            args.source,
            sequence=args.sequence,
            replay_cursor=args.cursor,
            idempotency_key=args.idempotency_key or None,
            metadata=metadata,
        )
        print(
            json.dumps(
                {**checkpoint.to_manifest(), "payload_path": str(checkpoint.path)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.checkpoint_action == "latest":
        checkpoint = store.latest_valid()
        if checkpoint is None:
            return 1
        if args.path_only:
            print(checkpoint.path)
        else:
            print(
                json.dumps(
                    {**checkpoint.to_manifest(), "payload_path": str(checkpoint.path)},
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0
    if args.checkpoint_action == "list":
        rows = [
            {**item.to_manifest(), "payload_path": str(item.path)}
            for item in store.iter_valid()
        ]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if args.checkpoint_action == "cleanup":
        print(
            json.dumps(
                {"removed": store.cleanup(keep=args.keep)}, indent=2, sort_keys=True
            )
        )
        return 0
    if args.checkpoint_action == "completed":
        payload = CompletionJournal(store).get(args.key)
        if payload is None:
            return 1
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise SystemExit("missing checkpoint action")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")

    submit = sub.add_parser("submit", help="enqueue a GPU command (default)")
    submit.add_argument("--sync", action="append", default=None)
    submit.add_argument("--workdir", default="/app")
    submit.add_argument("--timeout", type=int, default=3600)
    submit.add_argument("--note", default="")
    submit.add_argument("--max-attempts", type=int, default=3)
    submit.add_argument("--retry-backoff", type=float, default=10)
    submit.add_argument("--retry-backoff-max", type=float, default=120)
    submit.add_argument("--heartbeat-interval", type=int, default=5)
    submit.add_argument("--heartbeat-timeout", type=int, default=45)
    submit.add_argument(
        "--interruption-grace",
        type=float,
        default=20,
        help="Seconds a signaled worker gives the child to finish checkpointing",
    )
    submit.add_argument("--checkpoint-dir", default="")
    submit.add_argument("--progress-file", default="")
    submit.add_argument(
        "--resume-arg",
        default="",
        help="Append FLAG CHECKPOINT on replacement attempts",
    )
    submit.add_argument("command", nargs=argparse.REMAINDER)

    status = sub.add_parser("status")
    status.add_argument("job_id", nargs="?")

    wait = sub.add_parser("wait")
    wait.add_argument("job_id", nargs="?")
    wait.add_argument("--timeout", type=int, default=3600)

    logs = sub.add_parser("logs")
    logs.add_argument("job_id", nargs="?")

    checkpoint = sub.add_parser(
        "checkpoint", help="publish or inspect atomic durable checkpoints"
    )
    checkpoint.add_argument(
        "--root", default="", help="Checkpoint directory (defaults to worker env)"
    )
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_action")
    checkpoint_save = checkpoint_sub.add_parser("save")
    checkpoint_save.add_argument(
        "source",
        help="Complete state required to resume the submitted process",
    )
    checkpoint_save.add_argument("--sequence", type=int, required=True)
    checkpoint_save.add_argument("--cursor", default=None)
    checkpoint_save.add_argument("--idempotency-key", default="")
    checkpoint_save.add_argument("--metadata", default="")
    checkpoint_latest = checkpoint_sub.add_parser("latest")
    checkpoint_latest.add_argument("--path-only", action="store_true")
    checkpoint_sub.add_parser("list")
    checkpoint_cleanup = checkpoint_sub.add_parser("cleanup")
    checkpoint_cleanup.add_argument("--keep", type=int, default=3)
    checkpoint_completed = checkpoint_sub.add_parser("completed")
    checkpoint_completed.add_argument("key")

    # Allow: event gpu -- python train.py
    # and:   event gpu submit -- python train.py
    argv = sys.argv[1:]
    if argv and argv[0] not in {
        "submit",
        "status",
        "wait",
        "logs",
        "checkpoint",
        "-h",
        "--help",
    }:
        argv = ["submit", *argv]

    args = parser.parse_args(argv)
    if args.action in (None, "submit"):
        # argparse REMAINDER keeps a leading '--' when present.
        command = list(getattr(args, "command", []) or [])
        if command and command[0] == "--":
            command = command[1:]
        args.command = command
        return cmd_submit(args)
    if args.action == "status":
        return cmd_status(args)
    if args.action == "wait":
        return cmd_wait(args)
    if args.action == "logs":
        return cmd_logs(args)
    if args.action == "checkpoint":
        return cmd_checkpoint(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
