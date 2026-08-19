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

from event_runtime.container.sprint_resilience import CheckpointStore, CompletionJournal

AGENT_MIRROR_ROOT = Path("/run/sprint-gpu-mirror")
AGENT_WORKSPACE_ROOT = Path("/app")
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "terminated"})
MAX_OUTPUT_ARTIFACTS = 8


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


def _read_status_candidate(path: Path, source: str) -> tuple[dict, str] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return (payload, source) if isinstance(payload, dict) else None


def _attempt_number(payload: dict) -> int:
    try:
        return int(payload.get("attempt") or 0)
    except (TypeError, ValueError):
        return 0


def _event_epoch(payload: dict) -> float:
    for key in (
        "finished_at_epoch_s",
        "terminated_at_epoch_s",
        "updated_at_epoch_s",
        "started_at_epoch_s",
        "dispatched_at_epoch_s",
        "claimed_at_epoch_s",
        "created_at_epoch_s",
    ):
        try:
            value = float(payload.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _same_attempt(left: dict, right: dict) -> bool:
    if _attempt_number(left) != _attempt_number(right):
        return False
    left_lease = str(left.get("lease_id") or left.get("claim_id") or "")
    right_lease = str(right.get("lease_id") or right.get("claim_id") or "")
    return not left_lease or not right_lease or left_lease == right_lease


def reconcile_status_candidates(
    candidates: list[tuple[dict, str]],
) -> dict:
    """Resolve mutable delivery mirrors against immutable attempt outcomes.

    A completed worker attempt is authoritative for its lease unless the host
    fenced that lease before the worker reported completion.  This prevents a
    later batch-stop pass from rewriting a job that had already succeeded,
    while preserving a real stop/completion race in favour of the fence.
    """
    if not candidates:
        raise ValueError("no GPU status candidates")
    summaries = [item for item in candidates if item[1] != "durable_attempt"]
    attempts = [item for item in candidates if item[1] == "durable_attempt"]
    source_rank = {"volume_status": 1, "host_mirror": 2}
    selected, selected_source = max(
        summaries or attempts,
        key=lambda item: (
            _attempt_number(item[0]),
            str(item[0].get("status") or "") in TERMINAL_STATUSES,
            _event_epoch(item[0]),
            source_rank.get(item[1], 3),
        ),
    )
    matching_attempts = [
        item
        for item in attempts
        if _same_attempt(item[0], selected)
        and str(item[0].get("status") or "") in TERMINAL_STATUSES
    ]
    if matching_attempts:
        attempt, attempt_source = max(
            matching_attempts,
            key=lambda item: (_attempt_number(item[0]), _event_epoch(item[0])),
        )
        summary_status = str(selected.get("status") or "")
        fence_epoch = 0.0
        if summary_status == "terminated":
            try:
                fence_epoch = float(
                    selected.get("terminated_at_epoch_s")
                    or selected.get("finished_at_epoch_s")
                    or 0
                )
            except (TypeError, ValueError):
                fence_epoch = 0.0
        # A fence that predates completion wins. A later operator cleanup does
        # not mutate the already-durable outcome of this exact lease.
        if not fence_epoch or fence_epoch >= _event_epoch(attempt):
            selected, selected_source = attempt, attempt_source

    result = dict(selected)
    conflicts = [
        {
            "source": source,
            "attempt": _attempt_number(payload),
            "lease_id": payload.get("lease_id") or payload.get("claim_id"),
            "status": payload.get("status"),
        }
        for payload, source in candidates
        if (
            _attempt_number(payload),
            str(payload.get("lease_id") or payload.get("claim_id") or ""),
            str(payload.get("status") or ""),
        )
        != (
            _attempt_number(selected),
            str(selected.get("lease_id") or selected.get("claim_id") or ""),
            str(selected.get("status") or ""),
        )
    ]
    if conflicts:
        result["status_reconciliation"] = {
            "selected_source": selected_source,
            "conflicts": conflicts,
        }
    return result


def read_status(root: Path, job_id: str) -> dict:
    candidates: list[tuple[dict, str]] = []
    for path, source in (
        (AGENT_MIRROR_ROOT / "status" / f"{job_id}.json", "host_mirror"),
        (root / "status" / f"{job_id}.json", "volume_status"),
    ):
        candidate = _read_status_candidate(path, source)
        if candidate is not None:
            candidates.append(candidate)
    attempt_dir = root / "attempts" / job_id
    if attempt_dir.is_dir():
        for path in attempt_dir.glob("*.json"):
            candidate = _read_status_candidate(path, "durable_attempt")
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        raise SystemExit(f"unknown job: {job_id}")
    return reconcile_status_candidates(candidates)


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
    parts = Path(name).parts
    # The published verifier is already present in the immutable worker image.
    # Never let a mutable workspace copy replace its /app/verifier convenience
    # link or enter agent-authored training bytes.
    if len(parts) >= 2 and parts[:2] == ("app", "verifier"):
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
    if any(part in skip_parts for part in parts):
        return None
    if name.endswith((".pt", ".pth", ".ckpt")) and tarinfo.size > 80_000_000:
        # Large checkpoints should already live on /durable; skip huge copies.
        return None
    return tarinfo


def validate_output_paths(raw_paths: list[str] | None) -> list[str]:
    """Validate bounded files that the GPU worker must return to the agent."""
    paths: list[str] = []
    seen_names: set[str] = set()
    for raw in raw_paths or []:
        path = Path(raw)
        if not path.is_absolute() or path == AGENT_WORKSPACE_ROOT:
            raise SystemExit(f"--output must name an absolute file under /app: {raw}")
        try:
            relative = path.relative_to(AGENT_WORKSPACE_ROOT)
        except ValueError as exc:
            raise SystemExit(f"--output must be under /app: {raw}") from exc
        if ".." in relative.parts or not relative.name:
            raise SystemExit(f"invalid --output path: {raw}")
        if relative.name in seen_names:
            raise SystemExit(
                f"--output basenames must be unique (duplicate {relative.name!r})"
            )
        seen_names.add(relative.name)
        paths.append(str(AGENT_WORKSPACE_ROOT / relative))
    if len(paths) > MAX_OUTPUT_ARTIFACTS:
        raise SystemExit(f"at most {MAX_OUTPUT_ARTIFACTS} --output files are allowed")
    return paths


def cmd_submit(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("missing command after --")
    root = jobs_root()
    run_id = run_id_from_env(root)
    job_id = uuid.uuid4().hex[:12]
    created_epoch = time.time()
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
        "created_at_epoch_s": created_epoch,
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
        "job_kind": args.job_kind,
        "output_paths": validate_output_paths(getattr(args, "output", None)),
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


def cmd_get(args: argparse.Namespace) -> int:
    root = jobs_root()
    job_id = args.job_id or latest_job_id(root)
    if not job_id:
        raise SystemExit("no gpu jobs yet")
    payload = read_status(root, job_id)
    raw_source = str(payload.get("agent_policy_mirror_path") or "").strip()
    if not raw_source:
        raise SystemExit(
            f"job {job_id} has no mirrored policy yet; declare it with "
            "--output /app/POLICY.pt and wait for completion"
        )
    source = Path(raw_source)
    try:
        source.relative_to(AGENT_MIRROR_ROOT / "artifacts" / job_id)
    except ValueError as exc:
        raise SystemExit("GPU policy mirror path is outside the trusted job scope") from exc
    if not source.is_file():
        raise SystemExit(f"mirrored policy is not available yet: {source}")
    expected_size = int(payload.get("agent_policy_size_bytes") or 0)
    expected_sha = str(payload.get("agent_policy_sha256") or "")
    actual_size = source.stat().st_size
    with source.open("rb") as handle:
        actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_size != expected_size or actual_sha != expected_sha:
        raise SystemExit("mirrored policy failed size/digest verification")
    destination = Path(args.destination or (AGENT_WORKSPACE_ROOT / source.name))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)
    print(destination)
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    root = jobs_root()
    job_id = args.job_id or latest_job_id(root)
    if not job_id:
        raise SystemExit("no gpu jobs yet")
    payload = read_status(root, job_id)
    status = str(payload.get("status") or "")
    if status not in {"pending", "retry_wait", "claiming"} or payload.get(
        "sandbox_id"
    ):
        raise SystemExit(
            f"job {job_id} is {status or 'unknown'}; only undispatched jobs can "
            "be cancelled from the agent sandbox"
        )
    cancelled = {
        **payload,
        "status": "terminated",
        "termination_reason": "agent_cancelled_before_dispatch",
        "terminated_at": utc_now(),
        "terminated_at_epoch_s": time.time(),
    }
    for directory in (root / "queue", root / "status"):
        visible = directory / f"{job_id}.json"
        marker = directory / f".cancelled-{job_id}.json"
        atomic_write_json(marker, cancelled)
        visible.unlink(missing_ok=True)
    flush_durable(root)
    print(json.dumps(cancelled, indent=2, sort_keys=True))
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
    submit.add_argument(
        "--output",
        action="append",
        default=None,
        help="Required file under /app to return (repeatable; .pt/.pth is mirrored)",
    )
    submit.add_argument("--workdir", default="/app")
    submit.add_argument("--timeout", type=int, default=3600)
    submit.add_argument("--note", default="")
    submit.add_argument(
        "--job-kind",
        choices=("auto", "train", "evaluate", "verify"),
        default="auto",
        help="Select phase-aware worker watchdogs (default: infer from command)",
    )
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

    get = sub.add_parser("get", help="copy a mirrored policy back into /app")
    get.add_argument("job_id", nargs="?")
    get.add_argument("destination", nargs="?")

    cancel = sub.add_parser("cancel", help="cancel an undispatched queued job")
    cancel.add_argument("job_id", nargs="?")

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
        "get",
        "cancel",
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
    if args.action == "get":
        return cmd_get(args)
    if args.action == "cancel":
        return cmd_cancel(args)
    if args.action == "checkpoint":
        return cmd_checkpoint(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
