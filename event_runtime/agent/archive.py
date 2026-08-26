#!/usr/bin/env python3
"""Submit a policy for asynchronous scoring. Returns immediately.

    event archive mypolicy.pt --note "candidate-a"

Prints a request id and exits. The request and receipt are staged on the per-run
durable Volume. ``event history`` distinguishes local staging from sanitized
Harbor acceptance/rejection, while scores, gates, traces, queue progress, and
completion timing remain in the trusted archive. Every accepted policy is
retained in the final trajectory and Pareto analysis.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

IN_GPU_WORKER = bool(os.environ.get("SPRINT_GPU_JOB_ID"))
GPU_BRIDGE_ROOT = os.environ.get(
    "SPRINT_GPU_SUBMISSION_BRIDGE_ROOT", "/run/sprint-submission-bridge"
)
GPU_DURABLE_BRIDGE_ROOT = os.environ.get(
    "SPRINT_GPU_DURABLE_SUBMISSION_BRIDGE_ROOT",
    (
        f"/durable/runs/{os.environ.get('SPRINT_RUN_ID')}/submission-bridge"
        if os.environ.get("SPRINT_RUN_ID")
        else ""
    ),
)
_DEFAULT_ROOT = (
    GPU_BRIDGE_ROOT
    if IN_GPU_WORKER
    else ("/durable/submissions" if Path("/durable").is_dir() else "/app/submissions")
)
SUBMISSIONS_ROOT = os.environ.get("SPRINT_SUBMISSIONS_ROOT", _DEFAULT_ROOT)
QUEUE = os.path.join(SUBMISSIONS_ROOT, "outbox" if IN_GPU_WORKER else "queue")
NOTES = os.path.join(SUBMISSIONS_ROOT, "notes")
RECEIPTS = os.path.join(SUBMISSIONS_ROOT, "receipts")
ACKNOWLEDGMENTS = os.path.join(SUBMISSIONS_ROOT, "acknowledgments")
LOCK = os.path.join(SUBMISSIONS_ROOT, "submit.lock")
REQUEST_ID_RE = re.compile(r"^[0-9]{6}-[0-9a-f]{4}$")
MAX_POLICY_BYTES = 32 * 1024 * 1024


def atomic_json(path: str, payload: dict) -> None:
    staged = f"{path}.{os.getpid()}.tmp"
    with open(staged, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)


def read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def acknowledgment_for(receipt: dict) -> dict:
    queue_name = str(receipt.get("queue_name") or "")
    if not queue_name:
        submission_id = str(receipt.get("submission_id") or "")
        queue_name = f"{submission_id}.pt" if submission_id else ""
    return read_json(os.path.join(ACKNOWLEDGMENTS, f"{queue_name}.json"))


def admission_backpressure(policy_sha256: str) -> str | None:
    """Return an advisory local refusal backed by Harbor acknowledgments."""
    try:
        names = sorted(name for name in os.listdir(RECEIPTS) if name.endswith(".json"))
    except (FileNotFoundError, PermissionError):
        return None
    for name in reversed(names):
        receipt = read_json(os.path.join(RECEIPTS, name))
        if not receipt:
            continue
        ack = acknowledgment_for(receipt)
        state = str(ack.get("state") or "staged")
        submission_id = str(receipt.get("submission_id") or name[:-5])
        if receipt.get("policy_sha256") == policy_sha256 and state not in {
            "rejected",
            "ingestion_failed",
        }:
            return f"identical policy already {state} as {submission_id}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="TorchScript policy to score")
    parser.add_argument("--note", default="", help="local receipt annotation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="stage despite local duplicate backpressure",
    )
    args = parser.parse_args()
    policy = args.policy
    note = args.note

    if not os.path.exists(policy):
        print(f"no such file: {policy}", file=sys.stderr)
        return 1

    for directory in (QUEUE, NOTES, RECEIPTS, ACKNOWLEDGMENTS):
        os.makedirs(directory, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        source_size = os.path.getsize(policy)
        if source_size <= 0 or source_size > MAX_POLICY_BYTES:
            print(
                f"not submitted: policy is {source_size} bytes; the limit is "
                f"{MAX_POLICY_BYTES} bytes (32 MiB)",
                file=sys.stderr,
            )
            return 1
        with open(policy, "rb") as handle:
            source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        requested_id = str(os.environ.get("SPRINT_HOST_ARCHIVE_REQUEST_ID") or "")
        if requested_id and not REQUEST_ID_RE.fullmatch(requested_id):
            print("invalid host archive request id", file=sys.stderr)
            return 2
        if requested_id:
            prior = read_json(os.path.join(RECEIPTS, f"{requested_id}.json"))
            if prior:
                if prior.get("policy_sha256") != source_sha256:
                    print(
                        f"not staged: request id {requested_id} already names "
                        "different policy bytes",
                        file=sys.stderr,
                    )
                    return 2
                print(f"staged locally {requested_id}  (idempotent replay)")
                return 0
        submitted_at = datetime.now(timezone.utc)
        job_id = requested_id or f"{submitted_at:%H%M%S}-{uuid.uuid4().hex[:4]}"

        # Copy under a hidden name and rename into place. Copying rather than
        # referencing freezes the exact bytes while training continues.
        staged = os.path.join(QUEUE, f".{job_id}.pt")
        shutil.copy2(policy, staged)
        staged_size = os.path.getsize(staged)
        with open(staged, "rb") as handle:
            policy_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if staged_size != source_size or policy_sha256 != source_sha256:
            os.unlink(staged)
            print("not submitted: policy changed while being staged", file=sys.stderr)
            return 1
        # Validate the frozen bytes, not the mutable source pathname. The
        # trusted verifier repeats this contract check before the submission
        # consumes one of the trial's official slots.
        check = subprocess.run(
            ["/usr/local/bin/event", "check", staged],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            os.unlink(staged)
            sys.stderr.write(check.stdout + check.stderr)
            print(
                "not submitted: the policy does not meet the interface contract.",
                file=sys.stderr,
            )
            return 1
        if not args.force:
            refusal = admission_backpressure(policy_sha256)
            if refusal:
                os.unlink(staged)
                print(
                    f"not staged: {refusal}; use --force to override",
                    file=sys.stderr,
                )
                return 2
        queued = os.path.join(QUEUE, f"{job_id}.pt")
        os.replace(staged, queued)

        if note:
            with open(
                os.path.join(NOTES, f"{job_id}.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write(note + "\n")

        receipt = {
            "schema_version": 2 if IN_GPU_WORKER else 1,
            "submission_id": job_id,
            "submitted_at": submitted_at.isoformat(),
            "policy_sha256": policy_sha256,
            "policy_size_bytes": os.path.getsize(queued),
            "queue_name": os.path.basename(queued),
            "state": "staged",
            "note": note,
        }
        if IN_GPU_WORKER:
            receipt.update(
                {
                    "bridge": "host_owned_gpu_submission_v1",
                    "run_id": os.environ.get("SPRINT_RUN_ID"),
                    "gpu_job_id": os.environ.get("SPRINT_GPU_JOB_ID"),
                    "gpu_attempt": int(os.environ.get("SPRINT_GPU_ATTEMPT") or 0),
                    "gpu_lease_id": os.environ.get("SPRINT_GPU_LEASE_ID"),
                }
            )
        atomic_json(os.path.join(RECEIPTS, f"{job_id}.json"), receipt)
        if IN_GPU_WORKER and GPU_DURABLE_BRIDGE_ROOT:
            durable_queue = os.path.join(GPU_DURABLE_BRIDGE_ROOT, "outbox")
            durable_receipts = os.path.join(GPU_DURABLE_BRIDGE_ROOT, "receipts")
            durable_notes = os.path.join(GPU_DURABLE_BRIDGE_ROOT, "notes")
            for directory in (durable_queue, durable_receipts, durable_notes):
                os.makedirs(directory, exist_ok=True)
            durable_staged = os.path.join(durable_queue, f".{job_id}.pt")
            shutil.copy2(queued, durable_staged)
            os.replace(durable_staged, os.path.join(durable_queue, f"{job_id}.pt"))
            atomic_json(
                os.path.join(durable_receipts, f"{job_id}.json"), receipt
            )
            if note:
                durable_note = os.path.join(durable_notes, f"{job_id}.txt")
                with open(durable_note, "w", encoding="utf-8") as handle:
                    handle.write(note + "\n")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    if IN_GPU_WORKER:
        print(
            f"queued for host submission bridge {job_id}"
            + (f"  ({note})" if note else "")
        )
        print(
            "The trusted host will transfer these immutable bytes into Harbor; "
            "run 'event history' to inspect bridge and admission state."
        )
    else:
        print(f"staged locally {job_id}" + (f"  ({note})" if note else ""))
        print(
            "This is not accepted until Harbor acknowledges the immutable bytes; "
            "run 'event history' to inspect admission state."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
