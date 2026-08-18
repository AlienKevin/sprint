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
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_ROOT = (
    "/durable/submissions" if Path("/durable").is_dir() else "/app/submissions"
)
SUBMISSIONS_ROOT = os.environ.get("SPRINT_SUBMISSIONS_ROOT", _DEFAULT_ROOT)
QUEUE = os.path.join(SUBMISSIONS_ROOT, "queue")
NOTES = os.path.join(SUBMISSIONS_ROOT, "notes")
RECEIPTS = os.path.join(SUBMISSIONS_ROOT, "receipts")
ACKNOWLEDGMENTS = os.path.join(SUBMISSIONS_ROOT, "acknowledgments")
LOCK = os.path.join(SUBMISSIONS_ROOT, "submit.lock")
MINIMUM_INTERVAL_SECONDS = float(
    os.environ.get("SPRINT_SUBMISSION_MIN_INTERVAL_SEC", "300")
)


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
    now = datetime.now(timezone.utc)
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
        if state == "staged":
            return f"submission {submission_id} is still awaiting Harbor ingestion"
        if state != "accepted":
            continue
        raw_accepted = ack.get("accepted_at")
        try:
            accepted_at = datetime.fromisoformat(str(raw_accepted))
            elapsed = (now - accepted_at).total_seconds()
        except (TypeError, ValueError):
            continue
        remaining = MINIMUM_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            return (
                f"Harbor cooldown is active for about {int(remaining + 0.999)} "
                "more second(s)"
            )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="TorchScript policy to score")
    parser.add_argument("--note", default="", help="local receipt annotation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="stage despite local duplicate/cooldown backpressure",
    )
    args = parser.parse_args()
    policy = args.policy
    note = args.note

    if not os.path.exists(policy):
        print(f"no such file: {policy}", file=sys.stderr)
        return 1

    # Reject malformed TorchScript locally before spending verifier compute.
    check = subprocess.run(
        ["/usr/local/bin/event", "check", policy],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        sys.stderr.write(check.stdout + check.stderr)
        print(
            "not submitted: the policy does not meet the interface contract.",
            file=sys.stderr,
        )
        return 1

    for directory in (QUEUE, NOTES, RECEIPTS, ACKNOWLEDGMENTS):
        os.makedirs(directory, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        with open(policy, "rb") as handle:
            source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if not args.force:
            refusal = admission_backpressure(source_sha256)
            if refusal:
                print(
                    f"not staged: {refusal}; use --force to override", file=sys.stderr
                )
                return 2
        submitted_at = datetime.now(timezone.utc)
        job_id = f"{submitted_at:%H%M%S}-{uuid.uuid4().hex[:4]}"

        # Copy under a hidden name and rename into place. Copying rather than
        # referencing freezes the exact bytes while training continues.
        staged = os.path.join(QUEUE, f".{job_id}.pt")
        shutil.copy2(policy, staged)
        with open(staged, "rb") as handle:
            policy_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        queued = os.path.join(QUEUE, f"{job_id}.pt")
        os.replace(staged, queued)

        if note:
            with open(
                os.path.join(NOTES, f"{job_id}.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write(note + "\n")

        receipt = {
            "schema_version": 1,
            "submission_id": job_id,
            "submitted_at": submitted_at.isoformat(),
            "policy_sha256": policy_sha256,
            "policy_size_bytes": os.path.getsize(queued),
            "queue_name": os.path.basename(queued),
            "state": "staged",
            "note": note,
        }
        atomic_json(os.path.join(RECEIPTS, f"{job_id}.json"), receipt)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    print(f"staged locally {job_id}" + (f"  ({note})" if note else ""))
    print(
        "This is not accepted until Harbor acknowledges the immutable bytes; "
        "run 'event history' to inspect admission state."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
