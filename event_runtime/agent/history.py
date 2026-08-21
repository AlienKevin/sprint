#!/usr/bin/env python3
"""List staged policies and sanitized Harbor acknowledgments; scores stay blind."""

import json
import os
import sys

from pathlib import Path

_DEFAULT_ROOT = (
    os.environ.get("SPRINT_GPU_SUBMISSION_BRIDGE_ROOT", "/run/sprint-submission-bridge")
    if os.environ.get("SPRINT_GPU_JOB_ID")
    else ("/durable/submissions" if Path("/durable").is_dir() else "/app/submissions")
)
SUBMISSIONS_ROOT = os.environ.get("SPRINT_SUBMISSIONS_ROOT", _DEFAULT_ROOT)
RECEIPTS = os.path.join(SUBMISSIONS_ROOT, "receipts")
ACKNOWLEDGMENTS = os.path.join(SUBMISSIONS_ROOT, "acknowledgments")


def read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__)
        return 0
    try:
        names = sorted(name for name in os.listdir(RECEIPTS) if name.endswith(".json"))
    except (FileNotFoundError, PermissionError):
        names = []
    if not names:
        print("no policies submitted yet")
        return 0

    print(f"{len(names)} local submission request(s)")
    for name in names:
        receipt = read_json(os.path.join(RECEIPTS, name))
        if not receipt:
            continue
        note = str(receipt.get("note") or "")
        submission_id = receipt.get("submission_id", name[:-5])
        queue_name = str(receipt.get("queue_name") or f"{submission_id}.pt")
        ack = read_json(os.path.join(ACKNOWLEDGMENTS, f"{queue_name}.json"))
        state = str(ack.get("state") or "staged")
        if state == "accepted":
            detail = "accepted by Harbor; official result hidden"
        elif state == "forwarded":
            detail = "forwarded by trusted host; awaiting Harbor acknowledgment"
        elif state == "rejected":
            reason = str(ack.get("reason") or "admission rejected")
            retry = ack.get("retry_after_sec")
            suffix = f"; retry after {int(float(retry))}s" if retry else ""
            detail = f"rejected by Harbor: {reason}{suffix}"
        elif state == "ingestion_failed":
            detail = "Harbor ingestion failed; safe to retry"
        else:
            detail = "staged locally; awaiting Harbor acknowledgment"
        prefix = f"{submission_id}" + (f"  {note}" if note else "")
        print(f"{prefix}  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
