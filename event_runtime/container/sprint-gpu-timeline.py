#!/usr/bin/env python3
"""GPU-active time accounting for CPU-agent / GPU-worker lane runs.

Phases (enter/exit pairs):
  gpu_queue_wait      — waiting for a GPU worker (preempt / queue); no util expected
  gpu_worker_starting — Modal allocating / sandbox booting
  isaac_starting      — Isaac Lab / sim stack init before train steps
  gpu_active          — training/inference steps on GPU
  gpu_idle_assigned   — optional: GPU held but not training

Artifacts (durable + local):
  .../telemetry/gpu_timeline.jsonl
  .../telemetry/gpu_timeline/events/<ts>_<seq>.json   # concurrent-safe shards
  .../telemetry/gpu_time_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

PHASES = (
    "gpu_queue_wait",
    "gpu_worker_starting",
    "isaac_starting",
    "gpu_active",
    "gpu_idle_assigned",
)

WAIT_PHASES = frozenset({"gpu_queue_wait"})
STARTUP_PHASES = frozenset({"gpu_worker_starting", "isaac_starting"})
ACTIVE_PHASES = frozenset({"gpu_active"})


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def telemetry_root(run_id: str, durable_dir: str = "/durable") -> Path:
    return Path(durable_dir) / "runs" / run_id / "telemetry"


def _flush(path: Path) -> None:
    try:
        os.sync()
    except OSError:
        pass
    # Best-effort Volume v2 sync.
    mount = path
    while mount != mount.parent and mount.name != "durable":
        # walk up looking for /durable
        if mount.name == "runs" and mount.parent.name == "durable":
            mount = mount.parent
            break
        mount = mount.parent
    if mount.name == "durable":
        os.system(f"sync {mount} >/dev/null 2>&1")


def append_event(
    run_id: str,
    *,
    phase: str,
    action: str,
    job_id: str = "",
    attempt: int = 0,
    lease_id: str = "",
    detail: dict[str, Any] | None = None,
    durable_dir: str = "/durable",
    also_local: Path | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    if action not in {"enter", "exit"}:
        raise ValueError("action must be enter|exit")
    event = {
        "schema_version": 1,
        "ts_utc": utc_now(),
        "epoch_s": int(time.time()),
        "run_id": run_id,
        "job_id": job_id or None,
        "attempt": int(attempt),
        "lease_id": lease_id or None,
        "phase": phase,
        "action": action,
        "detail": detail or {},
        "event_id": uuid.uuid4().hex[:12],
    }
    roots = [telemetry_root(run_id, durable_dir)]
    if also_local is not None:
        roots.append(also_local)
    line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            jsonl = root / "gpu_timeline.jsonl"
            fd = os.open(jsonl, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            shard_dir = root / "gpu_timeline" / "events"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard = shard_dir / f"{event['epoch_s']}_{event['event_id']}.json"
            tmp = shard.with_name(f".{shard.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, shard)
            _flush(root)
        except OSError:
            continue
    return event


def _load_events(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    jsonl = root / "gpu_timeline.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    shard_dir = root / "gpu_timeline" / "events"
    if shard_dir.is_dir():
        for path in sorted(shard_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                events.append(payload)
    # Deduplicate by event_id
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda e: (e.get("epoch_s") or 0, e.get("event_id") or "")):
        eid = str(event.get("event_id") or "")
        if eid and eid in seen:
            continue
        if eid:
            seen.add(eid)
        unique.append(event)
    return unique


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair events by logical job, attempt, and phase."""
    open_stack: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    durations = {phase: 0.0 for phase in PHASES}
    segments: list[dict[str, Any]] = []
    first_ts: int | None = None
    last_ts: int | None = None
    ignored_duplicate_enters = 0

    for event in events:
        epoch = int(event.get("epoch_s") or 0)
        if not epoch:
            continue
        first_ts = epoch if first_ts is None else min(first_ts, epoch)
        last_ts = epoch if last_ts is None else max(last_ts, epoch)
        phase = str(event.get("phase") or "")
        action = str(event.get("action") or "")
        job_id = str(event.get("job_id") or "")
        attempt = int(event.get("attempt") or 0)
        key = (job_id, attempt, phase)
        if action == "enter":
            stack = open_stack.setdefault(key, [])
            # Ignore re-enter only within the same logical attempt.
            if stack:
                ignored_duplicate_enters += 1
                continue
            stack.append(event)
        elif action == "exit":
            stack = open_stack.get(key) or []
            if not stack:
                continue
            start_event = stack.pop(0)
            start = int(start_event.get("epoch_s") or epoch)
            dur = max(0, epoch - start)
            if phase in durations:
                durations[phase] += float(dur)
            segments.append(
                {
                    "job_id": job_id or None,
                    "attempt": attempt,
                    "lease_id": start_event.get("lease_id")
                    or event.get("lease_id"),
                    "phase": phase,
                    "start_epoch_s": start,
                    "end_epoch_s": epoch,
                    "duration_s": dur,
                }
            )

    # Still-open intervals: count through last_ts (or now).
    now = int(time.time())
    end_ref = last_ts or now
    for (job_id, attempt, phase), starts in open_stack.items():
        for start_event in starts:
            start = int(start_event.get("epoch_s") or end_ref)
            dur = max(0, end_ref - start)
            if phase in durations:
                durations[phase] += float(dur)
            segments.append(
                {
                    "job_id": job_id or None,
                    "attempt": attempt,
                    "lease_id": start_event.get("lease_id"),
                    "phase": phase,
                    "start_epoch_s": start,
                    "end_epoch_s": end_ref,
                    "duration_s": dur,
                    "open": True,
                }
            )

    wall = float((last_ts - first_ts) if first_ts is not None and last_ts is not None else 0)
    gpu_wait_s = durations["gpu_queue_wait"]
    gpu_startup_s = durations["gpu_worker_starting"]
    isaac_startup_s = durations["isaac_starting"]
    gpu_active_interval_s = durations["gpu_active"]
    gpu_idle_s = durations["gpu_idle_assigned"]

    by_attempt: dict[tuple[str, int], dict[str, float]] = {}
    for seg in segments:
        jid = str(seg.get("job_id") or "")
        attempt = int(seg.get("attempt") or 0)
        bucket = by_attempt.setdefault(
            (jid, attempt),
            {
                "first": float("inf"),
                "last": 0.0,
                "wait": 0.0,
                "startup": 0.0,
                "isaac": 0.0,
                "active_interval": 0.0,
            },
        )
        start = float(seg.get("start_epoch_s") or 0)
        end = float(seg.get("end_epoch_s") or 0)
        bucket["first"] = min(bucket["first"], start)
        bucket["last"] = max(bucket["last"], end)
        phase = str(seg.get("phase") or "")
        dur = float(seg.get("duration_s") or 0)
        if phase == "gpu_queue_wait":
            bucket["wait"] += dur
        elif phase == "gpu_worker_starting":
            bucket["startup"] += dur
        elif phase == "isaac_starting":
            bucket["isaac"] += dur
        elif phase == "gpu_active":
            bucket["active_interval"] += dur

    per_attempt: list[dict[str, Any]] = []
    gpu_active_s = 0.0
    by_job_totals: dict[str, dict[str, float]] = {}
    for (jid, attempt), bucket in sorted(by_attempt.items()):
        if bucket["first"] == float("inf"):
            continue
        wall_j = max(0.0, bucket["last"] - bucket["first"])
        wall_minus_wait = max(
            0.0,
            wall_j - bucket["wait"] - bucket["startup"] - bucket["isaac"],
        )
        # Explicit active intervals survive retries without counting heartbeat
        # grace or queue gaps as GPU work.
        active_j = bucket["active_interval"]
        gpu_active_s += active_j
        per_attempt.append(
            {
                "job_id": jid or None,
                "attempt": attempt,
                "wall_time_s": wall_j,
                "gpu_wait_s": bucket["wait"],
                "gpu_startup_s": bucket["startup"],
                "isaac_startup_s": bucket["isaac"],
                "gpu_active_s": active_j,
                "gpu_active_interval_s": bucket["active_interval"],
                "wall_minus_wait_s": wall_minus_wait,
            }
        )
        total = by_job_totals.setdefault(
            jid,
            {
                "attempts": 0.0,
                "wall_time_s": 0.0,
                "gpu_wait_s": 0.0,
                "gpu_startup_s": 0.0,
                "isaac_startup_s": 0.0,
                "gpu_active_s": 0.0,
                "gpu_active_interval_s": 0.0,
            },
        )
        total["attempts"] += 1
        for field in (
            "wall_time_s",
            "gpu_wait_s",
            "gpu_startup_s",
            "isaac_startup_s",
            "gpu_active_s",
            "gpu_active_interval_s",
        ):
            total[field] += float(per_attempt[-1][field])

    per_job = [
        {"job_id": jid or None, **values}
        for jid, values in sorted(by_job_totals.items())
    ]

    return {
        "schema_version": 3,
        "updated_at": utc_now(),
        "first_epoch_s": first_ts,
        "last_epoch_s": last_ts,
        "wall_time_s": wall,
        "gpu_wait_s": gpu_wait_s,
        "gpu_startup_s": gpu_startup_s,
        "isaac_startup_s": isaac_startup_s,
        "gpu_active_s": gpu_active_s,
        "gpu_active_interval_s": gpu_active_interval_s,
        "gpu_idle_assigned_s": gpu_idle_s,
        "wall_minus_wait_s": sum(
            float(item["wall_minus_wait_s"]) for item in per_attempt
        ),
        "per_attempt": per_attempt,
        "per_job": per_job,
        "ignored_duplicate_enters": ignored_duplicate_enters,
        "event_count": len(events),
        "segment_count": len(segments),
        "segments": segments,
        "accounting": {
            "gpu_active_s": (
                "sum of closed gpu_active intervals across attempts"
            ),
            "gpu_active_interval_s": "sum of gpu_active enter/exit intervals",
            "wall_minus_wait_s": (
                "sum over attempts of max(0, wall - wait - startup - isaac)"
            ),
            "wall_time_s": "global first→last event (may span many jobs)",
            "duplicate_enters": (
                "ignored while phase open for (job_id, attempt, phase)"
            ),
            "fairness": "prefer gpu_active_s over wall_time_s",
        },
    }


def write_summary(
    run_id: str,
    *,
    durable_dir: str = "/durable",
    also_local: Path | None = None,
) -> dict[str, Any]:
    roots = [telemetry_root(run_id, durable_dir)]
    if also_local is not None:
        roots.append(also_local)
    # Merge events from all roots.
    merged: list[dict[str, Any]] = []
    for root in roots:
        if root.is_dir():
            merged.extend(_load_events(root))
    # Dedup
    seen: set[str] = set()
    events: list[dict[str, Any]] = []
    for event in sorted(
        merged, key=lambda e: (e.get("epoch_s") or 0, e.get("event_id") or "")
    ):
        eid = str(event.get("event_id") or "")
        if eid and eid in seen:
            continue
        if eid:
            seen.add(eid)
        events.append(event)
    summary = summarize_events(events)
    summary["run_id"] = run_id
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / "gpu_time_summary.json"
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(text)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            _flush(root)
        except OSError:
            continue
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--run-id", required=True)
    emit.add_argument("--phase", required=True, choices=PHASES)
    emit.add_argument("--action", required=True, choices=["enter", "exit"])
    emit.add_argument("--job-id", default="")
    emit.add_argument("--attempt", type=int, default=0)
    emit.add_argument("--lease-id", default="")
    emit.add_argument("--durable-dir", default="/durable")
    emit.add_argument("--detail-json", default="")
    summ = sub.add_parser("summarize")
    summ.add_argument("--run-id", required=True)
    summ.add_argument("--durable-dir", default="/durable")
    args = parser.parse_args()
    if args.cmd == "emit":
        detail = json.loads(args.detail_json) if args.detail_json else {}
        event = append_event(
            args.run_id,
            phase=args.phase,
            action=args.action,
            job_id=args.job_id,
            attempt=args.attempt,
            lease_id=args.lease_id,
            detail=detail,
            durable_dir=args.durable_dir,
        )
        print(json.dumps(event, indent=2, sort_keys=True))
        return 0
    summary = write_summary(args.run_id, durable_dir=args.durable_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
