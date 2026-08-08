"""Pure lease, retry, and host-lock helpers for durable GPU jobs."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

TERMINAL = frozenset({"succeeded", "failed", "terminated"})
OWNED = frozenset({"claiming", "dispatched", "running", "death_observed"})
CLAIMABLE = frozenset({"pending", "retry_wait"})

DEFAULT_CLAIM_STALE_SEC = 900
DEFAULT_HEARTBEAT_TIMEOUT_SEC = 45
DEFAULT_STARTUP_GRACE_SEC = 600
DEFAULT_DEAD_GRACE_SEC = 20
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SEC = 10
DEFAULT_RETRY_BACKOFF_MAX_SEC = 120


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        import calendar

        return float(
            calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
        )
    except (TypeError, ValueError):
        return None


def heartbeat_epoch(heartbeat: dict[str, Any] | None) -> float | None:
    if not heartbeat:
        return None
    raw = heartbeat.get("updated_at_epoch_s")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return parse_ts(heartbeat.get("updated_at"))


def heartbeat_matches(
    job: dict[str, Any], heartbeat: dict[str, Any] | None
) -> bool:
    if not heartbeat:
        return False
    try:
        attempt_matches = int(heartbeat.get("attempt") or 0) == int(
            job.get("attempt") or 0
        )
    except (TypeError, ValueError):
        return False
    return attempt_matches and str(heartbeat.get("lease_id") or "") == str(
        job.get("lease_id") or ""
    )


def assess_worker_liveness(
    job: dict[str, Any],
    heartbeat: dict[str, Any] | None,
    *,
    probe_state: str,
    now: float | None = None,
    heartbeat_timeout_sec: int = DEFAULT_HEARTBEAT_TIMEOUT_SEC,
    startup_grace_sec: int = DEFAULT_STARTUP_GRACE_SEC,
    dead_grace_sec: int = DEFAULT_DEAD_GRACE_SEC,
    standing: bool = False,
) -> str:
    """Return alive, grace, observe, dead, or ignore.

    A failed/unknown Modal probe alone never fences a worker. The lease must
    also be stale, and a second observation must survive ``dead_grace_sec``.

    ``standing`` marks a job running inside a *standing* sandbox that outlives
    individual jobs (see gpu_worker.ensure_standing_sandbox). There, a live
    sandbox says nothing about whether this job's process is still alive -- the
    container stays up between jobs by design -- so a live probe must NOT
    short-circuit to alive, or a crashed job would never be reaped and the
    logical job would hang forever. The heartbeat governs instead; the sandbox
    probe only downgrades (a dead standing sandbox still means a dead job).
    """
    status = str(job.get("status") or "")
    if status not in OWNED:
        return "ignore"
    if probe_state not in {"alive", "exited", "unknown"}:
        raise ValueError(f"invalid probe_state: {probe_state}")
    ref = now if now is not None else time.time()
    hb_at = heartbeat_epoch(heartbeat) if heartbeat_matches(job, heartbeat) else None
    hb_age = None if hb_at is None else max(0.0, ref - hb_at)
    heartbeat_fresh = hb_age is not None and hb_age <= heartbeat_timeout_sec

    if probe_state == "alive" and (not standing or heartbeat_fresh):
        return "alive"
    if heartbeat_fresh:
        # Modal may report an exit before the last Volume commit is visible.
        return "grace"

    started = (
        float(job.get("claimed_at_epoch_s") or 0)
        or float(job.get("dispatched_at_epoch_s") or 0)
        or parse_ts(job.get("claimed_at") or job.get("dispatched_at"))
        or ref
    )
    if status in {"claiming", "dispatched"} and ref - started < startup_grace_sec:
        return "grace"

    observed = job.get("death_observed_epoch_s")
    try:
        observed_at = float(observed) if observed is not None else None
    except (TypeError, ValueError):
        observed_at = None
    if observed_at is None:
        return "observe"
    if ref - observed_at < dead_grace_sec:
        return "grace"

    # Unknown probes get one extra lease window. This avoids replacing a live
    # worker during a short Modal control-plane or Volume read outage.
    if probe_state == "unknown":
        stale_for = hb_age if hb_age is not None else ref - started
        if stale_for < heartbeat_timeout_sec + dead_grace_sec:
            return "grace"
    return "dead"


def select_claim_action(
    job: dict[str, Any] | None,
    *,
    claim_id: str,
    now: float | None = None,
    stale_sec: int = DEFAULT_CLAIM_STALE_SEC,
) -> str:
    """Return claim or skip for a loaded job snapshot."""
    if not job:
        return "skip"
    status = str(job.get("status") or "pending")
    if status in TERMINAL:
        return "skip"
    if status == "pending" and not job.get("sandbox_id"):
        return "claim"
    if status == "retry_wait":
        ref = now if now is not None else time.time()
        try:
            due = float(job.get("retry_not_before_epoch_s") or 0)
        except (TypeError, ValueError):
            due = 0.0
        return "claim" if ref >= due else "skip"
    if status in OWNED and job.get("lease_id") == claim_id:
        return "claim"
    return "skip"


def build_claim_payload(
    job: dict[str, Any],
    *,
    claim_id: str,
    owner: str = "host",
) -> dict[str, Any]:
    payload = dict(job)
    current_attempt = int(payload.get("attempt") or 0)
    if str(payload.get("status") or "") == "retry_wait":
        attempt = int(payload.get("next_attempt") or current_attempt + 1)
    elif current_attempt <= 0:
        attempt = 1
    else:
        attempt = current_attempt
    now_epoch = time.time()
    payload.update(
        {
            "status": "claiming",
            "claim_id": claim_id,
            "lease_id": claim_id,
            "attempt": attempt,
            "fence_epoch": int(payload.get("fence_epoch") or 0) + 1,
            "claimed_at": utc_now(),
            "claimed_at_epoch_s": now_epoch,
            "claim_owner": owner,
            "gpu_type": payload.get("gpu_type") or "A10G",
        }
    )
    for key in (
        "sandbox_id",
        "dispatched_at",
        "dispatched_at_epoch_s",
        "death_observed_at",
        "death_observed_epoch_s",
        "next_attempt",
        "retry_not_before",
        "retry_not_before_epoch_s",
    ):
        payload.pop(key, None)
    return payload


def ownership_matches(job: dict[str, Any] | None, claim_id: str) -> bool:
    if not job:
        return False
    return (
        str(job.get("claim_id") or "") == claim_id
        and str(job.get("lease_id") or "") == claim_id
        and str(job.get("status") or "") in OWNED
    )


def retry_delay_s(
    completed_attempt: int,
    *,
    base_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
    max_sec: float = DEFAULT_RETRY_BACKOFF_MAX_SEC,
) -> float:
    completed_attempt = max(1, int(completed_attempt))
    return min(float(max_sec), float(base_sec) * (2 ** (completed_attempt - 1)))


def retry_allowed(job: dict[str, Any]) -> bool:
    attempt = int(job.get("attempt") or 0)
    max_attempts = int(job.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    return attempt < max(1, max_attempts)


@contextmanager
def dispatch_lock(state_dir: Path, *, timeout_sec: float = 30.0) -> Iterator[bool]:
    """Exclusive flock for one host dispatch loop. Yields True if acquired."""
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "gpu-dispatch.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.time() + timeout_sec
    got = False
    try:
        while time.time() < deadline:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except BlockingIOError:
                time.sleep(0.05)
        yield got
    finally:
        if got:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
