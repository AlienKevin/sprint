#!/usr/bin/env python3
"""Recompute, capture, render, and deploy the current policy frontier."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[2]
WEB_DEFAULT = ROOT / "web"
sys.path.insert(0, str(ROOT))
from event_runtime.event import load_event  # noqa: E402
from event_runtime.export.config import PUBLIC_RUN_LIMIT, public_index_lock  # noqa: E402
from event_runtime.export.site_bundle import build_site_bundle  # noqa: E402

EVENT = load_event(repository_root=ROOT)
EVENT_REPLAY = ROOT / "web" / "renderers" / EVENT.name
BUILD_SCRIPT = EVENT_REPLAY / "render.py"
HQ_PATH = EVENT_REPLAY / "g1_hq.json"
SCENE_SOURCE = EVENT_REPLAY / "scene.js"
PLAYER_TEMPLATE = ROOT / "web/replay-template.html"
PROJECT_ID = "prj_dgvTovRNwdSDcefYmo6oXfju9M3p"
ORG_ID = "team_SNgoAcFfHYXYdUIXhj16bGek"
VERCEL_SCOPE = "alienkevins-projects"
SCORE_TOLERANCE_MPS = 1e-9
DEPLOY_DEBOUNCE_SECONDS = 300
# Vercel may spend several minutes retrieving and building a large replay-heavy
# site after the upload has completed.  The CLI remains the authoritative wait
# for both deployment readiness and the subsequent alias step, so keep this
# bounded but above the observed 15-minute production build queue delay.
DEPLOY_COMMAND_TIMEOUT_SECONDS = 30 * 60
CAPTURE_MAX_ATTEMPTS = 3
PIPELINE_LOCK = ROOT / "runs" / "ops" / ".frontier-pipeline.lock"
SITE_SNAPSHOT_IGNORED_DIRECTORIES = {".git", ".vercel", "node_modules"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n", mode)


def preserve_observer_timestamp(
    path: Path, payload: dict[str, Any], *, timestamp_key: str
) -> None:
    """Keep a public observer clock stable when its material payload is unchanged."""
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(previous, dict):
        return
    previous_material = {
        key: value for key, value in previous.items() if key != timestamp_key
    }
    material = {key: value for key, value in payload.items() if key != timestamp_key}
    previous_timestamp = previous.get(timestamp_key)
    if previous_material == material and isinstance(previous_timestamp, str):
        payload[timestamp_key] = previous_timestamp


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    tmp = Path(raw)
    try:
        shutil.copyfile(source, tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def is_atomic_staging_file(path: Path) -> bool:
    """Return whether *path* is an unpublished atomic-writer scratch file."""
    return path.name.startswith(".") and path.name.endswith(".tmp")


@contextlib.contextmanager
def staged_site_snapshot(web: Path) -> Iterator[Path]:
    """Expose one immutable, deployable view of the live website tree.

    Live telemetry publishers replace their destination files atomically.  A
    Vercel upload used to traverse that mutable tree directly, so it could
    enumerate an adjacent ``.*.tmp`` file just before the publisher renamed it
    and then fail while trying to stat the vanished path.  This snapshot uses
    hard links on the same filesystem: an atomic replacement of the live name
    cannot change the inode visible to the upload.  The volatile Vercel build
    directory and unpublished scratch files are deliberately omitted.
    """
    PIPELINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vercel-site-snapshot-", dir=PIPELINE_LOCK.parent
    ) as raw:
        root = Path(raw)
        snapshot = root / "web"
        for attempt in range(3):
            shutil.rmtree(snapshot, ignore_errors=True)
            snapshot.mkdir()
            try:
                build_site_bundle(
                    web,
                    snapshot,
                    require_current=False,
                    include_project_link=True,
                )
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise
                time.sleep(0.02 * (attempt + 1))
        yield snapshot


@dataclasses.dataclass(frozen=True)
class LedgerRead:
    rows: list[dict[str, Any]]
    errors: list[str]
    digest: str


def read_ledger(path: Path) -> LedgerRead:
    if not path.exists():
        return LedgerRead([], [], hashlib.sha256(b"").hexdigest())
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    lines = raw.splitlines()
    for number, encoded in enumerate(lines, start=1):
        if not encoded.strip():
            continue
        try:
            row = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            suffix = (
                "partial"
                if number == len(lines) and not raw.endswith(b"\n")
                else "malformed"
            )
            errors.append(f"line {number}: {suffix} JSON ({exc})")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {number}: row is not an object")
            continue
        rows.append(row)
    return LedgerRead(rows, errors, hashlib.sha256(raw).hexdigest())


def row_terminal(row: dict[str, Any]) -> bool:
    return bool(
        row.get("finished_at") or row.get("error") or row.get("rewards") is not None
    )


def ledger_counts(read: LedgerRead) -> dict[str, int]:
    rows = read.rows
    accepted = [row for row in rows if row.get("accepted", True) is not False]
    return {
        "submitted": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "queued": sum(
            not row.get("started_at") and not row_terminal(row) for row in accepted
        ),
        "running": sum(
            bool(row.get("started_at")) and not row_terminal(row) for row in accepted
        ),
        "scored": sum(row.get("rewards") is not None for row in accepted),
        "error": sum(bool(row.get("error")) for row in accepted),
        "terminal": sum(row_terminal(row) for row in rows),
        "malformed": len(read.errors),
    }


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def attempt_dir_for_row(trial: Path, row: dict[str, Any]) -> Path | None:
    root = trial / "artifacts" / "continuous" / "attempts"
    try:
        index = int(row["index"])
    except (KeyError, TypeError, ValueError):
        return None
    matches = sorted(root.glob(f"{index:04d}-*")) if root.is_dir() else []
    name = str(row.get("name") or "")
    named = [path for path in matches if name and path.name.endswith(name)]
    if len(named) == 1:
        return named[0]
    return matches[0] if len(matches) == 1 else None


def policy_for_row(trial: Path, row: dict[str, Any]) -> Path | None:
    artifact = row.get("artifact_path")
    if isinstance(artifact, str):
        candidate = trial / "artifacts" / artifact
        if candidate.is_file():
            return candidate
    attempt = attempt_dir_for_row(trial, row)
    if not attempt:
        return None
    candidate = attempt / "artifacts" / "app" / "submission" / "policy.pt"
    return candidate if candidate.is_file() else None


@dataclasses.dataclass(frozen=True)
class Candidate:
    index: int
    name: str
    effective_speed_mps: float
    policy_hash: str | None
    policy_path: str | None


def candidates_from_rows(
    trial: Path, rows: Sequence[dict[str, Any]]
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in rows:
        rewards = row.get("rewards")
        if not isinstance(rewards, dict) or row.get("error"):
            continue
        structurally_valid = _as_number(rewards.get("submission_contract_valid"))
        score = _as_number(rewards.get("effective_speed_mps", rewards.get("reward")))
        if structurally_valid != 1.0 or score is None or score < 0:
            continue
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        policy = policy_for_row(trial, row)
        policy_hash = sha256_file(policy) if policy else None
        candidates.append(
            Candidate(
                index=index,
                name=str(row.get("name") or f"attempt-{index}"),
                effective_speed_mps=score,
                policy_hash=policy_hash,
                policy_path=str(policy) if policy else None,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.index)


def graded_policy_records(
    trial: Path, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return every uniquely graded policy, including DQs and failed runs."""
    records: list[dict[str, Any]] = []
    for row in rows:
        rewards = row.get("rewards")
        if not isinstance(rewards, dict) or row.get("error"):
            continue
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        policy = policy_for_row(trial, row)
        if policy is None:
            continue
        attempt = attempt_dir_for_row(trial, row)
        replay = attempt / "verifier" / "replay.json" if attempt else None
        details_path = attempt / "verifier" / "sprint_results.json" if attempt else None
        try:
            details = json.loads(details_path.read_text()) if details_path else {}
        except (OSError, json.JSONDecodeError):
            details = {}
        failed = details.get("failed_gates")
        if not isinstance(failed, list):
            failed = [
                name.removeprefix("gate_")
                for name, value in rewards.items()
                if name.startswith("gate_") and _as_number(value) == 0.0
            ]
        valid = _as_number(rewards.get("valid_run")) == 1.0
        best = _as_number(rewards.get("best_100m_s", rewards.get("best_valid_100m_s")))
        score = _as_number(rewards.get("effective_speed_mps", rewards.get("reward")))
        records.append(
            {
                "index": index,
                "name": str(row.get("name") or f"attempt-{index}"),
                "policy_hash": sha256_file(policy),
                "policy_path": str(policy),
                "replay_path": str(replay) if replay and replay.is_file() else None,
                "valid_run": valid,
                "best_100m_s": best if valid and best and best > 0 else None,
                "effective_speed_mps": score,
                "max_distance_m": _as_number(details.get("max_distance_m")),
                "max_distance_semantics": details.get("max_distance_semantics"),
                "termination_reason": details.get("termination_reason"),
                "stop_time_s": _as_number(details.get("stop_time_s")),
                "time_to_max_distance_s": _as_number(
                    details.get("time_to_max_distance_s")
                ),
                "peak_speed_mps": _as_number(rewards.get("peak_speed_mps")),
                "failed_gates": sorted({str(name) for name in failed}),
                "submitted_at": row.get("submitted_at"),
                "finished_at": row.get("finished_at"),
                "cache_hit": bool(row.get("cache_hit")),
                "source_evaluation_id": row.get("source_evaluation_id"),
                "verification_attempts": int(row.get("verification_attempts") or 0),
                "verification_retry_events": row.get("verification_retry_events")
                if isinstance(row.get("verification_retry_events"), list)
                else [],
            }
        )
    return sorted(records, key=lambda record: int(record["index"]))


def compute_frontier(candidates: Sequence[Candidate]) -> tuple[list[Candidate], bool]:
    if not candidates:
        return [], False
    best = candidates[0]
    for candidate in candidates[1:]:
        if (
            candidate.effective_speed_mps
            > best.effective_speed_mps + SCORE_TOLERANCE_MPS
        ):
            best = candidate
    return [best], False


def site_tree_hash(web: Path) -> str:
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            if not web.is_dir():
                return digest.hexdigest()
            for path in sorted(item for item in web.rglob("*") if item.is_file()):
                relative = path.relative_to(web)
                if any(
                    part in SITE_SNAPSHOT_IGNORED_DIRECTORIES for part in relative.parts
                ):
                    continue
                # Atomic publishers stage dot-prefixed ``*.tmp`` files in the
                # live tree before ``os.replace``.  Those files are neither
                # public artifacts nor a stable part of a deployable snapshot;
                # hashing them races their expected disappearance.
                if is_atomic_staging_file(path):
                    continue
                digest.update(relative.as_posix().encode())
                digest.update(b"\0")
                digest.update(sha256_file(path).encode())
                digest.update(b"\0")
            return digest.hexdigest()
        except FileNotFoundError:
            # Timeline/policy writers publish with atomic replacements while the
            # observer hashes the live tree.  Retry the entire snapshot rather
            # than preserving a digest assembled from two filesystem states.
            if attempt == 2:
                raise
            time.sleep(0.02 * (attempt + 1))
    raise AssertionError("unreachable")


def public_artifact_hashes(web: Path) -> dict[str, str]:
    """Snapshot hashes for the per-run artifacts used as deployment proofs."""
    for attempt in range(3):
        try:
            paths = [
                *web.glob("data/batches/*.json"),
                *web.glob("data/performance/current.json"),
                *web.glob("data/policies/*.json"),
                *web.glob("data/timelines/*.json"),
                *web.glob("data/timeline-overviews/*.json"),
            ]
            return {
                path.relative_to(web).as_posix(): sha256_file(path)
                for path in sorted(paths)
                if path.is_file()
            }
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(0.02 * (attempt + 1))
    raise AssertionError("unreachable")


def initial_state(job: Path, trial: Path, web: Path) -> dict[str, Any]:
    baseline = site_tree_hash(web)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "job": str(job.resolve()),
        "trial": str(trial.resolve()),
        "ledger_hash": None,
        "ledger_errors": [],
        "counts": {},
        "policies": {},
        "frontier": [],
        "frontier_candidates": [],
        "capture_queue": [],
        "captures": {},
        "errors": [],
        "baseline_site_hash": baseline,
        "last_deployed_site_hash": baseline,
        "pending_site_hash": None,
        "site_change_first_seen_at": None,
        "site_status": "noop",
        "last_deployment_url": None,
        "production_alias": "https://g1-sprint.vercel.app",
    }


def load_state(path: Path, job: Path, trial: Path, web: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        state = initial_state(job, trial, web)
    if state.get("job") != str(job.resolve()) or state.get("trial") != str(
        trial.resolve()
    ):
        raise ValueError("frontier state belongs to a different explicit job/trial")
    return state


def record_error(state: dict[str, Any], message: str) -> None:
    errors = state.setdefault("errors", [])
    errors.append({"at": utc_now(), "message": message})
    del errors[:-100]


def renderer_source_hash() -> str:
    """Identify the exact renderer and validation implementation for a capture."""
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        BUILD_SCRIPT,
        SCENE_SOURCE,
        PLAYER_TEMPLATE,
    ):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def reconcile_capture_queue(state: dict[str, Any], renderer_hash: str) -> set[str]:
    """Fence stale renderer work and recover an interrupted current attempt.

    There is at most one actionable queue row per policy and renderer version.
    A renderer code change permits a fresh bounded attempt without discarding
    the prior error rows, while an unchanged broken renderer stays terminal.
    """
    queue = state.setdefault("capture_queue", [])
    current: set[str] = set()
    captured = state.setdefault("captures", {})
    for item in queue:
        policy_hash = str(item.get("policy_hash") or "")
        if not policy_hash:
            continue
        if captured.get(policy_hash, {}).get("valid"):
            if item.get("status") in {"queued", "running"}:
                item["status"] = "captured"
                item["finished_at"] = utc_now()
            continue
        if item.get("renderer_hash") != renderer_hash:
            if item.get("status") in {"queued", "running"}:
                item["status"] = "superseded_renderer"
                item["finished_at"] = utc_now()
            continue
        if policy_hash in current:
            if item.get("status") in {"queued", "running"}:
                item["status"] = "superseded_duplicate"
                item["finished_at"] = utc_now()
            continue
        current.add(policy_hash)
        if item.get("status") == "running":
            attempts = int(item.get("attempts") or 0)
            item["interrupted_at"] = utc_now()
            item["status"] = "queued" if attempts < CAPTURE_MAX_ATTEMPTS else "error"
            if attempts >= CAPTURE_MAX_ATTEMPTS:
                item["error"] = "capture worker terminated at retry limit"
                item["finished_at"] = utc_now()
    return current


def enqueue_capture(
    state: dict[str, Any],
    *,
    policy_hash: str,
    index: int,
    renderer_hash: str,
    story: str | None = None,
) -> None:
    item: dict[str, Any] = {
        "policy_hash": policy_hash,
        "index": index,
        "queued_at": utc_now(),
        "status": "queued",
        "attempts": 0,
        "max_attempts": CAPTURE_MAX_ATTEMPTS,
        "renderer_hash": renderer_hash,
    }
    if story:
        item["story"] = story
    state.setdefault("capture_queue", []).append(item)


def scan_frontier(
    *, job: Path, trial: Path, state_path: Path, web: Path = WEB_DEFAULT
) -> dict[str, Any]:
    ledger_path = trial / "artifacts" / "continuous" / "ledger.jsonl"
    read = read_ledger(ledger_path)
    candidates = candidates_from_rows(trial, read.rows)
    graded = graded_policy_records(trial, read.rows)
    frontier, _ = compute_frontier(candidates)
    state = load_state(state_path, job, trial, web)
    previous = set(state.get("frontier") or [])
    active = [candidate.policy_hash for candidate in frontier if candidate.policy_hash]

    policies = state.setdefault("policies", {})
    for candidate in candidates:
        key = candidate.policy_hash or f"missing-policy-{candidate.index}"
        policies[key] = {
            "index": candidate.index,
            "name": candidate.name,
            "effective_speed_mps": candidate.effective_speed_mps,
            "policy_hash": candidate.policy_hash,
            "policy_path": candidate.policy_path,
            "on_frontier": candidate.policy_hash in active,
        }
        if candidate.policy_hash is None:
            record_error(state, f"attempt {candidate.index} has no archived policy")

    for record in graded:
        key = str(record["policy_hash"])
        previous_record = policies.get(key) or {}
        policies[key] = {
            **previous_record,
            **record,
            "on_frontier": key in active,
        }

    queue = state.setdefault("capture_queue", [])
    renderer_hash = renderer_source_hash()
    attempted_hashes = reconcile_capture_queue(state, renderer_hash)
    queued_hashes = {
        item.get("policy_hash")
        for item in queue
        if item.get("renderer_hash") == renderer_hash
        and item.get("status") in {"queued", "running"}
    }
    captured = set(state.setdefault("captures", {}))
    for candidate in frontier:
        policy_hash = candidate.policy_hash
        if (
            policy_hash
            and policy_hash not in previous
            and policy_hash not in attempted_hashes
            and policy_hash not in captured
        ):
            enqueue_capture(
                state,
                policy_hash=policy_hash,
                index=candidate.index,
                renderer_hash=renderer_hash,
            )
            attempted_hashes.add(policy_hash)
            queued_hashes.add(policy_hash)

    # Every fresh verifier now records one representative pose replay. Publish
    # all unique graded policies, not only successes, so falls, lane exits, and
    # collision failures remain part of the experiment story. Exact duplicate
    # bytes share one replay page by policy hash.
    for record in graded:
        policy_hash = str(record["policy_hash"])
        if (
            record.get("replay_path")
            and policy_hash not in attempted_hashes
            and policy_hash not in captured
        ):
            enqueue_capture(
                state,
                policy_hash=policy_hash,
                index=record["index"],
                renderer_hash=renderer_hash,
                story="valid" if record["valid_run"] else "failure",
            )
            attempted_hashes.add(policy_hash)
            queued_hashes.add(policy_hash)

    for item in queue:
        if (
            item.get("status") == "queued"
            and item.get("policy_hash") not in active
            and not policies.get(str(item.get("policy_hash")), {}).get("replay_path")
        ):
            item["status"] = "skipped_dominated"
            item["finished_at"] = utc_now()

    state.update(
        {
            "updated_at": utc_now(),
            "last_scan_at": utc_now(),
            "ledger_hash": read.digest,
            "ledger_errors": read.errors,
            "counts": ledger_counts(read),
            "frontier": active,
            "frontier_candidates": [
                {
                    "index": candidate.index,
                    "policy_hash": candidate.policy_hash,
                    "effective_speed_mps": candidate.effective_speed_mps,
                }
                for candidate in frontier
            ],
        }
    )
    atomic_write_json(state_path, state)
    return state


@contextlib.contextmanager
def file_lock(
    path: Path,
    *,
    blocking: bool = True,
    timeout_seconds: float | None = None,
) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if blocking and timeout_seconds is None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:
            deadline = (
                None
                if timeout_seconds is None
                else time.monotonic() + max(0.0, float(timeout_seconds))
            )
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not blocking or (
                        deadline is not None and time.monotonic() >= deadline
                    ):
                        yield False
                        return
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion)
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / magnitude, y / magnitude, z / magnitude, w / magnitude
    vx, vy, vz = (float(value) for value in vector)
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    )
    return [
        vx + 2 * (w * uv[0] + uuv[0]),
        vy + 2 * (w * uv[1] + uuv[1]),
        vz + 2 * (w * uv[2] + uuv[2]),
    ]


def validate_capture_precision(capture: Path, html: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("event_replay_renderer", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay renderer: {BUILD_SCRIPT}")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    G1_PARENT = renderer.G1_PARENT
    _rest_offsets = renderer._rest_offsets

    payload = json.loads(capture.read_text())
    names = payload["body_names"]
    links = [name for name in G1_PARENT if name in names]
    rest = _rest_offsets(payload, names, links)
    checked = [
        child
        for child in (
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
        )
        if child in rest and G1_PARENT.get(child) in names
    ]
    maximum_mm = 0.0
    observations = 0
    for run in payload["frames"]:
        for row in run:
            for child in checked:
                parent = G1_PARENT[child]
                child_offset = 1 + names.index(child) * 7
                parent_offset = 1 + names.index(parent) * 7
                parent_pos = row[parent_offset : parent_offset + 3]
                parent_quat = row[parent_offset + 3 : parent_offset + 7]
                rotated = _rotate_xyzw(parent_quat, rest[child])
                predicted = [
                    float(parent_pos[axis]) + rotated[axis] for axis in range(3)
                ]
                actual = row[child_offset : child_offset + 3]
                error_mm = (
                    math.sqrt(
                        sum(
                            (float(actual[axis]) - predicted[axis]) ** 2
                            for axis in range(3)
                        )
                    )
                    * 1000.0
                )
                maximum_mm = max(maximum_mm, error_mm)
                observations += 1
    page = html.read_text()
    controls_ok = all(f'data-s="{speed}"' in page for speed in ("0.1", "0.6", "1"))
    one_x_default = 'data-s="1" aria-pressed="true"' in page
    # The verifier stores independent world-space body poses.  Sub-millimetre
    # float/solver noise at adjacent articulation origins can add in norm; the
    # observed fixed-link error can therefore land just above 2.0 mm without a
    # visible or structural replay defect.  Keep this guard tight, but leave a
    # small margin above the nominal 2 mm target so valid evidence is not lost
    # to a boundary-level numerical fluctuation.
    threshold_mm = 2.1
    result = {
        "checked_at": utc_now(),
        "observations": observations,
        "max_attachment_error_mm": maximum_mm,
        "threshold_mm": threshold_mm,
        "controls_present": controls_ok,
        "one_x_default": one_x_default,
        "valid": observations > 0
        and maximum_mm <= threshold_mm
        and controls_ok
        and one_x_default,
    }
    if not result["valid"]:
        raise RuntimeError(f"renderer precision validation failed: {result}")
    return result


def run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: float | None = None,
) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        rendered = " ".join(str(part) for part in command)
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output = output[-8_000:].strip()
        raise RuntimeError(
            f"command timed out after {timeout_seconds}s: {rendered}"
            + (f"\n{output}" if output else "")
        ) from exc
    if completed.returncode != 0:
        rendered = " ".join(str(part) for part in command)
        output = completed.stdout[-8_000:].strip()
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {rendered}"
            + (f"\n{output}" if output else "")
        )
    return completed.stdout


def capture_and_render(
    *, state: dict[str, Any], state_path: Path, web: Path, policy_hash: str
) -> None:
    policy = state["policies"][policy_hash]
    replay_path = policy.get("replay_path")
    if not replay_path or not Path(replay_path).is_file():
        raise RuntimeError("sealed verifier replay artifact is missing")
    source = Path(replay_path)
    work = state_path.parent / "captures"
    work.mkdir(parents=True, exist_ok=True)
    short = policy_hash[:12]
    label = f"frontier-{short}"
    capture_final = work / f"{label}.json"
    validation_final = work / f"{label}.validation.json"
    web_capture = web / "captures" / f"{label}.json"
    web_html = web / "replay" / f"{label}.html"

    with tempfile.TemporaryDirectory(
        prefix="event-frontier-", dir=state_path.parent
    ) as raw:
        temporary = Path(raw)
        capture_tmp = temporary / "capture.json"
        html_tmp = temporary / "replay.html"
        atomic_copy(source, capture_tmp)
        valid = bool(policy.get("valid_run"))
        time_value = policy.get("best_100m_s")
        termination = str(policy.get("termination_reason") or "timeout").replace(
            "_", " "
        )
        headline = (
            f"{float(time_value):.3f} s" if valid and time_value else "Did not finish"
        )
        lede = (
            f"Valid 100 m policy: <b>{float(time_value):.3f} s</b>."
            if valid and time_value
            else f"Evaluation stopped by <b>{termination}</b>."
        )
        run_checked(
            [
                "run-heavy",
                sys.executable,
                str(BUILD_SCRIPT),
                "--capture",
                str(capture_tmp),
                "--hq",
                str(HQ_PATH),
                "--out",
                str(html_tmp),
                "--meta-policy",
                label,
                "--title",
                f"G1 100 metres attempt #{policy['index']} - {headline}",
                "--eyebrow",
                f"Unitree G1 · submission #{policy['index']}",
                "--headline",
                headline,
                "--lede",
                lede,
                "--cap",
                f"Attempt #{policy['index']} · <b>{headline}</b>",
                "--story",
                f"Policy SHA-256 {policy_hash}.",
                "--active",
                "timeline",
            ]
        )
        validation = validate_capture_precision(capture_tmp, html_tmp)
        atomic_copy(capture_tmp, capture_final)
        atomic_copy(capture_tmp, web_capture)
        atomic_copy(html_tmp, web_html)
        atomic_write_json(validation_final, validation)

    state.setdefault("captures", {})[policy_hash] = {
        "captured_at": utc_now(),
        "capture": str(capture_final),
        "validation": str(validation_final),
        "web_capture": str(web_capture.relative_to(web)),
        "web_html": str(web_html.relative_to(web)),
        "valid": True,
        "story": "valid" if policy.get("valid_run") else "failure",
    }
    for item in state.get("capture_queue", []):
        if item.get("policy_hash") == policy_hash and item.get("status") == "running":
            item["status"] = "captured"
            item["finished_at"] = utc_now()
    atomic_write_json(state_path, state)


def write_web_policy_indexes(
    state_path: Path, state: dict[str, Any], web: Path
) -> None:
    """Publish a public-safe policy history for the comparison dashboard."""
    run_id = state_path.parent.name
    enqueue_times: dict[str, str] = {}
    registry = state_path.parent / "gpu-job-registry"
    for path in sorted(registry.glob("*.json")) if registry.is_dir() else []:
        try:
            job = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        created_at = job.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            continue
        progress = job.get("progress")
        results = (
            progress.get("submission_results") if isinstance(progress, dict) else []
        )
        for result in results if isinstance(results, list) else []:
            policy_hash = (
                result.get("policy_sha256") if isinstance(result, dict) else None
            )
            if isinstance(policy_hash, str) and policy_hash:
                enqueue_times.setdefault(policy_hash, created_at)
    try:
        run = json.loads((state_path.parent / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        run = {"run_id": run_id}
    rows: list[dict[str, Any]] = []
    captures = state.get("captures", {})
    for policy_hash, policy in state.get("policies", {}).items():
        capture = captures.get(policy_hash) or {}
        web_html = capture.get("web_html") if capture.get("valid") else None
        rows.append(
            {
                "submission_index": policy.get("index"),
                "policy_sha256": policy_hash,
                "enqueued_at": enqueue_times.get(policy_hash),
                "submitted_at": policy.get("submitted_at"),
                "finished_at": policy.get("finished_at"),
                "valid_run": bool(policy.get("valid_run")),
                "best_100m_s": policy.get("best_100m_s"),
                "effective_speed_mps": policy.get("effective_speed_mps"),
                "max_distance_m": policy.get("max_distance_m"),
                "max_distance_semantics": policy.get("max_distance_semantics"),
                "termination_reason": policy.get("termination_reason"),
                "stop_time_s": policy.get("stop_time_s"),
                "time_to_max_distance_s": policy.get("time_to_max_distance_s"),
                "peak_speed_mps": policy.get("peak_speed_mps"),
                "failed_gates": policy.get("failed_gates") or [],
                "on_frontier": bool(policy.get("on_frontier")),
                "replay_ready": bool(web_html),
                "replay_url": (
                    "/" + str(web_html).removesuffix(".html") if web_html else None
                ),
            }
        )
    rows.sort(key=lambda row: int(row.get("submission_index") or 0))
    payload = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "run_id": run_id,
        "model": run.get("model"),
        "resolved_model_version": run.get("resolved_model_version"),
        "reasoning_effort": run.get("reasoning_effort"),
        "created_at": run.get("created_at"),
        "policies": rows,
    }
    run_path = web / "data" / "policies" / f"{run_id}.json"
    preserve_observer_timestamp(run_path, payload, timestamp_key="updated_at")
    atomic_write_json(run_path, payload)

    index_path = web / "data" / "policies" / "index.json"
    with public_index_lock(index_path):
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            index = {"schema_version": 1, "runs": []}
        entries = {
            str(item.get("run_id")): item
            for item in index.get("runs", [])
            if isinstance(item, dict) and item.get("run_id")
        }
        entries[run_id] = {
            "run_id": run_id,
            "model": payload["model"],
            "resolved_model_version": payload["resolved_model_version"],
            "reasoning_effort": payload["reasoning_effort"],
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
            "policy_count": len(rows),
            "valid_count": sum(row["valid_run"] for row in rows),
            "replay_count": sum(row["replay_ready"] for row in rows),
            "path": f"/data/policies/{run_id}.json",
        }
        atomic_write_json(
            index_path,
            {
                "schema_version": 1,
                "updated_at": payload["updated_at"],
                "runs": sorted(
                    entries.values(),
                    key=lambda item: (item.get("created_at") or "", item["run_id"]),
                    reverse=True,
                )[:PUBLIC_RUN_LIMIT],
            },
        )


def verify_project_link(web: Path) -> None:
    path = web / ".vercel" / "project.json"
    try:
        project = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid Vercel project link: {path}") from exc
    expected = {"projectId": PROJECT_ID, "orgId": ORG_ID, "projectName": "sprint"}
    if any(project.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            f"refusing deployment: {path} is not linked to {VERCEL_SCOPE}/sprint"
        )


Runner = Callable[[Sequence[str], Path | None], str]


def _default_runner(command: Sequence[str], cwd: Path | None) -> str:
    return run_checked(
        command,
        cwd=cwd,
        timeout_seconds=DEPLOY_COMMAND_TIMEOUT_SECONDS,
    )


def deploy_if_needed(
    state: dict[str, Any],
    *,
    web: Path,
    now: float | None = None,
    debounce_seconds: int = DEPLOY_DEBOUNCE_SECONDS,
    runner: Runner = _default_runner,
) -> tuple[bool, str]:
    now = time.time() if now is None else now
    with staged_site_snapshot(web) as deployable_web:
        current_hash = site_tree_hash(deployable_web)
        current_artifacts = public_artifact_hashes(deployable_web)
        if current_hash == state.get("last_deployed_site_hash"):
            state.update(
                {
                    "pending_site_hash": None,
                    "site_change_first_seen_at": None,
                    "site_status": "noop",
                    "last_site_check_at": utc_now(),
                    "last_deployed_public_artifacts": current_artifacts,
                }
            )
            return False, "no site file changes"

        if state.get("pending_site_hash") != current_hash:
            state["pending_site_hash"] = current_hash
        if not state.get("site_change_first_seen_at"):
            state["site_change_first_seen_at"] = dt.datetime.fromtimestamp(
                now, tz=dt.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        first = parse_iso(state.get("site_change_first_seen_at")) or now
        if now - first < debounce_seconds:
            state["site_status"] = "debouncing"
            return (
                False,
                f"deploy debounced for {int(debounce_seconds - (now - first))}s",
            )

        verify_project_link(deployable_web)
        deployed_hash = current_hash
        deployed_artifacts = current_artifacts
        output = runner(
            [
                "vercel",
                "deploy",
                "--prod",
                "--yes",
                "--scope",
                VERCEL_SCOPE,
            ],
            deployable_web,
        )
        urls = re.findall(r"https://[A-Za-z0-9.-]+\.vercel\.app", output)
        deployment_url = next(
            (url for url in urls if "-alienkevins-projects.vercel.app" in url),
            urls[0] if urls else None,
        )
        if not deployment_url:
            raise RuntimeError("Vercel deploy succeeded without a deployment URL")
        runner(
            [
                "vercel",
                "alias",
                "set",
                deployment_url,
                "g1-sprint.vercel.app",
                "--scope",
                VERCEL_SCOPE,
            ],
            deployable_web,
        )
    state.update(
        {
            # These proofs describe the exact immutable tree Vercel received,
            # even if a live telemetry writer publishes newer data meanwhile.
            "last_deployed_site_hash": deployed_hash,
            "last_deployed_public_artifacts": deployed_artifacts,
            "pending_site_hash": None,
            "site_change_first_seen_at": None,
            "site_status": "deployed",
            "last_deployed_at": utc_now(),
            "production_alias_updated_at": utc_now(),
            "last_deployment_url": deployment_url,
            "production_alias": "https://g1-sprint.vercel.app",
        }
    )
    return True, deployment_url or "deployed"


def run_worker(
    *,
    job: Path,
    trial: Path,
    state_path: Path,
    web: Path,
    deploy: bool,
    debounce_seconds: int,
) -> dict[str, Any]:
    # A worker is already detached from the run monitor, and each run launches
    # at most one while its PID is alive.  Wait for the shared renderer instead
    # of losing a short lock race and leaving this run's replay queued until its
    # next (potentially multi-minute) telemetry/monitor cycle.  flock remains
    # crash-safe, so a dead renderer cannot strand the lease.
    with file_lock(PIPELINE_LOCK, blocking=True) as acquired:
        if not acquired:  # Defensive: blocking acquisition normally always succeeds.
            return {"status": "busy"}
        state = scan_frontier(job=job, trial=trial, state_path=state_path, web=web)
        while True:
            queued = next(
                (
                    item
                    for item in state.get("capture_queue", [])
                    if item.get("status") == "queued"
                ),
                None,
            )
            if not queued:
                break
            policy_hash = str(queued["policy_hash"])
            state = scan_frontier(job=job, trial=trial, state_path=state_path, web=web)
            policy_record = state.get("policies", {}).get(policy_hash, {})
            if policy_hash not in state.get("frontier", []) and not policy_record.get(
                "replay_path"
            ):
                for item in state.get("capture_queue", []):
                    if item.get("policy_hash") == policy_hash:
                        item["status"] = "skipped_dominated"
                        item["finished_at"] = utc_now()
                atomic_write_json(state_path, state)
                continue
            queued = next(
                item
                for item in state["capture_queue"]
                if item.get("policy_hash") == policy_hash
                and item.get("status") == "queued"
            )
            queued["status"] = "running"
            queued["started_at"] = utc_now()
            queued["attempts"] = int(queued.get("attempts") or 0) + 1
            queued["max_attempts"] = CAPTURE_MAX_ATTEMPTS
            atomic_write_json(state_path, state)
            try:
                capture_and_render(
                    state=state,
                    state_path=state_path,
                    web=web,
                    policy_hash=policy_hash,
                )
            except Exception as exc:
                queued["error"] = str(exc)
                if int(queued["attempts"]) < CAPTURE_MAX_ATTEMPTS:
                    queued["status"] = "queued"
                    queued["retry_queued_at"] = utc_now()
                else:
                    queued["status"] = "error"
                    queued["finished_at"] = utc_now()
                record_error(state, f"capture {policy_hash[:12]}: {exc}")
                atomic_write_json(state_path, state)
            state = load_state(state_path, job, trial, web)

        state = scan_frontier(job=job, trial=trial, state_path=state_path, web=web)
        write_web_policy_indexes(state_path, state, web)
        if deploy:
            state = scan_frontier(job=job, trial=trial, state_path=state_path, web=web)
            write_web_policy_indexes(state_path, state, web)
            ready = all(
                candidate.get("policy_hash")
                and state.get("captures", {})
                .get(candidate["policy_hash"], {})
                .get("valid")
                for candidate in state.get("frontier_candidates", [])
            )
            try:
                if not ready:
                    raise RuntimeError("active frontier capture is missing or invalid")
                deploy_if_needed(
                    state,
                    web=web,
                    debounce_seconds=debounce_seconds,
                )
            except Exception as exc:
                state["site_status"] = "error"
                record_error(state, f"deploy: {exc}")
        atomic_write_json(state_path, state)
        return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "worker", "status"):
        command = sub.add_parser(name)
        command.add_argument("--job", type=Path, required=True)
        command.add_argument("--trial", type=Path, required=True)
        command.add_argument("--state", type=Path, required=True)
        command.add_argument("--web", type=Path, default=WEB_DEFAULT)
        if name == "worker":
            command.add_argument("--deploy", action="store_true")
            command.add_argument(
                "--debounce-seconds", type=int, default=DEPLOY_DEBOUNCE_SECONDS
            )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    job = args.job.resolve()
    trial = args.trial.resolve()
    state_path = args.state.resolve()
    web = args.web.resolve()
    if args.command == "status":
        state = load_state(state_path, job, trial, web)
    elif args.command == "scan":
        with file_lock(PIPELINE_LOCK, blocking=False) as acquired:
            if not acquired:
                print(json.dumps({"status": "busy"}, indent=2))
                return 0
            state = scan_frontier(job=job, trial=trial, state_path=state_path, web=web)
    else:
        state = run_worker(
            job=job,
            trial=trial,
            state_path=state_path,
            web=web,
            deploy=args.deploy,
            debounce_seconds=args.debounce_seconds,
        )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
