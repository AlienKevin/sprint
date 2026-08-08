#!/usr/bin/env python3
"""Rebuild per-attempt ATIF and one run-level usage ledger from durable Codex logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from harbor.agents.installed.codex import Codex
from harbor.utils.trajectory_utils import format_trajectory_json


ATTEMPT_RE = re.compile(r"cpu-attempt-(\d+)")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def source_groups(state_dir: Path) -> list[tuple[int, str, list[Path]]]:
    groups: dict[tuple[int, str], list[Path]] = {}
    roots = [state_dir / "durable-trace" / "raw", state_dir / "trace" / "raw"]
    roots.extend(state_dir.glob("recovery/*/*/trace/raw"))
    for root in roots:
        for chunks in root.glob("cpu-attempt-*/codex/*/chunks"):
            match = ATTEMPT_RE.fullmatch(chunks.parents[2].name)
            if not match:
                continue
            attempt = int(match.group(1))
            source_id = chunks.parent.name
            files = sorted(chunks.glob("*.jsonl"))
            if not files:
                continue
            key = (attempt, source_id)
            existing = groups.get(key)
            if existing is None or sum(path.stat().st_size for path in files) > sum(
                path.stat().st_size for path in existing
            ):
                # Recovery snapshots may contain an older prefix of the same
                # immutable stream. Use the most complete copy.
                groups[key] = files
    return [
        (attempt, source_id, groups[(attempt, source_id)])
        for attempt, source_id in sorted(groups)
    ]


def reconstruct_group(
    *,
    state_dir: Path,
    run: dict[str, Any],
    attempt: int,
    source_id: str,
    chunks: list[Path],
) -> dict[str, Any]:
    out = (
        state_dir / "trace" / "reconstructed" / f"cpu-attempt-{attempt:03d}" / source_id
    )
    sessions = out / "sessions"
    combined = b"".join(path.read_bytes() for path in chunks)
    session_path = sessions / "rollout.jsonl"
    atomic_text(session_path, combined.decode("utf-8", errors="replace"))

    kwargs: dict[str, Any] = {
        "logs_dir": out,
        "model_name": str(run["model"]),
        "reasoning_effort": str(run["reasoning_effort"]),
    }
    if str(run["model"]).split("/", 1)[-1] == "gpt-5.6-terra":
        kwargs["service_tier"] = "default"
    agent = Codex(**kwargs)
    trajectory = agent._convert_events_to_trajectory(sessions)
    audit = getattr(agent, "_last_usage_audit", None)
    if trajectory is not None:
        trajectory_path = out / "trajectory.json"
        atomic_text(trajectory_path, format_trajectory_json(trajectory.to_json_dict()))
    else:
        trajectory_path = None

    source = {
        "cpu_attempt": attempt,
        "source_id": source_id,
        "combined_session_sha256": hashlib.sha256(combined).hexdigest(),
        "chunks": [
            {
                "path": str(path.relative_to(state_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in chunks
        ],
        "trajectory_path": (
            str(trajectory_path.relative_to(state_dir)) if trajectory_path else None
        ),
        "trajectory_sha256": sha256_file(trajectory_path) if trajectory_path else None,
    }
    if not isinstance(audit, dict):
        source.update(
            {
                "session_id": None,
                "request_count": 0,
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.0,
                "requests": [],
                "pricing_snapshots": [],
                "note": "no completed model request was present",
            }
        )
        return source

    session_id = str(audit.get("session_id") or source_id)
    requests = []
    for request in audit.get("requests") or []:
        if not isinstance(request, dict):
            continue
        requests.append(
            {
                **request,
                "session_id": session_id,
                "cpu_attempt": attempt,
                "run_api_call_id": f"{session_id}:{request.get('api_call_id')}",
            }
        )
    source.update(
        {
            "session_id": session_id,
            "request_count": len(requests),
            "cost_reconstruction_complete": audit.get("cost_reconstruction_complete"),
            "calculated_api_usage_usd": audit.get("calculated_api_usage_usd"),
            "requests": requests,
            "pricing_snapshots": audit.get("pricing_snapshots") or [],
            "reconciliation_mismatches": audit.get("reconciliation_mismatches") or {},
        }
    )
    atomic_text(
        out / "usage-audit.json", json.dumps(source, indent=2, sort_keys=True) + "\n"
    )
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    run = json.loads((state_dir / "run.json").read_text())
    groups = source_groups(state_dir)
    if not groups:
        raise SystemExit("no durable Codex trace chunks are available")
    sources = [
        reconstruct_group(
            state_dir=state_dir,
            run=run,
            attempt=attempt,
            source_id=source_id,
            chunks=chunks,
        )
        for attempt, source_id, chunks in groups
    ]
    unique_sessions: dict[str, dict[str, Any]] = {}
    anonymous = []
    for source in sources:
        session_id = source.get("session_id")
        if session_id:
            previous = unique_sessions.get(str(session_id))
            if (
                previous
                and previous["combined_session_sha256"]
                != source["combined_session_sha256"]
            ):
                raise SystemExit(
                    f"conflicting durable copies for Codex session {session_id}"
                )
            unique_sessions[str(session_id)] = source
        else:
            anonymous.append(source)
    sessions = list(unique_sessions.values()) + anonymous
    requests = [request for source in sessions for request in source["requests"]]
    request_ids = [request["run_api_call_id"] for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise SystemExit("duplicate run-level Codex API call IDs")
    snapshots: dict[str, dict[str, Any]] = {}
    for source in sessions:
        for snapshot in source["pricing_snapshots"]:
            if isinstance(snapshot, dict) and snapshot.get("id"):
                snapshots[str(snapshot["id"])] = snapshot
    complete = all(source["cost_reconstruction_complete"] for source in sessions)
    expected_attempts = {
        int(row["attempt"])
        for row in run.get("cpu_launch_history") or []
        if isinstance(row, dict) and row.get("attempt") is not None
    }
    captured_attempts = {int(source["cpu_attempt"]) for source in sessions}
    payload = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "model": run["model"],
        "resolved_model_version": run.get("resolved_model_version"),
        "reasoning_effort": run["reasoning_effort"],
        "source": "durable_codex_session_chunks",
        "source_sessions": [
            {key: value for key, value in source.items() if key != "requests"}
            for source in sessions
        ],
        "expected_cpu_attempts": sorted(expected_attempts),
        "captured_cpu_attempts": sorted(captured_attempts),
        "attempt_coverage_complete": expected_attempts <= captured_attempts,
        "request_count": len(requests),
        "requests": requests,
        "pricing_snapshots": list(snapshots.values()),
        "cost_reconstruction_complete": complete and bool(requests),
        "calculated_api_usage_usd": (
            sum(float(request["calculated_cost_usd"]) for request in requests)
            if complete and requests
            else None
        ),
    }
    atomic_text(
        state_dir / "usage" / "run-usage-audit.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["cost_reconstruction_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
