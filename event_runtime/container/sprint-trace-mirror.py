#!/usr/bin/env python3
"""Mirror complete agent trace records to immutable durable chunks.

Agent CLIs write their native JSONL locally. This process tails those files
and places complete-line chunks on the run's durable volume. A chunk is
published before its cursor, so termination at any instruction leaves either
the previous cursor or a harmless replayable chunk. Consumers deduplicate
native records when CPU attempts are replayed.

Raw chunks are private forensic data and may contain tool arguments/output.
Only the summary from ``event_runtime.export.timeline`` is public-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Iterable


def _atomic_write(path: pathlib.Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _source_id(path: pathlib.Path, root: pathlib.Path) -> tuple[str, str]:
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = path.name
    with path.open("rb") as handle:
        # The first complete record is stable while the active file grows.
        # Hashing an arbitrary prefix would change the identity for files
        # smaller than that prefix on every append.
        prefix = handle.readline()
    identity = relative.encode() + b"\0" + prefix
    return hashlib.sha256(identity).hexdigest()[:20], relative


def discover_sources(
    *,
    agent_kind: str,
    codex_home: pathlib.Path,
    claude_home: pathlib.Path,
    agent_log_dir: pathlib.Path,
) -> Iterable[tuple[pathlib.Path, pathlib.Path]]:
    if agent_kind == "codex":
        for path in sorted((codex_home / "sessions").rglob("*.jsonl")):
            if path.is_file() and not path.is_symlink():
                yield path, codex_home
        return
    if agent_kind == "claude-code":
        for path in sorted((claude_home / "projects").rglob("*.jsonl")):
            if path.is_file() and not path.is_symlink():
                yield path, claude_home
        path = agent_log_dir / "claude-code.txt"
        if path.is_file() and not path.is_symlink():
            yield path, agent_log_dir
        return
    if agent_kind == "deepseek-harness":
        # The SDK's authoritative session JSONL already lives on /durable.
        # Mirror its real-time notification stream so the existing timeline
        # exporter sees progress before the next restic snapshot.
        path = agent_log_dir / "deepseek-harness-events.jsonl"
        if path.is_file() and not path.is_symlink():
            yield path, agent_log_dir


def mirror_source(
    source: pathlib.Path,
    source_root: pathlib.Path,
    destination_root: pathlib.Path,
    *,
    run_id: str,
    agent_kind: str,
    cpu_attempt: int,
) -> dict[str, Any]:
    source_id, relative = _source_id(source, source_root)
    destination = destination_root / agent_kind / source_id
    cursor_path = destination / "cursor.json"
    try:
        cursor = json.loads(cursor_path.read_text())
    except (OSError, json.JSONDecodeError):
        cursor = {}
    start = int(cursor.get("offset") or 0)
    size = source.stat().st_size
    if size < start:
        start = 0
    with source.open("rb") as handle:
        handle.seek(start)
        pending = handle.read()
    newline = pending.rfind(b"\n")
    if newline < 0:
        return {"source": relative, "offset": start, "bytes": 0}
    data = pending[: newline + 1]
    end = start + len(data)
    digest = hashlib.sha256(data).hexdigest()
    chunk = destination / "chunks" / f"{start:016x}-{end:016x}-{digest[:16]}.jsonl"
    if not chunk.exists():
        _atomic_write(chunk, data, mode=0o400)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "agent_kind": agent_kind,
        "cpu_attempt": cpu_attempt,
        "source": relative,
        "source_id": source_id,
        "offset": end,
        "last_chunk": chunk.name,
        "last_chunk_sha256": digest,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_write(cursor_path, _json_bytes(payload))
    return {"source": relative, "offset": end, "bytes": len(data), "chunk": str(chunk)}


def mirror_once(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = (
        pathlib.Path(args.durable_dir)
        / "runs"
        / args.run_id
        / "trace"
        / "raw"
        / f"cpu-attempt-{args.cpu_attempt:03d}"
    )
    rows = []
    for source, source_root in discover_sources(
        agent_kind=args.agent_kind,
        codex_home=pathlib.Path(args.codex_home),
        claude_home=pathlib.Path(args.claude_home),
        agent_log_dir=pathlib.Path(args.agent_log_dir),
    ):
        rows.append(
            mirror_source(
                source,
                source_root,
                root,
                run_id=args.run_id,
                agent_kind=args.agent_kind,
                cpu_attempt=args.cpu_attempt,
            )
        )
    return rows


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--run-id", required=True)
    out.add_argument(
        "--agent-kind",
        choices=("codex", "claude-code", "deepseek-harness"),
        required=True,
    )
    out.add_argument("--cpu-attempt", type=int, default=1)
    out.add_argument("--codex-home", default="/tmp/codex-home")
    out.add_argument("--claude-home", default="/root/.claude")
    out.add_argument("--agent-log-dir", default="/logs/agent")
    out.add_argument("--durable-dir", default="/durable")
    out.add_argument("--interval-seconds", type=float, default=5.0)
    out.add_argument("--once", action="store_true")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cpu_attempt < 1 or args.interval_seconds < 0.25:
        raise SystemExit("cpu attempt must be positive and interval at least 0.25s")
    while True:
        mirror_once(args)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
