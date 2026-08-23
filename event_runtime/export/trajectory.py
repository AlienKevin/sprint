#!/usr/bin/env python3
"""Export web-safe ATIF trajectories for the public experiment viewer.

The durable Harbor/Codex trajectory is intentionally richer than the public
resource timeline.  This exporter preserves the useful agent workflow—messages,
tool calls, observations, timing, and token/cost metrics—while omitting system
bootstrap context and failing closed if credential-like material survives the
redaction pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PUBLIC_RUN_LIMIT = 6
MAX_PUBLIC_STRING_CHARS = 200_000

_ATTEMPT_RE = re.compile(r"cpu-attempt-(\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|api[_-]?key|management[_-]?key|password|passwd|secret|"
    r"access[_-]?token|refresh[_-]?token|credential|cookie|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"(?im)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*)"
        r"\s*[:=]\s*([^\s'\"`]+)"
    ),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|token|secret|password|signature|sig)=)"
        r"[^&#\s]+"
    ),
)


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp, 0o644)
    os.replace(temp, path)


def _attempt_number(path: Path) -> int:
    match = _ATTEMPT_RE.search(str(path))
    return int(match.group(1)) if match else 1


def discover_trajectories(state_dir: Path) -> list[tuple[int, Path]]:
    """Return one reconstructed ATIF trajectory for each CPU attempt."""
    paths = sorted(
        state_dir.glob("trace/reconstructed/cpu-attempt-*/*/trajectory.json")
    )
    if not paths:
        paths = sorted(state_dir.glob("harbor-jobs/*/*/agent/trajectory.json"))
    by_attempt: dict[int, Path] = {}
    for path in paths:
        attempt = _attempt_number(path)
        current = by_attempt.get(attempt)
        if current is None or path.stat().st_mtime_ns > current.stat().st_mtime_ns:
            by_attempt[attempt] = path
    return sorted(by_attempt.items())


def _source_fingerprint(
    paths: list[tuple[int, Path]], *, metadata_paths: tuple[Path, ...] = ()
) -> str:
    digest = hashlib.sha256()
    fingerprint_paths = [(str(attempt), path) for attempt, path in paths]
    fingerprint_paths.extend((f"metadata:{path.name}", path) for path in metadata_paths)
    for label, path in fingerprint_paths:
        if not path.exists():
            digest.update(f"{label}:missing\n".encode())
            continue
        stat = path.stat()
        digest.update(f"{label}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _redact_string(value: str) -> str:
    text = _ANSI_RE.sub("", value).replace("\x00", "")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    if len(text) > MAX_PUBLIC_STRING_CHARS:
        omitted = len(text) - MAX_PUBLIC_STRING_CHARS
        text = (
            text[:MAX_PUBLIC_STRING_CHARS]
            + f"\n\n[TRUNCATED {omitted:,} CHARACTERS FOR THE PUBLIC VIEWER]"
        )
    return text


def redact(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact credentials and non-displayable image payloads."""
    if key and _SENSITIVE_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        kind = str(value.get("type") or "").lower()
        if kind in {"image", "input_image", "image_url"}:
            return {"type": kind, "content": "[IMAGE OMITTED FROM PUBLIC VIEW]"}
        return {str(name): redact(item, key=str(name)) for name, item in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_string(str(value))


def _public_user_message(message: Any) -> bool:
    if not isinstance(message, str):
        return False
    text = message.lstrip()
    return text.startswith("/goal ") or text.startswith("/goal\n")


def _call_id(run_id: str, raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    return hashlib.sha256(f"{run_id}\0{raw}".encode()).hexdigest()[:14]


def _public_step(run_id: str, attempt: int, step: dict[str, Any]) -> dict[str, Any] | None:
    source = str(step.get("source") or "agent")
    message = step.get("message")
    if source == "system" or (source == "user" and not _public_user_message(message)):
        return None

    tool_calls: list[dict[str, Any]] = []
    for call in step.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool_calls.append(
            {
                "tool_call_id": _call_id(run_id, call.get("tool_call_id")),
                "function_name": _redact_string(
                    str(call.get("function_name") or "tool")
                ),
                "arguments": redact(call.get("arguments") or {}),
            }
        )

    results: list[dict[str, Any]] = []
    observation = step.get("observation")
    if isinstance(observation, dict):
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                continue
            results.append(
                {
                    "source_call_id": _call_id(run_id, result.get("source_call_id")),
                    "content": redact(result.get("content")),
                }
            )

    metrics = step.get("metrics") if isinstance(step.get("metrics"), dict) else {}
    safe_metrics = {
        name: metrics.get(name)
        for name in (
            "cost_usd",
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
        )
        if isinstance(metrics.get(name), (int, float))
    }
    step_number = step.get("step_id")
    public: dict[str, Any] = {
        "step_id": f"a{attempt}-s{step_number}",
        "attempt": attempt,
        "attempt_step_id": step_number,
        "timestamp": step.get("timestamp"),
        "source": source,
    }
    if isinstance(step.get("model_name"), str):
        public["model_name"] = _redact_string(step["model_name"])
    if isinstance(message, str) and message:
        public["message"] = _redact_string(message)
    if tool_calls:
        public["tool_calls"] = tool_calls
    if results:
        public["observation"] = {"results": results}
    if safe_metrics:
        public["metrics"] = safe_metrics
    if len(public) <= 5:
        return None
    return public


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _timestamp_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None


def _iso_from_ms(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return (
        dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    try:
        return int(dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _assert_public_safe(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    checks = (
        re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    )
    if any(pattern.search(serialized) for pattern in checks):
        raise ValueError("credential-like material survived trajectory redaction")


def build_public_trajectory(state_dir: Path, *, web_dir: Path) -> dict[str, Any] | None:
    """Build one public multi-attempt trajectory and update its six-run index."""
    run = _read_json(state_dir / "run.json")
    run_id = str(run.get("run_id") or state_dir.name)
    sources = discover_trajectories(state_dir)
    if not sources:
        return None
    unified_path = state_dir / "telemetry" / "unified-timeline.json"
    unified = _read_json(unified_path)
    fingerprint = _source_fingerprint(
        sources,
        metadata_paths=(state_dir / "run.json", unified_path),
    )
    public_path = web_dir / "data" / "trajectories" / f"{run_id}.json"
    previous = _read_json(public_path)
    if previous.get("source_fingerprint") == fingerprint:
        payload = previous
    else:
        steps: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        omitted = 0
        cpu_intervals = {
            int(item["cpu_attempt"]): item
            for item in (
                (unified.get("coverage") or {})
                .get("cpu_metric_coverage", {})
                .get("attempts", [])
            )
            if isinstance(item, dict)
            and isinstance(item.get("cpu_attempt"), int)
        }
        for attempt, path in sources:
            trajectory = _read_json(path)
            attempt_steps: list[dict[str, Any]] = []
            for step in trajectory.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                public_step = _public_step(run_id, attempt, step)
                if public_step is None:
                    omitted += 1
                    continue
                attempt_steps.append(public_step)
            steps.extend(attempt_steps)
            timestamps = [
                item.get("timestamp")
                for item in attempt_steps
                if isinstance(item.get("timestamp"), str)
            ]
            interval = cpu_intervals.get(attempt) or {}
            attempts.append(
                {
                    "attempt": attempt,
                    "step_count": len(attempt_steps),
                    "started_at": _iso_from_ms(interval.get("start_epoch_ms"))
                    or (min(timestamps) if timestamps else None),
                    "ended_at": _iso_from_ms(interval.get("end_epoch_ms"))
                    or (max(timestamps) if timestamps else None),
                    "tool_call_count": sum(
                        len(item.get("tool_calls") or []) for item in attempt_steps
                    ),
                }
            )
        steps.sort(key=lambda item: (item.get("timestamp") or "", item["step_id"]))
        start_ms = _timestamp_ms(steps[0].get("timestamp")) if steps else None
        end_ms = _timestamp_ms(steps[-1].get("timestamp")) if steps else None
        usage = unified.get("usage_summary") or {}
        comparison = unified.get("comparison_summary") or {}
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "source_fingerprint": fingerprint,
            "run": {
                "run_id": run_id,
                "model": run.get("model"),
                "agent_kind": run.get("agent_kind"),
                "reasoning_effort": run.get("reasoning_effort"),
                "created_at": run.get("created_at"),
            },
            "summary": {
                "attempt_count": len(attempts),
                "step_count": len(steps),
                "message_count": sum(bool(item.get("message")) for item in steps),
                "tool_call_count": sum(len(item.get("tool_calls") or []) for item in steps),
                "duration_ms": end_ms - start_ms
                if start_ms is not None and end_ms is not None
                else None,
                "final_agent_cost_usd": comparison.get("final_agent_total_cost_usd"),
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                "omitted_bootstrap_steps": omitted,
            },
            "attempts": attempts,
            "steps": steps,
            "privacy": {
                "system_and_bootstrap_messages_omitted": True,
                "credential_like_values_redacted": True,
                "image_payloads_omitted": True,
            },
        }
        _assert_public_safe(payload)
        _atomic_json(public_path, payload)

    index_path = web_dir / "data" / "trajectories" / "index.json"
    index = _read_json(index_path) or {"schema_version": SCHEMA_VERSION, "runs": []}
    entries = {
        str(item.get("run_id")): item
        for item in index.get("runs") or []
        if isinstance(item, dict) and item.get("run_id")
    }
    entries[run_id] = {
        "run_id": run_id,
        "model": run.get("model"),
        "created_at": run.get("created_at"),
        "generated_at": payload.get("generated_at"),
        "path": f"/data/trajectories/{run_id}.json",
        "summary": payload.get("summary") or {},
        "attempts": payload.get("attempts") or [],
    }
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": payload.get("generated_at"),
        "runs": sorted(
            entries.values(),
            key=lambda item: (item.get("created_at") or "", item["run_id"]),
            reverse=True,
        )[:PUBLIC_RUN_LIMIT],
    }
    _atomic_json(index_path, index_payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=pathlib.Path, required=True)
    parser.add_argument("--web-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    payload = build_public_trajectory(args.state_dir, web_dir=args.web_dir)
    if payload is None:
        raise SystemExit("no ATIF trajectory found")
    print(
        json.dumps(
            {
                "run_id": payload["run"]["run_id"],
                "steps": payload["summary"]["step_count"],
                "attempts": payload["summary"]["attempt_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
