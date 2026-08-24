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
DEEPSEEK_ATIF_TRANSFORM_VERSION = 2
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


def _notification_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return one current structured DeepSeek Harness notification."""
    if (
        row.get("schema_version") == 1
        and isinstance(row.get("method"), str)
        and isinstance(row.get("payload"), dict)
    ):
        return {"method": row["method"], "payload": row["payload"]}
    return None


def _content_text(content: Any) -> str | None:
    texts: list[str] = []
    blocks = content if isinstance(content, list) else [content]
    for block in blocks:
        if isinstance(block, str):
            texts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            texts.append(text)
        nested = block.get("content")
        if isinstance(nested, (list, dict, str)):
            nested_text = _content_text(nested)
            if nested_text:
                texts.append(nested_text)
    joined = "\n".join(part for part in texts if part)
    return joined or None


def _tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value if value is not None else {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"input": value}


def _deepseek_events(chunk_paths: list[Path]) -> list[dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}
    unsequenced: list[dict[str, Any]] = []
    for path in chunk_paths:
        try:
            handle = path.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                notification = _notification_payload(row)
                if not notification or notification.get("method") != "session.event":
                    continue
                payload = notification.get("payload")
                event = payload.get("event") if isinstance(payload, dict) else None
                if not isinstance(event, dict):
                    continue
                seq = event.get("seq")
                if isinstance(seq, int):
                    events[seq] = event
                else:
                    unsequenced.append(event)
    return [events[key] for key in sorted(events)] + unsequenced


def _deepseek_atif(events: list[dict[str, Any]], *, model: str | None) -> dict[str, Any]:
    """Convert high-level DeepSeek Harness events to the ATIF subset we publish."""
    steps: list[dict[str, Any]] = []
    calls: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    assistant_steps: dict[int, dict[str, Any]] = {}

    for event in events:
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        timestamp = _iso_from_ms(event.get("time"))
        seq = event.get("seq")
        if not isinstance(seq, int):
            seq = len(steps) + 1

        if kind == "user/message":
            message = _content_text(data.get("content"))
            if message:
                steps.append(
                    {
                        "step_id": seq,
                        "timestamp": timestamp,
                        "source": "user",
                        "message": message,
                    }
                )
            continue

        if kind == "assistant/message":
            message_data = data.get("message")
            if not isinstance(message_data, dict):
                continue
            content = message_data.get("content")
            blocks = content if isinstance(content, list) else []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool-call":
                    continue
                call_id = block.get("id")
                call = {
                    "tool_call_id": call_id,
                    "function_name": str(block.get("name") or "tool"),
                    "arguments": _tool_arguments(block.get("arguments")),
                }
                tool_calls.append(call)
            source = message_data.get("source")
            source_model = source.get("model") if isinstance(source, dict) else None
            # Current DeepSeek Harness session events attach request usage to
            # the assistant/message event data, alongside ``message`` and
            # ``step``.  It is not part of the message object itself.
            usage = data.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            step: dict[str, Any] = {
                "step_id": seq,
                "timestamp": timestamp,
                "source": "agent",
                "model_name": source_model or model,
            }
            message = _content_text(
                [block for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
            )
            if message:
                step["message"] = message
            if tool_calls:
                step["tool_calls"] = tool_calls
                for call in tool_calls:
                    call_id = call.get("tool_call_id")
                    if isinstance(call_id, str):
                        calls[call_id] = (step, call)
            metrics = {
                "prompt_tokens": sum(
                    value
                    for value in (usage.get("inputTokens"), usage.get("cacheReadTokens"))
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ),
                "cached_tokens": usage.get("cacheReadTokens"),
                "completion_tokens": usage.get("outputTokens"),
            }
            step["metrics"] = {
                name: value
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            steps.append(step)
            harness_step = data.get("step")
            if isinstance(harness_step, int):
                assistant_steps[harness_step] = step
            continue

        if kind == "tool/call":
            call_id = data.get("callId")
            harness_step = data.get("step")
            step = assistant_steps.get(harness_step) if isinstance(harness_step, int) else None
            if not isinstance(call_id, str) or step is None or call_id in calls:
                continue
            call = {
                "tool_call_id": call_id,
                "function_name": str(data.get("name") or "tool"),
                "arguments": _tool_arguments(data.get("arguments")),
            }
            step.setdefault("tool_calls", []).append(call)
            calls[call_id] = (step, call)
            continue

        if kind == "tool/result":
            message_data = data.get("message")
            if not isinstance(message_data, dict):
                continue
            source = message_data.get("source")
            call_id = source.get("callId") if isinstance(source, dict) else None
            if not isinstance(call_id, str):
                continue
            target = calls.get(call_id)
            if target is None:
                continue
            step, _ = target
            result = {
                "source_call_id": call_id,
                "content": _content_text(message_data.get("content")),
            }
            step.setdefault("observation", {}).setdefault("results", []).append(result)

    return {"schema_version": "1.0", "steps": steps}


def _materialize_deepseek_trajectories(state_dir: Path) -> list[tuple[int, Path]]:
    grouped: dict[tuple[int, str], list[Path]] = {}
    for path in sorted(
        state_dir.glob(
            "durable-trace/raw/cpu-attempt-*/deepseek-harness/*/chunks/*.jsonl"
        )
    ):
        grouped.setdefault((_attempt_number(path), path.parent.parent.name), []).append(path)
    run = _read_json(state_dir / "run.json")
    materialized: list[tuple[int, Path]] = []
    for (attempt, source), chunk_paths in grouped.items():
        target = (
            state_dir
            / "trace"
            / "reconstructed"
            / f"cpu-attempt-{attempt:03d}"
            / f"deepseek-harness-{source}"
            / "trajectory.json"
        )
        source_fingerprint = _source_fingerprint(
            [(index, path) for index, path in enumerate(chunk_paths, start=1)]
        )
        # Derived trajectories must be rebuilt when the transform changes,
        # even if the immutable raw notification chunks have not changed.
        fingerprint = hashlib.sha256(
            (
                f"deepseek-atif-transform-v{DEEPSEEK_ATIF_TRANSFORM_VERSION}:"
                f"{source_fingerprint}"
            ).encode()
        ).hexdigest()
        previous = _read_json(target)
        if previous.get("deepseek_source_fingerprint") != fingerprint:
            payload = _deepseek_atif(
                _deepseek_events(chunk_paths), model=run.get("model")
            )
            payload["deepseek_source_fingerprint"] = fingerprint
            _atomic_json(target, payload)
        materialized.append((attempt, target))
    return materialized


def discover_trajectories(state_dir: Path) -> list[tuple[int, Path]]:
    """Return one reconstructed ATIF trajectory for each CPU attempt."""
    _materialize_deepseek_trajectories(state_dir)
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
    if not text:
        return False
    # Codex records bootstrap context as user-role XML blocks.  Those records
    # are implementation context, not operator messages.  The actual rollout
    # task may be delivered either as `/goal ...` or as a plain resume prompt
    # after the goal was created programmatically, so requiring the literal
    # slash command hides the task and makes the public trace begin mid-turn.
    private_bootstrap_prefixes = (
        "<app-context",
        "<codex_internal_context",
        "<environment_context",
        "<in-app-browser-context",
        "<permissions",
        "<skills_instructions",
    )
    return not text.startswith(private_bootstrap_prefixes)


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
    try:
        return int(
            dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            * 1000
        )
    except ValueError:
        return None


def _iso_from_ms(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return (
        dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


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
            for public_number, public_step in enumerate(attempt_steps, start=1):
                # Preserve attempt_step_id as the raw ATIF sequence for audit,
                # while presenting a gap-free public sequence after bootstrap
                # records have been removed.
                public_step["public_step_id"] = public_number
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
