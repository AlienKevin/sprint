#!/usr/bin/env python3
"""Run one persistent DeepSeek Harness native-goal benchmark session."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from typing import Any, Callable

from deepseek_harness import DeepSeekHarness


HARNESS_RUNTIME = "/usr/local/bin/dsh-jsonrpc-agent"
HARNESS_CONFIG = "/opt/deepseek-harness-minimal.cordis.yml"
MODEL = "deepseek/deepseek-v4-flash-vision-exp"
MAX_TOKENS = 384_000
TERMINAL_GOAL_PHASES = frozenset({"blocked", "complete"})
INFRA_FAILURE_EXIT = 70
DEFAULT_CONTINUATION_TIMEOUT_SECONDS = 600
MAX_RUNTIME_DIAGNOSTICS_CHARS = 16_000


@dataclass(slots=True)
class GoalSessionResult:
    exit_code: int
    final_response: str
    goal_status: str
    completed_turns: int
    rounds_started: int
    runner_state: str


@dataclass(slots=True)
class SignalState:
    signum: int | None = None

    def handle(self, signum: int, _frame: object) -> None:
        self.signum = signum


def append_event(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete JSONL notification to the forensic transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        handle.write("\n")
        handle.flush()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--workspace", type=Path, default=Path("/app"))
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--stop-file", type=Path, default=Path("/run/sprint-stop"))
    parser.add_argument(
        "--continuation-timeout-seconds",
        type=float,
        default=float(
            os.environ.get(
                "DSH_GOAL_CONTINUATION_TIMEOUT_SECONDS",
                str(DEFAULT_CONTINUATION_TIMEOUT_SECONDS),
            )
        ),
    )
    args = parser.parse_args()
    if args.continuation_timeout_seconds <= 0:
        parser.error("--continuation-timeout-seconds must be positive")
    if args.lifecycle is None:
        args.lifecycle = args.events.with_name("goal-lifecycle.json")
    return args


def notification_dict(notification: object) -> dict[str, Any]:
    if hasattr(notification, "model_dump"):
        raw = notification.model_dump(mode="json")
    elif isinstance(getattr(notification, "method", None), str) and isinstance(
        getattr(notification, "payload", None), dict
    ):
        raw = {
            "method": notification.method,
            "payload": notification.payload,
        }
    else:
        raise TypeError(
            "unsupported DeepSeek Harness notification schema; "
            "structured method/payload fields are required"
        )
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("method"), str)
        or not isinstance(raw.get("payload"), dict)
    ):
        raise TypeError(
            "unsupported DeepSeek Harness notification payload; "
            "structured method/payload fields are required"
        )
    return {
        "schema_version": 1,
        "method": raw["method"],
        "payload": raw["payload"],
    }


def session_event(notification: object, session_id: str) -> dict[str, Any] | None:
    raw = notification_dict(notification)
    if raw["method"] != "session.event":
        return None
    payload = raw["payload"]
    if payload.get("sessionId") != session_id:
        return None
    event = payload.get("event")
    return event if isinstance(event, dict) else None


def assistant_text(event: dict[str, Any]) -> str:
    data = event.get("data")
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def sanitized_runtime_diagnostics(harness: object) -> str:
    """Return a bounded, secret-redacted tail from the pinned SDK runtime."""
    client = getattr(harness, "client", None)
    read = getattr(client, "_runtime_diagnostics", None)
    if not callable(read):
        return ""
    try:
        diagnostics = str(read() or "")
    except Exception as exc:  # diagnostics must never hide the primary failure
        diagnostics = f"runtime diagnostics unavailable: {type(exc).__name__}: {exc}"
    for name in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            diagnostics = diagnostics.replace(secret, "[REDACTED]")
    return diagnostics[-MAX_RUNTIME_DIAGNOSTICS_CHARS:]


def run_goal_session(
    harness: object,
    *,
    session_id: str,
    objective: str,
    record: Callable[[object], None],
    stop_file: Path,
    lifecycle_path: Path,
    continuation_timeout_seconds: float,
    signals: SignalState | None = None,
) -> GoalSessionResult:
    """Own one runtime/session until its native goal or host stop is terminal.

    ``Session.run`` returns at the first idle edge. Native goal continuation is
    queued *from* that edge, so using it here races teardown against the goal
    round driver. This loop owns one session-tree subscription across every
    round and sends exactly one direct-human prompt.
    """
    signals = signals or SignalState()
    completed_turns = 0
    rounds_started = 0
    goal_status = "pending"
    final_response = ""
    session_status = "starting"
    initial_message_id: str | None = None
    initial_receipt_seen = False
    continuation_deadline: float | None = None
    delivery: queue.Queue[object | BaseException] = queue.Queue()
    subscription: object | None = None
    pump: threading.Thread | None = None

    def write_state(runner_state: str, **extra: object) -> None:
        payload: dict[str, object] = {
            "schema_version": 1,
            "session_id": session_id,
            "goal_status": goal_status,
            "completed_turns": completed_turns,
            "goal_rounds_started": rounds_started,
            "runner_state": runner_state,
            "updated_at": utc_now(),
        }
        payload.update(extra)
        atomic_json(lifecycle_path, payload)

    def finish(exit_code: int, runner_state: str, **extra: object) -> GoalSessionResult:
        write_state(runner_state, **extra)
        return GoalSessionResult(
            exit_code=exit_code,
            final_response=final_response,
            goal_status=goal_status,
            completed_turns=completed_turns,
            rounds_started=rounds_started,
            runner_state=runner_state,
        )

    def pump_notifications() -> None:
        assert subscription is not None
        while True:
            try:
                delivery.put(subscription.next())
            except BaseException as exc:  # transport closure must reach the owner
                delivery.put(exc)
                return

    write_state("starting")
    try:
        session = harness.start_session(session_id)
        client = harness.client
        subscription = client.subscribe_session_notifications(session.id)
        pump = threading.Thread(
            target=pump_notifications,
            name="deepseek-goal-notifications",
            daemon=True,
        )
        pump.start()
        initial_message_id = client.session_prompt(
            session.id,
            [{"type": "text", "text": objective}],
            notification_subscription=subscription,
        )
        write_state("running")

        while True:
            if signals.signum is not None:
                if stop_file.exists():
                    return finish(
                        0,
                        "interrupted",
                        signal=signals.signum,
                        stop_reason=stop_file.read_text(errors="replace").strip()
                        or "operator_stop",
                    )
                return finish(
                    128 + signals.signum,
                    "invalid_infrastructure",
                    signal=signals.signum,
                    failure_code="unexpected_signal",
                )

            try:
                item = delivery.get(timeout=0.1)
            except queue.Empty:
                if (
                    continuation_deadline is not None
                    and time.monotonic() >= continuation_deadline
                ):
                    return finish(
                        INFRA_FAILURE_EXIT,
                        "invalid_infrastructure",
                        failure_code="goal_continuation_timeout",
                        detail=(
                            "DeepSeek Harness stayed idle with an active native goal; "
                            "the goal-round driver did not start the next turn"
                        ),
                        continuation_timeout_seconds=continuation_timeout_seconds,
                        runtime_diagnostics=sanitized_runtime_diagnostics(harness),
                    )
                continue

            if isinstance(item, BaseException):
                if stop_file.exists():
                    # The supervisor will signal this still-live owner and write
                    # the durable STOP_ACK. Do not turn an expected host stop into
                    # an apparent natural agent exit.
                    write_state("awaiting_stop_signal", transport_error=str(item))
                    continue
                return finish(
                    INFRA_FAILURE_EXIT,
                    "invalid_infrastructure",
                    failure_code="harness_transport_closed",
                    detail=str(item),
                )

            notification = item
            record(notification)
            raw = notification_dict(notification)
            event = session_event(notification, session_id)
            if event is not None:
                event_type = event.get("type")
                data = event.get("data")
                data = data if isinstance(data, dict) else {}

                if event_type == "agent/inbox/spliced":
                    inserted = data.get("inserted")
                    if isinstance(inserted, list):
                        for message in inserted:
                            if not isinstance(message, dict):
                                continue
                            if message.get("id") == initial_message_id:
                                initial_receipt_seen = True
                            source = message.get("source")
                            if (
                                isinstance(source, dict)
                                and source.get("kind") == "goal"
                            ):
                                continuation_deadline = (
                                    time.monotonic() + continuation_timeout_seconds
                                )
                                write_state("continuation_queued")
                elif event_type == "goal/change":
                    if data.get("operation") == "clear":
                        goal_status = "cleared"
                    else:
                        goal = data.get("goal")
                        phase = goal.get("phase") if isinstance(goal, dict) else None
                        if isinstance(phase, str):
                            goal_status = phase
                        counter = data.get("roundsStarted")
                        if isinstance(counter, int):
                            rounds_started = max(rounds_started, counter)
                    write_state("running")
                elif event_type == "user/message":
                    source = data.get("source")
                    if isinstance(source, dict) and source.get("kind") == "goal":
                        round_number = source.get("round")
                        if isinstance(round_number, int):
                            rounds_started = max(rounds_started, round_number)
                elif event_type == "turn/start":
                    continuation_deadline = None
                    write_state("running")
                elif event_type == "assistant/message":
                    text = assistant_text(event)
                    if text:
                        final_response = text
                elif event_type == "turn/end":
                    turn = data.get("turn")
                    completed_turns = max(
                        completed_turns,
                        turn if isinstance(turn, int) else completed_turns + 1,
                    )

            if raw["method"] == "session.status":
                payload = raw["payload"]
                if payload.get("sessionId") == session_id:
                    status = payload.get("status")
                    if isinstance(status, str):
                        session_status = status
                    if status == "running":
                        continuation_deadline = None
                    elif status == "idle" and initial_receipt_seen:
                        if goal_status in TERMINAL_GOAL_PHASES:
                            return finish(0, "terminal")
                        if goal_status != "active":
                            return finish(
                                INFRA_FAILURE_EXIT,
                                "invalid_infrastructure",
                                failure_code="nonterminal_goal_not_runnable",
                                detail=f"idle session has goal status {goal_status!r}",
                            )
                        continuation_deadline = (
                            time.monotonic() + continuation_timeout_seconds
                        )
                        write_state(
                            "awaiting_continuation",
                            continuation_timeout_seconds=continuation_timeout_seconds,
                        )

            if session_status == "idle" and goal_status in TERMINAL_GOAL_PHASES:
                return finish(0, "terminal")
    finally:
        # Close the runtime while the subscription is still registered so the
        # SDK wakes the notification pump instead of leaking a blocked thread.
        try:
            harness.close()
        finally:
            if subscription is not None:
                subscription.close()
            if pump is not None:
                pump.join(timeout=5)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    workspace = args.workspace.resolve()
    session_root = args.session_root.resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    objective = args.prompt.strip()
    if not objective:
        raise SystemExit("DeepSeek Harness goal objective must be non-empty")
    if objective == "/goal" or objective.startswith("/goal "):
        raise SystemExit(
            "DeepSeek Harness uses native goal mode, not /goal prompt text"
        )
    os.environ["DSH_GOAL_OBJECTIVE"] = objective

    def record(notification: object) -> None:
        append_event(args.events, notification_dict(notification))

    signals = SignalState()
    signal.signal(signal.SIGINT, signals.handle)
    signal.signal(signal.SIGTERM, signals.handle)
    harness = DeepSeekHarness(
        provider="deepseek-official",
        model=MODEL,
        max_tokens=MAX_TOKENS,
        cwd=str(workspace),
        runtime_cwd=str(workspace),
        session_root=str(session_root),
        cordis=HARNESS_CONFIG,
        runtime_bin=HARNESS_RUNTIME,
        base_url=args.base_url,
        api_key=api_key,
        request_timeout_seconds=None,
        shutdown_timeout_seconds=30.0,
    )
    result = run_goal_session(
        harness,
        session_id=args.session_id,
        objective=objective,
        record=record,
        stop_file=args.stop_file,
        lifecycle_path=args.lifecycle,
        continuation_timeout_seconds=args.continuation_timeout_seconds,
        signals=signals,
    )
    if result.final_response:
        print(result.final_response, flush=True)
    if result.exit_code != 0:
        print(
            "DeepSeek Harness goal session ended before a host stop or terminal goal "
            f"(goal={result.goal_status!r}, state={result.runner_state!r})",
            file=sys.stderr,
        )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
