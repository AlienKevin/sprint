#!/usr/bin/env python3
"""Offline protocol probe for the pinned DeepSeek Harness runtime image."""

from __future__ import annotations

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

from deepseek_harness import DeepSeekHarness


MODEL = "deepseek/deepseek-v4-flash-vision-exp"
RUNTIME = os.environ.get("DSH_PROBE_RUNTIME", "/usr/local/bin/dsh-jsonrpc-agent")
CORDIS = os.environ.get("DSH_PROBE_CORDIS", "/opt/deepseek-harness-minimal.cordis.yml")
RUNNER = Path(
    os.environ.get(
        "DSH_PROBE_RUNNER",
        "/opt/event_runtime/container/sprint-deepseek-harness-runner.py",
    )
)
FAILED_PARTIAL_TEXT = "partial-stream-content-must-not-surface"
STRESS_CHUNKS = int(os.environ.get("DSH_PROBE_STRESS_CHUNKS", "20000"))


class Handler(BaseHTTPRequestHandler):
    request_payloads: list[dict[str, object]] = []
    request_times: list[float] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_events(self, events: list[dict[str, object]]) -> None:
        body = (
            b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        type(self).request_payloads.append(json.loads(self.rfile.read(length)))
        type(self).request_times.append(time.monotonic())
        request_number = len(type(self).request_payloads)
        if request_number == 1:
            # Simulate the observed provider failure: valid SSE content arrives,
            # but the stream closes before the required [DONE] sentinel. The
            # harness must retry the same durable request surface and must not
            # carry this failed partial output into the retry.
            event = {
                "id": "probe-failed-generation",
                "object": "chat.completion.chunk",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": FAILED_PARTIAL_TEXT,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            body = f"data: {json.dumps(event)}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if request_number not in {2, 3}:
            self.send_error(500, f"unexpected request {request_number}")
            return
        response_text = "smoke-ok" if request_number == 2 else "round-ok"
        events = []
        if request_number == 2:
            # Exercise continuation after a substantial streamed first turn,
            # not only the tiny happy path. Production sessions contain many
            # more events, but this is large enough to cover write-behind,
            # checkpoint, and queueing behavior in every warmed image.
            events.extend(
                {
                    "id": "probe-generation",
                    "object": "chat.completion.chunk",
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "x"},
                            "finish_reason": None,
                        }
                    ],
                }
                for _ in range(STRESS_CHUNKS)
            )
        events.extend(
            [
                {
                    "id": "probe-generation",
                    "object": "chat.completion.chunk",
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": response_text},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "probe-generation",
                    "object": "chat.completion.chunk",
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 2,
                        "total_tokens": 102,
                    },
                },
            ]
        )
        self._send_events(events)


def main() -> int:
    Handler.request_payloads = []
    Handler.request_times = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="dsh-probe-") as raw:
            root = Path(raw)
            objective = "Reply exactly smoke-ok."
            previous_objective = os.environ.get("DSH_GOAL_OBJECTIVE")
            previous_rounds = os.environ.get("DSH_GOAL_MAX_ROUNDS")
            os.environ["DSH_GOAL_OBJECTIVE"] = objective
            # One automatic goal round proves that the native driver is armed
            # without making this offline image probe loop 256 times.
            os.environ["DSH_GOAL_MAX_ROUNDS"] = "1"
            try:
                spec = importlib.util.spec_from_file_location(
                    "sprint_deepseek_harness_runner_probe", RUNNER
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"could not load DeepSeek runner from {RUNNER}")
                runner = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = runner
                spec.loader.exec_module(runner)
                harness = DeepSeekHarness(
                    provider="deepseek-official",
                    model=MODEL,
                    max_tokens=384_000,
                    cwd=str(root),
                    runtime_cwd=str(root),
                    session_root=str(root / "sessions"),
                    cordis=CORDIS,
                    runtime_bin=RUNTIME,
                    base_url=f"http://127.0.0.1:{server.server_port}/api/v1",
                    api_key="offline-probe-key",
                    request_timeout_seconds=30.0,
                    shutdown_timeout_seconds=10.0,
                )
                events: list[dict[str, object]] = []

                def record(notification: object) -> None:
                    raw_notification = runner.notification_dict(notification)
                    if raw_notification["method"] != "session.event":
                        return
                    payload = raw_notification["payload"]
                    event = payload.get("event")
                    if isinstance(event, dict):
                        events.append(event)

                result = runner.run_goal_session(
                    harness,
                    session_id="image-probe",
                    objective=objective,
                    record=record,
                    stop_file=root / "stop",
                    lifecycle_path=root / "goal-lifecycle.json",
                    continuation_timeout_seconds=120,
                )
            finally:
                if previous_objective is None:
                    os.environ.pop("DSH_GOAL_OBJECTIVE", None)
                else:
                    os.environ["DSH_GOAL_OBJECTIVE"] = previous_objective
                if previous_rounds is None:
                    os.environ.pop("DSH_GOAL_MAX_ROUNDS", None)
                else:
                    os.environ["DSH_GOAL_MAX_ROUNDS"] = previous_rounds
        if result.exit_code != 0 or result.runner_state != "terminal":
            raise AssertionError(
                json.dumps(
                    {
                        "exit_code": result.exit_code,
                        "runner_state": result.runner_state,
                        "goal_status": result.goal_status,
                        "final_response": result.final_response,
                        "events": events[-20:],
                    },
                    indent=2,
                    default=str,
                )
            )
        assert result.final_response == "round-ok", result.final_response
        assert result.goal_status == "blocked", result.goal_status
        assert result.completed_turns == 2, result.completed_turns
        assert result.rounds_started == 1, result.rounds_started
        # The first provider attempt closes without [DONE]. The finite retry
        # executor opens a retry turn over the same surface history; it is not a
        # second goal round and it does not require a CPU-agent relaunch.
        assert len(Handler.request_payloads) == 3, len(Handler.request_payloads)
        assert len(Handler.request_times) == 3, len(Handler.request_times)
        continuation_seconds = Handler.request_times[2] - Handler.request_times[1]
        assert continuation_seconds < 120, continuation_seconds
        first_request, retry_request, goal_round_request = Handler.request_payloads
        assert first_request == retry_request
        assert FAILED_PARTIAL_TEXT not in json.dumps(retry_request)
        request = first_request
        assert request.get("model") == MODEL
        assert request.get("stream") is True
        assert request.get("max_tokens") == 384_000
        assert request.get("reasoning_effort") == "max"
        messages = request.get("messages")
        assert isinstance(messages, list) and messages
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith(
            "You are a helpful software engineer assistant."
        )
        assert "Use goal tools" not in messages[0]["content"]
        assert "/goal" not in json.dumps(messages)
        tools = request.get("tools")
        assert isinstance(tools, list)
        names = {
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict)
        }
        assert names == {
            "bash",
            "str_replace_editor",
        }, names
        retry_events = [event for event in events if event.get("type") == "llm/retry"]
        retry_started_events = [
            event for event in events if event.get("type") == "llm/retry-started"
        ]
        assert len(retry_events) == 1, retry_events
        assert len(retry_started_events) == 1, retry_started_events
        retry = retry_events[0]["data"]
        assert retry["mode"] == "normal"
        assert retry["retry"] == 1
        assert retry["maxRetries"] == 5
        assert retry["failure"]["code"] == "STREAM_CLOSED"
        surface_messages = [
            event
            for event in events
            if event.get("type") in {"assistant/message", "user/message"}
        ]
        assert FAILED_PARTIAL_TEXT not in json.dumps(surface_messages)
        created_at = next(
            index
            for index, event in enumerate(events)
            if event.get("type") == "goal/change"
            and event.get("data", {}).get("operation") == "create"
        )
        first_assistant_at = next(
            index
            for index, event in enumerate(events)
            if event.get("type") == "assistant/message"
        )
        assert created_at < first_assistant_at
        goal = events[created_at]["data"]["goal"]
        assert goal["objective"] == objective
        assert goal["maxGoalRounds"] == 1
        goal_messages = [
            event
            for event in events
            if event.get("type") == "user/message"
            and event.get("data", {}).get("source", {}).get("kind") == "goal"
        ]
        assert len(goal_messages) == 1, goal_messages
        assert goal_messages[0]["data"]["source"]["round"] == 1
        assert (
            len([event for event in events if event.get("type") == "turn/start"]) == 2
        )
        assert "<goal_round>" in json.dumps(goal_round_request)
        assert objective in json.dumps(goal_round_request)
        assert not any(
            name in json.dumps(Handler.request_payloads)
            for name in ("create_goal", "get_goal", "update_goal")
        )
        print(
            "DEEPSEEK_HARNESS_PROTOCOL_OK "
            f"stress_chunks={STRESS_CHUNKS} "
            f"continuation_seconds={continuation_seconds:.3f}"
        )
        return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
