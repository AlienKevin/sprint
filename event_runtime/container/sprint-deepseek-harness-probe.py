#!/usr/bin/env python3
"""Offline protocol probe for the pinned DeepSeek Harness runtime image."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import tempfile
import threading

from deepseek_harness import DeepSeekHarness


MODEL = "deepseek/deepseek-v4-flash-vision-exp"
RUNTIME = os.environ.get("DSH_PROBE_RUNTIME", "/usr/local/bin/dsh-jsonrpc-agent")
CORDIS = os.environ.get("DSH_PROBE_CORDIS", "/opt/deepseek-harness-minimal.cordis.yml")
FAILED_PARTIAL_TEXT = "partial-stream-content-must-not-surface"


class Handler(BaseHTTPRequestHandler):
    request_payloads: list[dict[str, object]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_events(self, events: list[dict[str, object]]) -> None:
        body = b"".join(
            f"data: {json.dumps(event)}\n\n".encode() for event in events
        ) + b"data: [DONE]\n\n"
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
        if request_number != 2:
            self.send_error(500, f"unexpected request {request_number}")
            return
        events = [
            {
                "id": "probe-generation",
                "object": "chat.completion.chunk",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "smoke-ok"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "probe-generation",
                "object": "chat.completion.chunk",
                "model": MODEL,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "total_tokens": 102,
                },
            },
        ]
        self._send_events(events)


def main() -> int:
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
                with DeepSeekHarness(
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
                ) as harness:
                    result = harness.start_session("image-probe").run(objective)
            finally:
                if previous_objective is None:
                    os.environ.pop("DSH_GOAL_OBJECTIVE", None)
                else:
                    os.environ["DSH_GOAL_OBJECTIVE"] = previous_objective
                if previous_rounds is None:
                    os.environ.pop("DSH_GOAL_MAX_ROUNDS", None)
                else:
                    os.environ["DSH_GOAL_MAX_ROUNDS"] = previous_rounds
        if result.finish_reason != "completed":
            raise AssertionError(
                json.dumps(
                    {
                        "finish_reason": result.finish_reason,
                        "final_response": result.final_response,
                        "events": result.events[-20:],
                    },
                    indent=2,
                    default=str,
                )
            )
        assert result.final_response == "smoke-ok", result.final_response
        # The first provider attempt closes without [DONE]. The finite retry
        # executor opens a retry turn over the same surface history; it is not a
        # second goal round and it does not require a CPU-agent relaunch.
        assert len(Handler.request_payloads) == 2, len(Handler.request_payloads)
        first_request, retry_request = Handler.request_payloads
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
        events = result.events
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
        assert not any(
            name in json.dumps(Handler.request_payloads)
            for name in ("create_goal", "get_goal", "update_goal")
        )
        print("DEEPSEEK_HARNESS_PROTOCOL_OK")
        return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
