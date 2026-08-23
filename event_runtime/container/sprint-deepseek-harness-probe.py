#!/usr/bin/env python3
"""Offline protocol probe for the pinned DeepSeek Harness runtime image."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading

from deepseek_harness import DeepSeekHarness


MODEL = "deepseek/deepseek-v4-flash-vision-exp"
RUNTIME = "/usr/local/bin/dsh-jsonrpc-agent"
CORDIS = "/opt/deepseek-harness-minimal.cordis.yml"


class Handler(BaseHTTPRequestHandler):
    request_payload: dict[str, object] | None = None

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        type(self).request_payload = json.loads(self.rfile.read(length))
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
        body = b"".join(
            f"data: {json.dumps(event)}\n\n".encode() for event in events
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="dsh-probe-") as raw:
            root = Path(raw)
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
                result = harness.start_session("image-probe").run(
                    "Reply exactly smoke-ok."
                )
        assert result.finish_reason == "completed", result.finish_reason
        assert result.final_response == "smoke-ok", result.final_response
        request = Handler.request_payload
        assert isinstance(request, dict)
        assert request.get("model") == MODEL
        assert request.get("stream") is True
        assert request.get("max_tokens") == 384_000
        assert request.get("reasoning_effort") == "max"
        messages = request.get("messages")
        assert isinstance(messages, list) and messages
        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful software engineer assistant.",
        }
        tools = request.get("tools")
        assert isinstance(tools, list)
        names = {
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict)
        }
        assert names == {"bash", "str_replace_editor"}, names
        print("DEEPSEEK_HARNESS_PROTOCOL_OK")
        return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
