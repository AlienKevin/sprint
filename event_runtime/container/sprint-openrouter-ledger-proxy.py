#!/usr/bin/env python3
"""Transparent OpenRouter proxy with a durable per-request billing ledger."""

from __future__ import annotations

import argparse
import datetime as dt
from http import HTTPStatus
import http.client
import http.server
import json
import math
import os
from pathlib import Path
import secrets
import ssl
import threading
from typing import Any
from urllib.parse import urlsplit


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def usage_from_event(event: object) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(event, dict):
        return None, {}
    response = event.get("response")
    if not isinstance(response, dict):
        response = event
    usage = response.get("usage")
    return (usage if isinstance(usage, dict) else None), response


class LedgerProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SprintOpenRouterLedger/1"

    def log_message(self, _format: str, *_args: object) -> None:
        # Request paths are intentionally not copied into the agent transcript.
        return

    @property
    def ledger_server(self) -> "LedgerProxyServer":
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            body = b'{"status":"ok"}\n'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward(record_usage=False)

    def do_POST(self) -> None:  # noqa: N802
        self._forward(record_usage=self.path.split("?", 1)[0].endswith("/responses"))

    def _request_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        length = int(raw_length)
        if length < 0 or length > 64 * 1024 * 1024:
            raise ValueError("invalid request content length")
        return self.rfile.read(length)

    def _forward_headers(self, body: bytes) -> dict[str, str]:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP | {"host", "content-length"}
        }
        headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        return headers

    def _forward(self, *, record_usage: bool) -> None:
        if record_usage:
            # Codex is normally serial, but enforce that invariant here so two
            # simultaneous calls cannot both pass the budget gate.
            with self.ledger_server.billing_lock:
                self._forward_once(record_usage=True)
            return
        self._forward_once(record_usage=False)

    def _forward_once(self, *, record_usage: bool) -> None:
        if record_usage:
            try:
                allowed, snapshot = self.ledger_server.budget_snapshot()
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                allowed, snapshot = False, {"reason": "budget_telemetry_unavailable"}
            if not allowed:
                self.ledger_server.write_stop(snapshot)
                body = b'{"error":{"message":"agent cost budget exhausted"}}\n'
                self.send_response(HTTPStatus.PAYMENT_REQUIRED)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
                return
        try:
            body = self._request_body()
        except (TypeError, ValueError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        request_id = secrets.token_hex(16)
        record_path = self.ledger_server.requests_dir / f"{request_id}.json"
        requested_model = None
        stream = False
        if record_usage:
            try:
                request_payload = json.loads(body)
            except json.JSONDecodeError:
                request_payload = {}
            if isinstance(request_payload, dict):
                requested_model = request_payload.get("model")
                stream = bool(request_payload.get("stream"))
            record = {
                "schema_version": 1,
                "ledger_request_id": request_id,
                "run_id": self.ledger_server.run_id,
                "cpu_attempt": self.ledger_server.cpu_attempt,
                "requested_at": utc_now(),
                "requested_model": requested_model,
                "stream": stream,
                "state": "in_flight",
                "generation_id": None,
                "provider_reported_cost_usd": None,
            }
            atomic_json(record_path, record)
        else:
            record = {}

        connection = http.client.HTTPSConnection(
            self.ledger_server.upstream_host,
            self.ledger_server.upstream_port,
            timeout=600,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=self._forward_headers(body),
            )
            upstream = connection.getresponse()
            generation_id = upstream.getheader("X-Generation-Id")
            if record_usage:
                record.update(
                    {
                        "generation_id": generation_id,
                        "upstream_http_status": upstream.status,
                        "response_started_at": utc_now(),
                    }
                )
                atomic_json(record_path, record)

            self.send_response(upstream.status, upstream.reason)
            for name, value in upstream.getheaders():
                if name.lower() in HOP_BY_HOP | {"content-length"}:
                    continue
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            content_type = upstream.getheader("Content-Type") or ""
            line_buffer = bytearray()
            response_buffer = bytearray()
            terminal_usage: dict[str, Any] | None = None
            terminal_response: dict[str, Any] = {}
            downstream_open = True
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                if downstream_open:
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        downstream_open = False
                if not record_usage:
                    continue
                if "text/event-stream" in content_type:
                    line_buffer.extend(chunk)
                    while b"\n" in line_buffer:
                        raw_line, _, remainder = line_buffer.partition(b"\n")
                        line_buffer = bytearray(remainder)
                        line = raw_line.rstrip(b"\r")
                        if not line.startswith(b"data: ") or line == b"data: [DONE]":
                            continue
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        usage, response = usage_from_event(event)
                        if usage is not None:
                            terminal_usage, terminal_response = usage, response
                else:
                    response_buffer.extend(chunk)
            if record_usage and "text/event-stream" not in content_type:
                try:
                    event = json.loads(response_buffer)
                except json.JSONDecodeError:
                    event = None
                terminal_usage, terminal_response = usage_from_event(event)

            if record_usage:
                cost = (
                    terminal_usage.get("cost")
                    if isinstance(terminal_usage, dict)
                    else None
                )
                valid_cost = (
                    isinstance(cost, (int, float))
                    and not isinstance(cost, bool)
                    and float(cost) >= 0
                )
                if valid_cost:
                    record.update(
                        {
                            "state": "complete",
                            "completed_at": utc_now(),
                            "generation_id": generation_id
                            or terminal_response.get("id"),
                            "provider_reported_cost_usd": float(cost),
                            "usage": terminal_usage,
                            "response_id": terminal_response.get("id"),
                            "response_model": terminal_response.get("model"),
                            "response_status": terminal_response.get("status"),
                        }
                    )
                elif upstream.status >= 400 and not generation_id:
                    record.update(
                        {
                            "state": "rejected_not_billed",
                            "completed_at": utc_now(),
                            "provider_reported_cost_usd": 0.0,
                        }
                    )
                else:
                    record.update(
                        {
                            "state": "cost_recovery_required",
                            "response_ended_at": utc_now(),
                        }
                    )
                atomic_json(record_path, record)
                if valid_cost:
                    try:
                        _allowed, snapshot = self.ledger_server.budget_snapshot()
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        _allowed, snapshot = (
                            False,
                            {
                                "schema_version": 2,
                                "run_id": self.ledger_server.run_id,
                                "reason": "budget_telemetry_unavailable",
                                "status": "fail_closed",
                            },
                        )
                    if not _allowed:
                        self.ledger_server.write_stop(snapshot)
        except Exception as exc:  # noqa: BLE001
            if record_usage:
                record.update(
                    {
                        "state": "cost_recovery_required",
                        "response_ended_at": utc_now(),
                        "proxy_error_type": type(exc).__name__,
                    }
                )
                atomic_json(record_path, record)
            if not self.wfile.closed:
                try:
                    self.send_error(HTTPStatus.BAD_GATEWAY, "upstream request failed")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
        finally:
            connection.close()


class LedgerProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        upstream: str,
        ledger_root: Path,
        run_id: str,
        cpu_attempt: int,
        runtime_dir: Path = Path("/run"),
    ) -> None:
        parsed = urlsplit(upstream)
        if parsed.scheme != "https" or parsed.hostname != "openrouter.ai":
            raise ValueError("upstream must be https://openrouter.ai")
        if parsed.path.rstrip("/") != "/api/v1" or parsed.query or parsed.fragment:
            raise ValueError("upstream must be the OpenRouter /api/v1 root")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 443
        self.run_id = run_id
        self.cpu_attempt = cpu_attempt
        self.run_root = ledger_root.parent
        self.requests_dir = ledger_root / "requests"
        self.runtime_dir = runtime_dir
        self.billing_lock = threading.Lock()
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(address, LedgerProxyHandler)

    def budget_snapshot(self) -> tuple[bool, dict[str, Any]]:
        """Return whether another paid request may start under the run cap."""
        run = json.loads((self.run_root / "state/run.json").read_text())
        if run.get("run_id") != self.run_id:
            raise ValueError("run identity mismatch")
        budget = float(run["agent_cost_budget_usd"])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("invalid agent cost budget")
        api_cost = 0.0
        for path in sorted(self.requests_dir.glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("run_id") != self.run_id:
                raise ValueError("ledger identity mismatch")
            cost = record.get("provider_reported_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                value = float(cost)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid provider-reported cost")
                api_cost += value
            elif record.get("state") not in {"rejected_not_billed"}:
                # Never admit another paid request while an earlier charge is
                # unknown. The watchdog can recover it by generation ID.
                return False, {
                    "schema_version": 2,
                    "run_id": self.run_id,
                    "reason": "budget_telemetry_unavailable",
                    "status": "fail_closed",
                }
        modal_cost = 0.0
        watchdog_path = self.run_root / "budget/watchdog.json"
        if watchdog_path.is_file():
            watchdog = json.loads(watchdog_path.read_text())
            if watchdog.get("run_id") != self.run_id:
                raise ValueError("watchdog identity mismatch")
            if watchdog.get("status") == "fail_closed":
                return False, {
                    "schema_version": 2,
                    "run_id": self.run_id,
                    "reason": "budget_telemetry_unavailable",
                    "status": "fail_closed",
                }
            if watchdog.get("schema_version") != 2:
                raise ValueError("watchdog schema mismatch")
            components = watchdog.get("components") or {}
            for name in ("cpu_agent", "training_sandboxes"):
                value = float((components.get(name) or {}).get("cost_usd") or 0)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid Modal cost")
                modal_cost += value
        total = api_cost + modal_cost
        payload = {
            "schema_version": 2,
            "run_id": self.run_id,
            "reason": "agent_cost_budget_exhausted",
            "status": "stop_requested" if total >= budget else "within_budget",
            "budget_usd": budget,
            "total_usd": total,
            "component_totals_usd": {
                "model_api_usd": api_cost,
                "modal_live_estimate_usd": modal_cost,
            },
        }
        return total < budget, payload

    def write_stop(self, payload: dict[str, Any]) -> None:
        marker = self.run_root / "BUDGET_STOP_REQUESTED.json"
        if not marker.exists():
            atomic_json(marker, payload)
        stop = self.runtime_dir / "sprint-stop"
        stop.parent.mkdir(parents=True, exist_ok=True)
        temporary = stop.with_name(f".{stop.name}.{os.getpid()}.tmp")
        temporary.write_text(
            str(payload.get("reason") or "agent_cost_budget_exhausted") + "\n"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--upstream", default="https://openrouter.ai/api/v1")
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu-attempt", type=int, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("/run"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = LedgerProxyServer(
        (args.listen, args.port),
        upstream=args.upstream,
        ledger_root=args.ledger_root,
        run_id=args.run_id,
        cpu_attempt=args.cpu_attempt,
        runtime_dir=args.runtime_dir,
    )
    server.serve_forever(poll_interval=0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
