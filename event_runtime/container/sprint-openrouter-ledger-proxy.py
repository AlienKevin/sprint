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
import sys
import threading
from typing import Any
from urllib.parse import urlsplit


sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprint_openrouter_pricing import (  # noqa: E402
    BENCHMARK_COST_BASIS,
    PROVIDER_COST_BASIS,
    OpenRouterPricingError,
    benchmark_cost_usd,
    capture_endpoint_discount_snapshot,
    undiscounted_cost_usd,
)


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


def valid_ledger_request_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


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
    """Extract usage from either Responses or Chat Completions payloads."""
    if not isinstance(event, dict):
        return None, {}
    response = event.get("response")
    if not isinstance(response, dict):
        response = event
    usage = response.get("usage")
    return (usage if isinstance(usage, dict) else None), response


def seal_goal_tool_schema(payload: dict[str, Any]) -> None:
    """Remove model-controlled token budgets from Codex's goal tool.

    ``/goal`` asks the model to create the persistent goal through a local
    Codex tool.  The optional ``token_budget`` argument is an operator control,
    not part of the benchmark task budget.  Exposing it lets a model
    accidentally terminate its own run after a handful of tokens.  Keep the
    goal objective model-authored while making the unbudgeted form the only
    callable schema presented to every routed model.
    """

    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        containers = [tool]
        if isinstance(function, dict):
            containers.append(function)
        for container in containers:
            if container.get("name") != "create_goal":
                continue
            parameters = container.get("parameters")
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties")
            if isinstance(properties, dict):
                properties.pop("token_budget", None)
            required = parameters.get("required")
            if isinstance(required, list):
                parameters["required"] = [
                    name for name in required if name != "token_budget"
                ]


def pin_provider_route(
    body: bytes,
    *,
    provider_endpoint: str,
    quantization: str | None,
    request_contract: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Replace any caller routing preference with the sealed eval route."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("request body is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body is not an object")
    provider: dict[str, Any] = {
        "only": [provider_endpoint],
        "order": [provider_endpoint],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    if quantization:
        provider["quantizations"] = [quantization]
    payload["provider"] = provider
    if request_contract:
        payload.update(request_contract)
        if request_contract.get("stream") is True:
            # Chat Completions emits usage only on its final stream event when
            # include_usage is enabled.  The proxy owns this bit because a
            # missing usage event would make a paid request unaccountable.
            stream_options = payload.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
            stream_options["include_usage"] = True
            payload["stream_options"] = stream_options
    # Codex emits this field even when the catalog disables parallel tool
    # calls. Some strict OpenRouter endpoints reject the parameter itself,
    # despite its false value. Omitting false is behaviorally equivalent and
    # keeps this compatibility rule generic across routed models.
    if payload.get("parallel_tool_calls") is False:
        payload.pop("parallel_tool_calls")
    seal_goal_tool_schema(payload)
    return json.dumps(payload, separators=(",", ":")).encode(), payload


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
        path = self.path.split("?", 1)[0].rstrip("/")
        inference_path = (
            "responses"
            if path in {"/responses", "/api/v1/responses"}
            else "chat_completions"
            if path in {"/chat/completions", "/api/v1/chat/completions"}
            else None
        )
        if self.ledger_server.allowed_inference_path is not None and (
            inference_path != self.ledger_server.allowed_inference_path
        ):
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "inference path is sealed")
            return
        record_usage = inference_path is not None
        self._forward(record_usage=record_usage)

    def _request_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        length = int(raw_length)
        if length < 0 or length > 64 * 1024 * 1024:
            raise ValueError("invalid request content length")
        return self.rfile.read(length)

    def _forward_headers(self, body: bytes, *, router_metadata: bool) -> dict[str, str]:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower()
            not in HOP_BY_HOP | {"host", "content-length", "authorization"}
        }
        headers["Authorization"] = self.ledger_server.upstream_authorization or str(
            self.headers.get("Authorization") or ""
        )
        headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        if router_metadata:
            headers["X-OpenRouter-Metadata"] = "enabled"
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
        if record_usage and self.ledger_server.provider_endpoint:
            try:
                body, request_payload = pin_provider_route(
                    body,
                    provider_endpoint=self.ledger_server.provider_endpoint,
                    quantization=self.ledger_server.quantization,
                    request_contract=self.ledger_server.request_contract,
                )
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
        request_id = secrets.token_hex(16)
        record_path = self.ledger_server.requests_dir / f"{request_id}.json"
        requested_model = None
        stream = False
        promotion_snapshot: dict[str, Any] | None = None
        if record_usage:
            try:
                request_payload = json.loads(body)
            except json.JSONDecodeError:
                request_payload = {}
            if isinstance(request_payload, dict):
                requested_model = request_payload.get("model")
                stream = bool(request_payload.get("stream"))
            try:
                promotion_snapshot = capture_endpoint_discount_snapshot(
                    canonical_model=self.ledger_server.canonical_model,
                    requested_model=requested_model,
                    request_provider=(
                        request_payload.get("provider")
                        if isinstance(request_payload, dict)
                        else None
                    ),
                    authorization=self.ledger_server.upstream_authorization
                    or self.headers.get("Authorization"),
                )
                if requested_model != self.ledger_server.canonical_model:
                    raise OpenRouterPricingError(
                        "sealed request model does not match run contract"
                    )
                if (
                    promotion_snapshot.get("model")
                    != self.ledger_server.canonical_model
                    or promotion_snapshot.get("provider_tag")
                    != self.ledger_server.provider_endpoint
                ):
                    raise OpenRouterPricingError(
                        "sealed request route does not match run contract"
                    )
            except OpenRouterPricingError:
                self.ledger_server.write_stop(
                    {
                        "schema_version": 2,
                        "run_id": self.ledger_server.run_id,
                        "reason": "budget_telemetry_unavailable",
                        "status": "fail_closed",
                    }
                )
                body = b'{"error":{"message":"live list-price metadata unavailable"}}\n'
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
                return
            record = {
                "schema_version": 3,
                "ledger_request_id": request_id,
                "run_id": self.ledger_server.run_id,
                "cpu_attempt": self.ledger_server.cpu_attempt,
                "requested_at": utc_now(),
                "requested_model": requested_model,
                "stream": stream,
                "api_path": self.path.split("?", 1)[0],
                "request_contract": self.ledger_server.request_contract or None,
                "state": "in_flight",
                "generation_id": None,
                "provider_reported_cost_usd": None,
                "undiscounted_cost_usd": None,
                "benchmark_cost_usd": None,
                "promotion_snapshot": promotion_snapshot,
            }
            self.ledger_server.begin_request(request_id)
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
                headers=self._forward_headers(body, router_metadata=record_usage),
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
                    and math.isfinite(float(cost))
                    and float(cost) >= 0
                )
                if valid_cost:
                    response_model = terminal_response.get("model")
                    identity_verified = response_model in {
                        None,
                        self.ledger_server.canonical_model,
                        self.ledger_server.resolved_model,
                    }
                    list_cost = undiscounted_cost_usd(cost, promotion_snapshot)
                    benchmark_cost = benchmark_cost_usd(
                        cost, promotion_snapshot, terminal_usage
                    )
                    record.update(
                        {
                            "state": (
                                "complete"
                                if identity_verified
                                else "route_identity_mismatch"
                            ),
                            "completed_at": utc_now(),
                            "generation_id": generation_id
                            or terminal_response.get("id"),
                            "provider_reported_cost_usd": float(cost),
                            "undiscounted_cost_usd": list_cost,
                            "benchmark_cost_usd": benchmark_cost,
                            "promotion_adjustment_usd": list_cost - float(cost),
                            "deepseek_peak_adjustment_usd": benchmark_cost - list_cost,
                            "benchmark_adjustment_usd": benchmark_cost - float(cost),
                            "promotion_discount_fraction": promotion_snapshot[
                                "discount_fraction"
                            ],
                            "cost_basis": promotion_snapshot["cost_basis"],
                            "provider_cost_basis": PROVIDER_COST_BASIS,
                            "usage": terminal_usage,
                            "response_id": terminal_response.get("id"),
                            "response_model": response_model,
                            "route_identity_verified": identity_verified,
                            "response_status": terminal_response.get("status"),
                        }
                    )
                elif upstream.status >= 400 and not generation_id:
                    record.update(
                        {
                            "state": "rejected_not_billed",
                            "completed_at": utc_now(),
                            "provider_reported_cost_usd": 0.0,
                            "undiscounted_cost_usd": 0.0,
                            "benchmark_cost_usd": 0.0,
                            "promotion_adjustment_usd": 0.0,
                            "deepseek_peak_adjustment_usd": 0.0,
                            "benchmark_adjustment_usd": 0.0,
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
                    self.ledger_server.complete_request(
                        request_id, benchmark_cost, float(cost)
                    )
                    if not record.get("route_identity_verified", True):
                        self.ledger_server.write_stop(
                            {
                                "schema_version": 2,
                                "run_id": self.ledger_server.run_id,
                                "reason": "openrouter_route_identity_mismatch",
                                "status": "fail_closed",
                            }
                        )
                elif upstream.status >= 400 and not generation_id:
                    self.ledger_server.complete_request(request_id, 0.0, 0.0)
                else:
                    self.ledger_server.require_cost_recovery(request_id)
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
                cost = record.get("provider_reported_cost_usd")
                valid_recorded_cost = (
                    isinstance(cost, (int, float))
                    and not isinstance(cost, bool)
                    and math.isfinite(float(cost))
                    and float(cost) >= 0
                )
                if not valid_recorded_cost:
                    record.update(
                        {
                            "state": "cost_recovery_required",
                            "response_ended_at": utc_now(),
                            "proxy_error_type": type(exc).__name__,
                        }
                    )
                    atomic_json(record_path, record)
                    self.ledger_server.require_cost_recovery(request_id)
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
        provider_endpoint: str | None = None,
        quantization: str | None = None,
        request_contract: dict[str, Any] | None = None,
        allowed_inference_path: str | None = None,
        upstream_api_key: str | None = None,
    ) -> None:
        parsed = urlsplit(upstream)
        if parsed.scheme != "https" or parsed.hostname != "openrouter.ai":
            raise ValueError("upstream must be https://openrouter.ai")
        if parsed.path.rstrip("/") != "/api/v1" or parsed.query or parsed.fragment:
            raise ValueError("upstream must be the OpenRouter /api/v1 root")
        if provider_endpoint is not None and (
            not provider_endpoint
            or len(provider_endpoint) > 128
            or any(character.isspace() for character in provider_endpoint)
        ):
            raise ValueError("provider endpoint must be one OpenRouter slug")
        allowed_quantizations = {
            "int4",
            "int8",
            "fp4",
            "fp6",
            "fp8",
            "fp16",
            "bf16",
            "fp32",
            "unknown",
        }
        if quantization is not None and quantization not in allowed_quantizations:
            raise ValueError("invalid OpenRouter quantization")
        if request_contract is not None:
            allowed_contract_fields = {
                "max_output_tokens",
                "model",
                "max_tokens",
                "reasoning",
                "reasoning_effort",
                "service_tier",
                "stream",
                "temperature",
                "top_p",
            }
            unknown = set(request_contract) - allowed_contract_fields
            if unknown:
                raise ValueError(
                    "invalid request contract fields: " + ", ".join(sorted(unknown))
                )
            if not request_contract:
                raise ValueError("request contract must not be empty")
        if allowed_inference_path not in {None, "responses", "chat_completions"}:
            raise ValueError("invalid allowed inference path")
        if upstream_api_key is not None and len(upstream_api_key) < 16:
            raise ValueError("upstream API key is missing or too short")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 443
        self.run_id = run_id
        self.cpu_attempt = cpu_attempt
        self.provider_endpoint = provider_endpoint
        self.quantization = quantization
        self.request_contract = request_contract
        self.allowed_inference_path = allowed_inference_path
        self.upstream_authorization = (
            f"Bearer {upstream_api_key}" if upstream_api_key is not None else None
        )
        self.run_root = ledger_root.parent
        run = json.loads((self.run_root / "state/run.json").read_text())
        if run.get("run_id") != run_id:
            raise ValueError("run identity mismatch")
        self.canonical_model = str(run.get("model") or "")
        self.resolved_model = str(run.get("resolved_model_version") or "")
        self.requests_dir = ledger_root / "requests"
        self.runtime_dir = runtime_dir
        self.billing_lock = threading.Lock()
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        loaded = self._load_summary()
        if loaded is None:
            loaded = self._rebuild_totals()
        (
            self.api_cost_usd,
            self.provider_billed_api_cost_usd,
            self.completed_request_count,
            self.in_flight_request_ids,
            self.cost_recovery_required_request_ids,
        ) = loaded
        self._reconcile_pending_requests()
        self.write_summary()
        super().__init__(address, LedgerProxyHandler)

    @property
    def summary_path(self) -> Path:
        return self.requests_dir.parent / "summary.json"

    def _load_summary(
        self,
    ) -> tuple[float, float, int, set[str], set[str]] | None:
        if not self.summary_path.is_file():
            return None
        summary = json.loads(self.summary_path.read_text())
        schema = summary.get("schema_version")
        if schema not in {1, 2} or summary.get("run_id") != self.run_id:
            raise ValueError("ledger summary identity mismatch")
        total = float(summary["model_api_usd"])
        provider_total = float(summary.get("provider_billed_model_api_usd", total))
        completed = int(summary["completed_request_count"])
        pending = int(summary["pending_request_count"])
        in_flight_count = int(summary["in_flight_request_count"])
        recovery_count = int(summary["cost_recovery_required_count"])
        in_flight_raw = summary.get("in_flight_request_ids")
        recovery_raw = summary.get("cost_recovery_required_request_ids")
        if in_flight_raw is None and recovery_raw is None and pending == 0:
            in_flight_raw, recovery_raw = [], []
        if not isinstance(in_flight_raw, list) or not isinstance(recovery_raw, list):
            # A legacy pending summary lacks identities, so rebuild it once.
            return None
        if not all(
            valid_ledger_request_id(item) for item in in_flight_raw + recovery_raw
        ):
            raise ValueError("invalid pending ledger request identity")
        in_flight = set(in_flight_raw)
        recovery_required = set(recovery_raw)
        if not (
            math.isfinite(total)
            and total >= 0
            and math.isfinite(provider_total)
            and provider_total >= 0
            and provider_total <= total + 1e-12
            and completed >= 0
            and pending >= 0
            and in_flight_count >= 0
            and recovery_count >= 0
            and pending == in_flight_count + recovery_count
            and len(in_flight) == in_flight_count
            and len(recovery_required) == recovery_count
            and in_flight.isdisjoint(recovery_required)
            and all(in_flight | recovery_required)
        ):
            raise ValueError("invalid ledger summary")
        return total, provider_total, completed, in_flight, recovery_required

    def _rebuild_totals(self) -> tuple[float, float, int, set[str], set[str]]:
        total = 0.0
        provider_total = 0.0
        completed = 0
        in_flight: set[str] = set()
        recovery_required: set[str] = set()
        for path in sorted(self.requests_dir.glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("run_id") != self.run_id:
                raise ValueError("ledger identity mismatch")
            request_id = str(record.get("ledger_request_id") or path.stem)
            if not valid_ledger_request_id(request_id):
                raise ValueError("invalid ledger request identity")
            if request_id in in_flight or request_id in recovery_required:
                raise ValueError("duplicate pending ledger request identity")
            provider_cost = record.get("provider_reported_cost_usd")
            if isinstance(provider_cost, (int, float)) and not isinstance(
                provider_cost, bool
            ):
                provider_value = float(provider_cost)
                cost = record.get(
                    "benchmark_cost_usd",
                    record.get("undiscounted_cost_usd", provider_value),
                )
                if not isinstance(cost, (int, float)) or isinstance(cost, bool):
                    raise ValueError("invalid undiscounted cost")
                value = float(cost)
                if (
                    not math.isfinite(value)
                    or value < 0
                    or not math.isfinite(provider_value)
                    or provider_value < 0
                    or provider_value > value + 1e-12
                ):
                    raise ValueError("invalid provider-reported cost")
                total += value
                provider_total += provider_value
                completed += 1
            elif record.get("state") == "in_flight":
                in_flight.add(request_id)
            elif record.get("state") == "cost_recovery_required":
                recovery_required.add(request_id)
            else:
                raise ValueError("invalid unpriced ledger record")
        return total, provider_total, completed, in_flight, recovery_required

    def _reconcile_pending_requests(self, *, include_in_flight: bool = True) -> None:
        for request_id in (
            sorted(self.in_flight_request_ids) if include_in_flight else ()
        ):
            path = self.requests_dir / f"{request_id}.json"
            if not path.is_file():
                # The summary is persisted before the request record and before
                # any upstream I/O. A missing record therefore cannot be billed.
                atomic_json(
                    path,
                    {
                        "schema_version": 1,
                        "ledger_request_id": request_id,
                        "run_id": self.run_id,
                        "cpu_attempt": self.cpu_attempt,
                        "state": "rejected_not_billed",
                        "completed_at": utc_now(),
                        "provider_reported_cost_usd": 0.0,
                        "undiscounted_cost_usd": 0.0,
                        "benchmark_cost_usd": 0.0,
                        "recovered_after_proxy_restart": True,
                    },
                )
                self.in_flight_request_ids.remove(request_id)
                self.completed_request_count += 1
                continue
            record = json.loads(path.read_text())
            if record.get("run_id") != self.run_id:
                raise ValueError("pending ledger identity mismatch")
            cost = record.get("provider_reported_cost_usd")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and math.isfinite(float(cost))
                and float(cost) >= 0
            ):
                self.in_flight_request_ids.remove(request_id)
                self.completed_request_count += 1
                provider_cost = float(cost)
                benchmark_cost = float(
                    record.get(
                        "benchmark_cost_usd",
                        record.get("undiscounted_cost_usd", provider_cost),
                    )
                )
                if benchmark_cost < provider_cost or not math.isfinite(benchmark_cost):
                    raise ValueError("invalid benchmark cost")
                self.api_cost_usd += benchmark_cost
                self.provider_billed_api_cost_usd += provider_cost
                continue
            record.update(
                {
                    "state": "cost_recovery_required",
                    "response_ended_at": utc_now(),
                    "proxy_error_type": "ProxyRestart",
                }
            )
            atomic_json(path, record)
            self.in_flight_request_ids.remove(request_id)
            self.cost_recovery_required_request_ids.add(request_id)

        for request_id in sorted(self.cost_recovery_required_request_ids):
            path = self.requests_dir / f"{request_id}.json"
            record = json.loads(path.read_text())
            if record.get("run_id") != self.run_id:
                raise ValueError("recovery ledger identity mismatch")
            cost = record.get("provider_reported_cost_usd")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and math.isfinite(float(cost))
                and float(cost) >= 0
            ):
                self.cost_recovery_required_request_ids.remove(request_id)
                self.completed_request_count += 1
                provider_cost = float(cost)
                benchmark_cost = float(
                    record.get(
                        "benchmark_cost_usd",
                        record.get("undiscounted_cost_usd", provider_cost),
                    )
                )
                if benchmark_cost < provider_cost or not math.isfinite(benchmark_cost):
                    raise ValueError("invalid benchmark cost")
                self.api_cost_usd += benchmark_cost
                self.provider_billed_api_cost_usd += provider_cost
            elif record.get("state") != "cost_recovery_required":
                raise ValueError("invalid recovery ledger state")

    def write_summary(self) -> None:
        pending = len(self.in_flight_request_ids) + len(
            self.cost_recovery_required_request_ids
        )
        atomic_json(
            self.summary_path,
            {
                "schema_version": 2,
                "run_id": self.run_id,
                "updated_at": utc_now(),
                "model_api_usd": self.api_cost_usd,
                "provider_billed_model_api_usd": self.provider_billed_api_cost_usd,
                "promotion_savings_usd": (
                    self.api_cost_usd - self.provider_billed_api_cost_usd
                ),
                "model_api_cost_basis": BENCHMARK_COST_BASIS,
                "provider_billed_cost_basis": PROVIDER_COST_BASIS,
                "completed_request_count": self.completed_request_count,
                "pending_request_count": pending,
                "in_flight_request_count": len(self.in_flight_request_ids),
                "cost_recovery_required_count": len(
                    self.cost_recovery_required_request_ids
                ),
                "in_flight_request_ids": sorted(self.in_flight_request_ids),
                "cost_recovery_required_request_ids": sorted(
                    self.cost_recovery_required_request_ids
                ),
            },
        )

    def begin_request(self, request_id: str) -> None:
        if (
            not valid_ledger_request_id(request_id)
            or request_id in self.in_flight_request_ids
            or request_id in self.cost_recovery_required_request_ids
        ):
            raise ValueError("duplicate or invalid ledger request identity")
        self.in_flight_request_ids.add(request_id)
        self.write_summary()

    def complete_request(
        self, request_id: str, cost: float, provider_cost: float | None = None
    ) -> None:
        provider_cost = cost if provider_cost is None else provider_cost
        if (
            not math.isfinite(cost)
            or cost < 0
            or not math.isfinite(provider_cost)
            or provider_cost < 0
            or provider_cost > cost + 1e-12
        ):
            raise ValueError("invalid provider-reported cost")
        if request_id not in self.in_flight_request_ids:
            raise ValueError("ledger has no in-flight request to complete")
        self.in_flight_request_ids.remove(request_id)
        self.completed_request_count += 1
        self.api_cost_usd += cost
        self.provider_billed_api_cost_usd += provider_cost
        self.write_summary()

    def require_cost_recovery(self, request_id: str) -> None:
        if request_id not in self.in_flight_request_ids:
            return
        self.in_flight_request_ids.remove(request_id)
        self.cost_recovery_required_request_ids.add(request_id)
        self.write_summary()

    def budget_snapshot(self) -> tuple[bool, dict[str, Any]]:
        """Return whether another paid request may start under the run cap."""
        if self.in_flight_request_ids or self.cost_recovery_required_request_ids:
            # The independent watchdog can recover an interrupted OpenRouter
            # generation while this proxy remains alive. Reconcile only the
            # named pending records so the next request can safely proceed.
            before = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                frozenset(self.in_flight_request_ids),
                frozenset(self.cost_recovery_required_request_ids),
            )
            self._reconcile_pending_requests(include_in_flight=False)
            after = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                frozenset(self.in_flight_request_ids),
                frozenset(self.cost_recovery_required_request_ids),
            )
            if after != before:
                self.write_summary()
        run = json.loads((self.run_root / "state/run.json").read_text())
        if run.get("run_id") != self.run_id:
            raise ValueError("run identity mismatch")
        budget = float(run["agent_cost_budget_usd"])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("invalid agent cost budget")
        api_cost = self.api_cost_usd
        if self.in_flight_request_ids or self.cost_recovery_required_request_ids:
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
                "provider_billed_model_api_usd": self.provider_billed_api_cost_usd,
                "promotion_savings_usd": (api_cost - self.provider_billed_api_cost_usd),
                "modal_live_estimate_usd": modal_cost,
            },
        }
        return total < budget, payload

    def write_stop(self, payload: dict[str, Any]) -> None:
        marker = self.run_root / "BUDGET_STOP_REQUESTED.json"
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
    parser.add_argument("--provider-endpoint")
    parser.add_argument("--quantization")
    parser.add_argument("--request-contract-json")
    parser.add_argument(
        "--allowed-inference-path", choices=("responses", "chat_completions")
    )
    parser.add_argument("--upstream-api-key-stdin", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    upstream_api_key = None
    if args.upstream_api_key_stdin:
        upstream_api_key = sys.stdin.readline().rstrip("\r\n")
        if len(upstream_api_key) < 16:
            raise SystemExit("upstream API key was not provided on stdin")
    request_contract = None
    if args.request_contract_json:
        try:
            request_contract = json.loads(args.request_contract_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --request-contract-json: {exc}") from exc
        if not isinstance(request_contract, dict):
            raise SystemExit("--request-contract-json must be a JSON object")
    server = LedgerProxyServer(
        (args.listen, args.port),
        upstream=args.upstream,
        ledger_root=args.ledger_root,
        run_id=args.run_id,
        cpu_attempt=args.cpu_attempt,
        runtime_dir=args.runtime_dir,
        provider_endpoint=args.provider_endpoint,
        quantization=args.quantization,
        request_contract=request_contract,
        allowed_inference_path=args.allowed_inference_path,
        upstream_api_key=upstream_api_key,
    )
    server.serve_forever(poll_interval=0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
