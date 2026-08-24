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
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlsplit


sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprint_openrouter_pricing import (  # noqa: E402
    PROVIDER_COST_BASIS,
    OpenRouterPricingError,
    benchmark_cost_basis_for_model,
    benchmark_cost_usd,
    capture_endpoint_discount_snapshot,
    undiscounted_cost_usd,
)
from sprint_openrouter_usage import (  # noqa: E402
    add_token_usage,
    empty_token_usage,
    generation_usage_payload,
    normalize_token_usage,
    validate_token_usage_totals,
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

# Max-effort reasoning requests can legitimately spend well over ten minutes
# before OpenRouter returns the response headers or first SSE event.  Do not
# impose a proxy-local wall-clock timeout: the independent budget watchdog
# remains live while this socket is open and will terminate the sandbox at the
# run's dollar cap.  A finite timeout here converts a healthy long request into
# an unpriced interrupted generation, which must fail closed.
UPSTREAM_SOCKET_TIMEOUT_SECONDS: float | None = None
OPENROUTER_GENERATION_RECOVERY_TIMEOUT_SECONDS = 30.0
OPENROUTER_GENERATION_RECOVERY_POLL_SECONDS = 0.25


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


def recover_openrouter_generation(
    generation_id: str, authorization: str | None
) -> dict[str, Any] | None:
    """Fetch one exact generation audit with the sealed per-trial key."""
    if not generation_id or not authorization:
        return None
    url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode(
        {"id": generation_id}
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": authorization, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else None


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
            # Some routed models will still invent undeclared arguments.  The
            # response-side guard below is the enforcement boundary, but a
            # closed object schema prevents compliant providers from emitting
            # the operator-only field in the first place.
            parameters["additionalProperties"] = False


def sanitize_goal_arguments(arguments: object) -> object:
    """Strip the operator-only goal token budget from model output."""
    if not isinstance(arguments, str):
        return arguments
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(payload, dict) or "token_budget" not in payload:
        return arguments
    payload.pop("token_budget", None)
    return json.dumps(payload, separators=(",", ":"))


def sanitize_goal_tool_calls(payload: object) -> object:
    """Rewrite complete Responses objects at the trusted proxy boundary."""
    if isinstance(payload, list):
        for item in payload:
            sanitize_goal_tool_calls(item)
        return payload
    if not isinstance(payload, dict):
        return payload
    if payload.get("type") == "function_call" and payload.get("name") == "create_goal":
        payload["arguments"] = sanitize_goal_arguments(payload.get("arguments"))
    for value in payload.values():
        sanitize_goal_tool_calls(value)
    return payload


def expose_deepseek_reasoning_content(payload: object) -> object:
    """Alias OpenRouter's normalized reasoning field for DeepSeek Harness.

    OpenRouter normalizes Chat Completions reasoning text to ``reasoning`` (and
    ``reasoning_details``), while the official DeepSeek harness adapter reads
    the native DeepSeek field ``reasoning_content``.  Preserve OpenRouter's
    canonical fields and add the native alias at the trusted compatibility
    boundary so the pinned official harness records and carries reasoning
    across tool-call turns exactly as it does against DeepSeek's native API.
    """
    if not isinstance(payload, dict):
        return payload
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for field in ("delta", "message"):
            model_output = choice.get(field)
            if not isinstance(model_output, dict):
                continue
            if "reasoning_content" in model_output:
                continue
            reasoning = model_output.get("reasoning")
            if isinstance(reasoning, str):
                model_output["reasoning_content"] = reasoning
    return payload


class GoalToolStreamSanitizer:
    """Buffer only ``create_goal`` argument deltas and emit sealed JSON.

    Codex executes tool calls from streamed Responses events.  Editing the
    advertised schema is therefore insufficient: a model may hallucinate an
    undeclared ``token_budget`` and Codex's local tool will accept it.  This
    state machine withholds only that tool's argument deltas until its done
    event, then emits one sanitized delta plus sanitized terminal events.  All
    reasoning, text, and unrelated tool events remain fully streaming.
    """

    def __init__(self) -> None:
        self._tracked_ids: set[str] = set()
        self._tracked_indexes: set[int] = set()
        self._arguments: dict[str, list[str]] = {}
        self._delta_templates: dict[str, dict[str, Any]] = {}
        self._emitted: set[str] = set()

    @staticmethod
    def _event_key(event: dict[str, Any]) -> str | None:
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id:
            return f"id:{item_id}"
        item = event.get("item")
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                return f"id:{item_id}"
        index = event.get("output_index")
        if isinstance(index, int) and not isinstance(index, bool):
            return f"index:{index}"
        return None

    def _is_tracked(self, event: dict[str, Any]) -> bool:
        item_id = event.get("item_id")
        index = event.get("output_index")
        return bool(
            (isinstance(item_id, str) and item_id in self._tracked_ids)
            or (
                isinstance(index, int)
                and not isinstance(index, bool)
                and index in self._tracked_indexes
            )
        )

    def _track(self, event: dict[str, Any], item: dict[str, Any]) -> None:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            self._tracked_ids.add(item_id)
        index = event.get("output_index")
        if isinstance(index, int) and not isinstance(index, bool):
            self._tracked_indexes.add(index)

    def rewrite_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = event.get("type")
        item = event.get("item")
        if (
            event_type == "response.output_item.added"
            and isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "create_goal"
        ):
            self._track(event, item)
            sanitize_goal_tool_calls(event)
            return [event]

        if event_type == "response.function_call_arguments.delta" and self._is_tracked(
            event
        ):
            key = self._event_key(event)
            delta = event.get("delta")
            if key is not None and isinstance(delta, str):
                self._arguments.setdefault(key, []).append(delta)
                self._delta_templates[key] = dict(event)
            return []

        if event_type == "response.function_call_arguments.done" and self._is_tracked(
            event
        ):
            key = self._event_key(event)
            raw = event.get("arguments")
            if not isinstance(raw, str) and key is not None:
                raw = "".join(self._arguments.get(key, []))
            sealed = sanitize_goal_arguments(raw)
            event["arguments"] = sealed
            emitted: list[dict[str, Any]] = []
            if key is not None and key not in self._emitted and isinstance(sealed, str):
                delta_event = dict(self._delta_templates.get(key, {}))
                for key_name in ("item_id", "output_index"):
                    if key_name not in delta_event and key_name in event:
                        delta_event[key_name] = event[key_name]
                delta_event["type"] = "response.function_call_arguments.delta"
                delta_event["delta"] = sealed
                delta_event.pop("arguments", None)
                emitted.append(delta_event)
                self._emitted.add(key)
            emitted.append(event)
            return emitted

        if event_type == "response.output_item.done" and isinstance(item, dict):
            tracked = self._is_tracked(event) or (
                item.get("type") == "function_call"
                and item.get("name") == "create_goal"
            )
            if tracked:
                self._track(event, item)
                key = self._event_key(event)
                sealed = sanitize_goal_arguments(item.get("arguments"))
                item["arguments"] = sealed
                emitted = []
                if (
                    key is not None
                    and key not in self._emitted
                    and isinstance(sealed, str)
                ):
                    delta_event = {
                        key_name: event[key_name]
                        for key_name in ("sequence_number", "output_index")
                        if key_name in event
                    }
                    item_id = item.get("id")
                    if isinstance(item_id, str):
                        delta_event["item_id"] = item_id
                    delta_event.update(
                        {
                            "type": "response.function_call_arguments.delta",
                            "delta": sealed,
                        }
                    )
                    emitted.append(delta_event)
                    self._emitted.add(key)
                emitted.append(event)
                return emitted

        sanitize_goal_tool_calls(event)
        return [event]

    def rewrite_line(self, line: bytes) -> list[bytes]:
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            return [line]
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            return [line]
        if not isinstance(event, dict):
            return [line]
        return [
            b"data: " + json.dumps(item, separators=(",", ":")).encode()
            for item in self.rewrite_event(event)
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
    payload = sanitize_request_text(payload)
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
    # Codex 0.149.1 emits ``text.verbosity`` for every Responses turn.  The
    # pinned OpenRouter ``openai`` endpoint currently rejects that transport
    # field under ``require_parameters=true`` even though the same request is
    # accepted when it is omitted.  Verbosity is only a presentation hint; it
    # does not select the model, reasoning effort, token ceiling, tools, or
    # sampling contract.  Remove only this known hint and only for the official
    # OpenAI route, while preserving future text configuration with additional
    # semantics and every non-OpenAI provider payload.
    text_config = payload.get("text")
    if (
        provider_endpoint == "openai"
        and isinstance(text_config, dict)
        and set(text_config) == {"verbosity"}
    ):
        payload.pop("text")
    seal_goal_tool_schema(payload)
    return json.dumps(payload, separators=(",", ":")).encode(), payload


def _sanitize_model_string(value: str) -> str:
    """Make model-visible text valid Unicode without hiding useful whitespace.

    Terminal tools can observe arbitrary bytes (for example, when a model cats
    a binary checkpoint).  The DeepSeek Harness PTY replaces most malformed
    UTF-8, but a split terminal chunk can still surface a lone UTF-16 surrogate
    and C0 controls in the JSON request.  Python's JSON encoder preserves those
    as escapes, while model providers are allowed to reject them as invalid
    text.  Render controls visibly and replace lone surrogates before the
    request crosses the trusted proxy boundary.
    """
    rendered: list[str] = []
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            rendered.append("\N{REPLACEMENT CHARACTER}")
        elif codepoint < 0x20 and character not in "\t\n\r":
            rendered.append(f"\\x{codepoint:02x}")
        else:
            rendered.append(character)
    return "".join(rendered)


def sanitize_request_text(value: Any) -> Any:
    """Recursively sanitize string values in an inference request payload."""
    if isinstance(value, str):
        return _sanitize_model_string(value)
    if isinstance(value, list):
        return [sanitize_request_text(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_request_text(item) for key, item in value.items()}
    return value


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
        if self.path in {"/healthz", "/ledger-status"}:
            status = self.ledger_server.reconciliation_status()
            ready = bool(status["ready"])
            body = (json.dumps(status, sort_keys=True) + "\n").encode()
            self.send_response(
                HTTPStatus.OK
                if ready or self.path == "/ledger-status"
                else HTTPStatus.SERVICE_UNAVAILABLE
            )
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
            timeout=UPSTREAM_SOCKET_TIMEOUT_SECONDS,
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
            is_event_stream = "text/event-stream" in content_type
            goal_stream = GoalToolStreamSanitizer() if is_event_stream else None
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                if downstream_open and (not record_usage or not is_event_stream):
                    try:
                        # Non-stream ledger responses are buffered below so a
                        # complete function-call object can be sealed before
                        # Codex sees it. Unmetered passthroughs remain raw.
                        if not record_usage:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        downstream_open = False
                if not record_usage:
                    continue
                if is_event_stream:
                    line_buffer.extend(chunk)
                    while b"\n" in line_buffer:
                        raw_line, _, remainder = line_buffer.partition(b"\n")
                        line_buffer = bytearray(remainder)
                        line = raw_line.rstrip(b"\r")
                        if line.startswith(b"data: ") and line != b"data: [DONE]":
                            try:
                                event = json.loads(line[6:])
                            except json.JSONDecodeError:
                                event = None
                            if (
                                isinstance(event, dict)
                                and self.ledger_server.provider_endpoint == "deepseek"
                            ):
                                expose_deepseek_reasoning_content(event)
                                line = b"data: " + json.dumps(
                                    event, separators=(",", ":")
                                ).encode()
                            usage, response = usage_from_event(event)
                            if usage is not None:
                                terminal_usage, terminal_response = usage, response
                        rewritten = (
                            goal_stream.rewrite_line(line) if goal_stream else [line]
                        )
                        if downstream_open:
                            try:
                                for output_line in rewritten:
                                    self.wfile.write(output_line + b"\n")
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                downstream_open = False
                else:
                    response_buffer.extend(chunk)
            if record_usage and is_event_stream and line_buffer and downstream_open:
                try:
                    for output_line in goal_stream.rewrite_line(bytes(line_buffer)):
                        self.wfile.write(output_line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    downstream_open = False
            if record_usage and not is_event_stream:
                try:
                    event = json.loads(response_buffer)
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict):
                    if self.ledger_server.provider_endpoint == "deepseek":
                        expose_deepseek_reasoning_content(event)
                    sanitize_goal_tool_calls(event)
                    response_buffer = bytearray(
                        json.dumps(event, separators=(",", ":")).encode()
                    )
                if downstream_open:
                    try:
                        self.wfile.write(response_buffer)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        downstream_open = False
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
                    if (
                        promotion_snapshot.get("cost_basis")
                        != self.ledger_server.model_api_cost_basis
                    ):
                        raise ValueError("request pricing cost basis mismatch")
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
                            "cost_basis": self.ledger_server.model_api_cost_basis,
                            "provider_cost_basis": PROVIDER_COST_BASIS,
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
                        request_id,
                        benchmark_cost,
                        float(cost),
                        terminal_usage,
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
                    if not self.ledger_server.recover_request_until(
                        request_id,
                        timeout_seconds=OPENROUTER_GENERATION_RECOVERY_TIMEOUT_SECONDS,
                    ):
                        self.ledger_server.write_stop(
                            {
                                "schema_version": 2,
                                "run_id": self.ledger_server.run_id,
                                "reason": "budget_telemetry_unavailable",
                                "status": "fail_closed",
                            }
                        )
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
        self.model_api_cost_basis = str(
            (run.get("budget_enforcement") or {}).get("api_budget_cost_basis") or ""
        )
        snapshot_basis = str(
            (run.get("api_pricing_snapshot") or {}).get("cost_basis") or ""
        )
        if not self.model_api_cost_basis or (
            snapshot_basis and snapshot_basis != self.model_api_cost_basis
        ):
            raise ValueError("run and pricing snapshot cost basis mismatch")
        if self.model_api_cost_basis != benchmark_cost_basis_for_model(
            self.canonical_model
        ):
            raise ValueError("run model and cost basis mismatch")
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
            self.token_usage,
        ) = loaded
        self._reconcile_pending_requests()
        self.write_summary()
        super().__init__(address, LedgerProxyHandler)

    @property
    def summary_path(self) -> Path:
        return self.requests_dir.parent / "summary.json"

    def _load_summary(
        self,
    ) -> tuple[float, float, int, set[str], set[str], dict[str, int]] | None:
        if not self.summary_path.is_file():
            return None
        summary = json.loads(self.summary_path.read_text())
        schema = summary.get("schema_version")
        if schema != 3 or summary.get("run_id") != self.run_id:
            raise ValueError("ledger summary identity mismatch")
        if summary.get("model_api_cost_basis") != self.model_api_cost_basis:
            raise ValueError("ledger summary cost basis mismatch")
        total = float(summary["model_api_usd"])
        provider_total = float(summary.get("provider_billed_model_api_usd", total))
        completed = int(summary["completed_request_count"])
        pending = int(summary["pending_request_count"])
        in_flight_count = int(summary["in_flight_request_count"])
        recovery_count = int(summary["cost_recovery_required_count"])
        in_flight_raw = summary.get("in_flight_request_ids")
        recovery_raw = summary.get("cost_recovery_required_request_ids")
        if not isinstance(in_flight_raw, list) or not isinstance(recovery_raw, list):
            raise ValueError("ledger summary lacks pending request identities")
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
        token_usage = validate_token_usage_totals(summary.get("token_usage"))
        return (
            total,
            provider_total,
            completed,
            in_flight,
            recovery_required,
            token_usage,
        )

    def _rebuild_totals(
        self,
    ) -> tuple[float, float, int, set[str], set[str], dict[str, int]]:
        total = 0.0
        provider_total = 0.0
        completed = 0
        in_flight: set[str] = set()
        recovery_required: set[str] = set()
        token_usage = empty_token_usage()
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
                if record.get("cost_basis") != self.model_api_cost_basis:
                    raise ValueError("ledger record cost basis mismatch")
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
                usage = record.get("usage")
                if usage is not None:
                    add_token_usage(token_usage, usage)
            elif record.get("state") == "in_flight":
                in_flight.add(request_id)
            elif record.get("state") == "cost_recovery_required":
                recovery_required.add(request_id)
            else:
                raise ValueError("invalid unpriced ledger record")
        return (
            total,
            provider_total,
            completed,
            in_flight,
            recovery_required,
            token_usage,
        )

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
                        "cost_basis": self.model_api_cost_basis,
                        "provider_cost_basis": PROVIDER_COST_BASIS,
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
                if record.get("cost_basis") != self.model_api_cost_basis:
                    raise ValueError("pending ledger cost basis mismatch")
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
                usage = record.get("usage")
                if usage is not None:
                    add_token_usage(self.token_usage, usage)
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
            recovered: dict[str, Any] | None = None
            if not isinstance(cost, (int, float)) or isinstance(cost, bool):
                generation_id = record.get("generation_id")
                if isinstance(generation_id, str) and generation_id:
                    recovered = recover_openrouter_generation(
                        generation_id, self.upstream_authorization
                    )
                    candidate = recovered.get("total_cost") if recovered else None
                    if (
                        isinstance(candidate, (int, float))
                        and not isinstance(candidate, bool)
                        and math.isfinite(float(candidate))
                        and float(candidate) >= 0
                    ):
                        provider_cost = float(candidate)
                        recovered_usage = generation_usage_payload(recovered)
                        try:
                            if (
                                (record.get("promotion_snapshot") or {}).get(
                                    "cost_basis"
                                )
                                != self.model_api_cost_basis
                            ):
                                raise ValueError(
                                    "recovered request cost basis mismatch"
                                )
                            list_cost = undiscounted_cost_usd(
                                provider_cost, record.get("promotion_snapshot")
                            )
                            benchmark_cost = benchmark_cost_usd(
                                provider_cost,
                                record.get("promotion_snapshot"),
                                recovered,
                            )
                        except OpenRouterPricingError as exc:
                            raise ValueError(
                                "recovered request has incomplete pricing metadata"
                            ) from exc
                        record.update(
                            {
                                "state": "recovered_complete",
                                "completed_at": utc_now(),
                                "provider_reported_cost_usd": provider_cost,
                                "undiscounted_cost_usd": list_cost,
                                "benchmark_cost_usd": benchmark_cost,
                                "promotion_adjustment_usd": list_cost - provider_cost,
                                "deepseek_peak_adjustment_usd": benchmark_cost
                                - list_cost,
                                "benchmark_adjustment_usd": benchmark_cost
                                - provider_cost,
                                "promotion_discount_fraction": (
                                    record.get("promotion_snapshot") or {}
                                ).get("discount_fraction"),
                                "cost_basis": self.model_api_cost_basis,
                                "provider_cost_basis": PROVIDER_COST_BASIS,
                                "generation_audit": recovered,
                                "usage": recovered_usage,
                                "recovered_after_proxy_restart": True,
                            }
                        )
                        atomic_json(path, record)
                        cost = provider_cost
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and math.isfinite(float(cost))
                and float(cost) >= 0
            ):
                if record.get("cost_basis") != self.model_api_cost_basis:
                    raise ValueError("recovered ledger cost basis mismatch")
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
                usage = record.get("usage")
                if usage is not None:
                    add_token_usage(self.token_usage, usage)
            elif record.get("state") != "cost_recovery_required":
                raise ValueError("invalid recovery ledger state")

    def reconciliation_status(self) -> dict[str, Any]:
        """Reconcile named interrupted generations and expose drain readiness."""
        with self.billing_lock:
            before = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                tuple(self.token_usage.items()),
                frozenset(self.in_flight_request_ids),
                frozenset(self.cost_recovery_required_request_ids),
            )
            self._reconcile_pending_requests(include_in_flight=False)
            after = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                tuple(self.token_usage.items()),
                frozenset(self.in_flight_request_ids),
                frozenset(self.cost_recovery_required_request_ids),
            )
            if after != before:
                self.write_summary()
            pending = len(self.in_flight_request_ids) + len(
                self.cost_recovery_required_request_ids
            )
            return {
                "status": "ok" if pending == 0 else "reconciling",
                "ready": pending == 0,
                "pending_request_count": pending,
                "in_flight_request_count": len(self.in_flight_request_ids),
                "cost_recovery_required_count": len(
                    self.cost_recovery_required_request_ids
                ),
            }

    def write_summary(self) -> None:
        pending = len(self.in_flight_request_ids) + len(
            self.cost_recovery_required_request_ids
        )
        atomic_json(
            self.summary_path,
            {
                "schema_version": 3,
                "run_id": self.run_id,
                "updated_at": utc_now(),
                "model_api_usd": self.api_cost_usd,
                "provider_billed_model_api_usd": self.provider_billed_api_cost_usd,
                "promotion_savings_usd": (
                    self.api_cost_usd - self.provider_billed_api_cost_usd
                ),
                "model_api_cost_basis": self.model_api_cost_basis,
                "provider_billed_cost_basis": PROVIDER_COST_BASIS,
                "completed_request_count": self.completed_request_count,
                "token_usage": self.token_usage,
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
        self,
        request_id: str,
        cost: float,
        provider_cost: float | None = None,
        usage: object = None,
    ) -> None:
        provider_cost = cost if provider_cost is None else provider_cost
        normalized_usage = (
            normalize_token_usage(usage) if usage is not None else None
        )
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
        if normalized_usage is not None:
            add_token_usage(self.token_usage, normalized_usage, normalized=True)
        self.write_summary()

    def require_cost_recovery(self, request_id: str) -> None:
        if request_id not in self.in_flight_request_ids:
            return
        self.in_flight_request_ids.remove(request_id)
        self.cost_recovery_required_request_ids.add(request_id)
        self.write_summary()

    def recover_request_until(
        self, request_id: str, *, timeout_seconds: float
    ) -> bool:
        """Resolve a named OpenRouter charge before releasing the API turn.

        OpenRouter can occasionally close a successful stream before its final
        usage event reaches the client, while the generation audit becomes
        available a moment later.  The proxy retains the sealed per-trial key,
        so it is the only component that can safely recover that exact charge.
        Keep the billing lock and downstream turn open while polling; no later
        model request can overtake an unknown charge.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("invalid OpenRouter recovery timeout")
        deadline = time.monotonic() + timeout_seconds
        while request_id in self.cost_recovery_required_request_ids:
            before = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                tuple(self.token_usage.items()),
                frozenset(self.cost_recovery_required_request_ids),
            )
            self._reconcile_pending_requests(include_in_flight=False)
            after = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                tuple(self.token_usage.items()),
                frozenset(self.cost_recovery_required_request_ids),
            )
            if after != before:
                self.write_summary()
            if request_id not in self.cost_recovery_required_request_ids:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(OPENROUTER_GENERATION_RECOVERY_POLL_SECONDS, remaining))
        return True

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
                tuple(self.token_usage.items()),
                frozenset(self.in_flight_request_ids),
                frozenset(self.cost_recovery_required_request_ids),
            )
            self._reconcile_pending_requests(include_in_flight=False)
            after = (
                self.api_cost_usd,
                self.provider_billed_api_cost_usd,
                self.completed_request_count,
                tuple(self.token_usage.items()),
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
