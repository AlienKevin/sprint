"""Create a Codex thread goal before the thread's first model turn.

The interactive Codex clients interpret ``/goal`` themselves. ``codex exec``
does not: it sends the marker to the model as ordinary prompt text.  Harbor
therefore uses the app-server control plane to create the thread, set and verify
its persistent goal, and only then resumes that thread with ``codex exec``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 30.0


class AppServerError(RuntimeError):
    """The local Codex app-server did not honor the bootstrap contract."""


class AppServerClient:
    """Minimal line-delimited JSON-RPC client for Codex app-server."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if process.stdin is None or process.stdout is None:
            raise ValueError("app-server subprocess requires stdin and stdout pipes")
        self.process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.timeout_seconds = timeout_seconds
        self._next_request_id = 1

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.stdin.write(json.dumps({"method": method, "params": params}) + "\n")
        self.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        self.stdin.write(
            json.dumps({"id": request_id, "method": method, "params": params}) + "\n"
        )
        self.stdin.flush()

        selector = selectors.DefaultSelector()
        selector.register(self.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AppServerError(
                        f"timed out waiting for app-server response to {method}"
                    )
                if not selector.select(remaining):
                    continue
                line = self.stdout.readline()
                if not line:
                    raise AppServerError(
                        f"app-server exited before responding to {method}"
                    )
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AppServerError(
                        f"app-server returned malformed JSON for {method}"
                    ) from exc
                # Notifications may be interleaved with responses.
                if message.get("id") != request_id:
                    continue
                if message.get("error") is not None:
                    raise AppServerError(
                        f"app-server rejected {method}: {message['error']}"
                    )
                result = message.get("result")
                if not isinstance(result, dict):
                    raise AppServerError(
                        f"app-server returned no object result for {method}"
                    )
                return result
        finally:
            selector.close()


def _validated_goal(
    result: dict[str, Any], *, thread_id: str, objective: str
) -> dict[str, Any]:
    goal = result.get("goal")
    if not isinstance(goal, dict):
        raise AppServerError("goal response did not contain a goal object")
    if goal.get("threadId") != thread_id:
        raise AppServerError("goal response referenced the wrong thread")
    if goal.get("objective") != objective:
        raise AppServerError("goal response changed the objective")
    if goal.get("status") != "active":
        raise AppServerError("goal was not active before the first turn")
    if goal.get("tokenBudget") is not None:
        raise AppServerError("goal unexpectedly has a model-token budget")
    return goal


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def bootstrap_goal(
    *,
    model: str,
    cwd: str,
    objective: str,
    receipt_path: Path,
    app_server_args: list[str],
    expected_provider: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    if not objective.strip():
        raise AppServerError("refusing to create an empty goal")
    command = ["codex", "app-server", "--stdio", *app_server_args]
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        client = AppServerClient(process, timeout_seconds=timeout_seconds)
        try:
            client.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "harbor-codex-goal-bootstrap",
                        "version": "1",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            client.notify("initialized", {})
            started = client.request(
                "thread/start",
                {
                    "cwd": cwd,
                    "model": model,
                    "approvalPolicy": "never",
                    "sandbox": "danger-full-access",
                    "ephemeral": False,
                },
            )
            thread = started.get("thread")
            thread_id = thread.get("id") if isinstance(thread, dict) else None
            if not isinstance(thread_id, str) or not thread_id:
                raise AppServerError("thread/start did not return a thread id")
            if started.get("model") != model:
                raise AppServerError("thread/start selected a different model")
            model_provider = started.get("modelProvider")
            if expected_provider and model_provider != expected_provider:
                raise AppServerError("thread/start selected a different model provider")

            set_result = client.request(
                "thread/goal/set",
                {
                    "threadId": thread_id,
                    "objective": objective,
                    "status": "active",
                },
            )
            _validated_goal(set_result, thread_id=thread_id, objective=objective)
            get_result = client.request(
                "thread/goal/get", {"threadId": thread_id}
            )
            goal = _validated_goal(
                get_result, thread_id=thread_id, objective=objective
            )
            _atomic_write_json(
                receipt_path,
                {
                    "schema_version": 1,
                    "phase": "before_first_model_turn",
                    "thread_id": thread_id,
                    "model": started.get("model"),
                    "model_provider": model_provider,
                    "objective": objective,
                    "objective_sha256": hashlib.sha256(objective.encode()).hexdigest(),
                    "status": goal["status"],
                    "token_budget": goal.get("tokenBudget"),
                    "created_at": goal.get("createdAt"),
                    "verified_at": int(time.time()),
                },
            )
            return thread_id
        except Exception as exc:
            stderr.flush()
            stderr.seek(0)
            detail = stderr.read().strip()
            if detail:
                raise AppServerError(f"{exc}; app-server stderr: {detail}") from exc
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--objective-env", default="SPRINT_CODEX_GOAL_OBJECTIVE"
    )
    parser.add_argument(
        "--server-args-env", default="SPRINT_CODEX_APP_SERVER_ARGS_JSON"
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--expected-provider")
    args = parser.parse_args()

    objective = os.environ.get(args.objective_env, "")
    try:
        raw_server_args = os.environ.get(args.server_args_env, "[]")
        app_server_args = json.loads(raw_server_args)
        if not isinstance(app_server_args, list) or not all(
            isinstance(item, str) for item in app_server_args
        ):
            raise ValueError("app-server args must be a JSON string array")
        thread_id = bootstrap_goal(
            model=args.model,
            cwd=args.cwd,
            objective=objective,
            receipt_path=args.receipt,
            app_server_args=app_server_args,
            expected_provider=args.expected_provider,
            timeout_seconds=args.timeout_seconds,
        )
    except (AppServerError, ValueError, json.JSONDecodeError) as exc:
        print(f"Codex goal bootstrap failed: {exc}", file=sys.stderr)
        return 1
    print(thread_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
