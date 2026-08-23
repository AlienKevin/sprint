#!/usr/bin/env python3
"""Run one official DeepSeek Harness minimal-mode benchmark turn."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from deepseek_harness import DeepSeekHarness


HARNESS_RUNTIME = "/usr/local/bin/dsh-jsonrpc-agent"
HARNESS_CONFIG = "/opt/deepseek-harness-minimal.cordis.yml"
MODEL = "deepseek/deepseek-v4-flash-vision-exp"
MAX_TOKENS = 384_000


def append_event(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete JSONL notification to the forensic transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        handle.write("\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--workspace", type=Path, default=Path("/app"))
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--stop-file", type=Path, default=Path("/run/sprint-stop"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    workspace = args.workspace.resolve()
    session_root = args.session_root.resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    def record(notification: object) -> None:
        if hasattr(notification, "model_dump"):
            payload = notification.model_dump(mode="json")
        else:
            payload = {"notification": repr(notification)}
        append_event(args.events, payload)

    with DeepSeekHarness(
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
    ) as harness:
        session = harness.start_session(args.session_id)
        prompt = args.prompt
        round_number = 1
        while not args.stop_file.exists():
            result = session.run(prompt, on_notification=record)
            if result.final_response:
                print(result.final_response, flush=True)
            if result.finish_reason != "completed":
                if args.stop_file.exists():
                    return 0
                print(
                    f"DeepSeek Harness ended with {result.finish_reason!r}",
                    file=sys.stderr,
                )
                return 1
            round_number += 1
            prompt = (
                "Continue working autonomously on the benchmark task. Inspect the "
                "current workspace and durable run state, submit any improved policy, "
                "then keep improving until the operator stops the run. "
                f"This is continuation round {round_number}."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
