#!/usr/bin/env python3
"""Run one paid production-wrapper DeepSeek/OpenRouter/watchdog smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
DEFAULT_REPORT = ROOT / "runs/ops/deepseek-harness-live-smoke.json"
APP_NAME = "sprint-image-warmup"
SUCCESS_MARKER = "LIVE-SMOKE-OK"
RUN_ID = "deepseek-harness-live-smoke"
MODEL = "deepseek/deepseek-v4-flash-vision-exp"
CONTRACT = {
    "model": MODEL,
    "stream": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 384_000,
    "reasoning_effort": "max",
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY is required")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("completed") is not True:
        parser.error("the immutable image warmup is not complete")
    image_id = manifest["contexts"]["agent_training"]["image_id"]
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandbox: modal.Sandbox | None = None
    started = time.monotonic()
    env = {
        "SPRINT_RUN_ID": RUN_ID,
        "SPRINT_MODEL": MODEL,
        "SPRINT_DURABLE_DIR": "/tmp/durable",
        "SPRINT_RUNTIME_DIR": "/tmp/runtime",
        "SPRINT_AGENT_LOG_DIR": "/tmp/logs/agent",
        "SPRINT_CPU_LAUNCH_ATTEMPT": "1",
        "SPRINT_OPENROUTER_PROVIDER_ENDPOINT": "deepseek",
        "SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH": "chat_completions",
        "DSH_GOAL_MAX_ROUNDS": "1",
    }
    run = {
        "run_id": RUN_ID,
        "agent_kind": "deepseek-harness",
        "model": MODEL,
        "reasoning_effort": "max",
        "usage_audit_required": True,
        "standing_gpu_worker": False,
        "agent_cost_budget_usd": 10.0,
        "cpu_launch_attempt": 1,
        "budget_enforcement": {
            "api_cost_source": "openrouter_reported_per_request",
            "shutdown_reserve_usd": 0.0,
            "minimum_safe_shutdown_reserve_usd": 0.0,
        },
    }
    try:
        sandbox = modal.Sandbox.create(
            "python3",
            "-c",
            "import time; time.sleep(900)",
            app=app,
            image=modal.Image.from_id(image_id),
            cpu=2,
            memory=8192,
            timeout=900,
            block_network=False,
            outbound_domain_allowlist=["openrouter.ai"],
            secrets=[modal.Secret.from_dict({"OPENROUTER_API_KEY": key})],
            tags={"sprint.role": "deepseek-harness-live-smoke"},
        )
        initialize = sandbox.exec(
            "python3",
            "-c",
            "import json,pathlib; p=pathlib.Path('/tmp/durable/runs/"
            + RUN_ID
            + "/state/run.json'); p.parent.mkdir(parents=True); "
            + "p.write_text(json.dumps("
            + repr(run)
            + "))",
            timeout=30,
        )
        initialize.stdout.read()
        initialize.stderr.read()
        initialize.wait()
        if initialize.returncode != 0:
            raise RuntimeError("could not initialize the smoke run contract")

        process = sandbox.exec(
            "/opt/sprint-deepseek-harness-exec-wrapper.sh",
            "Reply exactly LIVE-SMOKE-OK. Do not call a tool.",
            env=env,
            timeout=720,
        )
        health = sandbox.exec(
            "python3",
            "-c",
            "import json,os,pathlib,subprocess,time; "
            "root=pathlib.Path('/tmp/runtime/sprint-agent'); "
            "deadline=time.monotonic()+30; "
            "exec(\"while time.monotonic() < deadline and not "
            "((root/'openrouter-proxy.pid').is_file() and "
            "(root/'agent-process').is_file()): time.sleep(0.1)\"); "
            "agent=int((root/'agent-process').read_text().split()[0]); "
            "proxy=int((root/'openrouter-proxy.pid').read_text()); "
            "os.kill(agent,0); os.kill(proxy,0); "
            "items=pathlib.Path(f'/proc/{agent}/environ').read_bytes().split(b'\\0'); "
            "token=next(x.split(b'=',1)[1] for x in items if x.startswith(b'OPENROUTER_API_KEY=')); "
            "assert token == b'sprint-local-proxy-token'; "
            "assert token != os.environ['OPENROUTER_API_KEY'].encode(); "
            "result=subprocess.run(['/opt/sprint-budget-watchdog.py','--run-id','"
            + RUN_ID
            + "','--durable-dir','/tmp/durable','--runtime-dir','/tmp/runtime',"
            "'--codex-home','/tmp/codex-home']); "
            "assert result.returncode == 0; "
            "watchdog=json.loads(pathlib.Path('/tmp/durable/runs/"
            + RUN_ID
            + "/budget/watchdog.json').read_text()); "
            "assert watchdog['status'] == 'within_budget'; "
            "print('TRUSTED_KEY_BOUNDARY_OK'); print('DEEPSEEK_WATCHDOG_LIVE_OK')",
            env=env,
            timeout=90,
        )
        health_stdout = health.stdout.read() or ""
        health_stderr = health.stderr.read() or ""
        health.wait()
        if health.returncode != 0 or "DEEPSEEK_WATCHDOG_LIVE_OK" not in health_stdout:
            raise RuntimeError(
                "DeepSeek runtime/watchdog health failed: "
                + (health_stderr or health_stdout)[-4000:]
            )

        stdout = process.stdout.read() or ""
        stderr = process.stderr.read() or ""
        process.wait()
        if process.returncode != 0 or SUCCESS_MARKER not in stdout:
            raise RuntimeError(
                f"live probe exited {process.returncode}: {(stderr or stdout)[-4000:]}"
            )

        inspect = sandbox.exec(
            "python3",
            "-c",
            "import json,pathlib; root=pathlib.Path('/tmp/durable/runs/"
            + RUN_ID
            + "/api-usage'); records=sorted((root/'requests').glob('*.json')); "
            + "assert records; parsed=[json.loads(p.read_text()) for p in records]; "
            + "print(json.dumps({'records':parsed,'summary':json.loads((root/'summary.json').read_text())}))",
            timeout=30,
        )
        raw = inspect.stdout.read() or ""
        inspect.wait()
        if inspect.returncode != 0:
            raise RuntimeError("could not inspect the DeepSeek smoke ledger")
        ledger = json.loads(raw.strip().splitlines()[-1])
        records = ledger["records"]
        record = records[-1]
        summary = ledger["summary"]
        snapshot = record.get("promotion_snapshot") or {}
        endpoints = snapshot.get("endpoints") or []
        endpoint = endpoints[0] if endpoints else {}
        checks = {
            "record_complete": record.get("state") == "complete",
            "chat_completions_path": record.get("api_path")
            == "/api/v1/chat/completions",
            "request_contract_sealed": record.get("request_contract") == CONTRACT,
            "official_provider_route": endpoint.get("tag") == "deepseek",
            "official_provider_name": endpoint.get("provider_name") == "DeepSeek",
            "peak_price_floor_present": bool(
                snapshot.get("deepseek_peak_pricing_usd_per_token")
            ),
            "trusted_key_boundary": "TRUSTED_KEY_BOUNDARY_OK" in health_stdout,
            "watchdog_live": "DEEPSEEK_WATCHDOG_LIVE_OK" in health_stdout,
            "all_requests_complete": summary.get("completed_request_count")
            == len(records),
            "no_pending_request": summary.get("pending_request_count") == 0,
            "benchmark_not_below_charge": float(record["benchmark_cost_usd"])
            >= float(record["provider_reported_cost_usd"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"live smoke checks failed: {checks}")
        usage = record.get("usage") or {}
        report = {
            "schema_version": 2,
            "completed": True,
            "image_id": image_id,
            "sandbox_id": sandbox.object_id,
            "elapsed_s": round(time.monotonic() - started, 3),
            "checks": checks,
            "provider_name": endpoint.get("provider_name"),
            "provider_tag": endpoint.get("tag"),
            "provider_reported_cost_usd": record.get("provider_reported_cost_usd"),
            "undiscounted_cost_usd": record.get("undiscounted_cost_usd"),
            "benchmark_cost_usd": record.get("benchmark_cost_usd"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        }
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        if sandbox is not None:
            sandbox.terminate(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
