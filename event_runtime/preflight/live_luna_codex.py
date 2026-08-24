#!/usr/bin/env python3
"""Run one tiny paid end-to-end Luna/Codex/OpenRouter smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import time
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
DEFAULT_REPORT = ROOT / "runs/ops/luna-codex-live-smoke.json"
APP_NAME = "sprint-image-warmup"
SUCCESS_MARKER = "LIVE-SMOKE-OK"
MODEL = "openai/gpt-5.6-luna"
RESOLVED_MODEL = "openai/gpt-5.6-luna-20260709"
CONTRACT = {
    "max_output_tokens": 128_000,
    "model": MODEL,
    "reasoning": {"effort": "max", "summary": "auto"},
    "service_tier": "default",
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
    run_id = "luna-codex-live-smoke"
    contract_json = json.dumps(CONTRACT, separators=(",", ":"))
    env = {
        "CODEX_HOME": "/tmp/codex-home",
        "SPRINT_RUN_ID": run_id,
        "SPRINT_MODEL": MODEL,
        "SPRINT_DURABLE_DIR": "/tmp/durable",
        "SPRINT_RUNTIME_DIR": "/tmp/runtime",
        "SPRINT_AGENT_LOG_DIR": "/tmp/logs/agent",
        "SPRINT_OPENROUTER_LEDGER_REQUIRED": "1",
        "SPRINT_OPENROUTER_PROVIDER_ENDPOINT": "openai",
        "SPRINT_OPENROUTER_ALLOWED_INFERENCE_PATH": "responses",
        "SPRINT_OPENROUTER_REQUEST_CONTRACT_JSON": contract_json,
        "SPRINT_CODEX_OPENAI_MODEL": "@preset/sprint-gpt-5-6-luna-openai-standard",
        "SPRINT_CODEX_OPENAI_MODEL_ID": "gpt-5.6-luna",
        "SPRINT_CODEX_OPENAI_MODEL_LOCK": "/opt/sprint-codex-luna-model-lock.json",
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
            tags={"sprint.role": "luna-codex-live-smoke"},
        )
        process = sandbox.exec(
            "bash",
            "-lc",
            "mkdir -p /app /tmp/durable/runs/"
            + run_id
            + "/state /tmp/runtime /tmp/logs/agent; "
            + "python3 -c "
            + shlex.quote(
                "import json,pathlib; "
                "pathlib.Path('/tmp/durable/runs/"
                + run_id
                + "/state/run.json').write_text(json.dumps({"
                + repr("run_id")
                + ":"
                + repr(run_id)
                + ","
                + repr("model")
                + ":"
                + repr(MODEL)
                + ","
                + repr("agent_cost_budget_usd")
                + ":10.0}))"
            )
            + "; "
            + "/opt/sprint-codex-exec-wrapper.sh \"$(command -v codex)\" "
            + "exec --dangerously-bypass-approvals-and-sandbox "
            + "--skip-git-repo-check --model gpt-5.6-luna --json -- "
            + shlex.quote("Reply exactly LIVE-SMOKE-OK. Do not call a tool.")
            + " </dev/null",
            env=env,
            timeout=720,
        )
        stdout = process.stdout.read() or ""
        stderr = process.stderr.read() or ""
        process.wait()
        if process.returncode != 0 or SUCCESS_MARKER not in stdout:
            log_process = sandbox.exec(
                "bash",
                "-lc",
                "tail -200 /tmp/logs/agent/openrouter-ledger-proxy.log 2>/dev/null || true",
                timeout=30,
            )
            proxy_log = log_process.stdout.read() or ""
            log_process.wait()
            raise RuntimeError(
                f"live probe exited {process.returncode}: "
                f"stderr={stderr[-2000:]} stdout={stdout[-4000:]} "
                f"proxy log: {proxy_log[-4000:]}"
            )
        trace_probe = sandbox.exec(
            "python3",
            "-c",
            "import json,pathlib; paths=list(pathlib.Path('/tmp/codex-home/sessions').rglob('*.jsonl')); "
            "assert paths; rows=[]; "
            "exec(\"for p in paths:\\n for line in p.read_text().splitlines():\\n  try: rows.append(json.loads(line))\\n  except json.JSONDecodeError: pass\"); "
            "items=[r.get('payload') for r in rows if r.get('type') == 'response_item']; "
            "summaries=[]; "
            "exec(\"for item in items:\\n if isinstance(item,dict) and item.get('type') == 'reasoning':\\n  for part in item.get('summary') or []:\\n   text=part.get('text') if isinstance(part,dict) else part\\n   if isinstance(text,str) and text.strip(): summaries.append(text)\"); "
            "print(json.dumps({'summary_count':len(summaries),'summary_chars':sum(map(len,summaries))}))",
            timeout=30,
        )
        trace_raw = trace_probe.stdout.read() or ""
        trace_probe.wait()
        if trace_probe.returncode != 0:
            raise RuntimeError("could not inspect the Luna reasoning summary trace")
        trace_reasoning = json.loads(trace_raw.strip().splitlines()[-1])
        inspect = sandbox.exec(
            "python3",
            "-c",
            "import json,pathlib; root=pathlib.Path('/tmp/durable/runs/"
            + run_id
            + "/api-usage'); records=sorted((root/'requests').glob('*.json')); "
            + "assert records; parsed=[json.loads(p.read_text()) for p in records]; "
            + "print(json.dumps({'records':parsed,'summary':json.loads((root/'summary.json').read_text())}))",
            timeout=30,
        )
        raw = inspect.stdout.read() or ""
        inspect.wait()
        if inspect.returncode != 0:
            raise RuntimeError("could not inspect the Luna smoke ledger")
        ledger = json.loads(raw.strip().splitlines()[-1])
        records = ledger["records"]
        record = records[-1]
        summary = ledger["summary"]
        snapshot = record.get("promotion_snapshot") or {}
        endpoints = snapshot.get("endpoints") or []
        endpoint = endpoints[0] if endpoints else {}
        checks = {
            "record_complete": record.get("state") == "complete",
            "responses_path": record.get("api_path") == "/api/v1/responses",
            "request_contract_sealed": record.get("request_contract") == CONTRACT,
            "official_provider_route": endpoint.get("tag") == "openai",
            "official_provider_name": endpoint.get("provider_name") == "OpenAI",
            "resolved_model_pinned": endpoint.get("model") in {None, RESOLVED_MODEL}
            and record.get("response_model") in {None, MODEL, RESOLVED_MODEL},
            "all_requests_complete": summary.get("completed_request_count")
            == len(records),
            "no_pending_request": summary.get("pending_request_count") == 0,
            "reasoning_summary_preserved": trace_reasoning["summary_count"] > 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"live smoke checks failed: {checks}")
        usage = record.get("usage") or {}
        report = {
            "schema_version": 1,
            "completed": True,
            "image_id": image_id,
            "sandbox_id": sandbox.object_id,
            "elapsed_s": round(time.monotonic() - started, 3),
            "checks": checks,
            "provider_name": endpoint.get("provider_name"),
            "provider_tag": endpoint.get("tag"),
            "response_model": record.get("response_model"),
            "provider_reported_cost_usd": record.get("provider_reported_cost_usd"),
            "benchmark_cost_usd": record.get("benchmark_cost_usd"),
            "input_tokens": usage.get("input_tokens"),
            "cached_tokens": (usage.get("input_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "reasoning_summary_count": trace_reasoning["summary_count"],
            "reasoning_summary_chars": trace_reasoning["summary_chars"],
        }
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        if sandbox is not None:
            sandbox.terminate(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
