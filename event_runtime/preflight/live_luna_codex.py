#!/usr/bin/env python3
"""Run one tiny paid end-to-end Luna/Codex/OpenRouter smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

import modal

from event_runtime.control.openrouter_credentials import (
    OpenRouterManagementClient,
    TrialCredentialSpec,
    provision_trial_credentials,
    revoke_trial_credentials,
)
from event_runtime.preflight.warm_images import stop_warmup_app


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
DEFAULT_REPORT = ROOT / "runs/ops/luna-codex-live-smoke.json"
GOAL_BOOTSTRAP = (
    ROOT / "harbor/src/harbor/agents/installed/codex_goal_bootstrap.py"
)
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
GOAL_OBJECTIVE = """Exercise persistent goal continuation in one sandbox.
Use exec_command to inspect /tmp/luna-goal-canary-turn. If it does not exist,
create it with the text 1, respond with CANARY-FIRST-TURN, and leave this goal
active. If it already exists, mark this goal complete with update_goal and make
the final response include LIVE-SMOKE-OK. Do not perform any other work."""


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
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY")
    if not management_key:
        parser.error("OPENROUTER_MANAGEMENT_KEY is required")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("completed") is not True:
        parser.error("the immutable image warmup is not complete")
    image_id = manifest["contexts"]["agent_training"]["image_id"]
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandbox: modal.Sandbox | None = None
    credential = None
    started = time.monotonic()
    run_id = "luna-codex-live-smoke"
    client = OpenRouterManagementClient(management_key)
    credential_journal = args.report.with_name("luna-codex-live-smoke-key.json")
    credential = provision_trial_credentials(
        client,
        [
            TrialCredentialSpec(
                run_id=run_id,
                model=MODEL,
                resolved_model=RESOLVED_MODEL,
                provider="openai",
                budget_usd=1.0,
            )
        ],
        journal_path=credential_journal,
        lifetime_hours=1,
    )[0]
    key = credential.api_key
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
    launch_contract = subprocess.run(
        [
            "bash",
            str(ROOT / "event_runtime/control/launch.sh"),
            "--dry-run",
            "--run-id",
            run_id,
            "--agent-kind",
            "codex",
            "--model",
            MODEL,
            "--endpoint",
            "https://openrouter.ai/api/v1",
            "--reasoning-effort",
            "max",
        ],
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    run = json.loads(launch_contract.stdout)
    run["standing_gpu_worker"] = False
    try:
        smoke_image = modal.Image.from_id(image_id).add_local_file(
            GOAL_BOOTSTRAP,
            "/opt/sprint-codex-goal-bootstrap.py",
            copy=True,
        )
        sandbox = modal.Sandbox.create(
            "python3",
            "-c",
            "import time; time.sleep(900)",
            app=app,
            image=smoke_image,
            cpu=2,
            memory=8192,
            timeout=900,
            block_network=False,
            outbound_domain_allowlist=["openrouter.ai"],
            secrets=[modal.Secret.from_dict({"OPENROUTER_API_KEY": key})],
            tags={"sprint.role": "luna-codex-live-smoke"},
        )
        shell_command = (
            "mkdir -p /app /tmp/durable/runs/"
            + run_id
            + "/state /tmp/runtime /tmp/logs/agent; "
            + "python3 -c "
            + shlex.quote(
                "import json,pathlib; p=pathlib.Path('/tmp/durable/runs/"
                + run_id
                + "/state/run.json'); p.parent.mkdir(parents=True, exist_ok=True); "
                + "p.write_text(json.dumps("
                + repr(run)
                + "))"
            )
            + "; "
            + "bash /opt/sprint-apply-openai-codex-config.sh >/dev/null; "
            + "export SPRINT_CODEX_GOAL_OBJECTIVE="
            + shlex.quote(GOAL_OBJECTIVE)
            + "; "
            + "sprint_codex_thread_id=$(python3 /opt/sprint-codex-goal-bootstrap.py "
            + "--model gpt-5.6-luna --cwd /app "
            + "--receipt /tmp/logs/agent/goal-bootstrap.json "
            + "--expected-provider sprint_openrouter); "
            + "export SPRINT_CODEX_GOAL_PERSIST=1; "
            + 'export SPRINT_CODEX_GOAL_THREAD_ID="$sprint_codex_thread_id"; '
            + "export SPRINT_CODEX_GOAL_RECEIPT=/tmp/logs/agent/goal-bootstrap.json; "
            + "unset SPRINT_CODEX_GOAL_OBJECTIVE; "
            + "/opt/sprint-codex-exec-wrapper.sh \"$(command -v codex)\" "
            + "exec resume --dangerously-bypass-approvals-and-sandbox "
            + "--skip-git-repo-check --model gpt-5.6-luna --json "
            + '"$sprint_codex_thread_id" -- '
            + shlex.quote(GOAL_OBJECTIVE)
            + " </dev/null"
        )
        process = sandbox.exec(
            "bash",
            "-lc",
            shell_command,
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
            "persistent_goal_crossed_turn_boundary": len(records) >= 2,
            "first_turn_marker_preserved": "CANARY-FIRST-TURN" in stdout,
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
        if credential is not None:
            errors = revoke_trial_credentials(
                client,
                [credential.public_metadata()],
                journal_path=credential_journal,
            )
            if errors:
                raise RuntimeError("; ".join(errors))
        stop_warmup_app(required=False)


if __name__ == "__main__":
    raise SystemExit(main())
