#!/usr/bin/env python3
"""Run one tiny paid end-to-end DeepSeek Harness/OpenRouter smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap
import time
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
DEFAULT_REPORT = ROOT / "runs/ops/deepseek-harness-live-smoke.json"
APP_NAME = "sprint-image-warmup"
SUCCESS_MARKER = "DEEPSEEK_HARNESS_LIVE_OK"

REMOTE_PROGRAM = textwrap.dedent(
    r"""
    import json
    import os
    from pathlib import Path
    import subprocess
    import tempfile
    import time
    import urllib.request

    from deepseek_harness import DeepSeekHarness

    model = "deepseek/deepseek-v4-flash-vision-exp"
    contract = {
        "model": model,
        "stream": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 384000,
        "reasoning_effort": "max",
    }
    with tempfile.TemporaryDirectory(prefix="dsh-live-") as raw:
        root = Path(raw)
        (root / "state").mkdir()
        (root / "state/run.json").write_text(json.dumps({
            "run_id": "deepseek-harness-live-smoke",
            "model": model,
            "agent_cost_budget_usd": 10.0,
        }))
        runtime = root / "runtime"
        runtime.mkdir()
        proxy = subprocess.Popen(
            [
                "/opt/sprint-openrouter-ledger-proxy.py",
                "--upstream", "https://openrouter.ai/api/v1",
                "--ledger-root", str(root / "api-usage"),
                "--run-id", "deepseek-harness-live-smoke",
                "--cpu-attempt", "1",
                "--runtime-dir", str(runtime),
                "--provider-endpoint", "deepseek",
                "--request-contract-json", json.dumps(contract, separators=(",", ":")),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(300):
                if proxy.poll() is not None:
                    raise RuntimeError("ledger proxy exited during startup")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18080/healthz", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError("ledger proxy did not become ready")

            with DeepSeekHarness(
                provider="deepseek-official",
                model=model,
                max_tokens=384000,
                cwd=str(root),
                runtime_cwd=str(root),
                session_root=str(root / "sessions"),
                cordis="/opt/deepseek-harness-minimal.cordis.yml",
                runtime_bin="/usr/local/bin/dsh-jsonrpc-agent",
                base_url="http://127.0.0.1:18080/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                request_timeout_seconds=600.0,
                shutdown_timeout_seconds=30.0,
            ) as harness:
                result = harness.start_session("live-smoke").run(
                    "Reply exactly LIVE-SMOKE-OK. Do not call a tool."
                )
            if result.finish_reason != "completed" or not result.final_response:
                raise RuntimeError(
                    f"harness did not complete: {result.finish_reason!r}"
                )
        finally:
            proxy.terminate()
            try:
                proxy.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait(timeout=10)

        records = sorted((root / "api-usage/requests").glob("*.json"))
        if len(records) != 1:
            raise RuntimeError(f"expected one ledger record, found {len(records)}")
        record = json.loads(records[0].read_text())
        summary = json.loads((root / "api-usage/summary.json").read_text())
        snapshot = record.get("promotion_snapshot") or {}
        endpoints = snapshot.get("endpoints") or []
        checks = {
            "record_complete": record.get("state") == "complete",
            "chat_completions_path": record.get("api_path")
                == "/api/v1/chat/completions",
            "request_contract_sealed": record.get("request_contract") == contract,
            "official_provider_route": bool(endpoints)
                and endpoints[0].get("tag") == "deepseek",
            "peak_price_floor_present": bool(
                snapshot.get("deepseek_peak_pricing_usd_per_token")
            ),
            "usage_present": isinstance(record.get("usage"), dict),
            "one_completed_request": summary.get("completed_request_count") == 1,
            "no_pending_request": summary.get("pending_request_count") == 0,
            "benchmark_not_below_charge": float(record["benchmark_cost_usd"])
                >= float(record["provider_reported_cost_usd"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"live smoke checks failed: {checks}")
        usage = record["usage"]
        print("DEEPSEEK_HARNESS_LIVE_OK")
        print(json.dumps({
            "checks": checks,
            "provider_name": endpoints[0].get("provider_name"),
            "provider_tag": endpoints[0].get("tag"),
            "provider_reported_cost_usd": record["provider_reported_cost_usd"],
            "undiscounted_cost_usd": record["undiscounted_cost_usd"],
            "benchmark_cost_usd": record["benchmark_cost_usd"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        }, sort_keys=True))
    """
)


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
        process = sandbox.exec("python3", "-c", REMOTE_PROGRAM, timeout=720)
        stdout = process.stdout.read() or ""
        stderr = process.stderr.read() or ""
        process.wait()
        if process.returncode != 0 or SUCCESS_MARKER not in stdout:
            raise RuntimeError(
                f"live probe exited {process.returncode}: {(stderr or stdout)[-4000:]}"
            )
        safe = json.loads(stdout.strip().splitlines()[-1])
        report = {
            "schema_version": 1,
            "completed": True,
            "image_id": image_id,
            "sandbox_id": sandbox.object_id,
            "elapsed_s": round(time.monotonic() - started, 3),
            **safe,
        }
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        if sandbox is not None:
            sandbox.terminate(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
