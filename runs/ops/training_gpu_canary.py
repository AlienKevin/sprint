#!/usr/bin/env python3
"""Run real Isaac Lab PPO iterations in the sealed training image."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

from warm_modal_images import APP_NAME, MANIFEST, VOLUME_NAME, run_sandbox


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "runs/ops/training-gpu-canary.json"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import agent_cost  # noqa: E402
import modal_cost  # noqa: E402


CANARY_RESOURCE_CONTRACT = {
    "cpu_agent": {
        "physical_cpu_cores": 2,
        "memory_mb": 8192,
        "gpu_count": 0,
    },
    "training_worker": {
        "physical_cpu_cores": 6,
        "memory_mb": 12288,
        "gpu_count": 1,
        "gpu_type": "A10G",
    },
    "verifier": {
        "physical_cpu_cores": 4,
        "memory_mb": 10240,
        "gpu_count": 1,
        "gpu_type": "A10G",
    },
}


def atomic_write(payload: dict) -> None:
    tmp = REPORT.with_name(f".{REPORT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, REPORT)


def _read_process(process: Any) -> tuple[int, str, str]:
    return_code = process.wait()
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return return_code, stdout, stderr


def _assert_cost_match(
    agent_payload: dict[str, Any],
    *,
    expected: dict[str, Any],
    tolerance_usd: float = 1e-9,
) -> dict[str, Any]:
    """Fail closed unless every comparison component matches host measurement."""
    comparisons = {
        "model_api_usd": (
            agent_payload["components"]["model_api"]["cost_usd"],
            expected["model_api_usd"],
        ),
        "cpu_agent_usd": (
            agent_payload["components"]["cpu_agent"]["cost_usd"],
            expected["cpu_agent_usd"],
        ),
        "training_sandboxes_usd": (
            agent_payload["components"]["training_sandboxes"]["cost_usd"],
            expected["training_sandboxes_usd"],
        ),
        "total_usd": (agent_payload["total_usd"], expected["total_usd"]),
    }
    deltas = {
        name: abs(float(observed) - float(target))
        for name, (observed, target) in comparisons.items()
    }
    if any(delta > tolerance_usd for delta in deltas.values()):
        raise RuntimeError(
            "sprint-cost differs from independent host measurement: "
            + json.dumps(deltas, sort_keys=True)
        )
    return {
        "verified": True,
        "tolerance_usd": tolerance_usd,
        "max_absolute_delta_usd": max(deltas.values(), default=0.0),
        "comparisons": {
            name: {
                "agent_usd": float(observed),
                "host_measured_usd": float(target),
                "absolute_delta_usd": deltas[name],
            }
            for name, (observed, target) in comparisons.items()
        },
    }


def run_cost_equivalence_canary(
    *, app: modal.App, image: modal.Image, training_allocated_s: float
) -> dict[str, Any]:
    """Query sprint-cost in the exact agent image and compare at one cutoff.

    CPU allocation time is measured around a real Modal sandbox. Training time
    comes from the real functional training sandbox that just produced the
    canary policy. The host independently applies the pinned tariff, while the
    agent sees only the normal sprint-cost JSON document.
    """
    sandbox: modal.Sandbox | None = None
    allocation_started = time.monotonic()
    try:
        sandbox = modal.Sandbox.create(
            "python3",
            "-c",
            "import time; time.sleep(300)",
            app=app,
            image=image,
            cpu=2,
            memory=8192,
            block_network=True,
            timeout=360,
            tags={"sprint.role": "cost-equivalence-canary", "sprint.warmup": "true"},
        )
        cutoff_epoch_ms = int(time.time() * 1000)
        cpu_allocated_ms = max(1, int((time.monotonic() - allocation_started) * 1000))
        training_allocated_ms = max(1, int(round(training_allocated_s * 1000)))
        modal_estimate = modal_cost.estimate_cost(
            resource_contract=CANARY_RESOURCE_CONTRACT,
            allocated_ms_by_role={
                "cpu_agent": cpu_allocated_ms,
                "training_gpu": training_allocated_ms,
                "verifier_gpu": 0,
            },
        )
        timeline = {
            "schema_version": 6,
            "generated_at": dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "run": {
                "run_id": "training-functional-cost-canary",
                "model": "none",
            },
            "clock": {
                "origin_epoch_ms": cutoff_epoch_ms - cpu_allocated_ms,
                "end_epoch_ms": cutoff_epoch_ms,
            },
            "events": [
                {
                    "kind": "cpu_allocated",
                    "epoch_ms": cutoff_epoch_ms - cpu_allocated_ms,
                }
            ],
            "usage_summary": {
                "request_count": 0,
                "ordinary_uncached_input_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
            },
            "resource_usage_summary": {
                "resource_contract": CANARY_RESOURCE_CONTRACT,
                "modal_estimate": modal_estimate,
            },
        }
        snapshot = agent_cost.build_snapshot(timeline)
        encoded = base64.b64encode(
            (json.dumps(snapshot, sort_keys=True) + "\n").encode()
        ).decode("ascii")
        install = sandbox.exec(
            "python3",
            "-c",
            (
                "import base64,os,pathlib,sys; "
                "p=pathlib.Path('/run/sprint-gpu-mirror/cost.json'); "
                "p.parent.mkdir(parents=True,exist_ok=True); "
                "q=p.with_name('.cost.canary.tmp'); "
                "q.write_bytes(base64.b64decode(sys.argv[1])); "
                "os.chmod(q,0o600); os.replace(q,p)"
            ),
            encoded,
            timeout=30,
        )
        return_code, stdout, stderr = _read_process(install)
        if return_code != 0:
            raise RuntimeError(
                f"cost canary snapshot install failed {return_code}: {stderr or stdout}"
            )
        query = sandbox.exec("sprint-cost", timeout=30)
        return_code, stdout, stderr = _read_process(query)
        if return_code != 0:
            raise RuntimeError(
                f"sprint-cost canary query failed {return_code}: {stderr or stdout}"
            )
        agent_payload = json.loads(stdout)

        rates = modal_cost.MODAL_SANDBOX_PRICING["rates_usd_per_second"]
        cpu_agent_usd = (cpu_allocated_ms / 1000.0) * (
            2 * float(rates["CPU"]) + 8 * float(rates["Memory"])
        )
        training_usd = (training_allocated_ms / 1000.0) * (
            6 * float(rates["CPU"]) + 12 * float(rates["Memory"]) + float(rates["A10G"])
        )
        expected = {
            "model_api_usd": 0.0,
            "cpu_agent_usd": cpu_agent_usd,
            "training_sandboxes_usd": training_usd,
            "total_usd": cpu_agent_usd + training_usd,
        }
        comparison = _assert_cost_match(agent_payload, expected=expected)
        return {
            "completed": True,
            "sandbox_id": sandbox.object_id,
            "snapshot_as_of_epoch_ms": cutoff_epoch_ms,
            "measured_allocated_ms": {
                "cpu_agent": cpu_allocated_ms,
                "training_gpu": training_allocated_ms,
            },
            "agent_total_usd": agent_payload["total_usd"],
            "host_measured_total_usd": expected["total_usd"],
            "comparison": comparison,
            "cost_basis": agent_payload["cost_basis"],
            "excluded": agent_payload["excluded"],
        }
    finally:
        if sandbox is not None:
            sandbox.terminate(wait=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.work_archive.resolve()
    if not archive.is_file():
        parser.error(f"work archive not found: {archive}")

    warmup = json.loads(MANIFEST.read_text())
    if not warmup.get("completed"):
        raise RuntimeError("Modal image warm-up is not complete")
    image_id = str(warmup["contexts"]["agent_training"]["image_id"])
    verifier_image_id = str(warmup["contexts"]["verifier"]["image_id"])
    canary_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote_root = f"/training-canary/{canary_id}"

    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with volume.batch_upload(force=True) as upload:
        upload.put_file(archive, f"{remote_root}/app.tar.gz", mode=0o444)

    command = (
        "set -euo pipefail; "
        f"tar -xzf /warm{remote_root}/app.tar.gz -C /; "
        f"mkdir -p /warm{remote_root}/checkpoints /warm{remote_root}/logs "
        f"/warm{remote_root}/telemetry; "
        "SPRINT_REQUESTED_CPU_CORES=6 SPRINT_REQUESTED_MEMORY_MIB=12288 "
        "python3 /opt/sprint-telemetry.py --role training-gpu "
        f"--run-id training-canary --interval-seconds 5 --out-dir /warm{remote_root}/telemetry "
        "--durable-dir /nonexistent --force & telemetry_pid=$!; "
        "trap 'kill $telemetry_pid 2>/dev/null || true' EXIT; "
        f"export SPRINT_GPU_CHECKPOINT_DIR=/warm{remote_root}/checkpoints; "
        f"export SPRINT_GPU_PROGRESS_FILE=/warm{remote_root}/progress.json; "
        "timeout --signal=TERM --kill-after=30 900 "
        "python3 /opt/sprint-isaac-bootstrap.py /app/train_sprint.py "
        "--headless --device cuda:0 --num_envs 128 --max_iters 10 "
        "--chunk_iters 1 --save_interval 1 --episode_length 2 "
        f"--exp_name sealed_training_canary --log_root /warm{remote_root}/logs "
        f"2>&1 | tee /warm{remote_root}/training.log; "
        "kill $telemetry_pid 2>/dev/null || true; wait $telemetry_pid 2>/dev/null || true; "
        f"test -s /warm{remote_root}/checkpoints/policy_final.pt; "
        f"python3 -c \"import json; p=json.load(open('/warm{remote_root}/progress.json')); "
        "assert p['finished'] is True and p['iteration'] >= 10\"; "
        f"grep -F '[sprint] iter=10' /warm{remote_root}/training.log; "
        f'python3 -c "import json; rows=[json.loads(x) for x in '
        f"open('/warm{remote_root}/telemetry/samples.jsonl') if x.strip()]; "
        "gpus=[g for p in rows for g in p['gpus']]; "
        "assert gpus and all(g['gpu_name'] for g in gpus); "
        "assert max(g['mem_used_mib'] for g in gpus) > 1000; "
        "assert max(g['util_gpu_pct'] for g in gpus) >= 10\"; "
        f"mkdir -p /warm{remote_root}/agent-verifier; "
        "timeout --signal=TERM --kill-after=30 180 "
        "python3 /opt/sprint-verifier/verify.py "
        f"--policy /warm{remote_root}/checkpoints/policy_final.pt "
        f"--logs /warm{remote_root}/agent-verifier "
        "--tests /opt/sprint-verifier --runs 1 --distance 1 --max-seconds 2 "
        "--headless; "
        "echo AGENT_PUBLISHED_VERIFIER_COMPLETED"
    )

    verifier_command = (
        "set -euo pipefail; "
        f"mkdir -p /warm{remote_root}/verifier/telemetry; "
        "SPRINT_REQUESTED_CPU_CORES=8 SPRINT_REQUESTED_MEMORY_MIB=32768 "
        "python3 /tests/verifier_telemetry.py "
        f"--out-dir /warm{remote_root}/verifier/telemetry --interval-seconds 2 & "
        "telemetry_pid=$!; trap 'kill $telemetry_pid 2>/dev/null || true' EXIT; "
        "timeout --signal=TERM --kill-after=30 180 "
        f"python3 /tests/verify.py --policy /warm{remote_root}/checkpoints/policy_final.pt "
        f"--logs /warm{remote_root}/verifier --tests /tests --runs 1 "
        "--distance 1 --max-seconds 2 --headless "
        f"2>&1 | tee /warm{remote_root}/verifier/isaac-stdout.txt; "
        "kill $telemetry_pid 2>/dev/null || true; wait $telemetry_pid 2>/dev/null || true; "
        f"test -s /warm{remote_root}/verifier/sprint_results.json; "
        f"test -s /warm{remote_root}/verifier/lanes.json; "
        f"test -s /warm{remote_root}/verifier/replay.json; "
        f"! grep -Eq 'GPU solver pipeline failed|GPU Bp pipeline failed|switching to software' "
        f"/warm{remote_root}/verifier/isaac-stdout.txt; "
        f'python3 -c "import json; rows=[json.loads(x) for x in '
        f"open('/warm{remote_root}/verifier/telemetry/samples.jsonl') if x.strip()]; "
        "gpus=[g for p in rows for g in p['gpus']]; "
        "assert gpus and all(g['gpu_name'] for g in gpus); "
        f"result=json.load(open('/warm{remote_root}/verifier/sprint_results.json')); "
        "assert result['runs'] == 1\"; "
        "python3 /tests/compare_results.py "
        f"--agent /warm{remote_root}/agent-verifier/sprint_results.json "
        f"--official /warm{remote_root}/verifier/sprint_results.json "
        f"--out /warm{remote_root}/verifier-equivalence.json; "
        "echo SEALED_GENERATED_POLICY_VERIFIED"
    )

    started = time.time()
    report = {
        "schema_version": 4,
        "completed": False,
        "canary_id": canary_id,
        "image_id": image_id,
        "verifier_image_id": verifier_image_id,
        "work_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "remote_root": remote_root,
        "started_at_epoch_s": started,
    }
    try:
        with modal.enable_output():
            report["training_sandbox"] = run_sandbox(
                app=app,
                image=modal.Image.from_id(image_id),
                role="training-gpu-functional-canary",
                gpu="A10G",
                volume=volume,
                command=command,
                timeout=1200,
                required_output_substrings=(
                    "[sprint] iter=10",
                    "AGENT_PUBLISHED_VERIFIER_COMPLETED",
                ),
            )
            report["cost_equivalence"] = run_cost_equivalence_canary(
                app=app,
                image=modal.Image.from_id(image_id),
                training_allocated_s=float(report["training_sandbox"]["total_s"]),
            )
            report["verifier_sandbox"] = run_sandbox(
                app=app,
                image=modal.Image.from_id(verifier_image_id),
                role="generated-policy-sealed-verifier-canary",
                gpu="A10G",
                volume=volume,
                command=verifier_command,
                timeout=300,
                cpu=4,
                memory=10240,
                required_output_substrings=(
                    "SEALED_GENERATED_POLICY_VERIFIED",
                    "VERIFIER_EQUIVALENCE_OK",
                ),
            )
        report["verifier_equivalence_verified"] = True
        report["cost_equivalence_verified"] = bool(
            report["cost_equivalence"]["comparison"]["verified"]
        )
        report["full_path_verified"] = True
        report["completed"] = True
        report["completed_at_epoch_s"] = time.time()
        report["elapsed_s"] = round(time.time() - started, 3)
        return 0
    finally:
        atomic_write(report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
