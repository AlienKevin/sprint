#!/usr/bin/env python3
"""Force-kill GPU attempt 1 and prove automatic same-job recovery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable
import sys

ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = ROOT / "runs/ops"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPS_DIR))

from event_runtime.compute import worker as gpu_worker  # noqa: E402
import sprintctl  # noqa: E402
from smoke_cpu_gpu_split import agent_codex_alive  # noqa: E402


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wait_for(
    fn: Callable[[], tuple[bool, Any]],
    *,
    timeout: float,
    label: str,
    sleep: float = 5,
) -> Any:
    deadline = time.time() + timeout
    last: Any = None
    while time.time() < deadline:
        Path("/data/.keepalive").touch()
        try:
            ok, last = fn()
        except Exception as exc:  # noqa: BLE001
            ok = False
            last = f"{type(exc).__name__}: {exc}"
        if ok:
            return last
        time.sleep(sleep)
    raise TimeoutError(f"timeout waiting for {label}: {last!r}")


def enqueue_probe(run: dict[str, Any], container_id: str) -> str:
    probe = r"""
import json
import os
import pathlib
import time

import torch

attempt = int(os.environ["SPRINT_GPU_ATTEMPT"])
progress_path = pathlib.Path(os.environ["SPRINT_GPU_PROGRESS_FILE"])
checkpoint_dir = pathlib.Path(os.environ["SPRINT_GPU_CHECKPOINT_DIR"])
checkpoint_dir.mkdir(parents=True, exist_ok=True)
previous = {}
if progress_path.is_file():
    previous = json.loads(progress_path.read_text())

def durable_write(path, payload):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    os.sync()

assert torch.cuda.is_available()
x = torch.randn(512, 512, device="cuda")
assert torch.isfinite(x @ x).all().item()

if attempt == 1:
    checkpoint = checkpoint_dir / "checkpoint_1.json"
    durable_write(checkpoint, {"completed_step": 1, "attempt": attempt})
    durable_write(progress_path, {
        "step": 1,
        "attempt": attempt,
        "ready_for_kill": True,
    })
    print("ATTEMPT1_PROGRESS_WRITTEN", flush=True)
    while True:
        x = torch.randn(512, 512, device="cuda")
        _ = x @ x
        time.sleep(1)

assert attempt == 2, attempt
checkpoint = checkpoint_dir / "checkpoint_1.json"
saved = json.loads(checkpoint.read_text())
assert previous.get("step") == 1, previous
assert saved.get("completed_step") == 1, saved
final = {
    "step": 2,
    "attempt": attempt,
    "resumed_from_attempt": previous.get("attempt"),
    "checkpoint": str(checkpoint),
    "cuda_available": True,
}
durable_write(progress_path, final)
marker = checkpoint_dir.parent.parent / "recovery-smoke-marker.json"
durable_write(marker, final)
print("ATTEMPT2_RESUMED_AND_SUCCEEDED", json.dumps(final, sort_keys=True), flush=True)
"""
    shell = (
        "set -euo pipefail\n"
        "cat > /app/gpu_recovery_probe.py <<'PY'\n" + probe.strip() + "\nPY\n"
        "sprint-gpu-train --max-attempts 3 --retry-backoff 2 "
        "--retry-backoff-max 5 --heartbeat-interval 5 "
        "--heartbeat-timeout 15 --note forced-recovery-smoke -- "
        "python3 -u /app/gpu_recovery_probe.py\n"
    )
    result = sprintctl.run_command(
        sprintctl.modal_command(
            "container",
            "exec",
            "--no-pty",
            container_id,
            "--",
            "bash",
            "-lc",
            shell,
        ),
        run=run,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"probe enqueue failed: {(result.stdout or '')[-1000:]}"
            f"{(result.stderr or '')[-1000:]}"
        )
    for line in (result.stdout or "").splitlines():
        value = line.strip()
        if len(value) == 12 and all(ch in "0123456789abcdef" for ch in value):
            return value
    raise RuntimeError(f"job id missing from enqueue output: {result.stdout!r}")


def load_status(run: dict[str, Any], job_id: str) -> dict[str, Any]:
    payload = gpu_worker.load_job(run, job_id)
    return payload or {}


def write_report(path: Path, result: dict[str, Any]) -> None:
    evidence = result["evidence"]
    lines = [
        f"# Automatic GPU recovery smoke — `{result['run_id']}`",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        f"Written: {result['written_at']}",
        f"Logical job: `{result['job_id']}`",
        "",
        "## Evidence",
        "",
        f"- Attempt 1: `{evidence.get('attempt_1_sandbox')}` wrote progress step 1.",
        f"- Forced termination return code: `{evidence.get('forced_termination_code')}`.",
        f"- Host detection: `{evidence.get('detection')}`.",
        f"- Attempt 2: `{evidence.get('attempt_2_sandbox')}` resumed step "
        f"{evidence.get('resumed_from_step')} and ended `{evidence.get('final_status')}`.",
        f"- CPU Codex harness alive after kill: `{evidence.get('cpu_harness_alive')}`.",
        f"- Attempts in telemetry: `{evidence.get('telemetry_attempts')}`.",
        f"- Timeline active seconds: `{evidence.get('gpu_active_s')}`; "
        f"per-attempt rows: `{evidence.get('timeline_attempts')}`.",
        f"- Error: `{result.get('error')}`.",
        "",
        "No manual requeue was used. The smoke driver only called the normal host "
        "dispatch/monitor pass. It did not run a bakeoff.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    state_dir = Path(__file__).resolve().parent / args.run_id

    # The launcher runs in the background, so the smoke driver can start a few
    # milliseconds before run.json is published.  Treat that as normal launch
    # startup, not as an unknown-run failure.
    wait_for(
        lambda: (
            (state_dir / "run.json").is_file(),
            str(state_dir / "run.json"),
        ),
        timeout=args.timeout,
        label="run metadata",
        sleep=1,
    )
    state_dir, run = sprintctl.load_run(args.run_id)
    result: dict[str, Any] = {
        "run_id": args.run_id,
        "written_at": utc_now(),
        "verdict": "FAIL",
        "job_id": None,
        "evidence": {},
    }
    try:

        def ready() -> tuple[bool, Any]:
            _, current_run = sprintctl.load_run(args.run_id)
            container = sprintctl.discover_agent_container(state_dir, current_run)
            if not container:
                return False, "agent container missing"
            alive, output, codex_hint = agent_codex_alive(
                current_run, container, timeout=45
            )
            return alive, {
                "container": container,
                "output": output[-500:],
                "codex_hint": codex_hint,
            }

        cpu = wait_for(ready, timeout=args.timeout, label="CPU Codex harness", sleep=10)
        container = str(cpu["container"])
        _, run = sprintctl.load_run(args.run_id)
        job_id = enqueue_probe(run, container)
        result["job_id"] = job_id
        transitions: list[dict[str, Any]] = []

        def attempt_one_ready() -> tuple[bool, Any]:
            dispatch = gpu_worker.dispatch_once(args.run_id)
            job = load_status(run, job_id)
            heartbeat = gpu_worker.load_heartbeat(run, job) if job else None
            snapshot = {
                "attempt": job.get("attempt"),
                "status": job.get("status"),
                "sandbox_id": job.get("sandbox_id"),
                "heartbeat": heartbeat,
                "dispatch": dispatch,
            }
            transitions.append(snapshot)
            progress = (heartbeat or {}).get("progress") or {}
            return (
                int(job.get("attempt") or 0) == 1
                and str(job.get("status") or "") == "running"
                and progress.get("step") == 1,
                snapshot,
            )

        first = wait_for(
            attempt_one_ready,
            timeout=args.timeout,
            label="attempt 1 durable progress",
        )
        first_sandbox = str(first["sandbox_id"])
        import modal

        kill_code = modal.Sandbox.from_id(first_sandbox).terminate(wait=True)
        killed_at = utc_now()

        def recovered() -> tuple[bool, Any]:
            dispatch = gpu_worker.dispatch_once(args.run_id)
            job = load_status(run, job_id)
            snapshot = {
                "attempt": job.get("attempt"),
                "status": job.get("status"),
                "sandbox_id": job.get("sandbox_id"),
                "dispatch": dispatch,
            }
            transitions.append(snapshot)
            return (
                int(job.get("attempt") or 0) == 2
                and str(job.get("status") or "") == "succeeded",
                snapshot,
            )

        final = wait_for(
            recovered,
            timeout=args.timeout,
            label="automatic replacement success",
        )
        marker_text = sprintctl.volume_get_text(
            run, f"runs/{args.run_id}/gpu-jobs/recovery-smoke-marker.json"
        )
        marker = json.loads(marker_text or "{}")
        alive, cpu_output, codex_hint = agent_codex_alive(run, container, timeout=45)
        telemetry = (
            sprintctl.volume_get_text(
                run, f"runs/{args.run_id}/telemetry/by-job/{job_id}/samples.jsonl"
            )
            or ""
        )
        telemetry_attempts = sorted(
            {
                int(row.get("attempt"))
                for line in telemetry.splitlines()
                if line.strip()
                for row in [json.loads(line)]
                if row.get("attempt") is not None
            }
        )
        from event_runtime.telemetry import timeline as gpu_timeline_host

        summary = gpu_timeline_host.host_write_summary(run)
        timeline_attempts = sorted(
            {
                int(item.get("attempt") or 0)
                for item in summary.get("per_attempt") or []
                if item.get("job_id") == job_id
            }
        )
        detection = [
            item["dispatch"].get("reconciled")
            for item in transitions
            if item.get("dispatch", {}).get("reconciled")
        ]
        passed = bool(
            marker.get("step") == 2
            and marker.get("resumed_from_attempt") == 1
            and alive
            and codex_hint
            and {1, 2}.issubset(set(telemetry_attempts))
            and {1, 2}.issubset(set(timeline_attempts))
            and final.get("status") == "succeeded"
        )
        result.update(
            {
                "written_at": utc_now(),
                "verdict": "PASS" if passed else "FAIL",
                "evidence": {
                    "attempt_1_sandbox": first_sandbox,
                    "attempt_2_sandbox": final.get("sandbox_id"),
                    "forced_termination_code": kill_code,
                    "forced_termination_at": killed_at,
                    "detection": detection,
                    "resumed_from_step": marker.get("resumed_from_attempt"),
                    "final_status": final.get("status"),
                    "cpu_harness_alive": alive,
                    "cpu_output": cpu_output[-500:],
                    "telemetry_attempts": telemetry_attempts,
                    "timeline_attempts": timeline_attempts,
                    "gpu_active_s": summary.get("gpu_active_s"),
                    "marker": marker,
                    "transitions": transitions,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            result["stop"] = sprintctl.request_stop(args.run_id)
        except Exception as exc:  # noqa: BLE001
            result["stop_error"] = f"{type(exc).__name__}: {exc}"
        result["written_at"] = utc_now()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_report(args.report, result)
        sprintctl.atomic_write_json(
            state_dir / "gpu-recovery-smoke-result.json", result, mode=0o600
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
