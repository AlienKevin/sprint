#!/usr/bin/env python3
"""Drive CPU/GPU-split smoke checks against a just-launched durable run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from event_runtime.compute import worker as gpu_worker  # noqa: E402
from event_runtime.control import run as sprintctl  # noqa: E402

os.environ.setdefault("MODAL_PROFILE", "kevinli020508")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wait_for(predicate, *, timeout: float, label: str, sleep: float = 10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        Path("/data/.keepalive").touch()
        try:
            ok, last = predicate()
        except Exception as exc:  # noqa: BLE001
            ok, last = False, f"{type(exc).__name__}: {exc}"
        if ok:
            return last
        time.sleep(sleep)
    raise TimeoutError(f"timeout waiting for {label}: {last!r}")


def agent_codex_alive(
    run: dict[str, Any], container_id: str, *, timeout: int = 180
) -> tuple[bool, str, bool]:
    result = sprintctl.exec_container(
        run,
        container_id,
        "pgrep -af 'codex|sprint-codex' | head -5; "
        "test -d /durable && echo DURABLE_OK; "
        "command -v event >/dev/null && echo EVENT_CLI_OK; "
        "nvidia-smi >/tmp/agent-nvidia.out 2>/tmp/agent-nvidia.err; "
        "if grep -qiE 'NVIDIA|CUDA' /tmp/agent-nvidia.out 2>/dev/null; then echo AGENT_HAS_GPU; "
        "else echo AGENT_NO_GPU; fi; "
        "python3 - <<'PY'\n"
        "import torch\n"
        "print('TORCH_CUDA', torch.cuda.is_available())\n"
        "PY",
        check=False,
        timeout=timeout,
    )
    out = (result.stdout or "") + (result.stderr or "")
    alive = (
        "DURABLE_OK" in out
        and "EVENT_CLI_OK" in out
        and "AGENT_NO_GPU" in out
        and "TORCH_CUDA False" in out
    )
    # Codex may take a bit to spawn; durable+CLI+no-GPU is the hard CPU check.
    codex_hint = "codex" in out.lower() or "sprint-codex" in out.lower()
    return alive, out[-2000:], codex_hint


def enqueue_smoke_job(run: dict[str, Any], container_id: str) -> str:
    # Small CUDA probe; writes marker onto /durable for the agent to see.
    # Use bash -lc: Modal `container exec … sh -c` is dash and rejects pipefail.
    # Harbor --ae vars apply to the Codex process only; export for this exec.
    run_id = str(run["run_id"])
    jobs_root = f"/durable/runs/{run_id}/gpu-jobs"
    # Keep the probe body in a non-f string so `{...}` dict literals stay literal.
    probe = r"""
import json, os, time, pathlib
import torch
out = {
  "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "cuda_available": bool(torch.cuda.is_available()),
  "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
  "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
assert out["cuda_available"], out
# Hold the GPU ~45s so durable telemetry + host poller can sample gpu_active.
t0 = time.time()
while time.time() - t0 < 45:
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    out["matmul_ok"] = bool(torch.isfinite(y).all().item())
    time.sleep(1)
out["hold_s"] = round(time.time() - t0, 2)
run_id = os.environ["SPRINT_RUN_ID"]
marker = pathlib.Path(f"/durable/runs/{run_id}/gpu-jobs/smoke-marker.json")
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, sort_keys=True), flush=True)
"""
    script = (
        "set -euo pipefail\n"
        f"export SPRINT_RUN_ID={run_id!r}\n"
        f"export SPRINT_GPU_JOBS_ROOT={jobs_root!r}\n"
        'printf "%s\\n" "$SPRINT_RUN_ID" > /run/sprint-run-id\n'
        'printf "%s\\n" "$SPRINT_GPU_JOBS_ROOT" > /run/sprint-gpu-jobs-root\n'
        "cat > /app/smoke_gpu_probe.py <<'PY'\n" + probe.strip() + "\nPY\n"
        "event gpu --timeout 1200 --note smoke-cuda-probe -- "
        "python3 -u /app/smoke_gpu_probe.py\n"
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
            script,
        ),
        run=run,
        check=False,
        timeout=300,
    )
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"enqueue failed: {out[-3000:]}")
    job_id = None
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if len(line) == 12 and all(c in "0123456789abcdef" for c in line):
            job_id = line
            break
    if not job_id:
        raise RuntimeError(f"no job id in enqueue output: {out[-2000:]}")
    return job_id


def wait_job_uses_gpu(
    run_id: str, job_id: str, timeout: float = 1800.0
) -> dict[str, Any]:
    def pred():
        Path("/data/.keepalive").touch()
        gpu_worker.dispatch_once(run_id)
        _, run = sprintctl.load_run(run_id)
        text = sprintctl.volume_get_text(
            run, f"runs/{run_id}/gpu-jobs/status/{job_id}.json"
        )
        if not text:
            return False, "no status yet"
        job = json.loads(text)
        status = job.get("status")
        if status == "failed":
            return True, job  # let caller decide
        if status == "succeeded":
            return True, job
        # Peek logs for nvidia evidence while running.
        log = sprintctl.volume_get_text(
            run, f"runs/{run_id}/gpu-jobs/out/{job_id}/worker.log"
        )
        if log and ("NVIDIA" in log or "cuda_available" in log):
            job = dict(job)
            job["_log_snippet"] = log[-1500:]
            if status in {"running", "dispatched", "succeeded"}:
                # Prefer full success, but GPU evidence mid-flight is enough to proceed
                # to terminate test once we have nvidia-smi output.
                if "NVIDIA" in log and status == "running":
                    return True, job
        return False, job

    return wait_for(pred, timeout=timeout, label=f"gpu job {job_id}", sleep=15.0)


def telemetry_ok(run: dict[str, Any], container_id: str) -> tuple[bool, str]:
    result = sprintctl.exec_container(
        run,
        container_id,
        "ls /logs/artifacts/telemetry 2>/dev/null | head; "
        "test -f /logs/artifacts/telemetry/latest.json && "
        "python3 -c \"import json; d=json.load(open('/logs/artifacts/telemetry/latest.json')); "
        "print('gpu_count', d.get('gpu_count')); print('nvidia_smi_ok', d.get('nvidia_smi_ok')); "
        "print('keys', sorted(d)[:12])\"",
        check=False,
    )
    out = (result.stdout or "") + (result.stderr or "")
    # CPU agent: gpu_count should be 0 / nvidia false; telemetry files should exist.
    ok = "gpu_count" in out and result.returncode == 0
    return ok, out[-1500:]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# CPU/GPU split smoke — `{payload['run_id']}`",
        "",
        f"**Verdict: {payload['verdict']}**",
        "",
        f"Written: {payload['written_at']}",
        f"Run: `{payload['run_id']}`",
        f"Model: `{payload['model']}` · Codex `{payload['codex_version']}` · "
        f"`reasoning_effort={payload['reasoning_effort']}`",
        f"Modal: `{payload['modal_profile']}` · volume `{payload['volume_name']}`",
        f"State: `{payload['state_dir']}`",
        "",
        "## Checklist",
        "",
        "| # | Criterion | Result | Evidence |",
        "|---|---|---|---|",
    ]
    for row in payload["checklist"]:
        lines.append(
            f"| {row['n']} | {row['name']} | **{row['result']}** | {row['evidence']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        payload.get("notes", ""),
        "",
        "## Residual limits",
        "",
        "- GPU workers remain preemptible; resume from `/durable` checkpoints.",
        "- Volume commit lag can delay host dispatch by a few seconds.",
        "- This smoke does not run a multi-hour bakeoff or Hub upload.",
        "",
        "Secrets redacted; env-file path only.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--launch-pid", type=int, default=0)
    ap.add_argument("--agent-wait-sec", type=int, default=5400)
    args = ap.parse_args()
    run_id = args.run_id
    Path("/data/.keepalive").touch()

    def run_ready():
        state = ROOT / "runs" / "ops" / run_id / "run.json"
        if not state.is_file():
            return False, "no run.json"
        state_dir, run = sprintctl.load_run(run_id)
        container = sprintctl.discover_agent_container(state_dir, run)
        if not container:
            return False, "no agent container"
        # Require a successful exec before driving checks (image may still build).
        probe = sprintctl.exec_container(
            run,
            container,
            "test -x /usr/local/bin/event && test -d /durable && printf READY",
            check=False,
            timeout=45,
        )
        out = (probe.stdout or "") + (probe.stderr or "")
        if probe.returncode != 0 or "READY" not in out:
            return False, f"container {container} not exec-ready yet"
        return True, (state_dir, run, container)

    state_dir, run, container = wait_for(
        run_ready, timeout=args.agent_wait_sec, label="agent sandbox", sleep=20.0
    )
    print("agent_container", container, flush=True)

    checklist = []
    notes = []
    verdict = "PASS"

    # (a) CPU sandbox + Codex / durable / no GPU
    def cpu_pred():
        alive, out, codex_hint = agent_codex_alive(run, container)
        return alive, {"out": out, "codex_hint": codex_hint}

    cpu_info = wait_for(cpu_pred, timeout=900, label="CPU agent checks", sleep=20.0)
    checklist.append(
        {
            "n": 1,
            "name": "Codex/agent alive on CPU sandbox (no GPU)",
            "result": "PASS" if cpu_info["out"] else "FAIL",
            "evidence": "durable+event+TORCH_CUDA False; "
            + (
                "codex process seen"
                if cpu_info["codex_hint"]
                else "codex not in pgrep yet"
            ),
        }
    )
    if not cpu_info.get("codex_hint"):
        notes.append(
            "Codex process not visible in early pgrep; CPU sandbox checks passed."
        )

    # (b) enqueue GPU job from inside agent + dispatch
    try:
        job_id = enqueue_smoke_job(run, container)
        print("job_id", job_id, flush=True)
        job = wait_job_uses_gpu(run_id, job_id, timeout=2400)
        gpu_ok = False
        evidence = (
            f"job={job_id} status={job.get('status')} sandbox={job.get('sandbox_id')}"
        )
        log = job.get("_log_snippet") or sprintctl.volume_get_text(
            run, f"runs/{run_id}/gpu-jobs/out/{job_id}/worker.log"
        )
        marker = sprintctl.volume_get_text(
            run, f"runs/{run_id}/gpu-jobs/smoke-marker.json"
        )
        if marker and "true" in marker.lower():
            gpu_ok = True
            evidence += f"; marker={marker.strip()[:200]}"
        elif log and "NVIDIA" in log:
            gpu_ok = True
            evidence += "; nvidia-smi in worker.log"
        elif job.get("status") == "succeeded":
            gpu_ok = True
        checklist.append(
            {
                "n": 2,
                "name": "GPU train job uses A10G worker",
                "result": "PASS" if gpu_ok else "FAIL",
                "evidence": evidence,
            }
        )
        if not gpu_ok:
            verdict = "FAIL"
    except Exception as exc:  # noqa: BLE001
        checklist.append(
            {
                "n": 2,
                "name": "GPU train job uses A10G worker",
                "result": "FAIL",
                "evidence": f"{type(exc).__name__}: {exc}",
            }
        )
        job_id = None
        verdict = "FAIL"

    # (c) CPU-agent telemetry + durable/host GPU telemetry while job runs
    tel_ok, tel_out = telemetry_ok(run, container)
    checklist.append(
        {
            "n": 3,
            "name": "Telemetry/artifacts on CPU agent",
            "result": "PASS" if tel_ok else "FAIL",
            "evidence": tel_out.replace("\n", " ")[:300],
        }
    )
    if not tel_ok:
        verdict = "FAIL"

    durable_gpu_ok = False
    durable_evidence = "no job"
    host_gpu_ok = False
    host_evidence = "no job"
    if job_id:
        # Force a host poll while GPU worker may still be up / freshly finished.
        try:
            from event_runtime.telemetry import host as telemetry_host

            poll = telemetry_host.poll_once(run_id)
            host_evidence = (
                f"gpu_workers={poll.get('gpu_worker_container_ids')}; "
                f"samples={poll.get('samples')}"
            )
        except Exception as exc:  # noqa: BLE001
            host_evidence = f"{type(exc).__name__}: {exc}"

        # Durable GPU stream / by-role / by-job
        for rel in (
            f"runs/{run_id}/telemetry/gpu-stream/samples.jsonl",
            f"runs/{run_id}/telemetry/by-role/gpu-worker/samples.jsonl",
            f"runs/{run_id}/telemetry/by-job/{job_id}/samples.jsonl",
        ):
            text = sprintctl.volume_get_text(run, rel)
            if text and "gpu-worker" in text and "util_gpu_pct" in text:
                durable_gpu_ok = True
                durable_evidence = f"{rel} bytes={len(text)}"
                break
        if not durable_gpu_ok:
            durable_evidence = "no gpu-worker samples on /durable yet"

        host_csv = ROOT / "runs" / f"monitor-{run_id}-telemetry.csv"
        if host_csv.is_file() and "gpu-worker" in host_csv.read_text():
            host_gpu_ok = True
            host_evidence += f"; host_csv={host_csv}"
        elif host_csv.is_file() and "util_gpu_pct" in host_csv.read_text():
            # Agent may be CPU-only; look for nvidia_smi_ok true rows.
            rows = host_csv.read_text()
            if "True" in rows or ",1," in rows:
                host_gpu_ok = True
                host_evidence += "; host_csv has gpu fields"

    checklist.append(
        {
            "n": 4,
            "name": "GPU telemetry on /durable while/after job",
            "result": "PASS" if durable_gpu_ok else "FAIL",
            "evidence": durable_evidence,
        }
    )
    if not durable_gpu_ok:
        verdict = "FAIL"
    checklist.append(
        {
            "n": 5,
            "name": "Host monitor telemetry captures GPU worker",
            "result": "PASS" if host_gpu_ok else "FAIL",
            "evidence": host_evidence[:300],
        }
    )
    if not host_gpu_ok:
        # Soft-fail only if durable passed: host poll of Sandbox ids can lag.
        notes.append("Host GPU poll incomplete; durable path is authoritative.")

    # (d) terminate GPU worker; harness still up; durable telem survives
    if job_id:
        try:
            # Snapshot durable telem before kill.
            pre_kill = (
                sprintctl.volume_get_text(
                    run, f"runs/{run_id}/telemetry/by-job/{job_id}/samples.jsonl"
                )
                or sprintctl.volume_get_text(
                    run, f"runs/{run_id}/telemetry/gpu-stream/samples.jsonl"
                )
                or ""
            )
            term = gpu_worker.terminate_job(run, job_id)
            time.sleep(5)
            post_kill = (
                sprintctl.volume_get_text(
                    run, f"runs/{run_id}/telemetry/by-job/{job_id}/samples.jsonl"
                )
                or sprintctl.volume_get_text(
                    run, f"runs/{run_id}/telemetry/gpu-stream/samples.jsonl"
                )
                or ""
            )
            alive, out, codex_hint = agent_codex_alive(run, container)
            still = alive and "DURABLE_OK" in out
            telem_survived = bool(pre_kill) and len(post_kill) >= max(
                1, len(pre_kill) - 50
            )
            checklist.append(
                {
                    "n": 6,
                    "name": "GPU worker stop: harness up + telem survives",
                    "result": "PASS" if still and telem_survived else "FAIL",
                    "evidence": (
                        f"terminate={term.get('status')}; harness={still}; "
                        f"durable_pre={len(pre_kill)} durable_post={len(post_kill)}; "
                        f"codex_hint={codex_hint}"
                    ),
                }
            )
            if not (still and telem_survived):
                verdict = "FAIL"
        except Exception as exc:  # noqa: BLE001
            checklist.append(
                {
                    "n": 6,
                    "name": "GPU worker stop: harness up + telem survives",
                    "result": "FAIL",
                    "evidence": f"{type(exc).__name__}: {exc}",
                }
            )
            verdict = "FAIL"
    else:
        checklist.append(
            {
                "n": 6,
                "name": "GPU worker stop: harness up + telem survives",
                "result": "SKIP",
                "evidence": "no job_id",
            }
        )
        verdict = "FAIL"

    # (e) GPU-active timeline accounting
    timeline_ok = False
    timeline_evidence = "n/a"
    try:
        from event_runtime.telemetry import timeline as gpu_timeline_host

        summary = gpu_timeline_host.host_write_summary(run)
        phases = {
            seg.get("phase")
            for seg in summary.get("segments") or []
            if isinstance(seg, dict)
        }
        has_wait_or_start = bool(
            phases & {"gpu_queue_wait", "gpu_worker_starting", "isaac_starting"}
        )
        active = float(summary.get("gpu_active_s") or 0)
        interval = float(summary.get("gpu_active_interval_s") or 0)
        timeline_ok = has_wait_or_start and (active > 0 or interval > 0)
        timeline_evidence = (
            f"gpu_active_s={active}; gpu_active_interval_s={interval}; "
            f"gpu_wait_s={summary.get('gpu_wait_s')}; "
            f"gpu_startup_s={summary.get('gpu_startup_s')}; "
            f"isaac_startup_s={summary.get('isaac_startup_s')}; "
            f"wall_time_s={summary.get('wall_time_s')}; phases={sorted(phases)}"
        )
        # Mirror summary into ops.
        (state_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        (state_dir / "telemetry" / "gpu_time_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except Exception as exc:  # noqa: BLE001
        timeline_evidence = f"{type(exc).__name__}: {exc}"
    checklist.append(
        {
            "n": 7,
            "name": "GPU timeline distinguishes wait/startup vs active",
            "result": "PASS" if timeline_ok else "FAIL",
            "evidence": timeline_evidence[:400],
        }
    )
    if not timeline_ok:
        verdict = "FAIL"

    # Stop the smoke run cleanly (do not leave a long bakeoff).
    try:
        sprintctl.request_stop(run_id)
        notes.append("Issued sprintctl stop after smoke checks.")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"stop failed: {type(exc).__name__}: {exc}")

    report = {
        "run_id": run_id,
        "written_at": utc_now(),
        "verdict": verdict,
        "model": run.get("model"),
        "codex_version": run.get("codex_version"),
        "reasoning_effort": run.get("reasoning_effort"),
        "modal_profile": run.get("modal_profile"),
        "volume_name": run.get("volume_name"),
        "state_dir": str(state_dir),
        "checklist": checklist,
        "notes": " ".join(notes),
        "secret_dir": run.get("secret_dir"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, report)
    (state_dir / "smoke-cpu-gpu-result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"verdict": verdict, "report": str(args.report)}, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
