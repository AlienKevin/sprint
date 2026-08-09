#!/usr/bin/env python3
"""Run real Isaac Lab PPO iterations in the sealed training image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import modal

from warm_modal_images import APP_NAME, MANIFEST, VOLUME_NAME, run_sandbox


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "runs/ops/training-gpu-canary.json"


def atomic_write(payload: dict) -> None:
    tmp = REPORT.with_name(f".{REPORT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, REPORT)


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
        "SPRINT_REQUESTED_CPU_CORES=8 SPRINT_REQUESTED_MEMORY_MIB=32768 "
        "python3 /opt/sprint-telemetry.py --role training-gpu "
        f"--run-id training-canary --interval-seconds 5 --out-dir /warm{remote_root}/telemetry "
        "--durable-dir /nonexistent --force & telemetry_pid=$!; "
        "trap 'kill $telemetry_pid 2>/dev/null || true' EXIT; "
        f"export SPRINT_GPU_CHECKPOINT_DIR=/warm{remote_root}/checkpoints; "
        f"export SPRINT_GPU_PROGRESS_FILE=/warm{remote_root}/progress.json; "
        "timeout --signal=TERM --kill-after=30 900 "
        "python3 /opt/sprint-isaac-bootstrap.py /app/train_sprint.py "
        "--headless --device cuda:0 --num_envs 32 --max_iters 3 "
        "--chunk_iters 1 --save_interval 1 --episode_length 2 "
        f"--exp_name sealed_training_canary --log_root /warm{remote_root}/logs "
        f"2>&1 | tee /warm{remote_root}/training.log; "
        "kill $telemetry_pid 2>/dev/null || true; wait $telemetry_pid 2>/dev/null || true; "
        f"test -s /warm{remote_root}/checkpoints/policy_final.pt; "
        f"python3 -c \"import json; p=json.load(open('/warm{remote_root}/progress.json')); "
        "assert p['finished'] is True and p['iteration'] >= 3\"; "
        f"grep -F '[sprint] iter=3' /warm{remote_root}/training.log; "
        f"python3 -c \"import json; rows=[json.loads(x) for x in "
        f"open('/warm{remote_root}/telemetry/samples.jsonl') if x.strip()]; "
        "gpus=[g for p in rows for g in p['gpus']]; "
        "assert gpus and all(g['gpu_name'] for g in gpus); "
        "assert max(g['mem_used_mib'] for g in gpus) > 1000\""
    )

    started = time.time()
    report = {
        "schema_version": 1,
        "completed": False,
        "canary_id": canary_id,
        "image_id": image_id,
        "work_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "remote_root": remote_root,
        "started_at_epoch_s": started,
    }
    try:
        with modal.enable_output():
            report["sandbox"] = run_sandbox(
                app=app,
                image=modal.Image.from_id(image_id),
                role="training-gpu-functional-canary",
                gpu="A10G",
                volume=volume,
                command=command,
                timeout=1200,
                required_output_substrings=("[sprint] iter=3",),
            )
        report["completed"] = True
        report["completed_at_epoch_s"] = time.time()
        report["elapsed_s"] = round(time.time() - started, 3)
        return 0
    finally:
        atomic_write(report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
