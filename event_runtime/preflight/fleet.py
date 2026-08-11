#!/usr/bin/env python3
"""Probe one fresh sealed Isaac training GPU allocation per planned trial."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.preflight.warm_images import MANIFEST, run_sandbox  # noqa: E402


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--worker-id", action="append", required=True)
    args = parser.parse_args()

    worker_ids = list(args.worker_id)
    if len(set(worker_ids)) != len(worker_ids):
        parser.error("worker IDs must be unique")
    warmup = json.loads(MANIFEST.read_text())
    if warmup.get("completed") is not True:
        raise RuntimeError("Modal image warm-up is incomplete")
    image_id = str(warmup["contexts"]["agent_training"]["image_id"])
    probe_hash = hashlib.sha256(
        (args.batch_id + "\0" + "\0".join(worker_ids)).encode()
    ).hexdigest()[:12]
    app_name = f"sprint-training-fleet-{probe_hash}"
    app = modal.App(app_name)
    image = modal.Image.from_id(image_id)
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "completed": False,
        "batch_id": args.batch_id,
        "app_name": app_name,
        "image_id": image_id,
        "modal_profile": os.environ.get("MODAL_PROFILE"),
        "started_at_epoch_s": started,
        "worker_ids": worker_ids,
        "workers": [],
    }

    command = (
        "set -euo pipefail; "
        'state=/tmp/sprint-app-launcher-state.json; rm -f "$state"; '
        'export SPRINT_APP_LAUNCHER_STATE_FILE="$state"; '
        "timeout --signal=TERM --kill-after=20 150 "
        "python3 /opt/sprint-isaac-bootstrap.py "
        "/app/train/asset_probe.py --headless --device cuda:0; "
        'python3 -c "import json; p=json.load(open('
        "'/tmp/sprint-app-launcher-state.json')); "
        "assert p == {'schema_version': 1, 'state': 'completed'}, p\"; "
        "echo SPRINT_FLEET_WORKER_READY"
    )

    def probe(worker_id: str) -> dict[str, Any]:
        result = run_sandbox(
            app=app,
            image=image,
            role="training-gpu-fleet-probe",
            gpu="A10G",
            command=command,
            timeout=240,
            required_output_substrings=(
                "LOCAL_G1=/opt/assets/",
                "LOCAL_DEBUG_MARKERS=ok",
                "SPRINT_FLEET_WORKER_READY",
            ),
        )
        return {"worker_id": worker_id, "ready": True, **result}

    failures: list[dict[str, str]] = []
    try:
        # App.run is temporary: leaving this context stops the App after every
        # sandbox has already been terminated by run_sandbox's finally block.
        with app.run():
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(worker_ids)
            ) as executor:
                futures = {
                    executor.submit(probe, worker_id): worker_id
                    for worker_id in worker_ids
                }
                results: dict[str, dict[str, Any]] = {}
                for future in concurrent.futures.as_completed(futures):
                    worker_id = futures[future]
                    try:
                        results[worker_id] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            {
                                "worker_id": worker_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[-4000:],
                            }
                        )
                report["workers"] = [
                    results[worker_id]
                    for worker_id in worker_ids
                    if worker_id in results
                ]
        report["failures"] = failures
        report["completed"] = not failures and len(report["workers"]) == len(worker_ids)
        report["completed_at_epoch_s"] = time.time()
        report["elapsed_s"] = round(time.time() - started, 3)
        report["cleanup"] = (
            "all fleet-probe sandboxes terminated; temporary App stopped"
        )
        if not report["completed"]:
            raise RuntimeError(
                f"{len(failures)} of {len(worker_ids)} training GPU probes failed"
            )
        return 0
    finally:
        atomic_write(args.report.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
