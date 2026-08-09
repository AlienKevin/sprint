#!/usr/bin/env python3
"""Prove nominal scoring and held-out draws in fresh sealed Isaac processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal

from warm_modal_images import APP_NAME, MANIFEST, VOLUME_NAME, run_sandbox


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "runs/ops/verifier-robustness-canary.json"


def atomic_write(payload: dict) -> None:
    temporary = REPORT.with_name(f".{REPORT.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(REPORT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    policy = args.policy.resolve()
    if not policy.is_file():
        parser.error(f"policy not found: {policy}")

    warmup = json.loads(MANIFEST.read_text())
    if warmup.get("completed") is not True:
        raise RuntimeError("Modal image warm-up is incomplete")
    verifier_image_id = str(warmup["contexts"]["verifier"]["image_id"])
    canary_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote_root = f"/verifier-robustness-canary/{canary_id}"
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with volume.batch_upload(force=True) as upload:
        upload.put_file(policy, f"{remote_root}/policy.pt", mode=0o444)

    command = (
        "set -euo pipefail; "
        "mkdir -p /app/submission "
        f"/warm{remote_root}/logs; "
        f"cp /warm{remote_root}/policy.pt /app/submission/policy.pt; "
        "LOGS_DIR="
        f"/warm{remote_root}/logs "
        "SPRINT_VERIFIER_TELEMETRY_INTERVAL_S=2 "
        "NOMINAL_TIMEOUT_S=240 ROBUSTNESS_SEED_TIMEOUT_S=180 "
        "timeout --signal=TERM --kill-after=30 700 /tests/test.sh; "
        f"python3 -c \"import json,pathlib; root=pathlib.Path('/warm{remote_root}/logs'); "
        "r=json.load(open(root/'sprint_results.json')); "
        "assert r['valid_run'] is True; assert r['robustness_complete'] is True; "
        "assert r['robustness_seeds_completed']==r['robustness_seeds_total']==2; "
        "rows=[json.load(open(p)) for p in sorted((root/'robustness').glob('seed-*.json'))]; "
        "assert len(rows)==2 and all(x['completed'] is True for x in rows); "
        "assert {x['physics']['added_mass'] for x in rows}=={-1.8,2.4}; "
        "assert all(x['performance']['trials'][0]['total_s']>0 for x in rows); "
        "t=[json.loads(x) for x in open(root/'telemetry/samples.jsonl') if x.strip()]; "
        "g=[gpu for sample in t for gpu in sample['gpus']]; assert g; "
        "assert max(x['mem_used_mib'] for x in g)>1000\"; "
        "echo SEALED_FRESH_PROCESS_ROBUSTNESS_VERIFIED"
    )

    started = time.time()
    report = {
        "schema_version": 1,
        "completed": False,
        "canary_id": canary_id,
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "verifier_image_id": verifier_image_id,
        "remote_root": remote_root,
        "started_at_epoch_s": started,
    }
    try:
        with modal.enable_output():
            report["sandbox"] = run_sandbox(
                app=app,
                image=modal.Image.from_id(verifier_image_id),
                role="verifier-fresh-process-robustness-canary",
                gpu="A10G",
                volume=volume,
                command=command,
                timeout=780,
                required_output_substrings=(
                    "robustness summary: 2/2 seeds complete",
                    "SEALED_FRESH_PROCESS_ROBUSTNESS_VERIFIED",
                ),
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
