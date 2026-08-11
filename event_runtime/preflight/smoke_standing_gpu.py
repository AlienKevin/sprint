#!/usr/bin/env python3
"""Live smoke for the standing-GPU path. Creates ONE A10G, then tears it down.

Exercises exactly what unit tests cannot: that ensure_standing_sandbox really
creates a Modal sandbox, that a second call reuses it rather than leaking a
second GPU, that a replacement is created after the first is terminated
(the preemption path), and that exec into it actually runs.

Everything is torn down in a finally block, including on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from event_runtime.compute import worker as gpu_worker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="smoke-standing")
    ap.add_argument("--keep", action="store_true", help="skip teardown")
    args = ap.parse_args()

    import modal

    rid = args.run_id
    vol_name = f"sprint-{rid}"
    run = {
        "run_id": rid,
        "app_name": f"sprint-{rid}",
        "volume_name": vol_name,
        "sandbox_timeout_secs": 1800,
        "standing_gpu_worker": True,
    }
    results: dict[str, object] = {"run_id": rid}
    created: list[str] = []
    state_dir = ROOT / "runs" / "ops" / rid
    state_dir.mkdir(parents=True, exist_ok=True)

    try:
        # from_name(create_if_missing=True) is lazy in modal 1.x and does not
        # actually create the volume until something hydrates it, so the
        # sandbox's own from_name() then 404s. Create it eagerly like the
        # durable launcher does.
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "create",
                "--version",
                "1",
                vol_name,
            ],
            check=False,
            capture_output=True,
            timeout=180,
        )
        print(f"  volume {vol_name} ready")

        assert gpu_worker.standing_enabled(run), "flag not honoured"
        results["flag_honoured"] = True

        t0 = time.time()
        first = gpu_worker.ensure_standing_sandbox(run)
        created.append(first["sandbox_id"])
        print(
            f"  create: {first['action']} {first['sandbox_id']} "
            f"({time.time() - t0:.0f}s)"
        )
        results["create"] = first

        second = gpu_worker.ensure_standing_sandbox(run)
        print(f"  reuse : {second['action']} {second['sandbox_id']}")
        results["reuse"] = second
        results["reused_same_sandbox"] = (
            second["action"] == "reuse" and second["sandbox_id"] == first["sandbox_id"]
        )

        sb = modal.Sandbox.from_id(first["sandbox_id"])
        proc = sb.exec(
            "bash", "-c", "nvidia-smi --query-gpu=name --format=csv,noheader"
        )
        out = (proc.stdout.read() or "").strip()
        proc.wait()
        print(f"  exec  : gpu={out!r}")
        results["exec_gpu"] = out
        results["exec_saw_gpu"] = "A10" in out.upper()

        # preemption path: kill it, then confirm a replacement is created
        print("  terminating to simulate preemption ...")
        sb.terminate()
        for _ in range(30):
            if modal.Sandbox.from_id(first["sandbox_id"]).poll() is not None:
                break
            time.sleep(2)
        third = gpu_worker.ensure_standing_sandbox(run)
        created.append(third["sandbox_id"])
        print(f"  replace: {third['action']} {third['sandbox_id']}")
        results["replace"] = third
        results["replaced_after_kill"] = (
            third["sandbox_id"] != first["sandbox_id"] and third["action"] == "replaced"
        )

        ok = all(
            [
                results.get("flag_honoured"),
                results.get("reused_same_sandbox"),
                results.get("exec_saw_gpu"),
                results.get("replaced_after_kill"),
            ]
        )
        results["PASS"] = ok
        print("\n  " + ("PASS" if ok else "FAIL"))
        print(json.dumps(results, indent=1, default=str))
        return 0 if ok else 1
    finally:
        if not args.keep:
            for sid in created:
                try:
                    modal.Sandbox.from_id(sid).terminate()
                    print(f"  torn down {sid}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  teardown {sid}: {type(exc).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
