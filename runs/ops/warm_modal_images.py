#!/usr/bin/env python3
"""Eagerly build and exercise every Modal image used by Sprint evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[2]
AGENT_CONTEXT = ROOT / "challenge/g1-sprint-100m-lane/environment"
VERIFIER_CONTEXT = ROOT / "challenge/g1-sprint-100m-lane/tests"
MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
APP_NAME = "sprint-image-warmup"
VOLUME_NAME = "sprint-image-warmup-artifacts"


def context_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def build_image(
    image: modal.Image, app: modal.App
) -> tuple[modal.Image, dict[str, Any]]:
    started = time.monotonic()
    image.build(app)
    first_s = time.monotonic() - started
    image_id = image.object_id

    # Resolve subsequent probes by immutable image ID. Re-resolving the original
    # Dockerfile definition can needlessly repeat its build graph in a fresh
    # resolver even though the resulting layers already exist.
    built_image = modal.Image.from_id(image_id)
    started = time.monotonic()
    built_image.build(app)
    cached_s = time.monotonic() - started
    if not image_id or built_image.object_id != image_id:
        raise RuntimeError("built Modal image ID was not retained")
    return built_image, {
        "image_id": image_id,
        "first_build_s": round(first_s, 3),
        "cached_build_s": round(cached_s, 3),
    }


def run_sandbox(
    *,
    app: modal.App,
    image: modal.Image,
    role: str,
    command: str,
    gpu: str | None = None,
    volume: modal.Volume | None = None,
    timeout: int = 300,
    cpu: int = 8,
    memory: int = 32768,
) -> dict[str, Any]:
    sandbox: modal.Sandbox | None = None
    started = time.monotonic()
    try:
        kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "cpu": cpu,
            "memory": memory,
            "block_network": True,
            "timeout": timeout,
            "tags": {"sprint.role": role, "sprint.warmup": "true"},
        }
        if gpu:
            kwargs["gpu"] = gpu
        if volume:
            kwargs["volumes"] = {"/warm": volume}
        create_started = time.monotonic()
        sandbox = modal.Sandbox.create("bash", "-lc", command, **kwargs)
        create_s = time.monotonic() - create_started
        output = sandbox.stdout.read()
        sandbox.wait(raise_on_termination=False)
        exit_code = sandbox.returncode
        elapsed_s = time.monotonic() - started
        if exit_code != 0:
            raise RuntimeError(f"{role} warmup exited {exit_code}: {output[-4000:]}")
        return {
            "sandbox_id": sandbox.object_id,
            "create_s": round(create_s, 3),
            "process_and_wait_s": round(elapsed_s - create_s, 3),
            "total_s": round(elapsed_s, 3),
            "output_tail": output[-1000:],
        }
    finally:
        if sandbox is not None:
            sandbox.terminate(wait=True)


def atomic_write(payload: dict[str, Any]) -> None:
    tmp = MANIFEST.with_name(f".{MANIFEST.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, MANIFEST)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    policy = args.policy.resolve()
    if not policy.is_file():
        parser.error(f"policy not found: {policy}")

    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with volume.batch_upload(force=True) as upload:
        upload.put_file(policy, "/policy.pt", mode=0o444)

    agent_image = modal.Image.from_dockerfile(
        AGENT_CONTEXT / "Dockerfile", context_dir=AGENT_CONTEXT
    )
    verifier_image = modal.Image.from_dockerfile(
        VERIFIER_CONTEXT / "Dockerfile", context_dir=VERIFIER_CONTEXT
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "completed": False,
        "started_at_epoch_s": time.time(),
        "modal_profile": os.environ.get("MODAL_PROFILE"),
        "app_name": APP_NAME,
        "contexts": {
            "agent_training": {
                "path": str(AGENT_CONTEXT),
                "sha256": context_digest(AGENT_CONTEXT),
            },
            "verifier": {
                "path": str(VERIFIER_CONTEXT),
                "sha256": context_digest(VERIFIER_CONTEXT),
            },
        },
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
    }

    with modal.enable_output():
        agent_image, agent_build = build_image(agent_image, app)
        payload["contexts"]["agent_training"].update(agent_build)
        verifier_image, verifier_build = build_image(verifier_image, app)
        payload["contexts"]["verifier"].update(verifier_build)

        payload["cpu_agent_probe"] = run_sandbox(
            app=app,
            image=agent_image,
            role="cpu-agent-warmup",
            command=(
                "python3 -c 'import torch; print(torch.__version__)' && "
                "test \"$(codex --version)\" = 'codex-cli 0.147.0' && "
                "mkdir -p /tmp/cpu-telemetry && "
                "SPRINT_REQUESTED_CPU_CORES=4 SPRINT_REQUESTED_MEMORY_MIB=16384 "
                "python3 /opt/sprint-telemetry.py --once --role cpu-agent "
                "--run-id warmup --out-dir /tmp/cpu-telemetry "
                "--durable-dir /nonexistent --force && "
                "python3 -c \"import json; p=json.load(open('/tmp/cpu-telemetry/latest.json')); "
                "assert p['resource_accounting_scope'] in ('cgroup-v1','cgroup-v2'); "
                "assert p['cpu_requested_cores']==4.0; "
                "assert p['mem_requested_kib']==16777216; assert p['mem_used_kib']>0\""
            ),
            cpu=4,
            memory=16384,
        )
        payload["training_gpu_probe"] = run_sandbox(
            app=app,
            image=agent_image,
            role="training-gpu-warmup",
            gpu="A10G",
            command=(
                'python3 -c "import torch; assert torch.cuda.is_available(); '
                'print(torch.cuda.get_device_name(0))" && '
                "mkdir -p /tmp/training-telemetry && "
                "SPRINT_REQUESTED_CPU_CORES=8 SPRINT_REQUESTED_MEMORY_MIB=32768 "
                "python3 /opt/sprint-telemetry.py --once --role training-gpu "
                "--run-id warmup --out-dir /tmp/training-telemetry "
                "--durable-dir /nonexistent --force && "
                "python3 -c \"import json; p=json.load(open('/tmp/training-telemetry/latest.json')); "
                "assert p['resource_accounting_scope'] in ('cgroup-v1','cgroup-v2'); "
                "assert p['cpu_requested_cores']==8.0; "
                "assert p['mem_requested_kib']==33554432; assert p['mem_used_kib']>0\""
            ),
        )

        verifier_command = (
            "mkdir -p /tmp/verifier-warm && "
            "mkdir -p /tmp/verifier-telemetry && "
            "SPRINT_REQUESTED_CPU_CORES=8 SPRINT_REQUESTED_MEMORY_MIB=32768 "
            "timeout --preserve-status --signal=TERM --kill-after=5 2 "
            "python3 /tests/verifier_telemetry.py "
            "--out-dir /tmp/verifier-telemetry --interval-seconds 0.5 && "
            "python3 -c \"import json; p=json.load(open('/tmp/verifier-telemetry/latest.json')); "
            "assert p['resource_accounting_scope'] in ('cgroup-v1','cgroup-v2'); "
            "assert p['cpu_requested_cores']==8.0; "
            "assert p['mem_requested_kib']==33554432; assert p['mem_used_kib']>0\" && "
            "timeout --signal=TERM --kill-after=30 180 "
            "python3 /tests/verify.py --policy /warm/policy.pt "
            "--logs /tmp/verifier-warm --tests /tests --runs 1 "
            "--distance 1 --max-seconds 2 --skip-robustness --headless && "
            "python3 -c \"import json; p=json.load(open('/tmp/verifier-warm/replay.json')); "
            "assert p['schema_version']==1; assert len(p['frames'])==1; "
            "assert p['frames'][0]; assert p['policy_sha256']\""
        )
        payload["verifier_probes"] = [
            run_sandbox(
                app=app,
                image=verifier_image,
                role=f"verifier-warmup-{index}",
                gpu="A10G",
                volume=volume,
                command=verifier_command,
            )
            for index in (1, 2)
        ]

    payload["completed"] = True
    payload["completed_at_epoch_s"] = time.time()
    payload["unique_image_count"] = 2
    payload["cleanup"] = "all warmup sandboxes terminated"
    atomic_write(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
