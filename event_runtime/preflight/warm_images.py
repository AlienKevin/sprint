#!/usr/bin/env python3
"""Eagerly build and exercise every Modal image used by an event."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.event import load_event  # noqa: E402
from event_runtime.image import (  # noqa: E402
    agent_context_roots,
    agent_image as compose_agent_image,
    context_digest,
    verifier_context_roots,
    verifier_image as compose_verifier_image,
)
from event_runtime.sync_verifier import materialize_public_verifier  # noqa: E402


EVENT = load_event(repository_root=ROOT)
VERIFIER_CONTEXT = EVENT.verifier
MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
APP_NAME = "sprint-image-warmup"
VOLUME_NAME = "sprint-image-warmup-artifacts"
FATAL_SANDBOX_OUTPUT = (
    "Failed to resolve extension dependencies",
    "Failed to startup python app",
    "ModuleNotFoundError:",
    "Traceback (most recent call last):",
)


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


def build_or_reuse_image(
    image: modal.Image,
    app: modal.App,
    *,
    context_name: str,
    context_sha256: str,
    previous_manifest: dict[str, Any] | None,
) -> tuple[modal.Image, dict[str, Any]]:
    """Reuse an exact immutable image when this context is byte-identical."""
    previous = (
        previous_manifest.get("contexts", {}).get(context_name, {})
        if isinstance(previous_manifest, dict)
        and (
            previous_manifest.get("completed") is True
            or previous_manifest.get("images_built") is True
        )
        else {}
    )
    previous_id = previous.get("image_id")
    if previous.get("sha256") == context_sha256 and isinstance(previous_id, str):
        started = time.monotonic()
        try:
            reused = modal.Image.from_id(previous_id)
            reused.build(app)
        except Exception:  # noqa: BLE001
            # Modal may have garbage-collected an old image. Fall back to the
            # Dockerfile build rather than treating an expired cache as an
            # evaluation failure.
            pass
        else:
            elapsed = time.monotonic() - started
            if reused.object_id != previous_id:
                raise RuntimeError("reused Modal image ID changed unexpectedly")
            return reused, {
                "image_id": previous_id,
                "first_build_s": 0.0,
                "cached_build_s": round(elapsed, 3),
                "reused_from_manifest": True,
            }
    built, metadata = build_image(image, app)
    metadata["reused_from_manifest"] = False
    return built, metadata


def run_sandbox(
    *,
    app: modal.App,
    image: modal.Image,
    role: str,
    command: str,
    gpu: str | None = None,
    volume: modal.Volume | None = None,
    timeout: int = 300,
    cpu: int = 6,
    memory: int = 12288,
    required_output_substrings: tuple[str, ...] = (),
    forbidden_output_substrings: tuple[str, ...] = FATAL_SANDBOX_OUTPUT,
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
        # Kit has paths that log a fatal startup error to stderr and still
        # return zero. Merge both streams and validate semantic success below.
        sandbox = modal.Sandbox.create("bash", "-lc", f"exec 2>&1; {command}", **kwargs)
        create_s = time.monotonic() - create_started
        output = sandbox.stdout.read()
        sandbox.wait(raise_on_termination=False)
        exit_code = sandbox.returncode
        elapsed_s = time.monotonic() - started
        if exit_code != 0:
            raise RuntimeError(f"{role} warmup exited {exit_code}: {output[-4000:]}")
        forbidden = [item for item in forbidden_output_substrings if item in output]
        if forbidden:
            raise RuntimeError(
                f"{role} warmup emitted fatal output {forbidden}: {output[-4000:]}"
            )
        missing = [item for item in required_output_substrings if item not in output]
        if missing:
            raise RuntimeError(
                f"{role} warmup missed success markers {missing}: {output[-4000:]}"
            )
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

    previous_manifest: dict[str, Any] | None = None
    try:
        candidate = json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    else:
        if isinstance(candidate, dict):
            previous_manifest = candidate

    agent_roots = agent_context_roots(EVENT)
    agent_sha256 = context_digest(*agent_roots)
    verifier_roots = verifier_context_roots(EVENT)
    verifier_sha256 = context_digest(*verifier_roots)
    public_temp = tempfile.TemporaryDirectory(prefix="event-public-verifier-")
    public_verifier = Path(public_temp.name) / "verifier"
    materialize_public_verifier(EVENT, public_verifier)
    agent_image = compose_agent_image(EVENT, public_verifier)
    verifier_image = compose_verifier_image(EVENT)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "completed": False,
        "started_at_epoch_s": time.time(),
        "modal_profile": os.environ.get("MODAL_PROFILE"),
        "app_name": APP_NAME,
        "contexts": {
            "agent_training": {
                "paths": [str(path) for path in agent_roots],
                "sha256": agent_sha256,
            },
            "verifier": {
                "paths": [str(path) for path in verifier_roots],
                "sha256": verifier_sha256,
            },
        },
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
    }

    with modal.enable_output():
        agent_image, agent_build = build_or_reuse_image(
            agent_image,
            app,
            context_name="agent_training",
            context_sha256=agent_sha256,
            previous_manifest=previous_manifest,
        )
        payload["contexts"]["agent_training"].update(agent_build)
        verifier_image, verifier_build = build_or_reuse_image(
            verifier_image,
            app,
            context_name="verifier",
            context_sha256=verifier_sha256,
            previous_manifest=previous_manifest,
        )
        payload["contexts"]["verifier"].update(verifier_build)
        # Image construction is the expensive, resumable phase. Persist exact
        # immutable IDs before disposable probes so a failed assertion can be
        # fixed and rerun without rebuilding ~20 GiB of unchanged dependencies.
        payload["images_built"] = True
        atomic_write(payload)

        payload["cpu_agent_probe"] = run_sandbox(
            app=app,
            image=agent_image,
            role="cpu-agent-warmup",
            command=(
                "python3 -c 'import torch; print(torch.__version__)' && "
                "test \"$(codex --version)\" = 'codex-cli 0.149.1' && "
                "test -x /opt/sprint-codex-exec-wrapper.sh && "
                "test -x /opt/sprint-deepseek-harness-exec-wrapper.sh && "
                "test -x /opt/sprint-deepseek-harness-runner.py && "
                "test -r /opt/event_runtime/container/sprint-deepseek-goal-bootstrap.mjs && "
                "node -e 'const v=require(\"/opt/deepseek-harness/node_modules/@deepseek-ai/dsh-sdk-jsonrpc-demo/package.json\").version; "
                "if (v !== process.argv[1]) throw new Error(`unexpected DeepSeek Harness version ${v}`)' 0.1.1-rc.2 && "
                "python3 -c \"import importlib.metadata; assert importlib.metadata.version('deepseek-harness-sdk') == '0.1.1rc1'\" && "
                "python3 /opt/event_runtime/container/sprint-deepseek-harness-probe.py && "
                "test -x /opt/sprint-apply-deepseek-codex-config.sh && "
                "test -x /opt/sprint-apply-openai-codex-config.sh && "
                "test -x /opt/sprint-apply-luna-codex-config.sh && "
                "test -x /opt/sprint-agent-shell-env.sh && "
                "echo AGENT_SHELL_ENTRYPOINTS_EXECUTABLE && "
                "mkdir -p /tmp/cpu-telemetry && "
                "SPRINT_REQUESTED_CPU_CORES=2 SPRINT_REQUESTED_MEMORY_MIB=8192 "
                "python3 /opt/sprint-telemetry.py --once --role cpu-agent "
                "--run-id warmup --out-dir /tmp/cpu-telemetry "
                "--durable-dir /nonexistent --force && "
                "python3 -c \"import json; p=json.load(open('/tmp/cpu-telemetry/latest.json')); "
                "assert p['resource_accounting_scope'] in ('cgroup-v1','cgroup-v2'); "
                "assert p['cpu_requested_cores']==2.0; "
                "assert p['mem_requested_kib']==8388608; assert p['mem_used_kib']>0\""
            ),
            cpu=2,
            memory=8192,
            required_output_substrings=(
                "DEEPSEEK_HARNESS_PROTOCOL_OK",
                "AGENT_SHELL_ENTRYPOINTS_EXECUTABLE",
            ),
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
                "SPRINT_REQUESTED_CPU_CORES=6 SPRINT_REQUESTED_MEMORY_MIB=12288 "
                "python3 /opt/sprint-telemetry.py --once --role training-gpu "
                "--run-id warmup --out-dir /tmp/training-telemetry "
                "--durable-dir /nonexistent --force && "
                "python3 -c \"import json; p=json.load(open('/tmp/training-telemetry/latest.json')); "
                "assert p['resource_accounting_scope'] in ('cgroup-v1','cgroup-v2'); "
                "assert p['cpu_requested_cores']==6.0; "
                "assert p['mem_requested_kib']==12582912; assert p['mem_used_kib']>0; "
                "g=p['gpus'][0]; assert g['pipeline_metrics_status']=='ok'; "
                "assert g['pipeline_metrics_sample_count']>0; "
                "assert g['pipeline_metrics_group'] in (0,1,2); "
                "assert isinstance(g['dram_throughput_pct'], float)\" && "
                "timeout --signal=TERM --kill-after=10 120 "
                "python3 /opt/sprint-isaac-bootstrap.py "
                "/app/train/asset_probe.py --headless --device cuda:0"
            ),
            required_output_substrings=(
                "LOCAL_G1=/opt/assets/",
                "LOCAL_DEBUG_MARKERS=ok",
            ),
        )

        verifier_command = (
            "mkdir -p /tmp/verifier-warm && "
            "mkdir -p /tmp/verifier-telemetry && "
            "SPRINT_REQUESTED_CPU_CORES=4 SPRINT_REQUESTED_MEMORY_MIB=10240 "
            "timeout --preserve-status --signal=TERM --kill-after=5 5 "
            "python3 /opt/event_runtime/container/verifier_telemetry.py "
            "--out-dir /tmp/verifier-telemetry --interval-seconds 0.5 && "
            "python3 -c \"import json; p=json.load(open('/tmp/verifier-telemetry/latest.json')); "
            "assert p['resource_accounting_scope'] in ('cgroup-v1','cgroup-v2'); "
            "assert p['cpu_requested_cores']==4.0; "
            "assert p['mem_requested_kib']==10485760; assert p['mem_used_kib']>0; "
            "rows=[json.loads(x) for x in open('/tmp/verifier-telemetry/samples.jsonl')]; "
            "gs=[g for r in rows for g in r['gpus'] if g.get('pipeline_metrics_status')=='ok']; "
            "assert gs; assert all(g['pipeline_metrics_sample_count']>0 for g in gs); "
            "assert {g['pipeline_metrics_group'] for g in gs} <= {0,1,2}; "
            "assert all(isinstance(g['dram_throughput_pct'], float) for g in gs)\" && "
            "timeout --signal=TERM --kill-after=30 180 "
            "python3 /tests/verify.py --policy /warm/policy.pt "
            "--logs /tmp/verifier-warm --tests /tests --runs 1 "
            "--distance 1 --max-seconds 2 --headless && "
            "python3 -c \"import json; p=json.load(open('/tmp/verifier-warm/replay.json')); "
            "assert p['schema_version']==2; "
            "assert p['lane_gate']['semantics']=='whole_body_collision_envelope'; "
            "assert p['lane_gate']['collision_samples']==1960; "
            "assert len(p['frames'])==1; "
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
                cpu=4,
                memory=10240,
            )
            for index in (1, 2)
        ]

    payload["completed"] = True
    payload["completed_at_epoch_s"] = time.time()
    payload["unique_image_count"] = 2
    payload["cleanup"] = "all warmup sandboxes terminated"
    atomic_write(payload)
    public_temp.cleanup()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
