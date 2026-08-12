"""Compose the shared agent image with one event's frozen contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import modal

from event_runtime.event import EventLayout


ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "event_runtime" / "container"
MODELS = ROOT / "event_runtime" / "models"
AGENT = ROOT / "event_runtime" / "agent"

_CONTAINER_LINKS = (
    "sprint-snapshot-loop.sh",
    "sprint-trace-mirror.py",
    "sprint-telemetry.sh",
    "sprint-telemetry.py",
    "sprint_gpu_pipeline.py",
    "sprint-codex-exec-wrapper.sh",
    "sprint-apply-deepseek-codex-config.sh",
    "sprint-apply-luna-codex-config.sh",
    "sprint-agent-shell-env.sh",
    "sprint-gpu-worker-run.py",
    "sprint-isaac-bootstrap.py",
    "sprint-gpu-timeline.py",
    "sprint_resilience.py",
    "sprint_assets.py",
)


def context_digest(*roots: Path) -> str:
    """Hash image inputs while ignoring interpreter caches."""
    digest = hashlib.sha256()
    for root in roots:
        digest.update(root.name.encode())
        digest.update(b"\0")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            digest.update(b"\0")
    return digest.hexdigest()


def agent_context_roots(event: EventLayout) -> tuple[Path, ...]:
    """Return every repository root that contributes bytes to the agent image."""
    return event.environment, CONTAINER, MODELS, AGENT, event.verifier


def agent_image(event: EventLayout, public_verifier: Path) -> modal.Image:
    """Build one event image from shared runtime files and event-owned inputs."""
    image = modal.Image.from_dockerfile(
        event.environment / "Dockerfile", context_dir=event.environment
    )
    image = image.add_local_dir(
        CONTAINER,
        "/opt/event_runtime/container",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    image = image.add_local_dir(MODELS, "/opt/event_runtime/models", copy=True)
    image = image.add_local_dir(
        AGENT,
        "/opt/event_runtime/agent",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    image = image.add_local_dir(public_verifier, "/opt/event-verifier", copy=True)
    image = image.add_local_file(
        event.verifier / "check_submission.py",
        "/opt/event/check_policy.py",
        copy=True,
    )

    links = " ".join(
        f"ln -sf /opt/event_runtime/container/{name} /opt/{name};"
        for name in _CONTAINER_LINKS
    )
    return image.run_commands(
        "python3 /opt/event_runtime/container/localize_assets.py",
        "bash /opt/event_runtime/container/setup.sh",
        "python3 /opt/event_runtime/container/sprint_gpu_pipeline.py "
        "--build /usr/local/cuda/extras/CUPTI/samples/pm_sampling "
        "--output /opt/sprint-pm-sampling",
        "mkdir -p /usr/local/bin /app /opt/event; "
        "ln -sf /opt/event_runtime/container/bin/event /usr/local/bin/event; "
        "ln -sf /opt/event_runtime/models/deepseek.json "
        "/opt/sprint-codex-deepseek-models.json; "
        "ln -sf /opt/event_runtime/models/luna.json "
        "/opt/sprint-codex-luna-model-lock.json; "
        f"{links} "
        "chmod 0755 /opt/event_runtime/container/bin/event "
        "/opt/event/check_policy.py; "
        "chmod -R a-w /opt/event-verifier; "
        "ln -sfn /opt/event-verifier /app/verifier",
    )
