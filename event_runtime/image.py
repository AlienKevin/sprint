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
CODEX_COST = (
    ROOT / "harbor" / "src" / "harbor" / "agents" / "installed" / "codex_cost.py"
)
DEEPSEEK_HARNESS_NODE = CONTAINER / "deepseek-harness-node"
NODE_VERSION = "22.22.0"
CODEX_VERSION = "0.149.1"
DEEPSEEK_HARNESS_VERSION = "0.1.1-rc.2"

_CONTAINER_LINKS = (
    "sprint-agent-supervisor.sh",
    "sprint-trace-mirror.py",
    "sprint-telemetry.sh",
    "sprint-telemetry.py",
    "sprint_gpu_pipeline.py",
    "sprint-codex-exec-wrapper.sh",
    "sprint-deepseek-harness-exec-wrapper.sh",
    "sprint-deepseek-harness-runner.py",
    "sprint-deepseek-harness-probe.py",
    "sprint-openrouter-ledger-proxy.py",
    "sprint-apply-openai-codex-config.sh",
    "sprint-agent-shell-env.sh",
    "sprint-gpu-worker-run.py",
    "sprint-budget-watchdog.py",
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
        paths = (root,) if root.is_file() else tuple(root.rglob("*"))
        for path in sorted(item for item in paths if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            relative = path.name if path == root else path.relative_to(root).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            digest.update(b"\0")
    return digest.hexdigest()


def agent_context_roots(event: EventLayout) -> tuple[Path, ...]:
    """Return every repository root that contributes bytes to the agent image."""
    return (
        event.environment,
        CONTAINER,
        MODELS,
        AGENT,
        CODEX_COST,
        event.verifier,
        Path(__file__).resolve(),
    )


def verifier_context_roots(event: EventLayout) -> tuple[Path, ...]:
    """Return every repository root that contributes bytes to the verifier image."""
    return (
        event.verifier,
        CONTAINER / "verifier_telemetry.py",
        CONTAINER / "sprint_gpu_pipeline.py",
        ROOT / "event_runtime" / "preflight" / "compare_results.py",
        Path(__file__).resolve(),
    )


def agent_image(event: EventLayout, public_verifier: Path) -> modal.Image:
    """Build one event image from shared runtime files and event-owned inputs."""
    image = modal.Image.from_dockerfile(
        event.environment / "Dockerfile", context_dir=event.environment
    )
    image = image.add_local_dir(
        DEEPSEEK_HARNESS_NODE,
        "/opt/deepseek-harness",
        copy=True,
    )
    image = image.run_commands(
        "set -eu; "
        f"archive=node-v{NODE_VERSION}-linux-x64.tar.gz; "
        f"curl -fsSLO https://nodejs.org/dist/v{NODE_VERSION}/$archive; "
        f"curl -fsSLO https://nodejs.org/dist/v{NODE_VERSION}/SHASUMS256.txt; "
        "grep \"  $archive$\" SHASUMS256.txt | sha256sum -c -; "
        "tar -xzf $archive -C /usr/local --strip-components=1; "
        "rm -f $archive SHASUMS256.txt; "
        f"npm install -g @openai/codex@{CODEX_VERSION}; "
        "npm ci --prefix /opt/deepseek-harness --omit=dev --ignore-scripts; "
        "ln -sf /opt/deepseek-harness/node_modules/.bin/dsh-jsonrpc-agent "
        "/usr/local/bin/dsh-jsonrpc-agent; "
        'pip3 install --no-cache-dir "pydantic>=2.12,<3"; '
        'pip3 install --no-cache-dir --no-deps "deepseek-harness-sdk==0.1.1rc1"; '
        f"test \"$(codex --version | sed 's/^codex-cli //')\" = "
        f'"{CODEX_VERSION}"; '
        "node -e 'const v=require(\"/opt/deepseek-harness/node_modules/"
        "@deepseek-ai/dsh-sdk-jsonrpc-demo/package.json\").version; "
        "if (v !== process.argv[1]) throw new Error(`unexpected DeepSeek Harness "
        f"version ${{v}}`)' '{DEEPSEEK_HARNESS_VERSION}'; "
        "test \"$(find /opt/deepseek-harness/node_modules -path "
        "'*/@deepseek-ai/dsh-agent/package.json' | wc -l)\" = 1; "
        "test \"$(find /opt/deepseek-harness/node_modules -path "
        "'*/@deepseek-ai/dsh-scope/package.json' | wc -l)\" = 1; "
        'python3 -c "import deepseek_harness"',
    )
    image = image.add_local_dir(
        CONTAINER,
        "/opt/event_runtime/container",
        copy=True,
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            "deepseek-harness-node/**",
        ],
    )
    image = image.add_local_dir(MODELS, "/opt/event_runtime/models", copy=True)
    image = image.add_local_dir(
        AGENT,
        "/opt/event_runtime/agent",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    image = image.add_local_file(
        CODEX_COST,
        "/opt/sprint-codex-cost.py",
        copy=True,
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
        "/opt/sprint-codex-model-messages.json; "
        "ln -sf /opt/event_runtime/models/luna.json "
        "/opt/sprint-codex-luna-model-lock.json; "
        "ln -sf /opt/event_runtime/models/sol.json "
        "/opt/sprint-codex-sol-model-lock.json; "
        "ln -sf /opt/event_runtime/container/deepseek-harness-minimal.cordis.yml "
        "/opt/deepseek-harness-minimal.cordis.yml; "
        f"{links} "
        "chmod 0755 /opt/event_runtime/container/bin/event "
        "/opt/event/check_policy.py "
        "/opt/event_runtime/container/sprint-agent-supervisor.sh "
        "/opt/event_runtime/container/sprint-codex-exec-wrapper.sh "
        "/opt/event_runtime/container/sprint-deepseek-harness-exec-wrapper.sh "
        "/opt/event_runtime/container/sprint-deepseek-harness-runner.py "
        "/opt/event_runtime/container/sprint-openrouter-ledger-proxy.py "
        "/opt/event_runtime/container/sprint-apply-openai-codex-config.sh "
        "/opt/event_runtime/container/sprint-agent-shell-env.sh "
        "/opt/event_runtime/container/sprint-trace-mirror.py "
        "/opt/event_runtime/container/sprint-budget-watchdog.py; "
        "chmod -R a-w /opt/event-verifier; "
        "ln -sfn /opt/event-verifier /app/verifier",
    )


def verifier_image(event: EventLayout) -> modal.Image:
    """Build one sealed verifier from event rules and shared telemetry."""
    image = modal.Image.from_dockerfile(
        event.verifier / "Dockerfile", context_dir=event.verifier
    )
    image = image.add_local_file(
        CONTAINER / "verifier_telemetry.py",
        "/opt/event_runtime/container/verifier_telemetry.py",
        copy=True,
    )
    image = image.add_local_file(
        CONTAINER / "sprint_gpu_pipeline.py",
        "/opt/event_runtime/container/sprint_gpu_pipeline.py",
        copy=True,
    )
    image = image.add_local_file(
        ROOT / "event_runtime" / "preflight" / "compare_results.py",
        "/opt/event_runtime/preflight/compare_results.py",
        copy=True,
    )
    return image.run_commands(
        "python3 /opt/event_runtime/container/sprint_gpu_pipeline.py "
        "--build /usr/local/cuda/extras/CUPTI/samples/pm_sampling "
        "--output /opt/sprint-pm-sampling"
    )
