from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "event_runtime" / "preflight"


def load_script(name: str):
    path = PREFLIGHT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_digest_is_stable_and_ignores_python_cache(tmp_path: Path) -> None:
    warmer = load_script("warm_images.py")
    checker = load_script("check_images.py")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    expected = warmer.context_digest(tmp_path)
    assert checker.context_digest(tmp_path) == expected

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ephemeral")
    assert warmer.context_digest(tmp_path) == expected


def test_agent_command_source_is_part_of_image_digest(tmp_path: Path) -> None:
    warmer = load_script("warm_images.py")
    environment = tmp_path / "environment"
    commands = tmp_path / "agent"
    environment.mkdir()
    commands.mkdir()
    (environment / "Dockerfile").write_text("FROM scratch\n")
    (commands / "cli.py").write_text("VERSION = 1\n")
    before = warmer.context_digest(environment, commands)
    (commands / "cli.py").write_text("VERSION = 2\n")
    assert warmer.context_digest(environment, commands) != before


def test_agent_image_marks_codex_shell_entrypoints_executable() -> None:
    source = (ROOT / "event_runtime" / "image.py").read_text()
    for name in (
        "sprint-agent-supervisor.sh",
        "sprint-codex-exec-wrapper.sh",
        "sprint-codex-goal-runner.py",
        "sprint-apply-openai-codex-config.sh",
        "sprint-agent-shell-env.sh",
        "sprint-trace-mirror.py",
    ):
        assert f'"/opt/event_runtime/container/{name}' in source


def test_image_compositor_is_part_of_both_image_contexts() -> None:
    from event_runtime.event import load_event
    from event_runtime.image import agent_context_roots, verifier_context_roots

    event = load_event(repository_root=ROOT)
    compositor = (ROOT / "event_runtime" / "image.py").resolve()

    assert compositor in agent_context_roots(event)
    assert compositor in verifier_context_roots(event)


def test_warmup_checks_agent_shell_entrypoints() -> None:
    source = (PREFLIGHT / "warm_images.py").read_text()
    for name in (
        "sprint-agent-supervisor.sh",
        "sprint-codex-exec-wrapper.sh",
        "sprint-codex-goal-runner.py",
        "sprint-apply-openai-codex-config.sh",
        "sprint-agent-shell-env.sh",
    ):
        assert f"test -x /opt/{name}" in source


def test_warmup_builds_a_valid_dynamic_batch_torchscript_policy(
    tmp_path: Path,
) -> None:
    import torch

    warmer = load_script("warm_images.py")
    policy = tmp_path / "policy.pt"

    warmer.build_warmup_policy(policy)

    module = torch.jit.load(policy, map_location="cpu").eval()
    output = module(torch.randn(3, 122))
    assert tuple(output.shape) == (3, 37)
    assert torch.count_nonzero(output).item() == 0


def test_warmup_retires_its_zero_task_modal_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmer = load_script("warm_images.py")
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        warmer.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or Result(),
    )

    warmer.stop_warmup_app()

    assert calls == [
        [
            warmer.sys.executable,
            "-m",
            "modal",
            "app",
            "stop",
            "-y",
            warmer.APP_NAME,
        ]
    ]


def test_functional_canary_uses_current_training_cli_contract() -> None:
    canary = load_script("canary.py")

    assert canary.TRAINING_CANARY_CLI == (
        "--num_envs=128 --max_iters=10 --chunk_iters=10 --save_interval=10 "
        "--headless --device=cuda:0"
    )
    assert "--max_iterations" not in canary.TRAINING_CANARY_CLI
    assert "--task=" not in canary.TRAINING_CANARY_CLI
    assert "--seed=" not in canary.TRAINING_CANARY_CLI


def test_functional_canary_uses_pinned_repo_owned_training_fixture() -> None:
    canary = load_script("canary.py")
    source = (PREFLIGHT / "canary.py").read_text()

    assert canary.TRAINING_FIXTURE == (
        PREFLIGHT / "training_canary" / "train_sprint.py"
    )
    assert canary.TRAINING_FIXTURE.is_file()
    assert "--work-archive" not in source
    assert "app.tar.gz" not in source
    assert "training_fixture_sha256" in source


def test_functional_canary_reuses_the_published_rollout_in_one_process() -> None:
    canary = load_script("canary.py")
    source = (PREFLIGHT / "canary.py").read_text()
    fixture = canary.REPEATED_VERIFIER_FIXTURE.read_text()

    assert canary.REPEATED_VERIFIER_FIXTURE == (PREFLIGHT / "repeat_verifier_trial.py")
    assert canary.REPEATED_VERIFIER_FIXTURE.is_file()
    assert 'geometry="/opt/event-verifier/verifier/collision_geometry.json"' in fixture
    assert "repeat_verifier_trial.py --headless --device=cuda:0" in source
    assert 'repeat-verifier.log)" = 3' in source
    assert "env.close()" in fixture
    assert "app.close()" not in fixture
    assert "os._exit(exit_code)" in fixture


def test_asset_localization_avoids_cpu_only_isaac_shutdown() -> None:
    source = (
        ROOT / "event_runtime" / "container" / "localize_assets.py"
    ).read_text()

    assert "app.close()" not in source
    assert "os._exit(code)" in source


def test_functional_canary_uses_trainer_owned_final_checkpoint() -> None:
    source = (PREFLIGHT / "canary.py").read_text()

    assert "export SPRINT_GPU_CHECKPOINT_DIR=/warm{remote_root}/checkpoints" in source
    assert "export SPRINT_GPU_PROGRESS_FILE=/warm{remote_root}/progress.json" in source
    assert "cp /app/policy_train.pt" not in source
    assert '"[sprint] optimization canary complete"' in source
    assert '"EXPORTED /app/policy_train.pt"' not in source


def test_functional_canary_accepts_any_authoritative_gpu_activity_signal() -> None:
    source = (PREFLIGHT / "canary.py").read_text()

    assert "g.get('util_gpu_pct', 0) >= 1" in source
    assert "g.get('sm_active_pct', 0) >= 0.1" in source
    assert "g.get('sm_occupancy_pct', 0) >= 1" in source
    assert "g.get('dram_throughput_pct', 0) >= 0.1" in source


def test_successful_sandbox_uses_returncode_after_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmer = load_script("warm_images.py")

    class Output:
        @staticmethod
        def read() -> str:
            return "ok\n"

    class Sandbox:
        object_id = "sb-success"
        stdout = Output()
        returncode = 0
        terminated = False

        @staticmethod
        def wait(*, raise_on_termination: bool) -> None:
            assert raise_on_termination is False
            return None

        def terminate(self, *, wait: bool) -> None:
            assert wait is True
            self.terminated = True

    sandbox = Sandbox()
    monkeypatch.setattr(warmer.modal.Sandbox, "create", lambda *args, **kwargs: sandbox)
    result = warmer.run_sandbox(
        app=object(), image=object(), role="test", command="true"
    )

    assert result["sandbox_id"] == "sb-success"
    assert result["output_tail"] == "ok\n"
    assert sandbox.terminated is True


def test_warmup_reuses_unchanged_context_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmer = load_script("warm_images.py")

    class ReusedImage:
        object_id = "im-existing"
        build_calls = 0

        def build(self, app: object) -> None:
            assert app is not None
            self.build_calls += 1

    reused = ReusedImage()
    monkeypatch.setattr(warmer.modal.Image, "from_id", lambda image_id: reused)
    monkeypatch.setattr(
        warmer,
        "build_image",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )

    image, metadata = warmer.build_or_reuse_image(
        object(),
        object(),
        context_name="verifier",
        context_sha256="same",
        previous_manifest={
            "completed": True,
            "contexts": {"verifier": {"sha256": "same", "image_id": "im-existing"}},
        },
    )

    assert image is reused
    assert reused.build_calls == 1
    assert metadata["image_id"] == "im-existing"
    assert metadata["reused_from_manifest"] is True


def test_warmup_rebuilds_only_changed_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmer = load_script("warm_images.py")
    built = object()
    monkeypatch.setattr(
        warmer,
        "build_image",
        lambda image, app: (built, {"image_id": "im-new"}),
    )

    image, metadata = warmer.build_or_reuse_image(
        object(),
        object(),
        context_name="verifier",
        context_sha256="new",
        previous_manifest={
            "completed": True,
            "contexts": {"verifier": {"sha256": "old", "image_id": "im-existing"}},
        },
    )

    assert image is built
    assert metadata == {"image_id": "im-new", "reused_from_manifest": False}


@pytest.mark.parametrize(
    ("output", "required", "match"),
    [
        ("ordinary output\n", ("READY",), "missed success markers"),
        (
            "Failed to resolve extension dependencies\n",
            (),
            "emitted fatal output",
        ),
        ("Traceback (most recent call last):\n", (), "emitted fatal output"),
    ],
)
def test_sandbox_fails_closed_on_semantic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    required: tuple[str, ...],
    match: str,
) -> None:
    del tmp_path
    warmer = load_script("warm_images.py")

    class Output:
        @staticmethod
        def read() -> str:
            return output

    class Sandbox:
        object_id = "sb-semantic-failure"
        stdout = Output()
        returncode = 0

        @staticmethod
        def wait(*, raise_on_termination: bool) -> None:
            assert raise_on_termination is False

        @staticmethod
        def terminate(*, wait: bool) -> None:
            assert wait is True

    monkeypatch.setattr(
        warmer.modal.Sandbox, "create", lambda *args, **kwargs: Sandbox()
    )
    with pytest.raises(RuntimeError, match=match):
        warmer.run_sandbox(
            app=object(),
            image=object(),
            role="test",
            command="true",
            required_output_substrings=required,
        )


def test_checker_accepts_complete_current_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checker = load_script("check_images.py")
    agent = tmp_path / "agent"
    verifier = tmp_path / "verifier"
    agent.mkdir()
    verifier.mkdir()
    (agent / "Dockerfile").write_text("FROM scratch\n")
    (verifier / "Dockerfile").write_text("FROM scratch\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed": True,
                "unique_image_count": 2,
                "contexts": {
                    "agent_training": {
                        "sha256": checker.context_digest(agent),
                        "image_id": "im-agent",
                    },
                    "verifier": {
                        "sha256": checker.context_digest(verifier),
                        "image_id": "im-verifier",
                    },
                },
                "verifier_probes": [
                    {"sandbox_id": "sb-one"},
                    {"sandbox_id": "sb-two"},
                ],
            }
        )
    )
    monkeypatch.setattr(checker, "MANIFEST", manifest)
    monkeypatch.setattr(
        checker, "CONTEXTS", {"agent_training": (agent,), "verifier": (verifier,)}
    )

    assert checker.main() == 0
    assert "two verifier executions" in capsys.readouterr().out


def test_checker_rejects_changed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = load_script("check_images.py")
    agent = tmp_path / "agent"
    verifier = tmp_path / "verifier"
    agent.mkdir()
    verifier.mkdir()
    (agent / "Dockerfile").write_text("FROM scratch\n")
    (verifier / "Dockerfile").write_text("FROM scratch\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed": True,
                "unique_image_count": 2,
                "contexts": {
                    "agent_training": {
                        "sha256": checker.context_digest(agent),
                        "image_id": "im-agent",
                    },
                    "verifier": {
                        "sha256": checker.context_digest(verifier),
                        "image_id": "im-verifier",
                    },
                },
                "verifier_probes": [
                    {"sandbox_id": "sb-one"},
                    {"sandbox_id": "sb-two"},
                ],
            }
        )
    )
    (verifier / "verify.py").write_text("print('changed')\n")
    monkeypatch.setattr(checker, "MANIFEST", manifest)
    monkeypatch.setattr(
        checker, "CONTEXTS", {"agent_training": (agent,), "verifier": (verifier,)}
    )

    with pytest.raises(SystemExit, match="verifier image changed"):
        checker.main()
