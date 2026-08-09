from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


OPS = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = OPS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_digest_is_stable_and_ignores_python_cache(tmp_path: Path) -> None:
    warmer = load_script("warm_modal_images.py")
    checker = load_script("check_modal_image_warmup.py")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    expected = warmer.context_digest(tmp_path)
    assert checker.context_digest(tmp_path) == expected

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ephemeral")
    assert warmer.context_digest(tmp_path) == expected


def test_successful_sandbox_uses_returncode_after_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmer = load_script("warm_modal_images.py")

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
    warmer = load_script("warm_modal_images.py")

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
            "contexts": {
                "verifier": {"sha256": "same", "image_id": "im-existing"}
            },
        },
    )

    assert image is reused
    assert reused.build_calls == 1
    assert metadata["image_id"] == "im-existing"
    assert metadata["reused_from_manifest"] is True


def test_warmup_rebuilds_only_changed_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmer = load_script("warm_modal_images.py")
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
            "contexts": {
                "verifier": {"sha256": "old", "image_id": "im-existing"}
            },
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
    warmer = load_script("warm_modal_images.py")

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
    checker = load_script("check_modal_image_warmup.py")
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
        checker, "CONTEXTS", {"agent_training": agent, "verifier": verifier}
    )

    assert checker.main() == 0
    assert "two verifier executions" in capsys.readouterr().out


def test_checker_rejects_changed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = load_script("check_modal_image_warmup.py")
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
        checker, "CONTEXTS", {"agent_training": agent, "verifier": verifier}
    )

    with pytest.raises(SystemExit, match="verifier image changed"):
        checker.main()
