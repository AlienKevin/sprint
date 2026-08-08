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
