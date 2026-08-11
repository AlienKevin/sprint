from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "events/g1-100-metres/environment/bin/event"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("event_cli", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("gpu", "sprint-gpu-train"),
        ("check", "sprint-check"),
        ("test", "sprint-verify"),
        ("archive", "sprint-submit"),
        ("cost", "sprint-cost"),
        ("history", "sprint-board"),
    ],
)
def test_event_command_preserves_arguments(
    monkeypatch: pytest.MonkeyPatch, command: str, target: str
) -> None:
    module = load_cli()
    called: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        module.os, "execv", lambda path, argv: called.append((path, argv))
    )
    monkeypatch.setattr(sys, "argv", ["event", command, "one", "--two"])
    assert module.main() == 127
    path = f"/usr/local/bin/{target}"
    assert called == [(path, [path, "one", "--two"])]


def test_event_help_is_one_screen(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = load_cli()
    monkeypatch.setattr(sys, "argv", ["event", "--help"])
    assert module.main() == 0
    output = capsys.readouterr().out
    for command in module.COMMANDS:
        assert f"event {command}" in output


def test_event_rejects_unknown_command(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = load_cli()
    monkeypatch.setattr(sys, "argv", ["event", "profile"])
    assert module.main() == 2
    assert "unknown event command: profile" in capsys.readouterr().err
