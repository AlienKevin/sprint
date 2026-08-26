from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.agent import cli  # noqa: E402


@pytest.mark.parametrize(
    "command",
    [
        "gpu",
        "test",
        "archive",
        "cost",
        "history",
    ],
)
def test_event_command_preserves_arguments(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    called: list[list[str]] = []
    monkeypatch.setattr(
        cli, "resolve_command", lambda action: lambda: called.append(sys.argv[:]) or 0
    )
    monkeypatch.setattr(sys, "argv", ["event", command, "one", "--two"])
    assert cli.main() == 0
    assert called == [[f"event {command}", "one", "--two"]]


def test_event_check_uses_event_specific_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: called.append((path, argv)))
    monkeypatch.setattr(sys, "argv", ["event", "check", "policy.pt", "--device", "cpu"])
    assert cli.main() == 127
    assert called == [
        (
            sys.executable,
            [
                sys.executable,
                cli.CHECK_POLICY,
                "policy.pt",
                "--device",
                "cpu",
            ],
        )
    ]


def test_event_help_is_one_screen(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["event", "--help"])
    assert cli.main() == 0
    output = capsys.readouterr().out
    for command in ("gpu", "test", "cost", "history", "check"):
        assert f"event {command}" in output
    assert "event archive" not in output


def test_event_rejects_unknown_command(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["event", "profile"])
    assert cli.main() == 2
    assert "unknown event command: profile" in capsys.readouterr().err
