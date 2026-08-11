from __future__ import annotations

from pathlib import Path

import pytest

from event_runtime.event import load_event


ROOT = Path(__file__).resolve().parents[2]


def test_default_event_layout_is_complete() -> None:
    event = load_event(repository_root=ROOT)
    assert event.name == "g1-100-metres"
    assert event.root == ROOT / "events/g1-100-metres"
    assert event.environment.is_dir()
    assert event.verifier.is_dir()
    assert event.task.is_file()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "UPPER", "a", "x_y"])
def test_event_name_rejects_unsafe_or_noncanonical_values(name: str) -> None:
    with pytest.raises(ValueError):
        load_event(name, repository_root=ROOT)


def test_unknown_event_fails_closed() -> None:
    with pytest.raises(FileNotFoundError):
        load_event("not-an-event", repository_root=ROOT)
