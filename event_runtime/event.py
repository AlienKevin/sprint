"""Resolve one event package without coupling shared runtime code to its name."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


EVENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
DEFAULT_EVENT_NAME = "g1-100-metres"


@dataclass(frozen=True)
class EventLayout:
    name: str
    root: Path
    environment: Path
    verifier: Path
    task: Path


def load_event(
    name: str | None = None, *, repository_root: Path | None = None
) -> EventLayout:
    """Return validated paths for an event stored under ``events/<name>``."""
    selected = name or os.environ.get("EVENT_NAME") or DEFAULT_EVENT_NAME
    if not EVENT_NAME_RE.fullmatch(selected):
        raise ValueError(f"invalid event name: {selected!r}")
    repository = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    root = (repository / "events" / selected).resolve()
    if root.parent != (repository / "events").resolve() or not root.is_dir():
        raise FileNotFoundError(f"unknown event: {selected}")
    layout = EventLayout(
        name=selected,
        root=root,
        environment=root / "environment",
        verifier=root / "tests",
        task=root / "task.toml",
    )
    missing = [
        path
        for path in (layout.environment, layout.verifier, layout.task)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"event {selected} is incomplete: {', '.join(map(str, missing))}"
        )
    return layout
