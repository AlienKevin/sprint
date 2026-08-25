"""Shared coordination and limits for public experiment exports."""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterator
from pathlib import Path

# Keep enough recent runs for multi-model comparison batches. The deployment
# bundler still selects only runs referenced by the current batch, so this does
# not cause obsolete artifacts to be published.
PUBLIC_RUN_LIMIT = 64


@contextlib.contextmanager
def public_index_lock(index_path: Path) -> Iterator[None]:
    """Serialize a public index's read-modify-write transaction."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(f".{index_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
