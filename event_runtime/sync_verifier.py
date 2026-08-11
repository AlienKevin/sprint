#!/usr/bin/env python3
"""Regenerate an event's agent-visible verifier from its trusted source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from event_runtime.event import EventLayout, load_event  # noqa: E402


VERIFIER_ENTRYPOINTS = (
    Path("test.sh"),
    Path("verify.py"),
    Path("compare_results.py"),
    Path("verifier_telemetry.py"),
    Path("sprint_gpu_pipeline.py"),
    Path("collision_geometry.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verifier_source_files(source: Path) -> tuple[Path, ...]:
    """Return the public verifier contract without copying event test helpers."""
    course = tuple(
        Path("course") / path.name for path in sorted((source / "course").glob("*.py"))
    )
    files = VERIFIER_ENTRYPOINTS + course
    missing = [relative for relative in files if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "verifier source is incomplete: " + ", ".join(map(str, missing))
        )
    return files


def sync_event(event: EventLayout) -> Path:
    """Replace one event's published verifier mirror and return its manifest."""
    source = event.verifier
    target = event.environment / "verifier"
    train_start = event.environment / "train/standing_start.py"
    check_policy = event.environment / "check_policy.py"
    if target.exists():
        shutil.rmtree(target)
    manifest: dict[str, str] = {}
    for relative in verifier_source_files(source):
        source_file = source / relative
        target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        manifest[relative.as_posix()] = sha256(source_file)
    manifest_path = target / "SOURCE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str(source.relative_to(event.root.parents[1])),
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    shutil.copy2(source / "course/standing_start.py", train_start)
    shutil.copy2(source / "check_submission.py", check_policy)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", help="event name under events/")
    args = parser.parse_args()
    manifest = sync_event(load_event(args.event))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
