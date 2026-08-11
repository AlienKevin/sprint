#!/usr/bin/env python3
"""Regenerate the agent-visible verifier mirror from the trusted source tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "events/g1-100-metres/tests"
TARGET = ROOT / "events/g1-100-metres/environment/verifier"
TRAIN_START = ROOT / "events/g1-100-metres/environment/train/standing_start.py"
CHECK_POLICY = ROOT / "events/g1-100-metres/environment/check_policy.py"
FILES = (
    Path("test.sh"),
    Path("verify.py"),
    Path("compare_results.py"),
    Path("verifier_telemetry.py"),
    Path("sprint_gpu_pipeline.py"),
    Path("collision_geometry.json"),
    Path("course/__init__.py"),
    Path("course/assets.py"),
    Path("course/metrics.py"),
    Path("course/policy.py"),
    Path("course/replay.py"),
    Path("course/rollout.py"),
    Path("course/forward_command.py"),
    Path("course/environment.py"),
    Path("course/standing_start.py"),
    Path("course/tasks.py"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    manifest: dict[str, str] = {}
    for relative in FILES:
        source = SOURCE / relative
        target = TARGET / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = sha256(source)
    (TARGET / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "events/g1-100-metres/tests",
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    shutil.copy2(SOURCE / "course/standing_start.py", TRAIN_START)
    shutil.copy2(SOURCE / "check_submission.py", CHECK_POLICY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
