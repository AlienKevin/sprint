#!/usr/bin/env python3
"""Regenerate the agent-visible verifier mirror from the trusted source tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "challenge/g1-sprint-100m-lane/tests"
TARGET = ROOT / "challenge/g1-sprint-100m-lane/environment/verifier"
FILES = (
    Path("test.sh"),
    Path("verify.py"),
    Path("compare_results.py"),
    Path("verifier_telemetry.py"),
    Path("sprint_gpu_pipeline.py"),
    Path("collision_geometry.json"),
    Path("sprintbench/__init__.py"),
    Path("sprintbench/assets.py"),
    Path("sprintbench/metrics.py"),
    Path("sprintbench/policy.py"),
    Path("sprintbench/replay.py"),
    Path("sprintbench/rollout.py"),
    Path("sprintbench/sprint_command.py"),
    Path("sprintbench/sprint_env_cfg.py"),
    Path("sprintbench/tasks.py"),
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
                "source": "challenge/g1-sprint-100m-lane/tests",
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
