from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_TESTS))

from course.replay import (  # noqa: E402
    DEFAULT_FPS,
    PoseRecorder,
    atomic_write,
    failure_modes,
    representative_index,
)


def lane(
    *,
    valid: bool,
    distance: float,
    finish: float | None = None,
    peak: float = 0.0,
    failed: str | None = None,
) -> dict:
    checks = []
    if failed:
        checks.append({"name": failed, "gating": True, "passed": False})
    return {
        "valid": valid,
        "distance_m": distance,
        "finish_time_s": finish,
        "peak_speed_mps": peak,
        "checks": checks,
    }


def test_representative_prefers_fastest_valid_lane() -> None:
    rows = [
        lane(valid=False, distance=99.0, peak=9.0),
        lane(valid=True, distance=100.0, finish=12.0),
        lane(valid=True, distance=100.0, finish=9.5),
    ]
    assert representative_index(rows) == 2


def test_representative_failure_is_furthest_then_fastest() -> None:
    rows = [
        lane(valid=False, distance=25.0, peak=4.0),
        lane(valid=False, distance=40.0, peak=3.0, failed="in_lane"),
        lane(valid=False, distance=40.0, peak=5.0, failed="self_collision"),
    ]
    assert representative_index(rows) == 2
    assert failure_modes(rows[2]) == ["self_collision"]


def test_failure_without_explicit_gate_is_classified() -> None:
    assert failure_modes(lane(valid=False, distance=0.0)) == ["no_valid_finish"]


def test_replay_records_every_50_hz_policy_state() -> None:
    class Robot:
        body_names = ["pelvis"]

    class Origins:
        shape = (3, 3)

    recorder = PoseRecorder(Robot(), Origins(), control_hz=50.0)
    assert DEFAULT_FPS == 50.0
    assert recorder.sample_every == 1
    assert recorder.fps == 50.0


def test_web_replay_uses_authoritative_sample_hold() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    renderer = (ROOT / "web/renderers/g1-100-metres/render.py").read_text()

    assert "authoritative captured-state playback" in scene
    assert ".slerp(" not in scene
    assert "grp.add(node)" in scene
    assert '"interpolation": "none_authoritative_sample_hold"' in renderer


def test_atomic_replay_write_replaces_complete_json(tmp_path: Path) -> None:
    target = tmp_path / "replay.json"
    atomic_write(target, {"schema_version": 1, "frames": [[[0.0]]]})
    assert json.loads(target.read_text())["frames"] == [[[0.0]]]
    assert not list(tmp_path.glob(".replay.json.*"))
