from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_TESTS = ROOT / "challenge/g1-100-metres/tests"
sys.path.insert(0, str(TASK_TESTS))

from sprintbench.replay import atomic_write, failure_modes, representative_index  # noqa: E402


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


def test_atomic_replay_write_replaces_complete_json(tmp_path: Path) -> None:
    target = tmp_path / "replay.json"
    atomic_write(target, {"schema_version": 1, "frames": [[[0.0]]]})
    assert json.loads(target.read_text())["frames"] == [[[0.0]]]
    assert not list(tmp_path.glob(".replay.json.*"))
