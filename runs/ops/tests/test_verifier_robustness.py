from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "challenge/g1-sprint-100m-lane/tests/merge_robustness.py"
)
SPEC = importlib.util.spec_from_file_location("merge_robustness", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_merge_robustness_tracks_partial_and_complete_provenance(tmp_path: Path) -> None:
    results = tmp_path / "sprint_results.json"
    draws = tmp_path / "perturbations.json"
    result_dir = tmp_path / "robustness"
    physics = [
        {"seed": 1, "static_friction": 0.7, "added_mass": -1.0},
        {"seed": 2, "static_friction": 0.9, "added_mass": 2.0},
    ]
    write_json(results, {"valid_run": True, "performance": {"elapsed_s": 10}})
    write_json(draws, {"draws": physics})
    write_json(
        result_dir / "seed-1.json",
        {
            "seed": 1,
            "physics": physics[0],
            "completed": True,
            "valid_run": True,
            "performance": {"app_startup_s": 2},
        },
    )

    partial = MODULE.merge_results(results, draws, result_dir)
    assert partial["robustness_rate"] == 0.5
    assert partial["robustness_seeds_completed"] == 1
    assert partial["robustness_seeds_total"] == 2
    assert partial["robustness_complete"] is False
    assert partial["robustness_missing_seeds"] == [2]

    write_json(
        result_dir / "seed-2.json",
        {
            "seed": 2,
            "physics": physics[1],
            "completed": True,
            "valid_run": False,
            "performance": {"app_startup_s": 3},
        },
    )
    complete = MODULE.merge_results(results, draws, result_dir)
    assert complete["robustness_rate"] == 0.5
    assert complete["robustness_seeds_completed"] == 2
    assert complete["robustness_complete"] is True
    assert complete["robustness_missing_seeds"] == []
    assert len(complete["performance"]["robustness_processes"]) == 2


def test_merge_robustness_rejects_mismatched_seed_payload(tmp_path: Path) -> None:
    results = tmp_path / "sprint_results.json"
    draws = tmp_path / "perturbations.json"
    result_dir = tmp_path / "robustness"
    draw = {"seed": 7, "static_friction": 0.7}
    write_json(results, {"valid_run": True})
    write_json(draws, {"draws": [draw]})
    write_json(
        result_dir / "seed-7.json",
        {"seed": 8, "physics": draw, "completed": True, "valid_run": True},
    )

    try:
        MODULE.merge_results(results, draws, result_dir)
    except ValueError as exc:
        assert "provenance mismatch" in str(exc)
    else:
        raise AssertionError("mismatched robustness result was accepted")
