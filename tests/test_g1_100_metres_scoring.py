from __future__ import annotations

import sys
from pathlib import Path

import pytest

from event_runtime.export.performance import effective_speed


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "events/g1-100-metres"
sys.path.insert(0, str(TASK / "tests"))

from verifier.metrics import evaluate_run  # noqa: E402
from verifier.rollout import (  # noqa: E402
    max_lateral_extent_from_world_y,
    terminal_reason_for_sample,
)


OPTIONAL_METRICS = {
    "no_fall",
    "upright_posture",
    "no_ground_penetration",
    "alternating_gait",
    "feet_leave_ground",
    "foot_clearance",
    "steady_progress",
    "energy_accounted",
    "returned_to_standing",
    "robustness_rate",
}


def evaluate(
    *,
    x: list[float],
    t: list[float] | None = None,
    lateral_extent: list[float] | None = None,
    self_penetration: list[float] | None = None,
):
    count = len(x)
    return evaluate_run(
        env_id=0,
        commanded_speed=0.0,
        t=t or [float(index) for index in range(count)],
        x=x,
        lateral_extent_m=lateral_extent or [0.4] * count,
        vx=[1.0] * count,
        gates=(50.0, 100.0),
        finish_distance_m=100.0,
        self_penetration_m=self_penetration or [0.0] * count,
    )


def test_clean_finish_scores_effective_speed() -> None:
    result = evaluate(x=[0.0, 50.0, 100.0])
    assert result.valid is True
    assert result.finish_time_s == 2.0
    assert result.termination_reason == "finished"
    assert result.stop_time_s == 2.0
    assert result.time_to_max_distance_s == 2.0
    assert result.effective_speed_mps == 50.0
    assert [check.name for check in result.checks] == [
        "finished",
        "in_lane",
        "self_collision",
    ]
    assert not OPTIONAL_METRICS.intersection(result.to_dict())


def test_finish_is_scored_when_not_requested_as_an_intermediate_split() -> None:
    result = evaluate_run(
        env_id=0,
        commanded_speed=0.0,
        t=[0.0, 1.0, 2.0],
        x=[0.0, 50.0, 100.0],
        lateral_extent_m=[0.4, 0.4, 0.4],
        vx=[50.0, 50.0, 50.0],
        gates=(25.0, 50.0),
        finish_distance_m=100.0,
        self_penetration_m=[0.0, 0.0, 0.0],
    )

    assert result.termination_reason == "finished"
    assert result.effective_speed_mps == 50.0
    assert result.gate_times_s["100m"] == 2.0


def test_first_stop_condition_excludes_later_violations() -> None:
    result = evaluate(
        x=[0.0, 100.0],
        lateral_extent=[0.4, 0.62],
        self_penetration=[0.0, 0.011],
    )
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {"finished", "self_collision"}
    assert result.valid is False
    assert result.termination_reason == "self_collision"


def test_maximum_forward_distance_is_diagnostic_not_final_displacement() -> None:
    result = evaluate(x=[0.0, 14.5, 3.0])
    assert result.distance_m == 3.0
    assert result.max_distance_m == 14.5
    assert result.raw_max_distance_m == 14.5
    assert result.first_disqualification_gate is None
    assert result.termination_reason == "timeout"
    assert result.stop_time_s == 2.0
    assert result.time_to_max_distance_s == 1.0
    assert result.effective_speed_mps == 2.1025
    assert result.valid is False
    assert [check.name for check in result.checks] == [
        "finished",
        "in_lane",
        "self_collision",
    ]


def test_forward_distance_stops_at_first_lane_disqualification() -> None:
    result = evaluate(
        x=[0.0, 2.0, 44.93],
        lateral_extent=[0.4, 0.505, 3.9667],
    )

    # The trajectory continues far outside the lane for audit/replay, but only
    # The archived diagnostic stops at the interpolated 0.61 m boundary.
    expected_fraction = (0.61 - 0.505) / (3.9667 - 0.505)
    expected_distance = 2.0 + expected_fraction * (44.93 - 2.0)
    assert result.max_distance_m == round(expected_distance, 3)
    assert result.raw_max_distance_m == 44.93
    assert result.first_disqualification_gate == "in_lane"
    assert result.first_disqualification_time_s == round(1.0 + expected_fraction, 3)
    assert result.first_disqualification_distance_m == round(expected_distance, 3)
    expected_time = 1.0 + expected_fraction
    expected_speed = expected_distance**2 / (100.0 * expected_time)
    assert result.time_to_max_distance_s == round(expected_time, 3)
    assert result.effective_speed_mps == round(expected_speed, 6)
    assert result.effective_speed_mps == round(
        effective_speed(expected_distance, expected_time), 6
    )


def test_forward_distance_stops_at_first_self_collision() -> None:
    result = evaluate(
        x=[0.0, 4.0, 20.0],
        self_penetration=[0.0, 0.005, 0.015],
    )

    assert result.max_distance_m == 12.0
    assert result.raw_max_distance_m == 20.0
    assert result.first_disqualification_gate == "self_collision"
    assert result.first_disqualification_time_s == 1.5
    assert result.first_disqualification_distance_m == 12.0
    assert result.time_to_max_distance_s == 1.5
    assert result.effective_speed_mps == 0.96


def test_earliest_disqualification_controls_progress_cutoff() -> None:
    result = evaluate(
        x=[0.0, 3.0, 9.0],
        lateral_extent=[0.4, 0.7, 0.8],
        self_penetration=[0.0, 0.0, 0.02],
    )

    assert result.first_disqualification_gate == "in_lane"
    assert result.max_distance_m < 3.0
    assert result.raw_max_distance_m == 9.0


def test_effective_speed_uses_first_time_of_maximum_legal_progress() -> None:
    result = evaluate(
        t=[0.0, 1.0, 2.0, 3.0],
        x=[0.0, 20.0, 15.0, 10.0],
    )

    assert result.termination_reason == "timeout"
    assert result.stop_time_s == 3.0
    assert result.max_distance_m == 20.0
    assert result.time_to_max_distance_s == 1.0
    assert result.effective_speed_mps == 4.0


@pytest.mark.parametrize(
    ("forward", "lane", "self_penetration", "expected"),
    [
        (100.0, 0.4, 0.0, "finished"),
        (10.0, 0.611, 0.0, "in_lane"),
        (10.0, 0.4, 0.011, "self_collision"),
        (100.0, 0.611, 0.011, "in_lane"),
        (10.0, 0.4, 0.0, None),
    ],
)
def test_rollout_resolves_each_terminal_condition_immediately(
    forward: float,
    lane: float,
    self_penetration: float,
    expected: str | None,
) -> None:
    assert (
        terminal_reason_for_sample(
            forward_m=forward,
            lateral_extent_m=lane,
            self_penetration_m=self_penetration,
            distance_m=100.0,
        )
        == expected
    )


def test_whole_body_lane_extent_includes_collision_radii() -> None:
    import torch

    world_y = torch.tensor([[0.0, 0.50, -0.40], [6.0, 6.20, 5.50]])
    radii = torch.tensor([0.05, 0.12, 0.05])
    centres = torch.tensor([0.0, 6.0])

    extent = max_lateral_extent_from_world_y(world_y, radii, centres)

    assert extent.squeeze(1).tolist() == pytest.approx([0.62, 0.55])


def test_lane_geometry_covers_all_major_body_regions() -> None:
    import json

    geometry = json.loads(
        (TASK / "tests/verifier/collision_geometry.json").read_text()
    )["bodies"]
    required = {
        "head_link",
        "torso_link",
        "pelvis",
        "left_palm_link",
        "right_palm_link",
        "left_six_link",
        "right_six_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    }
    assert required <= geometry.keys()
    assert sum(len(body["points"]) for body in geometry.values()) == 1960


def test_submission_contract_has_one_lean_source() -> None:
    assert not (TASK / "environment" / "check_policy.py").exists()
    verifier = (TASK / "tests" / "check_submission.py").read_text()
    for name in OPTIONAL_METRICS:
        assert name not in verifier


def test_verifier_entrypoint_has_no_optional_sweep_or_reward_fields() -> None:
    entrypoint = (TASK / "tests" / "test.sh").read_text()
    verifier = (TASK / "tests" / "verify.py").read_text()
    for name in OPTIONAL_METRICS:
        assert name not in entrypoint
    assert "robustness" not in verifier.lower()
    assert "perturbations.json" not in verifier
    assert '"max_distance_m"' in entrypoint
    assert '"effective_speed_mps"' in entrypoint
    assert '"reward": score' in entrypoint
    assert '"reward": 1.0 if valid else 0.0' not in entrypoint
    assert '"max_distance_semantics"' in verifier
