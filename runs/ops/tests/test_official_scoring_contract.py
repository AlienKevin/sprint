from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "challenge" / "g1-100-metres"
sys.path.insert(0, str(TASK / "tests"))

from sprintbench.metrics import evaluate_run  # noqa: E402


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
    y: list[float] | None = None,
    self_penetration: list[float] | None = None,
):
    count = len(x)
    return evaluate_run(
        env_id=0,
        commanded_speed=0.0,
        t=[float(index) for index in range(count)],
        x=x,
        y=y or [0.0] * count,
        vx=[1.0] * count,
        gates=(50.0, 100.0),
        finish_distance_m=100.0,
        self_penetration_m=self_penetration or [0.0] * count,
    )


def test_only_the_three_public_gates_are_scored() -> None:
    result = evaluate(x=[0.0, 50.0, 100.0])
    assert result.valid is True
    assert result.finish_time_s == 2.0
    assert [check.name for check in result.checks] == [
        "finished",
        "in_lane",
        "self_collision",
    ]
    assert not OPTIONAL_METRICS.intersection(result.to_dict())


def test_lane_and_self_collision_are_independent_hard_gates() -> None:
    result = evaluate(
        x=[0.0, 100.0],
        y=[0.0, 0.62],
        self_penetration=[0.0, 0.011],
    )
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {"in_lane", "self_collision"}
    assert result.valid is False


def test_maximum_forward_distance_is_diagnostic_not_final_displacement() -> None:
    result = evaluate(x=[0.0, 14.5, 3.0])
    assert result.distance_m == 3.0
    assert result.max_distance_m == 14.5
    assert result.raw_max_distance_m == 14.5
    assert result.first_disqualification_gate is None
    assert result.valid is False
    assert [check.name for check in result.checks] == [
        "finished",
        "in_lane",
        "self_collision",
    ]


def test_forward_distance_stops_at_first_lane_disqualification() -> None:
    result = evaluate(
        x=[0.0, 2.0, 44.93],
        y=[0.0, 0.305, 3.9667],
    )

    # The trajectory continues far outside the lane for audit/replay, but only
    # The archived diagnostic stops at the interpolated 0.61 m boundary.
    expected_fraction = (0.61 - 0.305) / (3.9667 - 0.305)
    expected_distance = 2.0 + expected_fraction * (44.93 - 2.0)
    assert result.max_distance_m == round(expected_distance, 3)
    assert result.raw_max_distance_m == 44.93
    assert result.first_disqualification_gate == "in_lane"
    assert result.first_disqualification_time_s == round(1.0 + expected_fraction, 3)
    assert result.first_disqualification_distance_m == round(expected_distance, 3)


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


def test_earliest_disqualification_controls_progress_cutoff() -> None:
    result = evaluate(
        x=[0.0, 3.0, 9.0],
        y=[0.0, 0.7, 0.8],
        self_penetration=[0.0, 0.0, 0.02],
    )

    assert result.first_disqualification_gate == "in_lane"
    assert result.max_distance_m < 3.0
    assert result.raw_max_distance_m == 9.0


def test_agent_and_verifier_contracts_are_identical_and_lean() -> None:
    agent = (TASK / "environment" / "bin" / "sprint-check").read_text()
    verifier = (TASK / "tests" / "check_submission.py").read_text()
    normalized_agent = agent.replace(
        "sprint-check policy.pt", "python check_submission.py policy.pt"
    )
    assert normalized_agent == verifier
    for name in OPTIONAL_METRICS:
        assert name not in agent


def test_verifier_entrypoint_has_no_optional_sweep_or_reward_fields() -> None:
    entrypoint = (TASK / "tests" / "test.sh").read_text()
    verifier = (TASK / "tests" / "verify.py").read_text()
    for name in OPTIONAL_METRICS:
        assert name not in entrypoint
    assert "robustness" not in verifier.lower()
    assert "perturbations.json" not in verifier
    assert '"max_distance_m"' in entrypoint
    assert '"max_distance_semantics"' in verifier
