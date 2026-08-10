from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "challenge" / "g1-sprint-100m-lane"
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
    assert result.valid is False
    assert [check.name for check in result.checks] == [
        "finished",
        "in_lane",
        "self_collision",
    ]


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
