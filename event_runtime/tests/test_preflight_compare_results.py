from __future__ import annotations

from event_runtime.preflight.compare_results import canonical, equivalent


def result(**overrides):
    payload = {
        "effective_speed_mps": 0.0079,
        "termination_reason": "timeout",
        "stop_time_s": 60.0,
        "time_to_max_distance_s": 1.0,
        "valid_run": False,
        "best_valid_100m_s": None,
        "max_distance_m": 0.792,
        "max_distance_semantics": "legal_prefix_until_first_terminal_condition",
        "lane_containment_semantics": (
            "whole_body_collision_envelope_between_vertical_boundaries"
        ),
        "lanes_finished": 0,
        "lanes_valid": 0,
        "lanes_total": 1,
        "runs": 1,
        "all_valid_times_s": [],
        "failed_gates": ["finished"],
    }
    payload.update(overrides)
    return canonical(payload)


def test_equivalent_accepts_observed_cross_gpu_physics_jitter() -> None:
    matches, deltas = equivalent(result(), result(max_distance_m=0.796))

    assert matches is True
    assert 0 < deltas["max_distance_m"] <= 0.01


def test_equivalent_accepts_inclusive_one_centimetre_boundary() -> None:
    matches, deltas = equivalent(
        result(max_distance_m=0.802), result(max_distance_m=0.792)
    )

    assert matches is True
    assert round(deltas["max_distance_m"], 3) == 0.01


def test_equivalent_rejects_larger_numeric_difference() -> None:
    matches, _ = equivalent(result(), result(max_distance_m=0.803))

    assert matches is False


def test_equivalent_compares_the_effective_speed_reward() -> None:
    matches, deltas = equivalent(
        result(effective_speed_mps=0.0079),
        result(effective_speed_mps=0.0088),
    )
    assert matches is True
    assert round(deltas["effective_speed_mps"], 4) == 0.0009

    matches, _ = equivalent(
        result(effective_speed_mps=0.0079),
        result(effective_speed_mps=0.0091),
    )
    assert matches is False


def test_equivalent_uses_one_control_tick_for_finish_times() -> None:
    matches, _ = equivalent(
        result(
            valid_run=True,
            best_valid_100m_s=12.0,
            all_valid_times_s=[12.0],
            failed_gates=[],
            lanes_finished=1,
            lanes_valid=1,
        ),
        result(
            valid_run=True,
            best_valid_100m_s=12.02,
            all_valid_times_s=[12.02],
            failed_gates=[],
            lanes_finished=1,
            lanes_valid=1,
        ),
    )

    assert matches is True


def test_equivalent_requires_exact_verdicts_and_gates() -> None:
    matches, _ = equivalent(result(), result(failed_gates=["lane_containment"]))

    assert matches is False
