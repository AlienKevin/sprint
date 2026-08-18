from __future__ import annotations

from event_runtime.preflight.compare_results import canonical, equivalent


def result(**overrides):
    payload = {
        "valid_run": False,
        "best_valid_100m_s": None,
        "max_distance_m": 0.792,
        "max_distance_semantics": "legal_prefix_until_first_disqualification",
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


def test_equivalent_accepts_millimetre_scale_gpu_physics_jitter() -> None:
    matches, deltas = equivalent(result(), result(max_distance_m=0.793))

    assert matches is True
    assert 0 < deltas["max_distance_m"] <= 0.002


def test_equivalent_accepts_inclusive_two_millimetre_boundary() -> None:
    matches, deltas = equivalent(
        result(max_distance_m=0.796), result(max_distance_m=0.794)
    )

    assert matches is True
    assert round(deltas["max_distance_m"], 3) == 0.002


def test_equivalent_rejects_larger_numeric_difference() -> None:
    matches, _ = equivalent(result(), result(max_distance_m=0.795))

    assert matches is False


def test_equivalent_requires_exact_verdicts_and_gates() -> None:
    matches, _ = equivalent(result(), result(failed_gates=["lane_containment"]))

    assert matches is False
