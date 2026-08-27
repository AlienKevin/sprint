# Copyright (c) 2026 Sprint contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Turn one recorded lane into its official Effective Speed.

Evaluation ends at the first finish, timeout, lane exit, or self-collision.
Only legal progress through that point contributes to the score. Gait, posture,
energy, fall, ground-contact, and robustness heuristics are not part of the
benchmark and are neither computed nor reported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


LANE_HALF_WIDTH_M = 1.22 / 2.0
MAX_SELF_PENETRATION_M = 0.01


@dataclass
class Check:
    """One terminal-condition diagnostic retained in the trusted artifact."""

    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class RunResult:
    """One verifier lane's score and minimal outcome diagnostics."""

    env_id: int
    commanded_speed: float
    distance_m: float
    max_distance_m: float
    raw_max_distance_m: float
    duration_s: float
    finish_time_s: float | None
    stop_time_s: float
    time_to_max_distance_s: float
    effective_speed_mps: float
    termination_reason: str
    first_disqualification_gate: str | None = None
    first_disqualification_time_s: float | None = None
    first_disqualification_distance_m: float | None = None
    gate_times_s: dict[str, float | None] = field(default_factory=dict)
    mean_speed_mps: float = 0.0
    peak_speed_mps: float = 0.0
    tracking_error_mps: float = 0.0
    checks: list[Check] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


def gate_crossing_times(
    x: list[float], t: list[float], gates: tuple[float, ...]
) -> dict[str, float | None]:
    """First crossing time for every distance gate, linearly interpolated."""

    out: dict[str, float | None] = {f"{gate:g}m": None for gate in gates}
    gate_index = 0
    for index in range(1, len(x)):
        while gate_index < len(gates) and x[index] >= gates[gate_index]:
            x0, x1 = x[index - 1], x[index]
            fraction = 0.0 if x1 == x0 else (gates[gate_index] - x0) / (x1 - x0)
            out[f"{gates[gate_index]:g}m"] = t[index - 1] + fraction * (
                t[index] - t[index - 1]
            )
            gate_index += 1
        if gate_index >= len(gates):
            break
    return out


def _threshold_crossing(
    values: list[float],
    t: list[float],
    x: list[float],
    threshold: float,
) -> tuple[float, float] | None:
    """First threshold violation, interpolated to the boundary crossing."""

    if values[0] > threshold:
        return t[0], x[0]
    for index in range(1, len(values)):
        if values[index] <= threshold:
            continue
        previous = values[index - 1]
        current = values[index]
        fraction = (
            0.0
            if current == previous
            else (threshold - previous) / (current - previous)
        )
        fraction = min(max(fraction, 0.0), 1.0)
        return (
            t[index - 1] + fraction * (t[index] - t[index - 1]),
            x[index - 1] + fraction * (x[index] - x[index - 1]),
        )
    return None


def evaluate_run(
    *,
    env_id: int,
    commanded_speed: float,
    t: list[float],
    x: list[float],
    lateral_extent_m: list[float],
    vx: list[float],
    gates: tuple[float, ...],
    finish_distance_m: float,
    self_penetration_m: list[float] | None,
) -> RunResult:
    """Score one lane at its first public terminal condition."""

    if (
        not t
        or len(t) != len(x)
        or len(t) != len(lateral_extent_m)
        or len(t) != len(vx)
    ):
        raise ValueError(
            "time, position, and velocity traces must be nonempty and aligned"
        )
    if self_penetration_m is None or len(self_penetration_m) != len(t):
        raise ValueError("the official self-collision trace is required")

    # The finish line is authoritative even when a caller requests only a
    # reduced set of intermediate split markers.
    scoring_gates = tuple(sorted(set((*gates, finish_distance_m))))
    gate_times = gate_crossing_times(x, t, scoring_gates)
    raw_finish_time = gate_times.get(f"{finish_distance_m:g}m")
    raw_max_distance = max(0.0, max(x) - x[0])

    lane_crossing = _threshold_crossing(
        lateral_extent_m,
        t,
        x,
        LANE_HALF_WIDTH_M,
    )
    self_crossing = _threshold_crossing(
        self_penetration_m,
        t,
        x,
        MAX_SELF_PENETRATION_M,
    )

    # A DQ wins an exact tie with the finish line. This makes the boundary
    # deterministic and prevents a simultaneous lane exit from becoming a
    # clean finish merely because the finish event was appended first.
    terminal_events: list[tuple[float, int, str, float]] = []
    if lane_crossing is not None:
        terminal_events.append((lane_crossing[0], 0, "in_lane", lane_crossing[1]))
    if self_crossing is not None:
        terminal_events.append(
            (self_crossing[0], 0, "self_collision", self_crossing[1])
        )
    if raw_finish_time is not None:
        terminal_events.append(
            (raw_finish_time, 1, "finished", x[0] + finish_distance_m)
        )
    terminal_events.append((t[-1], 2, "timeout", x[-1]))
    stop_time, _, termination_reason, stop_x = min(terminal_events)

    first_disqualification = (
        (stop_time, termination_reason, stop_x)
        if termination_reason in {"in_lane", "self_collision"}
        else None
    )
    finish_time = stop_time if termination_reason == "finished" else None

    legal_t = [sample_time for sample_time in t if sample_time < stop_time]
    legal_x = [
        position
        for sample_time, position in zip(t, x, strict=True)
        if sample_time < stop_time
    ]
    if not legal_t or legal_t[-1] != stop_time:
        legal_t.append(stop_time)
        legal_x.append(stop_x)

    legal_progress = [max(0.0, position - x[0]) for position in legal_x]
    max_distance = min(finish_distance_m, max(legal_progress, default=0.0))
    max_index = legal_progress.index(max(legal_progress)) if legal_progress else 0
    time_to_max_distance = max(0.0, legal_t[max_index] - t[0])
    effective_speed = (
        max_distance * max_distance / (finish_distance_m * time_to_max_distance)
        if max_distance > 0.0 and time_to_max_distance > 0.0
        else 0.0
    )

    duration = max(0.0, stop_time - t[0])
    distance = stop_x - x[0]
    sampled_before_stop = [
        i for i, sample_time in enumerate(t) if sample_time <= stop_time
    ]
    count = max(1, len(sampled_before_stop))
    mean_speed = distance / duration if duration > 0 else 0.0
    peak_speed = max(vx[:count])
    achieved = sum(vx[:count]) / count
    lateral_before_stop = [
        value
        for sample_time, value in zip(t, lateral_extent_m, strict=True)
        if sample_time < stop_time
    ]
    self_before_stop = [
        value
        for sample_time, value in zip(t, self_penetration_m, strict=True)
        if sample_time < stop_time
    ]
    if termination_reason == "in_lane":
        lateral_before_stop.append(LANE_HALF_WIDTH_M)
    if termination_reason == "self_collision":
        self_before_stop.append(MAX_SELF_PENETRATION_M)
    max_lateral_extent = max(lateral_before_stop, default=lateral_extent_m[0])
    max_self = max(self_before_stop, default=self_penetration_m[0])

    lane_failed = lane_crossing is not None and lane_crossing[0] <= stop_time
    self_failed = self_crossing is not None and self_crossing[0] <= stop_time

    checks = [
        Check(
            "finished",
            termination_reason == "finished",
            max_distance,
            finish_distance_m,
            f"covered {max_distance:.1f} legal m of {finish_distance_m:g} m",
        ),
        Check(
            "in_lane",
            not lane_failed,
            max_lateral_extent,
            LANE_HALF_WIDTH_M,
            f"max whole-body lateral extent {max_lateral_extent:.2f} m "
            f"from lane centre (vertical boundaries at "
            f"+/-{LANE_HALF_WIDTH_M:.2f} m)",
        ),
        Check(
            "self_collision",
            not self_failed,
            max_self,
            MAX_SELF_PENETRATION_M,
            f"max non-adjacent link penetration {max_self * 100:.2f} cm "
            f"(2 cm radius pad; DQ above {MAX_SELF_PENETRATION_M * 100:.0f} cm)",
        ),
    ]

    return RunResult(
        env_id=env_id,
        commanded_speed=commanded_speed,
        distance_m=round(distance, 3),
        max_distance_m=round(max_distance, 3),
        raw_max_distance_m=round(raw_max_distance, 3),
        duration_s=round(duration, 3),
        finish_time_s=None if finish_time is None else round(finish_time, 3),
        stop_time_s=round(stop_time, 3),
        time_to_max_distance_s=round(time_to_max_distance, 3),
        effective_speed_mps=round(effective_speed, 6),
        termination_reason=termination_reason,
        first_disqualification_gate=(
            None if first_disqualification is None else first_disqualification[1]
        ),
        first_disqualification_time_s=(
            None
            if first_disqualification is None
            else round(first_disqualification[0], 3)
        ),
        first_disqualification_distance_m=(
            None
            if first_disqualification is None
            else round(first_disqualification[2] - x[0], 3)
        ),
        gate_times_s={
            key: None if value is None or value > stop_time else round(value, 3)
            for key, value in gate_times.items()
        },
        mean_speed_mps=round(mean_speed, 4),
        peak_speed_mps=round(peak_speed, 4),
        tracking_error_mps=round(abs(commanded_speed - achieved), 4),
        checks=checks,
    )
