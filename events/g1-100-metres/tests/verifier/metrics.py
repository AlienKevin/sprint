# Copyright (c) 2026 Sprint contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Turn a recorded lane into the official time and three gate verdicts.

This event scores only whether the policy reached 100 m inside the
window, stayed in its lane, and avoided disqualifying self-intersection.  Gait,
posture, energy, fall, ground-contact, and robustness heuristics are not part of
the benchmark and are neither computed nor reported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


LANE_HALF_WIDTH_M = 1.22 / 2.0
MAX_SELF_PENETRATION_M = 0.01


@dataclass
class Check:
    """One official gate with its measured value and threshold."""

    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""
    gating: bool = True


@dataclass
class RunResult:
    """One verifier lane's official result and minimal outcome diagnostics."""

    env_id: int
    commanded_speed: float
    distance_m: float
    max_distance_m: float
    raw_max_distance_m: float
    duration_s: float
    finish_time_s: float | None
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
    """Evaluate exactly the three public gates for one recorded lane."""

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

    gate_times = gate_crossing_times(x, t, gates)
    finish_time = gate_times.get(f"{finish_distance_m:g}m")
    end = len(t)
    if finish_time is not None:
        end = next((i + 1 for i, value in enumerate(t) if value >= finish_time), len(t))
    count = min(max(end, 2), len(t))

    duration = t[count - 1] - t[0]
    distance = x[count - 1] - x[0]
    raw_max_distance = max(x[:count]) - x[0]
    mean_speed = distance / duration if duration > 0 else 0.0
    peak_speed = max(vx[:count]) if count else 0.0
    achieved = sum(vx[:count]) / count if count else 0.0
    max_lateral_extent = max(lateral_extent_m[:count])
    max_self = max(self_penetration_m[:count], default=0.0)

    disqualifications: list[tuple[float, str, float]] = []
    lane_crossing = _threshold_crossing(
        lateral_extent_m[:count],
        t[:count],
        x[:count],
        LANE_HALF_WIDTH_M,
    )
    if lane_crossing is not None:
        disqualifications.append((*lane_crossing[:1], "in_lane", lane_crossing[1]))
    self_crossing = _threshold_crossing(
        self_penetration_m[:count],
        t[:count],
        x[:count],
        MAX_SELF_PENETRATION_M,
    )
    if self_crossing is not None:
        disqualifications.append(
            (*self_crossing[:1], "self_collision", self_crossing[1])
        )

    first_disqualification = min(disqualifications, default=None)
    if first_disqualification is None:
        max_distance = raw_max_distance
    else:
        disqualification_time, _, disqualification_x = first_disqualification
        legal_x = [
            position
            for sample_time, position in zip(t[:count], x[:count], strict=True)
            if sample_time < disqualification_time
        ]
        legal_x.append(disqualification_x)
        max_distance = max(legal_x) - x[0]

    checks = [
        Check(
            "finished",
            finish_time is not None,
            max_distance,
            finish_distance_m,
            f"covered {max_distance:.1f} legal m of {finish_distance_m:g} m",
        ),
        Check(
            "in_lane",
            max_lateral_extent <= LANE_HALF_WIDTH_M,
            max_lateral_extent,
            LANE_HALF_WIDTH_M,
            f"max whole-body lateral extent {max_lateral_extent:.2f} m "
            f"from lane centre (vertical boundaries at "
            f"+/-{LANE_HALF_WIDTH_M:.2f} m)",
        ),
        Check(
            "self_collision",
            max_self <= MAX_SELF_PENETRATION_M,
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
            key: None if value is None else round(value, 3)
            for key, value in gate_times.items()
        },
        mean_speed_mps=round(mean_speed, 4),
        peak_speed_mps=round(peak_speed, 4),
        tracking_error_mps=round(abs(commanded_speed - achieved), 4),
        checks=checks,
    )
