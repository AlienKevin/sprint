#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Score synthetic runs, including the ways of cheating the scoring is for.

Runs on the CPU with no simulator: the point is that the verdict logic is
checkable independently of anything Isaac Sim does.

    python tests/test_metrics.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sprintbench.metrics import evaluate_run, format_table, gate_crossing_times  # noqa: E402

DT = 0.02
GATES = (10.0, 20.0, 50.0, 100.0)


def synth(
    speed: float,
    seconds: float = 40.0,
    *,
    stride_hz: float = 1.6,
    clearance: float = 0.09,
    lateral: float = 0.05,
    tilt: float = 8.0,
    duty: float = 0.6,
    penetration: float = 0.0,
    stall: bool = False,
):
    """A plausible run: constant speed, alternating feet, small lateral weave."""
    n = int(seconds / DT)
    t, x, y, vx, tilt_deg, foot_z, foot_c = [], [], [], [], [], [], []
    pos = 0.0
    for i in range(n):
        tt = i * DT
        # a stalling gait covers the same ground but in lunges
        v = speed * (1.8 if math.sin(2 * math.pi * stride_hz * tt) > 0 else 0.2) if stall else speed
        pos += v * DT
        t.append(tt)
        x.append(pos)
        y.append(lateral * math.sin(2 * math.pi * 0.3 * tt))
        vx.append(v)
        tilt_deg.append(tilt)
        zs, cs = [], []
        for k in range(2):
            phase = (stride_hz * tt + 0.5 * k) % 1.0
            airborne = phase > duty
            # penetration starts after the first second: the reference height is
            # the standing pose at t=0, so a run that is already sunk at reset
            # has nothing to be measured against
            sink = penetration if tt > 1.0 else 0.0
            zs.append(clearance * math.sin(math.pi * (phase - duty) / (1 - duty)) if airborne else -sink)
            cs.append(not airborne)
        foot_z.append(tuple(zs))
        foot_c.append(tuple(cs))
    return dict(t=t, x=x, y=y, vx=vx, tilt_deg=tilt_deg, foot_z=foot_z, foot_contact=foot_c)


def score(trace, speed, **kw):
    return evaluate_run(
        env_id=0, commanded_speed=speed, gates=GATES, finish_distance_m=100.0,
        fell_at_s=kw.pop("fell_at_s", None), stood_after_finish=kw.pop("stood", True),
        foot_reference_m=kw.pop("foot_reference_m", 0.0),
        lowest_point_m=kw.pop("lowest_point_m", None),
        self_penetration_m=kw.pop("self_penetration_m", None), **trace, **kw,
    )


def check(name: str, condition, detail: str = "") -> bool:
    passed = bool(condition)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    return passed


def main() -> int:
    ok = True
    print("clean 3 m/s run")
    r = score(synth(3.0), 3.0)
    ok &= check("valid", r.valid, ", ".join(c.name for c in r.checks if not c.passed) or "all checks pass")
    ok &= check("finish time near 33.3 s", abs(r.finish_time_s - 100.0 / 3.0) < 0.1, f"{r.finish_time_s}")
    ok &= check("tracking error ~0", r.tracking_error_mps < 0.05, f"{r.tracking_error_mps}")
    ok &= check("all gates timed", all(v is not None for v in r.gate_times_s.values()))

    print("\ngate interpolation beats the 20 ms sample grid")
    times = gate_crossing_times([0.0, 0.9, 1.9], [0.0, 0.02, 0.04], (1.0,))
    ok &= check("interpolated", abs(times["1m"] - 0.022) < 1e-6, f"{times['1m']}")

    print("\ntoo slow to finish: 0.4 m/s over 40 s")
    r = score(synth(0.4), 0.4)
    ok &= check("DNF", r.finish_time_s is None)
    ok &= check("invalid", not r.valid)
    ok &= check("failure is 'finished'", [c.name for c in r.checks if not c.passed] == ["finished"])

    print("\nfell at 12 s")
    r = score(synth(3.0, seconds=12.0), 3.0, fell_at_s=12.0)
    ok &= check("no_fall fails", not next(c for c in r.checks if c.name == "no_fall").passed)

    print("\nskating: both feet never leave the ground")
    tr = synth(3.0, duty=1.01)
    r = score(tr, 3.0)
    names = {c.name for c in r.checks if not c.passed}
    ok &= check("caught", {"alternating_gait", "feet_leave_ground"} & names, f"failed: {sorted(names)}")

    print("\nshuffling: feet clear only 1 cm")
    r = score(synth(3.0, clearance=0.01), 3.0)
    ok &= check("caught", "foot_clearance" in {c.name for c in r.checks if not c.passed})

    print("\nfeet 5 cm through the floor")
    r = score(synth(3.0, penetration=0.05), 3.0)
    ok &= check("caught", "no_ground_penetration" in {c.name for c in r.checks if not c.passed})

    print("\none foot starts tilted — the reference must not be read per foot")
    # This is the bug that produced 17 mm of fictitious penetration on the real
    # robot: the left foot settled at an angle, so its link origin sat high, and
    # every later frame where it lay flat scored as the foot sinking.
    tr = synth(3.0)
    tr["foot_z"] = [(z[0] + 0.017, z[1]) if i < 25 else z for i, z in enumerate(tr["foot_z"])]
    r = score(tr, 3.0)
    pen = next(c for c in r.checks if c.name == "no_ground_penetration")
    ok &= check("a tilted foot at t=0 is not penetration", pen.passed,
                f"measured {pen.value * 1000:.1f} mm")

    print("\nwhole-body penetration: a knee through the floor is caught")
    tr = synth(3.0)
    low = [0.002] * len(tr["t"])
    low[400] = -0.031                      # one frame with a limb 3.1 cm under
    r = score(tr, 3.0, lowest_point_m=low)
    pen = next(c for c in r.checks if c.name == "no_ground_penetration")
    ok &= check("measured and reported", not pen.passed, pen.detail)
    # Penetration is a diagnostic, not a gate.  Keeping the simulator honest is
    # the simulator's job; the 2x-fidelity repeat is what catches a result that
    # depends on solver slack, whatever form the slack takes.
    ok &= check("but does not invalidate the time", r.valid and not pen.gating)

    print("\nwhole-body penetration: a clean run passes")
    r = score(tr, 3.0, lowest_point_m=[0.001] * len(tr["t"]))
    pen = next(c for c in r.checks if c.name == "no_ground_penetration")
    ok &= check("passes", pen.passed, pen.detail)

    print("\nreal penetration is still caught with a shared reference")
    tr = synth(3.0, penetration=0.05)
    r = score(tr, 3.0)
    ok &= check("caught", "no_ground_penetration" in {c.name for c in r.checks if not c.passed})

    print("\npogo: same mean speed, delivered in lunges")
    r = score(synth(3.0, stall=True), 3.0)
    ok &= check("caught", "steady_progress" in {c.name for c in r.checks if not c.passed},
                f"failed: {sorted(c.name for c in r.checks if not c.passed)}")

    print("\nveering out of lane")
    r = score(synth(3.0, lateral=1.2), 3.0)
    ok &= check("caught", "in_lane" in {c.name for c in r.checks if not c.passed})

    print("\nself-collision: 2 cm non-adjacent overlap DQs the time")
    tr = synth(3.0)
    self_pen = [0.0] * len(tr["t"])
    self_pen[200] = 0.02
    r = score(tr, 3.0, self_penetration_m=self_pen)
    sc = next(c for c in r.checks if c.name == "self_collision")
    ok &= check("caught", not sc.passed and sc.gating, sc.detail)
    ok &= check("invalidates the time", not r.valid)

    print("\nself-collision: sub-threshold overlap still valid")
    self_pen = [0.005] * len(tr["t"])
    r = score(tr, 3.0, self_penetration_m=self_pen)
    sc = next(c for c in r.checks if c.name == "self_collision")
    ok &= check("passes", sc.passed and r.valid, sc.detail)

    print("\nrunning folded at 70 degrees")
    r = score(synth(3.0, tilt=70.0), 3.0)
    ok &= check("caught", "upright_posture" in {c.name for c in r.checks if not c.passed})

    print("\ncollapsed after crossing the line")
    r = score(synth(3.0), 3.0, stood=False)
    ok &= check("caught", "returned_to_standing" in {c.name for c in r.checks if not c.passed})

    print("\ntable rendering")
    print(format_table([score(synth(s), s) for s in (1.0, 3.0, 5.0)]))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
