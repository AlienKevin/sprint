#!/usr/bin/env python3
"""Fail unless agent-published and trusted verifier outputs are equivalent."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


NUMERIC_TOLERANCES = {
    # One 50 Hz control tick. Separate GPU executions may resolve the finish
    # crossing on adjacent physics/control samples even with identical source.
    "best_valid_100m_s": Decimal("0.02"),
    # The canary's observed cross-GPU drift is 4 mm. A 1 cm bound is still far
    # below any meaningful configuration or scoring difference while covering
    # one control-step worth of motion for the deliberately slow probe policy.
    "max_distance_m": Decimal("0.01"),
    "stop_time_s": Decimal("0.02"),
    "time_to_max_distance_s": Decimal("0.02"),
    "effective_speed_mps": Decimal("0.001"),
}


def rounded(value: Any, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def canonical(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical score fields, at the precision published by the benchmark."""
    return {
        "effective_speed_mps": rounded(payload.get("effective_speed_mps"), 6),
        "termination_reason": payload.get("termination_reason"),
        "stop_time_s": rounded(payload.get("stop_time_s")),
        "time_to_max_distance_s": rounded(payload.get("time_to_max_distance_s")),
        "valid_run": bool(payload.get("valid_run")),
        "best_valid_100m_s": rounded(payload.get("best_valid_100m_s")),
        "max_distance_m": rounded(payload.get("max_distance_m")),
        "max_distance_semantics": payload.get("max_distance_semantics"),
        "lane_containment_semantics": payload.get("lane_containment_semantics"),
        "lanes_finished": int(payload.get("lanes_finished") or 0),
        "lanes_valid": int(payload.get("lanes_valid") or 0),
        "lanes_total": int(payload.get("lanes_total") or 0),
        "runs": int(payload.get("runs") or 0),
        "all_valid_times_s": [
            rounded(item) for item in payload.get("all_valid_times_s", [])
        ],
        "failed_gates": sorted(str(item) for item in payload.get("failed_gates", [])),
    }


def equivalent(
    agent: dict[str, Any], official: dict[str, Any]
) -> tuple[bool, dict[str, float]]:
    """Compare verifier results with a tight allowance for GPU-physics jitter."""
    numeric_fields = (
        "effective_speed_mps",
        "stop_time_s",
        "time_to_max_distance_s",
        "best_valid_100m_s",
        "max_distance_m",
    )
    numeric_list_fields = ("all_valid_times_s",)
    exact_agent = {
        key: value
        for key, value in agent.items()
        if key not in numeric_fields + numeric_list_fields
    }
    exact_official = {
        key: value
        for key, value in official.items()
        if key not in numeric_fields + numeric_list_fields
    }
    if exact_agent != exact_official:
        return False, {}

    deltas: dict[str, float] = {}
    for key in numeric_fields:
        left = agent[key]
        right = official[key]
        if (left is None) != (right is None):
            return False, deltas
        if left is not None:
            delta = abs(float(left) - float(right))
            deltas[key] = delta
            tolerance = NUMERIC_TOLERANCES[key]
            # Decimal avoids rejecting an inclusive boundary because binary
            # float subtraction may represent it just above the configured
            # tolerance.
            if Decimal(str(left)) - Decimal(str(right)) > tolerance or (
                Decimal(str(right)) - Decimal(str(left)) > tolerance
            ):
                return False, deltas

    for key in numeric_list_fields:
        left = agent[key]
        right = official[key]
        if len(left) != len(right):
            return False, deltas
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            if (left_item is None) != (right_item is None):
                return False, deltas
            if left_item is not None:
                delta = abs(float(left_item) - float(right_item))
                deltas[f"{key}[{index}]"] = delta
                if (
                    Decimal(str(left_item)) - Decimal(str(right_item))
                    > NUMERIC_TOLERANCES["best_valid_100m_s"]
                    or Decimal(str(right_item)) - Decimal(str(left_item))
                    > NUMERIC_TOLERANCES["best_valid_100m_s"]
                ):
                    return False, deltas
    return True, deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    agent = canonical(json.loads(args.agent.read_text()))
    official = canonical(json.loads(args.official.read_text()))
    is_equivalent, numeric_deltas = equivalent(agent, official)
    if not is_equivalent:
        raise SystemExit(
            "published verifier mismatch:\n"
            f"agent={json.dumps(agent, sort_keys=True)}\n"
            f"official={json.dumps(official, sort_keys=True)}"
        )
    payload = {
        "schema_version": 1,
        "equivalent": True,
        "canonical_result": official,
        "numeric_tolerances": {
            key: float(value) for key, value in NUMERIC_TOLERANCES.items()
        },
        "numeric_deltas": numeric_deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("VERIFIER_EQUIVALENCE_OK " + json.dumps(official, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
