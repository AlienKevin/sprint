#!/usr/bin/env python3
"""Fail unless agent-published and trusted verifier outputs are equivalent."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLOAT_TOLERANCE = 0.002


def rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def canonical(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical score fields, at the precision published by the benchmark."""
    return {
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
    numeric_fields = ("best_valid_100m_s", "max_distance_m")
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
            if not math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
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
                if not math.isclose(
                    float(left_item),
                    float(right_item),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
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
        "numeric_tolerance": FLOAT_TOLERANCE,
        "numeric_deltas": numeric_deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("VERIFIER_EQUIVALENCE_OK " + json.dumps(official, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
