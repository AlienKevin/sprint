#!/usr/bin/env python3
"""Fail unless agent-published and trusted verifier outputs are equivalent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def canonical(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical score fields, at the precision published by the benchmark."""
    return {
        "valid_run": bool(payload.get("valid_run")),
        "best_valid_100m_s": rounded(payload.get("best_valid_100m_s")),
        "max_distance_m": rounded(payload.get("max_distance_m")),
        "max_distance_semantics": payload.get("max_distance_semantics"),
        "lanes_finished": int(payload.get("lanes_finished") or 0),
        "lanes_valid": int(payload.get("lanes_valid") or 0),
        "lanes_total": int(payload.get("lanes_total") or 0),
        "runs": int(payload.get("runs") or 0),
        "all_valid_times_s": [
            rounded(item) for item in payload.get("all_valid_times_s", [])
        ],
        "failed_gates": sorted(str(item) for item in payload.get("failed_gates", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    agent = canonical(json.loads(args.agent.read_text()))
    official = canonical(json.loads(args.official.read_text()))
    if agent != official:
        raise SystemExit(
            "published verifier mismatch:\n"
            f"agent={json.dumps(agent, sort_keys=True)}\n"
            f"official={json.dumps(official, sort_keys=True)}"
        )
    payload = {
        "schema_version": 1,
        "equivalent": True,
        "canonical_result": official,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("VERIFIER_EQUIVALENCE_OK " + json.dumps(official, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
