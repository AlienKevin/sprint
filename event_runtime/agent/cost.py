#!/usr/bin/env python3
"""Print this trial's trusted cumulative comparison cost as one JSON document."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SNAPSHOT = Path("/run/sprint-gpu-mirror/cost.json")


def main() -> int:
    if len(sys.argv) != 1:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": "event cost takes no arguments",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        payload = json.loads(SNAPSHOT.read_text())
    except FileNotFoundError:
        payload = {
            "schema_version": 1,
            "status": "unavailable",
            "error": "the first trusted host cost snapshot is not available yet",
        }
        print(json.dumps(payload, sort_keys=True))
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": 1,
            "status": "error",
            "error": f"invalid trusted host cost snapshot: {type(exc).__name__}",
        }
        print(json.dumps(payload, sort_keys=True))
        return 1
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload.get("total_usd") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
