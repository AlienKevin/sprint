#!/usr/bin/env python3
"""Merge fresh-process robustness results into the nominal verifier result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge_results(results_path: Path, draws_path: Path, result_dir: Path) -> dict[str, Any]:
    result = json.loads(results_path.read_text())
    draws = json.loads(draws_path.read_text()).get("draws", [])
    if not isinstance(draws, list):
        raise ValueError("perturbations.draws must be a list")

    completed: list[dict[str, Any]] = []
    missing: list[int | str | None] = []
    for draw in draws:
        if not isinstance(draw, dict):
            raise ValueError("each robustness draw must be an object")
        seed = draw.get("seed")
        candidate = result_dir / f"seed-{seed}.json"
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            missing.append(seed)
            continue
        if payload.get("seed") != seed or payload.get("physics") != draw:
            raise ValueError(f"robustness result provenance mismatch for seed {seed}")
        if payload.get("completed") is not True:
            missing.append(seed)
            continue
        completed.append(payload)

    total = len(draws)
    survived = sum(payload.get("valid_run") is True for payload in completed)
    result["robustness_rate"] = round(survived / total, 4) if total else 0.0
    result["robustness_seeds"] = total
    result["robustness_seeds_total"] = total
    result["robustness_seeds_completed"] = len(completed)
    result["robustness_complete"] = bool(total and len(completed) == total)
    result["robustness_missing_seeds"] = missing
    performance = result.setdefault("performance", {})
    performance["robustness_processes"] = [
        payload.get("performance") for payload in completed
    ]
    atomic_write_json(results_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--draws", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge_results(args.results, args.draws, args.result_dir)
    print(
        "robustness summary: "
        f"{result['robustness_seeds_completed']}/{result['robustness_seeds_total']} "
        f"seeds complete, rate={result['robustness_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
