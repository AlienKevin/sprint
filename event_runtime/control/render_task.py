#!/usr/bin/env python3
"""Render the event task from its globally configured budget."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path


BUDGET_PLACEHOLDER = "{{ agent_cost_budget_usd }}"


def normalize_budget(value: str) -> str:
    budget = float(value)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("agent cost budget must be a positive finite number")
    return format(budget, "g")


def render_instruction(template: str, budget: str) -> str:
    if template.count(BUDGET_PLACEHOLDER) != 1:
        raise ValueError(
            f"instruction template must contain exactly one {BUDGET_PLACEHOLDER!r}"
        )
    return template.replace(BUDGET_PLACEHOLDER, normalize_budget(budget))


def render_task(source: Path, destination: Path, budget: str) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    rendered_instruction = render_instruction(
        (source / "instruction.md").read_text(), budget
    )
    if destination.exists():
        if (destination / "instruction.md").read_text() != rendered_instruction:
            raise ValueError(f"existing rendered task does not match: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copytree(
            source,
            temporary,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (temporary / "instruction.md").write_text(rendered_instruction)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--budget", required=True)
    args = parser.parse_args()
    render_task(args.source, args.destination, args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
