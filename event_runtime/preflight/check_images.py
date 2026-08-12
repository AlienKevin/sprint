#!/usr/bin/env python3
"""Reject an event launch if its Modal image warmup is missing or stale."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.event import load_event  # noqa: E402
from event_runtime.image import (  # noqa: E402
    agent_context_roots,
    context_digest,
    verifier_context_roots,
)


EVENT = load_event(repository_root=ROOT)
MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
CONTEXTS = {
    "agent_training": agent_context_roots(EVENT),
    "verifier": verifier_context_roots(EVENT),
}


def main() -> int:
    try:
        payload = json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "Modal images are not warmed; run event_runtime/preflight/warm_images.py "
            f"before launching ({exc})"
        ) from exc
    if payload.get("schema_version") != 1 or payload.get("completed") is not True:
        raise SystemExit("Modal image warmup manifest is incomplete")
    if payload.get("unique_image_count") != 2:
        raise SystemExit("Modal warmup did not cover both Sprint image definitions")
    recorded = payload.get("contexts") or {}
    for name, roots in CONTEXTS.items():
        expected = context_digest(*roots)
        entry = recorded.get(name) or {}
        if entry.get("sha256") != expected or not entry.get("image_id"):
            raise SystemExit(
                f"Modal {name} image changed or was not built; rerun image warmup"
            )
    probes = payload.get("verifier_probes") or []
    if len(probes) != 2 or any(not probe.get("sandbox_id") for probe in probes):
        raise SystemExit("Modal warmup did not complete two verifier probes")
    print(
        "Modal image warmup verified: agent/training + verifier definitions "
        "and two verifier executions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
