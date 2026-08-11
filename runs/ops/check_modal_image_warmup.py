#!/usr/bin/env python3
"""Reject an evaluation launch if its Modal image warmup is missing or stale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "runs/ops/modal-image-warmup.json"
CONTEXTS = {
    "agent_training": ROOT / "events/g1-100-metres/environment",
    "verifier": ROOT / "events/g1-100-metres/tests",
}


def context_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    try:
        payload = json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "Modal images are not warmed; run runs/ops/warm_modal_images.py "
            f"before launching ({exc})"
        ) from exc
    if payload.get("schema_version") != 1 or payload.get("completed") is not True:
        raise SystemExit("Modal image warmup manifest is incomplete")
    if payload.get("unique_image_count") != 2:
        raise SystemExit("Modal warmup did not cover both Sprint image definitions")
    recorded = payload.get("contexts") or {}
    for name, root in CONTEXTS.items():
        expected = context_digest(root)
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
