from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harbor.trial.trial import _reattest_scrubbed_usage_provenance


def test_post_scrub_usage_provenance_hashes_match_scrubbed_snapshots(
    tmp_path: Path,
) -> None:
    provenance_dir = tmp_path / "agent" / "usage-provenance"
    provenance_dir.mkdir(parents=True)
    source = provenance_dir / "source-session.jsonl"
    trajectory = provenance_dir / "trajectory.json"
    source.write_text('{"api_key":"[REDACTED]"}\n')
    trajectory.write_text('{"steps":[{"text":"[REDACTED]"}]}\n')
    audit = {
        "schema_version": 1,
        "provenance": {
            "source_session_path": "usage-provenance/source-session.jsonl",
            "source_session_sha256": "stale",
            "trajectory_path": "usage-provenance/trajectory.json",
            "trajectory_sha256": "stale",
        },
    }
    audit_path = tmp_path / "agent" / "usage-audit.json"
    audit_path.write_text(json.dumps(audit))

    assert _reattest_scrubbed_usage_provenance(tmp_path) is True

    repaired = json.loads(audit_path.read_text())
    provenance = repaired["provenance"]
    assert provenance["source_session_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert provenance["trajectory_sha256"] == hashlib.sha256(
        trajectory.read_bytes()
    ).hexdigest()
    assert provenance["attestation_phase"] == "post_secret_scrub"


def test_post_scrub_usage_provenance_rejects_escape(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n")
    (agent / "usage-audit.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "source_session_path": "../outside.jsonl",
                    "source_session_sha256": "stale",
                }
            }
        )
    )

    assert _reattest_scrubbed_usage_provenance(tmp_path) is False
