from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVENT = ROOT / "events/g1-100-metres"


def test_continuous_scoring_is_blind_bounded_and_globally_serialized() -> None:
    config = tomllib.loads((EVENT / "task.toml").read_text())
    continuous = config["verifier"]["continuous"]
    assert continuous["enabled"] is True
    assert continuous["watch_dir"] == "/durable/submissions/queue"
    assert continuous["max_concurrent"] == 1
    assert continuous["return_results_to_agent"] is False
    assert continuous["return_acknowledgments_to_agent"] is True
    assert continuous["acknowledgments_dir"] == ("/durable/submissions/acknowledgments")
    assert continuous["drain_pending_on_stop"] is True
    assert continuous["continuous_only"] is True
    assert "reuse_matching_result_for_final" not in continuous
    assert continuous["shared_scheduler_lock_path"].endswith("/verifier.lock")
    assert "minimum_submission_interval_sec" not in continuous
    assert "max_outstanding_submissions" not in continuous
    assert continuous["max_submissions"] == 32
    assert continuous["submission_limit_reward_key"] == "submission_contract_valid"
    assert continuous["submission_limit_unique_by_sha256"] is True
    assert continuous["shared_result_cache_dir"] == "$SPRINT_SHARED_CACHE_DIR"


def test_launcher_records_shared_blind_lane_and_submission_cap_provenance() -> None:
    launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
    assert '"scoring_queue_scope": "shared_blind_archival_queue"' in launcher
    assert '"scoring_queue_key": batch_id or "standalone"' in launcher
    assert '"scoring_max_concurrent_per_trial": 1' in launcher
    assert '"scoring_cross_trial_lease": True' in launcher
    assert (
        '"scoring_max_submissions_per_trial": int(submission_cap_per_trial)'
        in launcher
    )
    assert (
        '"scoring_submission_slot_policy": '
        '"unique_structurally_valid_policy"' in launcher
    )
    assert (
        '"scoring_feedback_policy": '
        '"official_results_hidden_agent_local_verification"' in launcher
    )
    assert '"evaluation_result_policy": "all_blind_archival_submissions"' in launcher
    assert '--ae "SPRINT_SCORING_QUEUE_KEY=${BATCH_ID:-standalone}"' in launcher
    assert '--ae "SPRINT_SUBMISSIONS_ROOT=/durable/submissions"' in launcher
    assert (
        '--ae "SPRINT_SUBMISSION_CAP_PER_TRIAL=$SUBMISSION_CAP_PER_TRIAL"'
        in launcher
    )
    assert 'config["verifier"]["continuous"].get("max_submissions")' in launcher
