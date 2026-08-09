from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_continuous_scoring_has_one_independent_rate_limited_lane_per_trial() -> None:
    config = tomllib.loads(
        (ROOT / "challenge" / "g1-sprint-100m-lane" / "task.toml").read_text()
    )
    continuous = config["verifier"]["continuous"]
    assert continuous["enabled"] is True
    assert continuous["watch_dir"] == "/app/submissions/queue"
    assert continuous["max_concurrent"] == 1
    assert continuous["return_results_to_agent"] is True
    assert continuous["drain_pending_on_stop"] is True
    assert continuous["continuous_only"] is True
    assert "reuse_matching_result_for_final" not in continuous
    assert "shared_scheduler_lock_path" not in continuous
    assert continuous["minimum_submission_interval_sec"] == 300.0
    assert continuous["max_outstanding_submissions"] == 1
    assert continuous["shared_result_cache_dir"].endswith("/result-cache")


def test_launcher_records_independent_lane_and_rate_limit_provenance() -> None:
    launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
    assert '"scoring_queue_scope": "independent_feedback_per_trial_queue"' in launcher
    assert '"scoring_queue_key": run_id' in launcher
    assert '"scoring_max_concurrent_per_trial": 1' in launcher
    assert '"scoring_cross_trial_lease": False' in launcher
    assert '"scoring_minimum_submission_interval_sec": 300' in launcher
    assert '"scoring_max_outstanding_submissions_per_trial": 1' in launcher
    assert '"scoring_feedback_policy": "score_and_gates_when_ready"' in launcher
    assert '"evaluation_result_policy": "all_feedback_submissions"' in launcher
    assert '--ae "SPRINT_SCORING_QUEUE_KEY=$RUN_ID"' in launcher
