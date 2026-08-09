from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_continuous_scoring_is_central_blind_and_bounded() -> None:
    config = tomllib.loads(
        (ROOT / "challenge" / "g1-sprint-100m-lane" / "task.toml").read_text()
    )
    continuous = config["verifier"]["continuous"]
    assert continuous["enabled"] is True
    assert continuous["watch_dir"] == "/app/submissions/queue"
    assert continuous["max_concurrent"] == 1
    assert continuous["return_results_to_agent"] is False
    assert continuous["drain_pending_on_stop"] is True
    assert continuous["continuous_only"] is True
    assert "reuse_matching_result_for_final" not in continuous
    assert continuous["shared_scheduler_lock_path"].endswith("/verifier.lock")
    assert continuous["shared_result_cache_dir"].endswith("/result-cache")


def test_launcher_records_central_blind_scheduler_and_per_run_provenance() -> None:
    launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
    assert '"scoring_queue_scope": "central_blind_per_run_queues"' in launcher
    assert '"scoring_queue_key": run_id' in launcher
    assert '"scoring_global_max_concurrent": 1' in launcher
    assert '"scoring_feedback_policy": "sealed_until_agent_exit"' in launcher
    assert (
        '"evaluation_result_policy": "all_blind_submissions_by_deadline"'
        in launcher
    )
    assert '--ae "SPRINT_SCORING_QUEUE_KEY=$RUN_ID"' in launcher
