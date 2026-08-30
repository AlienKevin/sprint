from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_runtime.export import frontier


DIGEST = "a" * 64
OTHER = "b" * 64


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path: Path, *, steps: list[dict] | None = None):
    root = tmp_path / "trial"
    web = tmp_path / "web"
    state_path = root / "frontier-state.json"
    state = {"policies": {DIGEST: {"index": 1, "name": "120000-first.pt"}}}
    write_json(web / "data/trajectories/trial.json", {"steps": steps or []})
    return root, web, state_path, state


def step(number: int, time: str, command: str, response: str) -> dict:
    return {
        "step_id": f"a1-s{number}",
        "public_step_id": number,
        "timestamp": time,
        "tool_calls": [{"arguments": {"command": command}}],
        "observation": {"results": [{"content": response}]},
    }


@pytest.mark.parametrize("response", [
    "queued gpu job job-1 timeout=3600s",
    "status: /durable/runs/trial/gpu-jobs/status/job-1.json",
    "job-1\nwaited",
])
def test_bridge_job_is_bound_to_original_request_not_late_observation(tmp_path, response):
    root, web, state_path, state = fixture(tmp_path, steps=[
        step(2, "2026-08-28T12:00:00Z", "event gpu --submit-output /app/p.pt -- true", response),
        step(9, "2026-08-28T13:00:00Z", "event history", "job-1 accepted"),
    ])
    write_json(root / "gpu-job-registry/job-1.json", {
        "job_id": "job-1", "created_at": "2026-08-28T12:00:00Z",
        "submission_paths": ["/app/p.pt"],
    })
    write_json(root / "submission-bridge/p.json", {
        "policy_sha256": DIGEST, "gpu_job_id": "job-1",
        "observed_at": "2026-08-28T13:01:00Z",
    })
    result = frontier.policy_queue_provenance(state_path, state, web)[DIGEST]
    assert result["queue_source_step_id"] == "a1-s2"
    assert result["enqueued_at"] == "2026-08-28T12:00:00Z"
    assert result["artifact_observed_at"] == "2026-08-28T13:01:00Z"


def test_output_artifact_join_is_limited_to_explicit_submission_paths(tmp_path):
    root, web, state_path, state = fixture(tmp_path, steps=[
        step(2, "2026-08-28T12:00:00Z", "event gpu --output /app/checkpoint.pt --submit-output /app/p.pt -- train", "queued gpu job job-1"),
    ])
    state["policies"][OTHER] = {"index": 2}
    write_json(root / "gpu-job-registry/job-1.json", {
        "created_at": "2026-08-28T12:00:00Z", "submission_paths": ["/app/p.pt"],
        "progress": {"output_artifacts": [
            {"source_path": "/app/p.pt", "sha256": DIGEST},
            {"source_path": "/app/checkpoint.pt", "sha256": OTHER},
        ]},
    })
    result = frontier.policy_queue_provenance(state_path, state, web)
    assert result[DIGEST]["queue_source_step_id"] == "a1-s2"
    assert result[OTHER]["queue_source_step_id"] is None
    assert result[OTHER]["queue_source_basis"] == "unresolved"


def test_recovered_attempt_zero_and_duplicate_hash_choose_earliest_queue(tmp_path):
    root, web, state_path, state = fixture(tmp_path, steps=[
        step(2, "2026-08-28T12:00:00Z", "event gpu --submit-output /app/p.pt -- true", "queued gpu job z-first"),
        step(4, "2026-08-28T12:10:00Z", "event gpu --submit-output /app/p.pt -- true", "queued gpu job a-later"),
    ])
    for name, time in [("z-first", "12:00"), ("a-later", "12:10")]:
        write_json(root / f"gpu-job-registry/{name}.json", {
            "created_at": f"2026-08-28T{time}:00Z", "attempt": 0,
            "progress": {"submission_results": [{"policy_sha256": DIGEST}]},
        })
    result = frontier.policy_queue_provenance(state_path, state, web)[DIGEST]
    assert result["queue_source_job_id"] == "z-first"
    assert result["queue_source_public_step_id"] == 2


def test_direct_archive_joins_immutable_request_name_not_acceptance(tmp_path):
    root, web, state_path, state = fixture(tmp_path, steps=[
        step(3, "2026-08-28T12:00:00Z", "event archive /app/p.pt", "staged locally 120000-first (candidate)"),
        step(8, "2026-08-28T13:00:00Z", "event history", "120000-first accepted"),
    ])
    result = frontier.policy_queue_provenance(state_path, state, web)[DIGEST]
    assert result["queue_source_step_id"] == "a1-s3"
    assert result["queue_source_basis"] == "direct_submission_response"
    assert result["enqueued_at_basis"] == "agent_direct_submission"


def test_unresolved_bridge_is_not_attached_to_last_step(tmp_path):
    root, web, state_path, state = fixture(tmp_path, steps=[
        step(8, "2026-08-28T13:00:00Z", "event history", "interrupted"),
    ])
    write_json(root / "submission-bridge/p.json", {
        "policy_sha256": DIGEST, "gpu_job_id": "missing",
        "observed_at": "2026-08-28T13:01:00Z",
    })
    result = frontier.policy_queue_provenance(state_path, state, web)[DIGEST]
    assert result["queue_source_step_id"] is None
    assert result["queue_source_basis"] == "unresolved"
    assert result["artifact_observed_at"] == "2026-08-28T13:01:00Z"


def test_discarded_batch_response_needs_unique_explicit_path_and_request_interval(tmp_path):
    root, web, state_path, state = fixture(tmp_path, steps=[
        step(2, "2026-08-28T12:00:00.500Z", 'const variants=["p.pt"]; event gpu --submit-output /app/${v} -- true', ""),
        step(3, "2026-08-28T12:00:10Z", "event cost", ""),
    ])
    write_json(root / "gpu-job-registry/job-1.json", {
        "created_at": "2026-08-28T12:00:01Z", "submission_paths": ["/app/p.pt"],
        "progress": {"submission_results": [{"policy_sha256": DIGEST}]},
    })
    result = frontier.policy_queue_provenance(state_path, state, web)[DIGEST]
    assert result["queue_source_step_id"] == "a1-s2"
    assert result["queue_source_basis"] == "gpu_explicit_path_request_interval"
    # Same filename much later is not permission to guess the nearest turn.
    write_json(root / "gpu-job-registry/job-1.json", {
        "created_at": "2026-08-28T12:20:00Z", "submission_paths": ["/app/p.pt"],
        "progress": {"submission_results": [{"policy_sha256": DIGEST}]},
    })
    assert frontier.policy_queue_provenance(state_path, state, web)[DIGEST]["queue_source_step_id"] is None


def test_live_luna_batched_submissions_have_original_queue_turns():
    root = Path(__file__).resolve().parents[2]
    run = "s10-vexp-r123-20260828-luna-4"
    policy_path = root / f"web/data/policies/{run}.json"
    if not policy_path.exists():
        pytest.skip("Local exported Luna policy fixture is not available")
    policies = json.loads(policy_path.read_text())["policies"]
    by_number = {policy["submission_index"]: policy for policy in policies}
    for number in (13, 14, 15):
        assert by_number[number]["queue_source_public_step_id"] == 590
        assert by_number[number]["queue_source_step_id"] == "a1-s593"
    for number in (16, 17, 30, 31, 32):
        assert by_number[number]["queue_source_public_step_id"] == 596
        assert by_number[number]["queue_source_step_id"] == "a1-s599"


def test_every_published_policy_has_a_valid_exact_queue_anchor():
    root = Path(__file__).resolve().parents[2]
    index_path = root / "web/data/policies/index.json"
    if not index_path.exists():
        pytest.skip("Local exported policy index fixture is not available")
    index = json.loads(index_path.read_text())
    required_paths = [
        root / f"web/data/{kind}/{run['run_id']}.json"
        for run in index["runs"]
        for kind in ("policies", "trajectories")
    ]
    if any(not path.exists() for path in required_paths):
        pytest.skip("Local exported policy/trajectory fixtures are incomplete")
    for run in index["runs"]:
        run_id = run["run_id"]
        policies = json.loads((root / f"web/data/policies/{run_id}.json").read_text())["policies"]
        trajectory = json.loads((root / f"web/data/trajectories/{run_id}.json").read_text())
        steps = {step["step_id"]: step for step in trajectory["steps"]}
        for policy in policies:
            source = steps[policy["queue_source_step_id"]]
            assert source["public_step_id"] == policy["queue_source_public_step_id"]
            assert source.get("tool_calls"), (run_id, policy["submission_index"])
            assert policy["queue_source_basis"] not in {"unresolved", "gpu_job_unmapped"}
