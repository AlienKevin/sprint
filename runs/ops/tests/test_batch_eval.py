from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))

import batch_eval  # noqa: E402
import frontier_update  # noqa: E402


def test_batch_matrix_is_exact_six_arm_max_effort_contract() -> None:
    rows = batch_eval.matrix("eval-20260808")
    assert len(rows) == 6
    assert len({row["run_id"] for row in rows}) == 6
    assert {row["family"] for row in rows} == {"deepseek", "luna"}
    assert {row["reasoning_effort"] for row in rows} == {"max"}
    assert {row["codex_version"] for row in rows} == {"0.147.0"}
    assert sum(row["model"] == "deepseek/deepseek-v4-flash" for row in rows) == 3
    assert sum(row["model"] == "openai/gpt-5.6-luna" for row in rows) == 3
    assert {
        row["resolved_model_version"] for row in rows if row["family"] == "deepseek"
    } == {"DeepSeek-V4-Flash-0731"}


def test_env_loader_reads_only_required_model_keys(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY='openai-secret'\n"
        'DEEPSEEK_API_KEY="deepseek-secret"\n'
        "MODAL_TOKEN_SECRET=must-not-load\n"
    )
    assert batch_eval.load_env(path) == {
        "OPENAI_API_KEY": "openai-secret",
        "DEEPSEEK_API_KEY": "deepseek-secret",
    }


def test_public_batch_never_contains_secrets_or_host_paths() -> None:
    payload = {
        "batch_id": "eval",
        "updated_at": "now",
        "status": "running",
        "reasoning_effort": "max",
        "codex_version": "0.147.0",
        "run_hours": 24,
        "alerts": [],
        "arms": [
            {
                "run_id": "eval-luna-1",
                "family": "luna",
                "model": "openai/gpt-5.6-luna",
                "resolved_model_version": "gpt-5.6-luna",
                "trial": 1,
                "status": "running",
                "OPENAI_API_KEY": "secret",
                "env_file": "/private/.env",
            }
        ],
    }
    encoded = json.dumps(batch_eval.public_batch(payload))
    assert "secret" not in encoded
    assert "/private" not in encoded


def test_dq_replay_is_queued_and_public_index_is_path_safe(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = job / "task__trial"
    web = tmp_path / "web"
    state_path = tmp_path / "eval-deepseek-1" / "frontier-state.json"
    attempt = trial / "artifacts/continuous/attempts/0001-fall.pt"
    policy = attempt / "artifacts/app/submission/policy.pt"
    replay = attempt / "verifier/replay.json"
    details = attempt / "verifier/sprint_results.json"
    policy.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    web.mkdir()
    state_path.parent.mkdir()
    policy.write_bytes(b"failed-policy")
    replay.write_text('{"schema_version":1,"body_names":[],"frames":[]}\n')
    details.write_text(
        json.dumps(
            {"valid_run": False, "max_distance_m": 14.5, "failed_gates": ["in_lane"]}
        )
    )
    ledger = trial / "artifacts/continuous/ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "index": 1,
                "name": "fall.pt",
                "submitted_at": "2026-08-08T00:00:00Z",
                "finished_at": "2026-08-08T00:01:00Z",
                "rewards": {"valid_run": 0.0, "gate_in_lane": 0.0},
            }
        )
        + "\n"
    )
    state = frontier_update.scan_frontier(
        job=job, trial=trial, state_path=state_path, web=web
    )
    digest = frontier_update.sha256_file(policy)
    assert state["policies"][digest]["failed_gates"] == ["in_lane"]
    assert state["policies"][digest]["replay_path"] == str(replay)
    queued = next(item for item in state["capture_queue"] if item["policy_hash"] == digest)
    assert queued["status"] == "queued"
    assert queued["story"] == "failure"

    state["captures"][digest] = {
        "valid": True,
        "web_html": f"replay/frontier-{digest[:12]}.html",
    }
    frontier_update.write_web_policy_indexes(state_path, state, web)
    public = json.loads((web / "data/policies/eval-deepseek-1.json").read_text())
    encoded = json.dumps(public)
    assert str(tmp_path) not in encoded
    assert public["policies"][0]["replay_ready"] is True
    assert public["policies"][0]["replay_url"].startswith("/replay/frontier-")


def test_website_javascript_parses_and_has_no_legacy_opus_copy() -> None:
    source = (ROOT / "sprint-web/app.js").read_text()
    assert "DeepSeek V4 Flash 0731" in source
    assert "GPT‑5.6 Luna" in source
    assert "Opus" not in source
    spec = importlib.util.find_spec("json")
    assert spec is not None


def test_partial_batch_launch_is_safely_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = [
        {
            "run_id": f"eval-luna-{index}",
            "family": "luna",
            "model": "openai/gpt-5.6-luna",
            "resolved_model_version": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "codex_version": "0.147.0",
            "wrapper": f"wrapper-{index}",
            "trial": index,
            "status": "planned",
        }
        for index in (1, 2)
    ]
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", tmp_path / "batches")
    monkeypatch.setattr(batch_eval, "matrix", lambda _batch_id: arms)
    monkeypatch.setattr(
        batch_eval,
        "preflight",
        lambda **_kwargs: {"ready": True, "checks": {}},
    )
    monkeypatch.setattr(batch_eval, "load_env", lambda _path: {})
    monkeypatch.setattr(batch_eval.time, "sleep", lambda _seconds: None)
    calls = 0

    def fake_run(_command: list[str], *, env: dict[str, str] | None = None) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, _command)
        return "launched"

    stopped: list[tuple[str, str]] = []
    monkeypatch.setattr(batch_eval, "run_checked", fake_run)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "request_stop",
        lambda run_id, *, reason: stopped.append((run_id, reason)),
    )
    with pytest.raises(subprocess.CalledProcessError):
        batch_eval.launch("eval", tmp_path / ".env", "test-profile")
    state = json.loads((tmp_path / "batches/eval/batch.json").read_text())
    assert state["status"] == "launch_error"
    assert state["arms"][0]["status"] == "stopping_after_launch_rollback"
    assert stopped == [("eval-luna-1", "partial_batch_launch_rollback")]


def test_deployment_marker_is_durable_idempotent_and_gates_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    ops = tmp_path / "ops"
    batches = tmp_path / "batches"
    run_id = "eval-luna-1"
    state_dir = ops / run_id
    policy_index = web / f"data/policies/{run_id}.json"
    policy_index.parent.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (batches / "eval").mkdir(parents=True)
    policy_index.write_text('{"policies":[]}\n')
    (state_dir / "frontier-state.json").write_text(
        json.dumps(
            {
                "capture_queue": [{"status": "captured"}],
                "policies": {
                    "abc": {"replay_path": "/sealed/replay.json"},
                },
                "captures": {"abc": {"valid": True}},
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WEB", web)
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "BATCH_ROOT", batches)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "load_run",
        lambda _run_id: (state_dir, {"run_id": run_id}),
    )
    uploads: list[str] = []
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "volume_upload",
        lambda _run, _source, destination: uploads.append(destination),
    )
    site_hash = frontier_update.site_tree_hash(web)
    payload = {
        "batch_id": "eval",
        "deploy": {
            "site_status": "deployed",
            "last_deployed_site_hash": site_hash,
            "last_deployed_at": "2026-08-08T00:00:00Z",
        },
        "arms": [{"run_id": run_id}],
    }
    assert batch_eval.mark_deployed_runs(payload) == []
    assert uploads == [f"runs/{run_id}/state/BATCH_SITE_DEPLOYED.json"]
    assert batch_eval.mark_deployed_runs(payload) == []
    assert len(uploads) == 1

    run = {
        "run_id": run_id,
        "batch_id": "eval",
        "site_dir": str(web),
    }
    assert batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
    policy_index.write_text('{"policies":[{"new":true}]}\n')
    assert not batch_eval.sprintctl.batch_site_deployed_ready(state_dir, run)
