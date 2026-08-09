from __future__ import annotations

import contextlib
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
    assert batch_eval.LIVE_SITE_DEPLOY_SECONDS == 20 * 60


def test_batch_matrix_can_launch_three_deepseek_trials_only() -> None:
    rows = batch_eval.matrix("eval-deepseek", families=("deepseek",))
    assert len(rows) == 3
    assert {row["family"] for row in rows} == {"deepseek"}
    assert [row["trial"] for row in rows] == [1, 2, 3]
    assert {row["model"] for row in rows} == {
        "deepseek/deepseek-v4-flash"
    }


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


def test_vercel_project_link_accepts_cli_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = tmp_path / "web"
    (web / ".vercel").mkdir(parents=True)
    (web / ".vercel/project.json").write_text(
        json.dumps(
            {
                "projectId": frontier_update.PROJECT_ID,
                "orgId": frontier_update.ORG_ID,
                "projectName": "sprint",
                "settings": {"nodeVersion": "24.x"},
            }
        )
    )
    monkeypatch.setattr(batch_eval, "WEB", web)
    assert batch_eval.vercel_project_link_ready()


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


def test_shared_verifier_stall_alert_requires_pending_work_and_old_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "scheduler-events.jsonl"
    monkeypatch.setattr(batch_eval, "SHARED_VERIFIER_EVENTS", events)
    payload = {
        "arms": [
            {
                "run_id": "eval-luna-1",
                "ledger": {"queued": 2, "running": 0},
            }
        ]
    }
    now = batch_eval.parse_time("2026-08-08T12:20:00Z")
    events.write_text(
        json.dumps(
            {
                "at": "2026-08-08T12:04:59Z",
                "event": "acquired",
                "queue_key": "eval-luna-1",
            }
        )
        + "\n"
    )

    alerts = batch_eval.shared_verifier_stall_alerts(payload, now=now)
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "shared_verifier_stalled"
    assert alerts[0]["pending_submissions"] == "2"

    events.write_text(
        json.dumps(
            {
                "at": "2026-08-08T12:05:01Z",
                "event": "released",
                "queue_key": "eval-luna-1",
            }
        )
        + "\n"
    )
    assert batch_eval.shared_verifier_stall_alerts(payload, now=now) == []
    payload["arms"][0]["ledger"] = {"queued": 0, "running": 0}
    events.write_text("")
    assert batch_eval.shared_verifier_stall_alerts(payload, now=now) == []


def test_fresh_pending_submission_outranks_old_scheduler_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ops = tmp_path / "ops"
    events = ops / "blind-verifier" / "scheduler-events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps({"at": "2026-08-08T10:00:00Z", "event": "released"}) + "\n"
    )
    ledger = (
        ops
        / "eval-deepseek-1/harbor-jobs/job/task__trial/artifacts/continuous/ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T12:19:00Z",
                "finished_at": None,
                "error": None,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", ops)
    monkeypatch.setattr(batch_eval, "SHARED_VERIFIER_EVENTS", events)
    payload = {
        "arms": [
            {
                "run_id": "eval-deepseek-1",
                "ledger": {"queued": 1, "running": 0},
            }
        ]
    }
    now = batch_eval.parse_time("2026-08-08T12:20:00Z")
    assert batch_eval.shared_verifier_stall_alerts(payload, now=now) == []

    ledger.write_text(
        json.dumps(
            {
                "submitted_at": "2026-08-08T12:00:00Z",
                "finished_at": None,
                "error": None,
            }
        )
        + "\n"
        + ledger.read_text()
    )
    alerts = batch_eval.shared_verifier_stall_alerts(payload, now=now)
    assert len(alerts) == 1
    assert alerts[0]["last_progress_at"] == "2026-08-08T12:00:00+00:00"


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
    queued = next(
        item for item in state["capture_queue"] if item["policy_hash"] == digest
    )
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


def test_replay_renderer_exposes_complete_cli() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "runs/build_lane_3d.py"), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "--capture" in completed.stdout
    assert "--meta-policy" in completed.stdout


def test_capture_retries_are_bounded_and_renderer_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    trial = job / "task__trial"
    web = tmp_path / "web"
    state_path = tmp_path / "eval-luna-1" / "frontier-state.json"
    attempt = trial / "artifacts/continuous/attempts/0001-fall.pt"
    policy = attempt / "artifacts/app/submission/policy.pt"
    replay = attempt / "verifier/replay.json"
    policy.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    state_path.parent.mkdir()
    web.mkdir()
    policy.write_bytes(b"failed-policy")
    replay.write_text('{"schema_version":1,"body_names":[],"frames":[]}\n')
    (attempt / "verifier/sprint_results.json").write_text(
        json.dumps({"valid_run": False, "failed_gates": ["finished"]})
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
                "rewards": {"valid_run": 0.0, "gate_finished": 0.0},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(frontier_update, "PIPELINE_LOCK", tmp_path / "pipeline.lock")
    monkeypatch.setattr(frontier_update, "renderer_source_hash", lambda: "renderer-a")
    calls = 0

    def broken_capture(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("renderer diagnostic")

    monkeypatch.setattr(frontier_update, "capture_and_render", broken_capture)
    state = frontier_update.run_worker(
        job=job,
        trial=trial,
        state_path=state_path,
        web=web,
        deploy=False,
        debounce_seconds=0,
    )
    current = [
        item
        for item in state["capture_queue"]
        if item.get("renderer_hash") == "renderer-a"
    ]
    assert calls == frontier_update.CAPTURE_MAX_ATTEMPTS
    assert len(current) == 1
    assert current[0]["status"] == "error"
    assert current[0]["attempts"] == frontier_update.CAPTURE_MAX_ATTEMPTS
    assert "renderer diagnostic" in current[0]["error"]

    frontier_update.run_worker(
        job=job,
        trial=trial,
        state_path=state_path,
        web=web,
        deploy=False,
        debounce_seconds=0,
    )
    assert calls == frontier_update.CAPTURE_MAX_ATTEMPTS

    monkeypatch.setattr(frontier_update, "renderer_source_hash", lambda: "renderer-b")
    updated = frontier_update.scan_frontier(
        job=job,
        trial=trial,
        state_path=state_path,
        web=web,
    )
    replacement = [
        item
        for item in updated["capture_queue"]
        if item.get("renderer_hash") == "renderer-b"
    ]
    assert len(replacement) == 1
    assert replacement[0]["status"] == "queued"
    assert replacement[0]["attempts"] == 0


def test_replay_worker_waits_for_shared_renderer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[bool] = []

    @contextlib.contextmanager
    def recording_lock(_path: Path, *, blocking: bool = True):
        observed.append(blocking)
        yield True

    state = {"capture_queue": [], "policies": {}, "captures": {}}
    monkeypatch.setattr(frontier_update, "file_lock", recording_lock)
    monkeypatch.setattr(frontier_update, "scan_frontier", lambda **_kwargs: state)
    monkeypatch.setattr(
        frontier_update, "write_web_policy_indexes", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        frontier_update, "atomic_write_json", lambda *_args, **_kwargs: None
    )

    result = frontier_update.run_worker(
        job=tmp_path / "job",
        trial=tmp_path / "trial",
        state_path=tmp_path / "state.json",
        web=tmp_path / "web",
        deploy=False,
        debounce_seconds=0,
    )
    assert result is state
    assert observed == [True]


def test_run_checked_preserves_failure_output() -> None:
    with pytest.raises(RuntimeError, match="renderer diagnostic"):
        frontier_update.run_checked(
            [sys.executable, "-c", "print('renderer diagnostic'); raise SystemExit(9)"]
        )


def test_batch_monitor_service_carries_absolute_uv_vercel_and_modal_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(batch_eval.shutil, "which", lambda name: "/tools/vercel")
    monkeypatch.setattr(
        batch_eval,
        "run_checked",
        lambda command, **_kwargs: commands.append(command) or "ok",
    )
    monkeypatch.setattr(batch_eval.subprocess, "run", lambda *args, **kwargs: None)
    batch_eval.start_monitor_service("eval", tmp_path / ".env", "profile-a")
    command = commands[0]
    assert "--setenv=UV=/home/ubuntu/.local/bin/uv" in command
    assert "--setenv=MODAL_PROFILE=profile-a" in command
    path_arg = next(item for item in command if item.startswith("--setenv=PATH="))
    assert "/tools" in path_arg
    assert str(Path(sys.executable).resolve().parent) in path_arg


def test_batch_monitor_reads_live_lane_status_without_duplicate_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "eval-luna-1"
    state_dir = tmp_path / run_id
    state_dir.mkdir()
    (state_dir / "monitor.pid").write_text("1234\n")
    expected = {
        "schema_version": 2,
        "run_id": run_id,
        "updated_at": "2026-08-08T23:00:00Z",
        "harbor_alive": True,
    }
    (state_dir / "status.json").write_text(json.dumps(expected))
    monkeypatch.setattr(batch_eval, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(batch_eval.sprintctl, "process_alive", lambda *_args: True)
    monkeypatch.setattr(
        batch_eval.sprintctl,
        "monitor_once",
        lambda *_args, **_kwargs: pytest.fail("duplicate monitor poll"),
    )
    assert batch_eval.live_run_monitor_status(run_id) == expected


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
    monkeypatch.setattr(
        batch_eval, "matrix", lambda _batch_id, **_kwargs: arms
    )
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
