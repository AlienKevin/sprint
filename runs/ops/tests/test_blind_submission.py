from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "challenge" / "g1-sprint-100m-lane" / "environment" / "bin"
OPS = ROOT / "runs" / "ops"
sys.path.insert(0, str(OPS))

import sprintctl  # noqa: E402


def load_script(name: str):
    path = BIN / name
    loader = importlib.machinery.SourceFileLoader(f"test_{name}", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def configure_paths(module, tmp_path: Path) -> None:
    module.QUEUE = str(tmp_path / "submissions" / "queue")
    module.NOTES = str(tmp_path / "submissions" / "notes")
    module.RECEIPTS = str(tmp_path / "submissions" / "receipts")
    module.LOCK = str(tmp_path / "submissions" / "submit.lock")


def test_submit_returns_async_receipt_without_designating_final(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    submit = load_script("sprint-submit")
    configure_paths(submit, tmp_path)
    monkeypatch.setattr(
        submit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="interface ok\n", stderr=""
        ),
    )
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"torchscript-policy")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sprint-submit", str(policy), "--note", "candidate"],
    )

    assert submit.main() == 0
    output = capsys.readouterr().out
    assert "queued" in output
    assert "300 seconds" in output
    assert "sprint-board" in output
    receipts = list(Path(submit.RECEIPTS).glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    expected_hash = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert "final" not in receipt
    assert receipt["policy_sha256"] == expected_hash
    queued = list(Path(submit.QUEUE).glob("*.pt"))
    assert len(queued) == 1
    assert queued[0].read_bytes() == policy.read_bytes()


def test_removed_final_flag_is_rejected(tmp_path: Path, monkeypatch) -> None:
    submit = load_script("sprint-submit")
    configure_paths(submit, tmp_path)
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"policy")
    monkeypatch.setattr(sys, "argv", ["sprint-submit", str(policy), "--final"])

    try:
        submit.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("removed --final flag was accepted")


def test_submission_accepts_after_sixty_prior_receipts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    submit = load_script("sprint-submit")
    configure_paths(submit, tmp_path)
    monkeypatch.setattr(
        submit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"policy")
    receipts = Path(submit.RECEIPTS)
    receipts.mkdir(parents=True)
    for index in range(60):
        (receipts / f"prior-{index}.json").write_text("{}\n")
    monkeypatch.setattr(sys, "argv", ["sprint-submit", str(policy)])

    assert submit.main() == 0
    assert "queued" in capsys.readouterr().out
    assert len(list(Path(submit.RECEIPTS).glob("*.json"))) == 61
    assert len(list(Path(submit.QUEUE).glob("*.pt"))) == 1


def test_board_exposes_pending_and_completed_verifier_feedback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    board = load_script("sprint-board")
    receipts = tmp_path / "receipts"
    results = tmp_path / "results"
    receipts.mkdir()
    results.mkdir()
    board.RECEIPTS = str(receipts)
    board.RESULTS = str(results)
    (receipts / "one.json").write_text(
        json.dumps(
            {
                "submission_id": "one",
                "note": "chosen",
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["sprint-board"])

    assert board.main() == 0
    output = capsys.readouterr().out
    assert "one  chosen  pending host acceptance/result" in output

    (results / "one.pt.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "rewards": {
                    "valid_run": 0.0,
                    "best_100m_s": 0.0,
                    "max_distance_m": 14.5,
                    "gate_finished": 1.0,
                    "gate_in_lane": 0.0,
                    "gate_self_collision": 1.0,
                },
            }
        )
    )
    assert board.main() == 0
    output = capsys.readouterr().out
    assert "one  chosen  DQ  left the lane" in output
    assert "max legal distance" not in output

    payload = json.loads((results / "one.pt.json").read_text())
    payload["rewards"]["gate_finished"] = 0.0
    (results / "one.pt.json").write_text(json.dumps(payload))
    assert board.main() == 0
    output = capsys.readouterr().out
    assert (
        "one  chosen  DQ  did not finish, left the lane; max legal distance: 14.500m"
        in output
    )


def test_board_exposes_valid_score_and_rate_rejection(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    board = load_script("sprint-board")
    receipts = tmp_path / "receipts"
    results = tmp_path / "results"
    receipts.mkdir()
    results.mkdir()
    board.RECEIPTS = str(receipts)
    board.RESULTS = str(results)
    for submission_id in ("valid", "limited"):
        (receipts / f"{submission_id}.json").write_text(
            json.dumps({"submission_id": submission_id, "note": ""})
        )
    (results / "valid.pt.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "rewards": {
                    "valid_run": 1.0,
                    "best_100m_s": 8.948,
                    "gate_finished": 1.0,
                    "gate_in_lane": 1.0,
                    "gate_self_collision": 1.0,
                },
            }
        )
    )
    (results / "limited.pt.json").write_text(
        json.dumps(
            {
                "accepted": False,
                "error": "submission cooldown active",
                "retry_after_sec": 173,
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["sprint-board"])

    assert board.main() == 0
    output = capsys.readouterr().out
    assert "valid  8.948s  valid" in output
    assert "limited  rejected: submission cooldown active; retry after 173s" in output


def test_finalization_requires_nonempty_host_frozen_policy(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    final_dir = trial / "artifacts" / "app" / "submission"
    final_dir.mkdir(parents=True)
    policy = final_dir / "policy.pt"
    policy.write_bytes(b"final-policy")
    assert sprintctl.final_policy_frozen_ready(trial) is True
    policy.write_bytes(b"")
    assert sprintctl.final_policy_frozen_ready(trial) is False


def test_all_submission_finalization_needs_no_final_policy_or_verifier(
    tmp_path: Path,
) -> None:
    state = tmp_path / "run"
    jobs = state / "harbor-jobs"
    job = jobs / "job"
    trial = job / "task__trial"
    continuous = trial / "artifacts" / "continuous"
    continuous.mkdir(parents=True)
    ledger = continuous / "ledger.jsonl"
    ledger.write_text("")
    (trial / "artifacts" / "manifest.json").write_text("[]\n")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-08-09T00:00:00Z",
                "continuous_verification": {"submissions": []},
                "verifier": None,
                "verifier_result": None,
            }
        )
    )
    (job / "result.json").write_text(
        json.dumps({"finished_at": "2026-08-09T00:00:00Z"})
    )
    (state / "STOP_ACK.json").write_text("{}\n")
    (state / "archive-manifest.json").write_text('{"attempts": {}}\n')
    ledger_digest = sprintctl.read_ledger(ledger).digest
    (state / "frontier-state.json").write_text(
        json.dumps(
            {
                "ledger_hash": ledger_digest,
                "capture_queue": [],
                "captures": {},
                "frontier_candidates": [],
                "pending_site_hash": None,
                "site_status": "noop",
            }
        )
    )
    raw_chunk = state / "trace" / "raw.jsonl"
    trajectory = state / "trace" / "trajectory.json"
    raw_chunk.parent.mkdir()
    raw_chunk.write_text('{"type":"provider_rejected"}\n')
    trajectory.write_text('{"schema_version":"ATIF-v1.7","steps":[]}\n')
    usage = state / "usage" / "run-usage-audit.json"
    usage.parent.mkdir()
    usage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "model": "openai/gpt-5.6-luna",
                "resolved_model_version": "gpt-5.6-luna",
                "reasoning_effort": "max",
                "expected_cpu_attempts": [1],
                "captured_cpu_attempts": [1],
                "attempt_coverage_complete": True,
                "request_count": 0,
                "requests": [],
                "pricing_snapshots": [],
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.0,
                "zero_request_reason": (
                    "no completed model request was present in any captured CPU attempt"
                ),
                "source_sessions": [
                    {
                        "cpu_attempt": 1,
                        "request_count": 0,
                        "cost_reconstruction_complete": True,
                        "chunks": [
                            {
                                "path": "trace/raw.jsonl",
                                "sha256": sprintctl.sha256_file(raw_chunk),
                            }
                        ],
                        "trajectory_path": "trace/trajectory.json",
                        "trajectory_sha256": sprintctl.sha256_file(trajectory),
                    }
                ],
            }
        )
    )
    run = {
        "run_id": "run",
        "state_dir": str(state),
        "jobs_root": str(jobs),
        "evaluation_result_policy": "all_blind_submissions_by_deadline",
        "model": "openai/gpt-5.6-luna",
        "resolved_model_version": "gpt-5.6-luna",
        "reasoning_effort": "max",
        "cpu_launch_history": [{"attempt": 1}],
        "usage_audit_required": True,
    }
    (state / "run.json").write_text(json.dumps(run))

    ready, conditions, details = sprintctl.final_conditions(state, run)

    assert ready, details
    assert conditions["continuous_result_set"] is True
    assert conditions["usage_audit_complete"] is True
    assert "final_verifier" not in conditions
    assert "final_policy_frozen" not in conditions
