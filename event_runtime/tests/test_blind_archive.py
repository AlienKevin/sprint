from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "event_runtime" / "agent"
sys.path.insert(0, str(ROOT))

from event_runtime.control import run as sprintctl  # noqa: E402


def load_script(name: str):
    path = AGENT / name
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
    module.ACKNOWLEDGMENTS = str(tmp_path / "submissions" / "acknowledgments")
    module.LOCK = str(tmp_path / "submissions" / "submit.lock")


def test_submit_returns_async_receipt_without_designating_final(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    submit = load_script("archive.py")
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
        ["event archive", str(policy), "--note", "candidate"],
    )

    assert submit.main() == 0
    output = capsys.readouterr().out
    assert "staged locally" in output
    assert "not accepted until Harbor acknowledges" in output
    assert "event history" in output
    receipts = list(Path(submit.RECEIPTS).glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    expected_hash = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert "final" not in receipt
    assert receipt["policy_sha256"] == expected_hash
    queued = list(Path(submit.QUEUE).glob("*.pt"))
    assert len(queued) == 1
    assert queued[0].read_bytes() == policy.read_bytes()


def test_gpu_archive_routes_to_host_bridge_instead_of_durable_queue(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("SPRINT_GPU_JOB_ID", "job-1")
    monkeypatch.setenv("SPRINT_GPU_ATTEMPT", "2")
    monkeypatch.setenv("SPRINT_GPU_LEASE_ID", "lease-2")
    monkeypatch.setenv("SPRINT_RUN_ID", "run-1")
    monkeypatch.setenv(
        "SPRINT_GPU_DURABLE_SUBMISSION_BRIDGE_ROOT", str(tmp_path / "durable-bridge")
    )
    submit = load_script("archive.py")
    configure_paths(submit, tmp_path)
    monkeypatch.setattr(
        submit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="interface ok\n", stderr=""
        ),
    )
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"gpu-policy")
    monkeypatch.setattr(sys, "argv", ["event archive", str(policy)])

    assert submit.main() == 0
    assert "queued for host submission bridge" in capsys.readouterr().out
    receipt_path = next(Path(submit.RECEIPTS).glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["bridge"] == "host_owned_gpu_submission_v1"
    assert receipt["run_id"] == "run-1"
    assert receipt["gpu_job_id"] == "job-1"
    assert receipt["gpu_attempt"] == 2
    assert receipt["gpu_lease_id"] == "lease-2"
    assert list((tmp_path / "durable-bridge/outbox").glob("*.pt"))


def test_host_archive_request_id_is_idempotent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    submit = load_script("archive.py")
    configure_paths(submit, tmp_path)
    monkeypatch.setattr(
        submit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setenv("SPRINT_HOST_ARCHIVE_REQUEST_ID", "123456-abcd")
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"same-policy")
    monkeypatch.setattr(sys, "argv", ["event archive", str(policy)])

    assert submit.main() == 0
    assert submit.main() == 0
    assert "idempotent replay" in capsys.readouterr().out
    assert len(list(Path(submit.RECEIPTS).glob("*.json"))) == 1
    assert len(list(Path(submit.QUEUE).glob("*.pt"))) == 1


def test_removed_final_flag_is_rejected(tmp_path: Path, monkeypatch) -> None:
    submit = load_script("archive.py")
    configure_paths(submit, tmp_path)
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"policy")
    monkeypatch.setattr(sys, "argv", ["event archive", str(policy), "--final"])

    try:
        submit.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("removed --final flag was accepted")


def test_submission_accepts_after_sixty_prior_receipts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    submit = load_script("archive.py")
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
    monkeypatch.setattr(sys, "argv", ["event archive", str(policy)])

    assert submit.main() == 0
    assert "staged locally" in capsys.readouterr().out
    assert len(list(Path(submit.RECEIPTS).glob("*.json"))) == 61
    assert len(list(Path(submit.QUEUE).glob("*.pt"))) == 1


def test_board_lists_receipts_without_reading_official_results(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    board = load_script("history.py")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    board.RECEIPTS = str(receipts)
    board.ACKNOWLEDGMENTS = str(tmp_path / "acknowledgments")
    (receipts / "one.json").write_text(
        json.dumps(
            {
                "submission_id": "one",
                "note": "chosen",
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["event history"])

    assert board.main() == 0
    output = capsys.readouterr().out
    assert "one  chosen  staged locally; awaiting Harbor acknowledgment" in output
    assert "DQ" not in output
    assert "valid" not in output


def test_duplicate_staged_policy_is_backpressured(
    tmp_path, monkeypatch, capsys
) -> None:
    submit = load_script("archive.py")
    configure_paths(submit, tmp_path)
    monkeypatch.setattr(
        submit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    policy = tmp_path / "candidate.pt"
    policy.write_bytes(b"same-policy")
    monkeypatch.setattr(sys, "argv", ["event archive", str(policy)])
    assert submit.main() == 0
    monkeypatch.setattr(sys, "argv", ["event archive", str(policy)])
    assert submit.main() == 2
    assert "identical policy already staged" in capsys.readouterr().err
    assert len(list(Path(submit.QUEUE).glob("*.pt"))) == 1


def test_history_shows_sanitized_acceptance_without_score(
    tmp_path, monkeypatch, capsys
) -> None:
    board = load_script("history.py")
    receipts = tmp_path / "receipts"
    acknowledgments = tmp_path / "acknowledgments"
    receipts.mkdir()
    acknowledgments.mkdir()
    board.RECEIPTS = str(receipts)
    board.ACKNOWLEDGMENTS = str(acknowledgments)
    (receipts / "one.json").write_text(
        json.dumps({"submission_id": "one", "queue_name": "one.pt"})
    )
    (acknowledgments / "one.pt.json").write_text(
        json.dumps({"state": "accepted", "accepted": True})
    )
    monkeypatch.setattr(sys, "argv", ["event history"])
    assert board.main() == 0
    output = capsys.readouterr().out
    assert "accepted by Harbor; official result hidden" in output
    assert "reward" not in output


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
                "request_count": 1,
                "requests": [
                    {
                        "model": "gpt-5.6-luna",
                        "reasoning_effort": "max",
                        "cost_reconstruction_status": "complete",
                        "usage_reported_at": "2026-08-09T00:00:00Z",
                        "calculated_cost_usd": 0.1,
                    }
                ],
                "pricing_snapshots": [],
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.1,
                "source_sessions": [
                    {
                        "cpu_attempt": 1,
                        "request_count": 1,
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
        "evaluation_result_policy": "all_blind_archival_submissions",
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
