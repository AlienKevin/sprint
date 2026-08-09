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


def test_submit_returns_blind_receipt_without_designating_final(
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
    assert "accepted" in output
    assert "sealed until this run ends" in output
    assert "score" not in output.lower()
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
    assert "accepted" in capsys.readouterr().out
    assert len(list(Path(submit.RECEIPTS).glob("*.json"))) == 61
    assert len(list(Path(submit.QUEUE).glob("*.pt"))) == 1


def test_board_exposes_receipts_but_no_verifier_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    board = load_script("sprint-board")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    board.RECEIPTS = str(receipts)
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
    assert "one  chosen" in output
    assert "FINAL" not in output
    assert "results are sealed" in output
    for forbidden in ("score", "queued", "running", "finished", "dq", "100 m"):
        assert forbidden not in output.lower()


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
    run = {
        "run_id": "run",
        "state_dir": str(state),
        "jobs_root": str(jobs),
        "evaluation_result_policy": "all_blind_submissions_by_deadline",
    }
    (state / "run.json").write_text(json.dumps(run))

    ready, conditions, details = sprintctl.final_conditions(state, run)

    assert ready, details
    assert conditions["continuous_result_set"] is True
    assert "final_verifier" not in conditions
    assert "final_policy_frozen" not in conditions
