"""Continuous verification: submissions scored while the agent still runs."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import logging
from unittest.mock import AsyncMock  # noqa: F401

import pytest

import harbor.trial.continuous as continuous_module
from harbor.environments.base import ExecResult
from harbor.models.task.config import ContinuousVerificationConfig, TaskOS
from harbor.models.trial.result import ContinuousSubmission
from harbor.models.verifier.result import VerifierResult
from harbor.trial.continuous import ContinuousVerificationService

POLICY = "/app/submission/policy.pt"


def test_continuous_only_requires_continuous_verification():
    with pytest.raises(ValueError, match="continuous_only requires"):
        ContinuousVerificationConfig(continuous_only=True)


def test_shared_paths_expand_environment_without_allowing_relative_paths(
    monkeypatch, tmp_path
):
    shared = tmp_path / "shared"
    monkeypatch.setenv("SPRINT_SHARED_STATE_DIR", str(shared))
    config = ContinuousVerificationConfig(
        shared_scheduler_lock_path="$SPRINT_SHARED_STATE_DIR/verifier.lock",
        shared_result_cache_dir="$SPRINT_SHARED_STATE_DIR/result-cache",
    )

    assert config.shared_scheduler_lock_path == str(shared / "verifier.lock")
    assert config.shared_result_cache_dir == str(shared / "result-cache")

    monkeypatch.delenv("SPRINT_SHARED_STATE_DIR")
    with pytest.raises(ValueError, match="must resolve to an absolute host path"):
        ContinuousVerificationConfig(
            shared_scheduler_lock_path="$SPRINT_SHARED_STATE_DIR/verifier.lock"
        )


class FakeAgentEnv:
    """An agent environment backed by a local directory.

    Only the four calls the service makes are modelled: listing the watch dir,
    downloading a submission, uploading a result, and creating directories.
    """

    os = TaskOS.LINUX

    def __init__(self, root: Path):
        self.root = root
        self.watch = root / "queue"
        self.results = root / "results"
        self.watch.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.uploads: list[str] = []
        self.upload_error: Exception | None = None
        self.list_error: Exception | None = None

    async def exec(self, command: str, **kwargs) -> ExecResult:
        if self.list_error is not None:
            raise self.list_error
        if command.startswith("mkdir"):
            return ExecResult(stdout="", stderr="", return_code=0)
        lines = [
            f"{p.name}\t{p.stat().st_size}"
            for p in sorted(self.watch.iterdir())
            if p.is_file()
        ]
        return ExecResult(stdout="\n".join(lines), stderr="", return_code=0)

    async def download_file(self, source_path, target_path: Path) -> None:
        name = str(source_path).rsplit("/", 1)[-1]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes((self.watch / name).read_bytes())

    async def upload_file(self, source_path: Path, target_path) -> None:
        if self.upload_error is not None:
            raise self.upload_error
        name = str(target_path).rsplit("/", 1)[-1]
        (self.results / name).write_bytes(Path(source_path).read_bytes())
        self.uploads.append(str(target_path))


def _service(tmp_path: Path, env, run_verifier, drain: float = 30.0, **overrides):
    verification_context = overrides.pop("verification_context", "task-v1")
    queue_key = overrides.pop("queue_key", "model-run")
    retryable_verifier_error = overrides.pop("retryable_verifier_error", None)
    config = ContinuousVerificationConfig(
        enabled=True,
        watch_dir=str(env.watch),
        results_dir=str(env.results),
        poll_interval_sec=overrides.pop("poll_interval_sec", 0.01),
        submission_path=POLICY,
        **overrides,
    )
    return ContinuousVerificationService(
        config=config,
        environment=env,
        artifacts_dir=tmp_path / "artifacts",
        run_verifier=run_verifier,
        verification_timeout_sec=drain,
        verification_context=verification_context,
        queue_key=queue_key,
        logger=logging.getLogger("continuous-test"),
        retryable_verifier_error=retryable_verifier_error,
    )


async def _settle(service, iterations: int = 6) -> None:
    """Give the watcher enough polls to see a file and its size settle."""
    for _ in range(iterations):
        await asyncio.sleep(0.02)
    while service._tasks:
        await asyncio.gather(*list(service._tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_submission_is_scored_and_result_returned(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    seen: list[tuple[str, Path]] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        seen.append((key, paths))
        return VerifierResult(rewards={"reward": 1.0, "best_100m_s": 12.5})

    service = _service(tmp_path, env, run_verifier)
    (env.watch / "attempt-1.pt").write_bytes(b"weights")

    async with service.running():
        await _settle(service)

    # Scored once, and the staged file sits where the verifier expects it.
    assert len(seen) == 1
    key, paths = seen[0]
    assert key == "continuous-0001"
    staged = paths.artifacts_dir / "app" / "submission" / "policy.pt"
    assert staged.read_bytes() == b"weights"
    # Each attempt gets its own verifier output dir, so one attempt cannot read
    # another's reward file, or the trial's.
    assert paths.verifier_dir.is_dir()

    record = service.summary.submissions[0]
    assert record.reward == 1.0
    assert record.rewards["best_100m_s"] == 12.5
    assert record.error is None
    assert record.duration_sec is not None

    # The agent gets the result back under the name it submitted.
    returned = json.loads((env.results / "attempt-1.pt.json").read_text())
    assert returned["rewards"]["reward"] == 1.0
    assert env.uploads == [f"{env.results}/attempt-1.pt.json"]


@pytest.mark.asyncio
async def test_retryable_verifier_loss_uses_fresh_sandbox_key(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls: list[str] = []

    async def run_verifier(key: str, _paths) -> VerifierResult:
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("provider lost sandbox")
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(
        tmp_path,
        env,
        run_verifier,
        max_verifier_attempts=3,
        verifier_retry_backoff_sec=0,
        retryable_verifier_error=lambda exc: "lost sandbox" in str(exc),
    )
    (env.watch / "policy.pt").write_bytes(b"weights")

    async with service.running():
        await _settle(service)

    record = service.summary.submissions[0]
    assert calls == ["continuous-0001", "continuous-0001-retry-2"]
    assert record.reward == 1.0
    assert record.verification_attempts == 2
    assert len(record.verification_retry_events) == 1
    assert record.verification_retry_events[0]["attempt"] == 1
    assert record.verification_retry_events[0]["error_type"] == "RuntimeError"
    assert record.verification_retry_events[0]["error"] == "provider lost sandbox"
    assert record.verification_retry_events[0]["failed_at"]
    assert record.error is None


@pytest.mark.asyncio
async def test_retryable_verifier_loss_exhaustion_is_terminal(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls = 0

    async def run_verifier(_key: str, _paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider lost sandbox")

    service = _service(
        tmp_path,
        env,
        run_verifier,
        max_verifier_attempts=2,
        verifier_retry_backoff_sec=0,
        retryable_verifier_error=lambda _exc: True,
    )
    (env.watch / "policy.pt").write_bytes(b"weights")

    async with service.running():
        await _settle(service)

    record = service.summary.submissions[0]
    assert calls == 2
    assert record.rewards is None
    assert record.verification_attempts == 2
    assert len(record.verification_retry_events) == 1
    assert record.verification_retry_events[0]["error"] == "provider lost sandbox"
    assert record.error == "RuntimeError: provider lost sandbox"


def _write_recovery_record(
    service: ContinuousVerificationService,
    *,
    name: str = "policy.pt",
    error: str | None = (
        "DownloadVerifierDirError: Failed to download verifier directory from "
        "environment"
    ),
    attempts: int = 1,
) -> ContinuousSubmission:
    record = ContinuousSubmission(
        name=name,
        index=1,
        submitted_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc) if error else None,
        verification_attempts=attempts,
        artifact_path=(
            f"continuous/attempts/0001-{name}/artifacts/app/submission/policy.pt"
        ),
        artifact_sha256=(
            "7abbc513b25fa4fe3d50caf371b6c0a0cac93f9ddf05e6bf9c2b04776bab4f2d"
        ),
        error=error,
    )
    artifact = service._artifacts_dir / record.artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"archived-policy")
    service._ledger.parent.mkdir(parents=True, exist_ok=True)
    service._ledger.write_text(record.model_dump_json() + "\n")
    return record


@pytest.mark.asyncio
async def test_restart_resumes_retryable_row_from_exact_archived_bytes(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls: list[tuple[str, bytes]] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        policy = paths.artifacts_dir / "app/submission/policy.pt"
        calls.append((key, policy.read_bytes()))
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(
        tmp_path,
        env,
        run_verifier,
        max_verifier_attempts=3,
        verifier_retry_backoff_sec=0,
        retryable_verifier_error=lambda _exc: True,
    )
    _write_recovery_record(service)
    # A restarted service must not trust mutable queue bytes or create a second
    # identity for the same filename.
    (env.watch / "policy.pt").write_bytes(b"mutated-agent-copy")

    async with service.running():
        await _settle(service)

    assert calls == [("continuous-0001-retry-2", b"archived-policy")]
    assert len(service.summary.submissions) == 1
    record = service.summary.submissions[0]
    assert record.rewards == {"reward": 1.0}
    assert record.error is None
    assert record.verification_attempts == 2
    assert record.verification_recovery_events[0]["resume_attempt"] == 2
    assert record.verification_recovery_events[0]["artifact_sha256"] == (
        record.artifact_sha256
    )


@pytest.mark.asyncio
async def test_restart_resumes_interrupted_row_without_duplicate(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls = 0

    async def run_verifier(_key: str, _paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": 0.5})

    service = _service(tmp_path, env, run_verifier, max_verifier_attempts=3)
    _write_recovery_record(service, error=None, attempts=1)

    async with service.running():
        await _settle(service)

    assert calls == 1
    assert len(service.summary.submissions) == 1
    assert service.summary.submissions[0].reward == 0.5


@pytest.mark.asyncio
async def test_restart_does_not_replay_deterministic_or_exhausted_rows(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls = 0

    async def run_verifier(_key: str, _paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier, max_verifier_attempts=3)
    record = _write_recovery_record(
        service, error="ValueError: invalid policy", attempts=1
    )
    async with service.running():
        await _settle(service)
    assert calls == 0
    assert service.summary.submissions[0].error == record.error

    exhausted = _service(
        tmp_path / "exhausted", env, run_verifier, max_verifier_attempts=3
    )
    _write_recovery_record(exhausted, attempts=3)
    async with exhausted.running():
        await _settle(exhausted)
    assert calls == 0


@pytest.mark.asyncio
async def test_restart_fails_closed_on_corrupt_archived_artifact(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(_key: str, _paths) -> VerifierResult:
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier, max_verifier_attempts=3)
    record = _write_recovery_record(service)
    (service._artifacts_dir / record.artifact_path).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        await service.start()


@pytest.mark.asyncio
async def test_non_retryable_verifier_error_runs_once(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls = 0

    async def run_verifier(_key: str, _paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        raise ValueError("bad verifier config")

    service = _service(
        tmp_path,
        env,
        run_verifier,
        verifier_retry_backoff_sec=0,
        retryable_verifier_error=lambda _exc: False,
    )
    (env.watch / "policy.pt").write_bytes(b"weights")

    async with service.running():
        await _settle(service)

    record = service.summary.submissions[0]
    assert calls == 1
    assert record.verification_attempts == 1
    assert record.verification_retry_events == []
    assert record.error == "ValueError: bad verifier config"


@pytest.mark.asyncio
async def test_submitted_bytes_are_archived_before_scoring(tmp_path):
    """The scored bytes are the archived ones, even if the agent overwrites."""
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(key: str, paths) -> VerifierResult:
        # Training continues and clobbers its own checkpoint mid-verification.
        (env.watch / "policy.pt").write_bytes(b"newer-and-different")
        return VerifierResult(rewards={"reward": 0.0})

    service = _service(tmp_path, env, run_verifier)
    (env.watch / "policy.pt").write_bytes(b"original")

    async with service.running():
        await _settle(service)

    record = service.summary.submissions[0]
    archived = tmp_path / "artifacts" / record.artifact_path
    assert archived.read_bytes() == b"original"
    assert record.result_path is not None
    assert (tmp_path / "artifacts" / record.result_path).exists()


@pytest.mark.asyncio
async def test_half_written_submission_is_not_scored(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    calls: list[Path] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        calls.append(paths.artifacts_dir)
        return VerifierResult(rewards={"reward": 1.0})

    # Poll slower than the writes, so consecutive polls always see the file
    # at a different size while it is still growing.
    service = _service(tmp_path, env, run_verifier, poll_interval_sec=0.05)
    await service.start()
    try:
        target = env.watch / "growing.pt"
        for chunk in range(30):
            target.write_bytes(b"x" * (100 * (chunk + 1)))
            await asyncio.sleep(0.01)
            assert calls == [], "scored a submission that was still being written"

        await _settle(service, iterations=20)
        assert len(calls) == 1
        assert (calls[0] / "app" / "submission" / "policy.pt").stat().st_size == 3000
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_agent_is_not_blocked_by_a_slow_verification(tmp_path):
    """Submitting costs a file write; scoring happens off to the side."""
    env = FakeAgentEnv(tmp_path / "env")
    released = asyncio.Event()

    async def run_verifier(key: str, paths) -> VerifierResult:
        await released.wait()
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier)
    agent_steps = 0

    async with service.running():
        (env.watch / "slow.pt").write_bytes(b"w")
        # The "agent" keeps working while verification is parked.
        for _ in range(20):
            await asyncio.sleep(0.01)
            agent_steps += 1
        assert agent_steps == 20
        assert service.summary.submissions[0].finished_at is None
        released.set()

    # stop() waits for the in-flight verification rather than dropping it.
    assert service.summary.submissions[0].reward == 1.0


@pytest.mark.asyncio
async def test_last_second_submission_still_counts(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(key: str, paths) -> VerifierResult:
        return VerifierResult(rewards={"reward": 0.5})

    service = _service(tmp_path, env, run_verifier)
    async with service.running():
        # Written with no poll interval left to settle in.
        (env.watch / "buzzer.pt").write_bytes(b"w")

    assert [s.name for s in service.summary.submissions] == ["buzzer.pt"]
    assert service.summary.best_reward == 0.5


@pytest.mark.asyncio
async def test_verifier_failure_is_recorded_not_raised(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(key: str, paths) -> VerifierResult:
        raise RuntimeError("verifier container died")

    service = _service(tmp_path, env, run_verifier)
    (env.watch / "broken.pt").write_bytes(b"w")

    async with service.running():
        await _settle(service)

    record = service.summary.submissions[0]
    assert record.rewards is None
    assert record.reward is None
    assert "verifier container died" in record.error
    # The agent is told why, rather than being left waiting on a file.
    assert json.loads((env.results / "broken.pt.json").read_text())["error"]


@pytest.mark.asyncio
async def test_poll_failure_does_not_kill_the_watcher(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(key: str, paths) -> VerifierResult:
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier)
    async with service.running():
        env.list_error = OSError("environment briefly unreachable")
        await asyncio.sleep(0.05)
        env.list_error = None
        (env.watch / "after.pt").write_bytes(b"w")
        await _settle(service)

    assert [s.name for s in service.summary.submissions] == ["after.pt"]


@pytest.mark.asyncio
async def test_submissions_past_the_cap_are_rejected_with_a_reason(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    scored: list[str] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        scored.append(key)
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier, max_submissions=2)
    async with service.running():
        for i in range(3):
            (env.watch / f"a{i}.pt").write_bytes(b"w")
            await _settle(service)

    assert len(scored) == 2
    rejected = service.summary.submissions[2]
    assert rejected.accepted is False
    assert rejected.rewards is None
    assert "limit of 2" in rejected.error
    assert json.loads((env.results / "a2.pt.json").read_text())["error"]


@pytest.mark.asyncio
async def test_one_outstanding_submission_is_enforced_by_trusted_host(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls: list[str] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        calls.append(key)
        first_started.set()
        await release_first.wait()
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(
        tmp_path,
        env,
        run_verifier,
        max_outstanding_submissions=1,
        minimum_submission_interval_sec=300,
    )
    async with service.running():
        (env.watch / "first.pt").write_bytes(b"one")
        await asyncio.wait_for(first_started.wait(), timeout=1)
        (env.watch / "second.pt").write_bytes(b"two")
        await asyncio.sleep(0.1)
        release_first.set()
        await _settle(service)

    assert calls == ["continuous-0001"]
    first, second = service.summary.submissions
    assert first.accepted is True
    assert first.rewards == {"reward": 1.0}
    assert second.accepted is False
    assert second.rewards is None
    assert "outstanding" in (second.error or "")
    assert json.loads((env.results / "second.pt.json").read_text())["accepted"] is False


@pytest.mark.asyncio
async def test_duplicate_bytes_are_new_requests_and_consume_cooldown(
    tmp_path, monkeypatch
):
    env = FakeAgentEnv(tmp_path / "env")
    now = [datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(continuous_module, "_now", lambda: now[0])
    calls = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(
        tmp_path,
        env,
        run_verifier,
        minimum_submission_interval_sec=300,
        max_outstanding_submissions=1,
        shared_result_cache_dir=str(tmp_path / "trusted-cache"),
    )
    async with service.running():
        (env.watch / "request-a.pt").write_bytes(b"same-policy")
        await _settle(service)

        now[0] += timedelta(seconds=299)
        (env.watch / "request-b.pt").write_bytes(b"same-policy")
        await _settle(service)

        now[0] += timedelta(seconds=1)
        (env.watch / "request-c.pt").write_bytes(b"same-policy")
        await _settle(service)

    first, rejected, third = service.summary.submissions
    assert first.accepted is True
    assert rejected.accepted is False
    assert rejected.retry_after_sec == 1
    assert "cooldown" in (rejected.error or "")
    assert third.accepted is True
    assert third.cache_hit is True
    assert first.artifact_sha256 == rejected.artifact_sha256 == third.artifact_sha256
    assert calls == 1


@pytest.mark.asyncio
async def test_request_id_replay_and_cooldown_survive_service_restart(
    tmp_path, monkeypatch
):
    env = FakeAgentEnv(tmp_path / "env")
    now = [datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(continuous_module, "_now", lambda: now[0])
    calls = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": 1.0})

    first = _service(
        tmp_path,
        env,
        run_verifier,
        minimum_submission_interval_sec=300,
        max_outstanding_submissions=1,
    )
    (env.watch / "stable-request.pt").write_bytes(b"policy")
    async with first.running():
        await _settle(first)

    now[0] += timedelta(seconds=100)
    restarted = _service(
        tmp_path,
        env,
        run_verifier,
        minimum_submission_interval_sec=300,
        max_outstanding_submissions=1,
    )
    async with restarted.running():
        # The original filename remains in the queue and is idempotently seen.
        await asyncio.sleep(0.05)
        (env.watch / "new-request.pt").write_bytes(b"new-policy")
        await _settle(restarted)

    assert calls == 1
    assert [row.name for row in restarted.summary.submissions] == [
        "stable-request.pt",
        "new-request.pt",
    ]
    assert restarted.summary.submissions[0].accepted is True
    assert restarted.summary.submissions[1].accepted is False
    assert restarted.summary.submissions[1].retry_after_sec == 200


@pytest.mark.asyncio
async def test_one_verification_at_a_time_by_default(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    concurrent = 0
    peak = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier)
    for i in range(3):
        (env.watch / f"p{i}.pt").write_bytes(b"w")

    async with service.running():
        await _settle(service, iterations=20)

    # Two timed trials sharing a GPU would measure each other.
    assert peak == 1
    assert len(service.summary.submissions) == 3


@pytest.mark.asyncio
async def test_result_survives_a_failed_handback(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(key: str, paths) -> VerifierResult:
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier)
    env.upload_error = OSError("no route to environment")
    (env.watch / "x.pt").write_bytes(b"w")

    async with service.running():
        await _settle(service)

    record = service.summary.submissions[0]
    assert record.reward == 1.0
    assert (tmp_path / "artifacts" / record.result_path).exists()


@pytest.mark.asyncio
async def test_ledger_records_every_attempt_in_order(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    rewards = iter([0.0, 1.0])

    async def run_verifier(key: str, paths) -> VerifierResult:
        return VerifierResult(rewards={"reward": next(rewards)})

    service = _service(tmp_path, env, run_verifier)
    async with service.running():
        for name in ("first.pt", "second.pt"):
            (env.watch / name).write_bytes(b"w")
            await _settle(service)

    ledger = tmp_path / "artifacts" / "continuous" / "ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [e["name"] for e in entries] == ["first.pt", "second.pt"]
    assert [e["index"] for e in entries] == [1, 2]
    assert service.summary.best_reward == 1.0


@pytest.mark.asyncio
async def test_dotfiles_are_ignored(tmp_path):
    """Write-then-rename is the safe way to submit, so a temp name is skipped."""
    env = FakeAgentEnv(tmp_path / "env")
    scored: list[str] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        scored.append(key)
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(tmp_path, env, run_verifier)
    async with service.running():
        staging = env.watch / ".job.pt"
        staging.write_bytes(b"partial")
        await _settle(service)
        assert scored == [], "scored a file still being staged"
        staging.rename(env.watch / "job.pt")
        await _settle(service)

    assert [s.name for s in service.summary.submissions] == ["job.pt"]
    assert len(scored) == 1


@pytest.mark.asyncio
async def test_a_backlog_does_not_hold_the_trial_open(tmp_path):
    """An agent that dumps its queue at the end must not stall the run.

    Verifications are serialized and each takes minutes, so waiting one out per
    queued submission would keep the trial alive for hours after the agent had
    nothing left to do.
    """
    env = FakeAgentEnv(tmp_path / "env")
    started = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal started
        started += 1
        await asyncio.sleep(0.15)
        return VerifierResult(rewards={"reward": 1.0})

    # A drain budget of roughly one verification, as the trial passes.
    service = _service(tmp_path, env, run_verifier, drain=0.25)
    for i in range(6):
        (env.watch / f"dump{i}.pt").write_bytes(b"w")

    async with service.running():
        # Long enough for the watcher to see them settle and spawn every task,
        # nowhere near long enough to score them. Not _settle(), which drains
        # the queue itself and would hide exactly what is under test.
        await asyncio.sleep(0.08)

    # The ones that never ran say so, rather than sitting unfinished.
    dropped = [s for s in service.summary.submissions if s.rewards is None]
    assert dropped, "expected some of the backlog to be dropped"
    assert all("not scored" in (s.error or "") for s in dropped)
    assert started < 6, "drained the whole backlog instead of bounding it"
    # And the ledger accounts for every submission either way.
    ledger = tmp_path / "artifacts" / "continuous" / "ledger.jsonl"
    assert len(ledger.read_text().splitlines()) == 6


@pytest.mark.asyncio
async def test_blind_mode_returns_no_score_or_completion_signal(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")

    async def run_verifier(key: str, paths) -> VerifierResult:
        return VerifierResult(rewards={"reward": 1.0, "best_100m_s": 9.5})

    service = _service(
        tmp_path,
        env,
        run_verifier,
        return_results_to_agent=False,
        drain_pending_on_stop=True,
    )
    (env.watch / "blind.pt").write_bytes(b"weights")

    async with service.running():
        await _settle(service)

    assert list(env.results.iterdir()) == []
    assert env.uploads == []
    record = service.summary.submissions[0]
    assert record.reward == 1.0
    archived = tmp_path / "artifacts" / record.result_path
    assert json.loads(archived.read_text())["rewards"]["best_100m_s"] == 9.5


@pytest.mark.asyncio
async def test_blind_mode_drains_every_submission_after_agent_exit(tmp_path):
    env = FakeAgentEnv(tmp_path / "env")
    scored: list[str] = []

    async def run_verifier(key: str, paths) -> VerifierResult:
        scored.append(key)
        await asyncio.sleep(0.03)
        return VerifierResult(rewards={"reward": 1.0})

    service = _service(
        tmp_path,
        env,
        run_verifier,
        drain=0.001,
        return_results_to_agent=False,
        drain_pending_on_stop=True,
    )
    for i in range(5):
        (env.watch / f"blind-{i}.pt").write_bytes(str(i).encode())

    async with service.running():
        await asyncio.sleep(0.08)

    assert len(scored) == 5
    assert all(row.rewards is not None for row in service.summary.submissions)


@pytest.mark.asyncio
async def test_shared_scheduler_serializes_independent_model_queues(tmp_path):
    concurrent = 0
    peak = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return VerifierResult(rewards={"reward": 1.0})

    lock = tmp_path / "trusted" / "verifier.lock"
    env_a = FakeAgentEnv(tmp_path / "env-a")
    env_b = FakeAgentEnv(tmp_path / "env-b")
    service_a = _service(
        tmp_path / "a",
        env_a,
        run_verifier,
        queue_key="terra-high",
        shared_scheduler_lock_path=str(lock),
        return_results_to_agent=False,
        drain_pending_on_stop=True,
    )
    service_b = _service(
        tmp_path / "b",
        env_b,
        run_verifier,
        queue_key="deepseek-max",
        shared_scheduler_lock_path=str(lock),
        return_results_to_agent=False,
        drain_pending_on_stop=True,
    )
    (env_a.watch / "a.pt").write_bytes(b"a")
    (env_b.watch / "b.pt").write_bytes(b"b")

    async with service_a.running(), service_b.running():
        await asyncio.sleep(0.3)

    assert peak == 1
    assert service_a.summary.submissions[0].queue_key == "terra-high"
    assert service_b.summary.submissions[0].queue_key == "deepseek-max"
    events = [
        json.loads(line)
        for line in (lock.parent / "scheduler-events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "acquired",
        "released",
        "acquired",
        "released",
    ]


@pytest.mark.asyncio
async def test_independent_trial_lanes_do_not_block_each_other(tmp_path):
    concurrent = 0
    peak = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return VerifierResult(rewards={"reward": 1.0})

    env_a = FakeAgentEnv(tmp_path / "env-a")
    env_b = FakeAgentEnv(tmp_path / "env-b")
    service_a = _service(
        tmp_path / "a",
        env_a,
        run_verifier,
        queue_key="trial-a",
        minimum_submission_interval_sec=300,
        max_outstanding_submissions=1,
    )
    service_b = _service(
        tmp_path / "b",
        env_b,
        run_verifier,
        queue_key="trial-b",
        minimum_submission_interval_sec=300,
        max_outstanding_submissions=1,
    )
    (env_a.watch / "a.pt").write_bytes(b"a")
    (env_b.watch / "b.pt").write_bytes(b"b")

    async with service_a.running(), service_b.running():
        await asyncio.sleep(0.3)

    assert peak == 2
    assert service_a.summary.submissions[0].queue_key == "trial-a"
    assert service_b.summary.submissions[0].queue_key == "trial-b"


@pytest.mark.asyncio
async def test_duplicate_policy_reuses_checksummed_trusted_result(tmp_path):
    calls = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": 1.0, "best_100m_s": 8.25})

    lock = tmp_path / "trusted" / "verifier.lock"
    cache = tmp_path / "trusted" / "cache"
    records = []
    for label in ("terra-high", "deepseek-max"):
        env = FakeAgentEnv(tmp_path / label)
        service = _service(
            tmp_path / f"artifacts-{label}",
            env,
            run_verifier,
            queue_key=label,
            verification_context="sealed-verifier-v7",
            shared_scheduler_lock_path=str(lock),
            shared_result_cache_dir=str(cache),
            return_results_to_agent=False,
            drain_pending_on_stop=True,
        )
        (env.watch / "same.pt").write_bytes(b"identical-policy")
        async with service.running():
            await _settle(service)
        records.append(service.summary.submissions[0])

    assert calls == 1
    assert records[0].cache_hit is False
    assert records[1].cache_hit is True
    assert records[0].artifact_sha256 == records[1].artifact_sha256
    assert records[0].evaluation_fingerprint == records[1].evaluation_fingerprint
    assert records[1].source_evaluation_id == records[0].source_evaluation_id
    assert records[1].rewards == records[0].rewards


@pytest.mark.asyncio
async def test_cache_is_not_reused_across_verifier_contexts(tmp_path):
    calls = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": float(calls)})

    lock = tmp_path / "trusted" / "verifier.lock"
    cache = tmp_path / "trusted" / "cache"
    for context in ("course-v1", "course-v2"):
        env = FakeAgentEnv(tmp_path / context)
        service = _service(
            tmp_path / f"artifacts-{context}",
            env,
            run_verifier,
            verification_context=context,
            shared_scheduler_lock_path=str(lock),
            shared_result_cache_dir=str(cache),
            return_results_to_agent=False,
            drain_pending_on_stop=True,
        )
        (env.watch / "same.pt").write_bytes(b"identical-policy")
        async with service.running():
            await _settle(service)

    assert calls == 2


@pytest.mark.asyncio
async def test_shared_lease_is_released_after_verifier_failure(tmp_path):
    calls = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("worker terminated")
        return VerifierResult(rewards={"reward": 1.0})

    lock = tmp_path / "trusted" / "verifier.lock"
    records = []
    for label in ("failed-run", "next-run"):
        env = FakeAgentEnv(tmp_path / label)
        service = _service(
            tmp_path / f"artifacts-{label}",
            env,
            run_verifier,
            queue_key=label,
            shared_scheduler_lock_path=str(lock),
            return_results_to_agent=False,
            drain_pending_on_stop=True,
        )
        (env.watch / "policy.pt").write_bytes(label.encode())
        async with service.running():
            await _settle(service)
        records.append(service.summary.submissions[0])

    assert "worker terminated" in (records[0].error or "")
    assert records[1].reward == 1.0
    assert calls == 2


@pytest.mark.asyncio
async def test_corrupt_shared_cache_is_rejected_and_recomputed(tmp_path):
    calls = 0

    async def run_verifier(key: str, paths) -> VerifierResult:
        nonlocal calls
        calls += 1
        return VerifierResult(rewards={"reward": float(calls)})

    lock = tmp_path / "trusted" / "verifier.lock"
    cache = tmp_path / "trusted" / "cache"

    async def evaluate(label: str):
        env = FakeAgentEnv(tmp_path / label)
        service = _service(
            tmp_path / f"artifacts-{label}",
            env,
            run_verifier,
            verification_context="course-v1",
            shared_scheduler_lock_path=str(lock),
            shared_result_cache_dir=str(cache),
            return_results_to_agent=False,
            drain_pending_on_stop=True,
        )
        (env.watch / "same.pt").write_bytes(b"same-policy")
        async with service.running():
            await _settle(service)
        return service.summary.submissions[0]

    first = await evaluate("first")
    cache_file = next(cache.rglob("*.json"))
    payload = json.loads(cache_file.read_text())
    payload["rewards"] = {"reward": 999.0}
    cache_file.write_text(json.dumps(payload))
    second = await evaluate("second")

    assert calls == 2
    assert first.reward == 1.0
    assert second.reward == 2.0
    assert second.cache_hit is False
