"""Trial-level wiring for [verifier.continuous]."""

import asyncio
import contextlib
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from harbor.environments.base import ExecResult
from harbor.models.task.config import TaskOS
from harbor.models.trial.config import TaskConfig as TrialTaskConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import (
    AgentInfo,
    ContinuousSubmission,
    ContinuousVerificationSummary,
)
from harbor.models.verifier.result import VerifierResult
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import Trial
from harbor.verifier.verifier import DownloadVerifierDirError


ModalNotFoundError = type(
    "NotFoundError",
    (Exception,),
    {"__module__": "modal.exception"},
)


def test_modal_sandbox_loss_is_retryable_but_verifier_errors_are_not():
    lost = ModalNotFoundError(
        "Modal Sandbox with container ID ta-x not found; Sandbox has shut down"
    )
    assert SingleStepTrial._retryable_continuous_verifier_error(lost)
    assert not SingleStepTrial._retryable_continuous_verifier_error(
        FileNotFoundError("verifier omitted reward.json")
    )


def test_wrapped_modal_sandbox_loss_is_retryable():
    lost = ModalNotFoundError("Sandbox ta-x not found")
    try:
        raise RuntimeError("download failed") from lost
    except RuntimeError as wrapped:
        assert SingleStepTrial._retryable_continuous_verifier_error(wrapped)


def test_verifier_output_download_loss_is_retryable():
    assert SingleStepTrial._retryable_continuous_verifier_error(
        DownloadVerifierDirError(
            "Failed to download verifier directory from environment"
        )
    )


SUBMISSION = "/app/submission/policy.pt"


def _task_dir(tmp: Path, *, separate: bool = True, extra: str = "") -> Path:
    task_dir = tmp / "task"
    task_dir.mkdir()
    mode = 'environment_mode = "separate"\n' if separate else ""
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 10.0\n"
        f"[verifier]\ntimeout_sec = 10.0\n{mode}"
        "[verifier.continuous]\nenabled = true\npoll_interval_sec = 0.01\n"
        f'submission_path = "{SUBMISSION}"\n'
        f"{extra}"
        "[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do nothing.\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


def _make_env(queue: dict[str, int] | None = None) -> AsyncMock:
    """An environment mock that reports *queue* as the contents of watch_dir."""
    env = AsyncMock()
    env.default_user = None
    env.capabilities.mounted = True
    env.capabilities.docker_compose = True
    env.os = TaskOS.LINUX

    async def exec_(command, **kwargs):
        if queue is not None and str(command).startswith("find "):
            listing = "\n".join(f"{n}\t{s}" for n, s in sorted(queue.items()))
            return ExecResult(stdout=listing, stderr="", return_code=0)
        return ExecResult(stdout="/", stderr="", return_code=0)

    env.exec = AsyncMock(side_effect=exec_)
    env.service_exec.return_value = ExecResult(stdout="", stderr="", return_code=0)
    env.is_dir = AsyncMock(return_value=False)
    env.service_is_dir = AsyncMock(return_value=False)
    env.validate_network_policy_support = MagicMock()
    for name in (
        "reset_dirs",
        "empty_dirs",
        "ensure_dirs",
        "start",
        "stop",
        "stop_service",
        "upload_dir",
        "upload_file",
    ):
        getattr(env, name).return_value = None

    async def download_file(source_path, target_path, **kwargs):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"policy-bytes")

    env.download_file = AsyncMock(side_effect=download_file)

    async def service_download_dir(source_dir, target_dir, service=None):
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    async def service_download_dir_with_exclusions(
        *, source_dir, target_dir, exclude, service=None
    ):
        await service_download_dir(source_dir, target_dir, service=service)

    async def service_download_file(source_path, target_path, service=None):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_path)

    env.service_download_dir = AsyncMock(side_effect=service_download_dir)
    env.service_download_dir_with_exclusions = AsyncMock(
        side_effect=service_download_dir_with_exclusions
    )
    env.service_download_file = AsyncMock(side_effect=service_download_file)

    @contextlib.contextmanager
    def with_default_user(user):
        previous = env.default_user
        env.default_user = user
        try:
            yield
        finally:
            env.default_user = previous

    env.with_default_user = with_default_user
    env.scoped_exec_env = MagicMock(side_effect=lambda _env: contextlib.nullcontext())
    return env


async def _run(task_dir: Path, trials_dir: Path, agent_env, verifier_env, *, agent_run):
    config = TrialConfig(
        task=TrialTaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(type="docker", delete=False),
        verifier=VerifierConfig(),
    )
    envs = [agent_env, verifier_env]
    index = [0]

    def fake_create(**kwargs):
        env = envs[min(index[0], len(envs) - 1)]
        index[0] += 1
        return env

    with (
        patch(
            "harbor.trial.trial.EnvironmentFactory.create_environment_from_config",
            side_effect=fake_create,
        ),
        patch(
            "harbor.trial.trial.AgentFactory.create_agent_from_config",
            return_value=MagicMock(
                name=lambda: "oracle",
                version=lambda: "1.0",
                SUPPORTS_ATIF=False,
                SUPPORTS_WINDOWS=True,
                setup=AsyncMock(),
                run=AsyncMock(side_effect=agent_run),
                to_agent_info=lambda: AgentInfo(name="oracle", version="1.0"),
            ),
        ),
    ):
        trial = await Trial.create(config)
        trial.paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        trial.paths.reward_text_path.write_text("1.0")
        await trial.run()
        return trial


class TestContinuousVerificationWiring:
    def test_matching_continuous_result_becomes_final_without_reexecution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = TrialPaths(trial_dir=root / "trial")
            paths.artifacts_dir.mkdir(parents=True)
            policy = paths.artifacts_dir / "app" / "submission" / "policy.pt"
            policy.parent.mkdir(parents=True)
            policy.write_bytes(b"final-policy")
            digest = hashlib.sha256(policy.read_bytes()).hexdigest()
            row = ContinuousSubmission(
                name="chosen.pt",
                index=2,
                submitted_at=datetime.now(timezone.utc),
                artifact_sha256=digest,
                source_evaluation_id="run-a:2:abcd",
                rewards={"reward": 1.0, "best_100m_s": 9.0},
            )
            result = SimpleNamespace(
                continuous_verification=ContinuousVerificationSummary(
                    submissions=[row]
                ),
                verifier_reused_continuous_evaluation_id=None,
            )
            continuous = SimpleNamespace(
                enabled=True,
                reuse_matching_result_for_final=True,
                submission_path="/app/submission/policy.pt",
            )
            trial = SimpleNamespace(
                task=SimpleNamespace(
                    config=SimpleNamespace(
                        verifier=SimpleNamespace(continuous=continuous)
                    )
                ),
                paths=paths,
                result=result,
            )

            reused = SingleStepTrial._reuse_matching_continuous_result(trial)

            assert reused is not None
            assert reused.rewards == row.rewards
            assert result.verifier_reused_continuous_evaluation_id == "run-a:2:abcd"
            assert json.loads(paths.reward_json_path.read_text()) == row.rewards
            provenance = json.loads(
                (paths.verifier_dir / "reused-result.json").read_text()
            )
            assert provenance["artifact_sha256"] == digest

    def test_unsubmitted_final_policy_falls_back_to_final_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = TrialPaths(trial_dir=root / "trial")
            paths.artifacts_dir.mkdir(parents=True)
            policy = paths.artifacts_dir / "app" / "submission" / "policy.pt"
            policy.parent.mkdir(parents=True)
            policy.write_bytes(b"never-continuously-submitted")
            continuous = SimpleNamespace(
                enabled=True,
                reuse_matching_result_for_final=True,
                submission_path="/app/submission/policy.pt",
            )
            trial = SimpleNamespace(
                task=SimpleNamespace(
                    config=SimpleNamespace(
                        verifier=SimpleNamespace(continuous=continuous)
                    )
                ),
                paths=paths,
                result=SimpleNamespace(
                    continuous_verification=ContinuousVerificationSummary(
                        submissions=[]
                    ),
                    verifier_reused_continuous_evaluation_id=None,
                ),
            )

            assert SingleStepTrial._reuse_matching_continuous_result(trial) is None

    def test_latest_identical_matching_submission_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = TrialPaths(trial_dir=root / "trial")
            paths.artifacts_dir.mkdir(parents=True)
            policy = paths.artifacts_dir / "app" / "submission" / "policy.pt"
            policy.parent.mkdir(parents=True)
            policy.write_bytes(b"same-policy")
            digest = hashlib.sha256(policy.read_bytes()).hexdigest()
            rewards = {"reward": 1.0, "best_100m_s": 8.0}
            rows = [
                ContinuousSubmission(
                    name=f"candidate-{index}.pt",
                    index=index,
                    submitted_at=datetime.now(timezone.utc),
                    artifact_sha256=digest,
                    source_evaluation_id=f"run-a:{index}:abcd",
                    rewards=rewards,
                )
                for index in (1, 3)
            ]
            continuous = SimpleNamespace(
                enabled=True,
                reuse_matching_result_for_final=True,
                submission_path="/app/submission/policy.pt",
            )
            result = SimpleNamespace(
                continuous_verification=ContinuousVerificationSummary(submissions=rows),
                verifier_reused_continuous_evaluation_id=None,
            )
            trial = SimpleNamespace(
                task=SimpleNamespace(
                    config=SimpleNamespace(
                        verifier=SimpleNamespace(continuous=continuous)
                    )
                ),
                paths=paths,
                result=result,
            )

            reused = SingleStepTrial._reuse_matching_continuous_result(trial)

            assert reused is not None
            assert result.verifier_reused_continuous_evaluation_id == "run-a:3:abcd"

    def test_disagreeing_matching_results_fall_back_to_final_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = TrialPaths(trial_dir=root / "trial")
            paths.artifacts_dir.mkdir(parents=True)
            policy = paths.artifacts_dir / "app" / "submission" / "policy.pt"
            policy.parent.mkdir(parents=True)
            policy.write_bytes(b"same-policy")
            digest = hashlib.sha256(policy.read_bytes()).hexdigest()
            rows = [
                ContinuousSubmission(
                    name=f"candidate-{index}.pt",
                    index=index,
                    submitted_at=datetime.now(timezone.utc),
                    artifact_sha256=digest,
                    rewards={"reward": float(index)},
                )
                for index in (1, 2)
            ]
            continuous = SimpleNamespace(
                enabled=True,
                reuse_matching_result_for_final=True,
                submission_path="/app/submission/policy.pt",
            )
            trial = SimpleNamespace(
                task=SimpleNamespace(
                    config=SimpleNamespace(
                        verifier=SimpleNamespace(continuous=continuous)
                    )
                ),
                paths=paths,
                result=SimpleNamespace(
                    continuous_verification=ContinuousVerificationSummary(
                        submissions=rows
                    ),
                    verifier_reused_continuous_evaluation_id=None,
                ),
            )

            assert SingleStepTrial._reuse_matching_continuous_result(trial) is None

    async def test_submission_scored_in_an_isolated_container_during_agent_run(self):
        """A submission made mid-run goes through the separate verifier path."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            queue: dict[str, int] = {}
            agent_env = _make_env(queue)
            verifier_env = _make_env()
            trials_dir = tmp_path / "trials"
            trials_dir.mkdir()

            calls = []

            async def agent_run(*args, **kwargs):
                # The agent submits and keeps going; two polls see a settled
                # size, which is what the watcher waits for.
                queue["attempt.pt"] = 128
                for _ in range(10):
                    await asyncio.sleep(0.02)

            original = Trial._run_separate_verifier

            async def spy(self, **kwargs):
                calls.append(kwargs)
                if kwargs["key"].startswith("continuous-"):
                    return VerifierResult(rewards={"reward": 1.0, "best_100m_s": 9.9})
                return await original(self, **kwargs)

            with patch.object(Trial, "_run_separate_verifier", spy):
                trial = await _run(
                    _task_dir(tmp_path),
                    trials_dir,
                    agent_env,
                    verifier_env,
                    agent_run=agent_run,
                )

            continuous = [c for c in calls if c["key"].startswith("continuous-")]
            assert len(continuous) == 1, "expected exactly one continuous verification"
            # Isolation: it goes through the separate-container path, and the
            # only thing crossing from the agent is the submitted file.
            assert continuous[0]["artifacts"] == [SUBMISSION]
            staged = continuous[0]["artifacts_dir"] / "app" / "submission" / "policy.pt"
            assert staged.read_bytes() == b"policy-bytes"
            # The attempt's verifier output dir is its own, not the trial's.
            attempt = continuous[0]["trial_paths"]
            assert attempt.verifier_dir != trial.paths.verifier_dir

            summary = trial.result.continuous_verification
            assert summary is not None
            assert [s.name for s in summary.submissions] == ["attempt.pt"]
            assert summary.best_reward == 1.0
            assert summary.submissions[0].rewards["best_100m_s"] == 9.9

            # And the final verifier still ran on its own.
            assert any(c["key"] == "trial" for c in calls)

    async def test_continuous_only_uses_complete_ledger_without_final_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            queue: dict[str, int] = {}
            trials_dir = tmp_path / "trials"
            trials_dir.mkdir()
            calls = []

            async def agent_run(*args, **kwargs):
                queue["candidate.pt"] = 128
                for _ in range(10):
                    await asyncio.sleep(0.02)

            async def spy(self, **kwargs):
                calls.append(kwargs)
                return VerifierResult(rewards={"reward": 1.0, "best_100m_s": 9.9})

            with patch.object(Trial, "_run_separate_verifier", spy):
                trial = await _run(
                    _task_dir(tmp_path, extra="continuous_only = true\n"),
                    trials_dir,
                    _make_env(queue),
                    _make_env(),
                    agent_run=agent_run,
                )

            assert [call["key"] for call in calls] == ["continuous-0001"]
            assert trial.result.verifier is None
            assert trial.result.verifier_result is None
            summary = trial.result.continuous_verification
            assert summary is not None
            assert [row.name for row in summary.submissions] == ["candidate.pt"]
            assert summary.submissions[0].rewards["best_100m_s"] == 9.9

    async def test_shared_mode_is_refused(self):
        """Shared mode cannot isolate the verifier, so it is an error, not a
        silent downgrade."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_dir = _task_dir(tmp_path, separate=False)
            trials_dir = tmp_path / "trials"
            trials_dir.mkdir()

            async def agent_run(*args, **kwargs):
                return None

            trial = await _run(
                task_dir,
                trials_dir,
                _make_env({}),
                _make_env(),
                agent_run=agent_run,
            )
            info = trial.result.exception_info
            assert info is not None
            assert "environment_mode = 'separate'" in info.exception_message

    async def test_archive_survives_in_the_trial_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            queue: dict[str, int] = {}
            trials_dir = tmp_path / "trials"
            trials_dir.mkdir()

            async def agent_run(*args, **kwargs):
                queue["run7.pt"] = 64
                for _ in range(10):
                    await asyncio.sleep(0.02)

            original = Trial._run_separate_verifier

            async def spy(self, **kwargs):
                if kwargs["key"].startswith("continuous-"):
                    return VerifierResult(rewards={"reward": 0.0})
                return await original(self, **kwargs)

            with patch.object(Trial, "_run_separate_verifier", spy):
                trial = await _run(
                    _task_dir(tmp_path),
                    trials_dir,
                    _make_env(queue),
                    _make_env(),
                    agent_run=agent_run,
                )

            root = trial.paths.artifacts_dir / "continuous"
            assert (root / "ledger.jsonl").read_text().strip()
            attempt = root / "attempts" / "0001-run7.pt"
            assert (
                attempt / "artifacts" / "app" / "submission" / "policy.pt"
            ).read_bytes() == b"policy-bytes"
            assert (attempt / "result.json").exists()
            assert (attempt / "verifier").is_dir()
