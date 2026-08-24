"""CPU-only failure simulations for the preemptible GPU job contract."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs" / "ops"
ENV = ROOT / "event_runtime" / "container"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENV))

from event_runtime.compute import claim as gpu_claim  # noqa: E402
from event_runtime.compute import worker as gpu_worker  # noqa: E402
from event_runtime.container import sprint_resilience as resilience  # noqa: E402

_worker_spec = importlib.util.spec_from_file_location(
    "sprint_gpu_worker_run_resilience", ENV / "sprint-gpu-worker-run.py"
)
assert _worker_spec and _worker_spec.loader
worker_run = importlib.util.module_from_spec(_worker_spec)
_worker_spec.loader.exec_module(worker_run)

_bootstrap_spec = importlib.util.spec_from_file_location(
    "sprint_isaac_bootstrap_resilience", ENV / "sprint-isaac-bootstrap.py"
)
assert _bootstrap_spec and _bootstrap_spec.loader
isaac_bootstrap = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(isaac_bootstrap)


class CheckpointStoreTests(unittest.TestCase):
    def test_interruption_during_checkpoint_keeps_previous_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.pt"
            source.write_bytes(b"generation-one")
            store = resilience.CheckpointStore(root / "checkpoints")
            first = store.commit(source, sequence=1, replay_cursor=1)

            source.write_bytes(b"half-written-next-generation")

            class SimulatedPreemption(BaseException):
                pass

            with self.assertRaises(SimulatedPreemption):
                store.commit(
                    source,
                    sequence=2,
                    _after_payload=lambda _path: (_ for _ in ()).throw(
                        SimulatedPreemption()
                    ),
                )

            latest = store.latest_valid()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.checkpoint_id, first.checkpoint_id)
            self.assertEqual(latest.path.read_bytes(), b"generation-one")
            self.assertFalse(
                any(
                    path.name.endswith(".partial")
                    for path in store.committed_dir.glob(".*")
                )
            )

    def test_corrupt_latest_falls_back_to_newest_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.pt"
            store = resilience.CheckpointStore(root / "checkpoints")
            source.write_bytes(b"one")
            first = store.commit(source, sequence=10)
            source.write_bytes(b"two")
            second = store.commit(source, sequence=20)
            second.path.write_bytes(b"corrupt")

            latest = store.latest_valid()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.checkpoint_id, first.checkpoint_id)
            pointer = json.loads(store.latest_path.read_text())
            self.assertEqual(pointer["checkpoint_id"], first.checkpoint_id)

    def test_resume_metadata_exposes_manifest_sequence_and_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.pt"
            source.write_bytes(b"state")
            checkpoint = resilience.CheckpointStore(root / "checkpoints").commit(
                source, sequence=12, replay_cursor="batch:12"
            )
            env = worker_run.checkpoint_resume_metadata(str(checkpoint.path))
            self.assertEqual(env["SPRINT_GPU_RESUME_SEQUENCE"], "12")
            self.assertEqual(env["SPRINT_GPU_REPLAY_CURSOR"], "batch:12")
            self.assertEqual(
                env["SPRINT_GPU_RESUME_CHECKPOINT_ID"], checkpoint.checkpoint_id
            )
            self.assertEqual(env["SPRINT_GPU_RESUME_SEQUENCE"], "12")
            self.assertEqual(env["SPRINT_GPU_REPLAY_CURSOR"], "batch:12")
            self.assertEqual(
                env["SPRINT_GPU_RESUME_CHECKPOINT_ID"], checkpoint.checkpoint_id
            )
            self.assertTrue(Path(env["SPRINT_GPU_RESUME_MANIFEST"]).is_file())

    def test_sprint_environment_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            preferred = str(Path(raw) / "preferred")
            env = {
                "SPRINT_GPU_CHECKPOINT_DIR": preferred,
                "SPRINT_GPU_ATTEMPT": "2",
                "SPRINT_GPU_LEASE_ID": "lease",
                "SPRINT_GPU_JOB_ID": "job",
                "SPRINT_GPU_FENCE_EPOCH": "3",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                store = resilience.CheckpointStore.from_env()
            self.assertEqual(str(store.root), preferred)
            self.assertEqual(store.lease.job_id, "job")

    def test_older_delayed_commit_cannot_regress_resume_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.pt"
            store = resilience.CheckpointStore(root / "checkpoints")
            source.write_bytes(b"new")
            newest = store.commit(source, sequence=50)
            source.write_bytes(b"stale")
            store.commit(source, sequence=40)
            latest = store.latest_valid()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.checkpoint_id, newest.checkpoint_id)
            self.assertEqual(latest.sequence, 50)

    def test_same_sequence_is_idempotent_only_for_identical_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "trainer.pt"
            source.write_bytes(b"trainer-state-a")
            store = resilience.CheckpointStore(root / "checkpoints")
            first = store.commit(source, sequence=7, replay_cursor=7)
            replay = store.commit(source, sequence=7, replay_cursor=7)
            self.assertEqual(replay.checkpoint_id, first.checkpoint_id)
            self.assertEqual(len(list(store.iter_valid())), 1)

            source.write_bytes(b"trainer-state-b")
            with self.assertRaisesRegex(ValueError, "sequences must advance"):
                store.commit(source, sequence=7, replay_cursor=7)
            self.assertEqual(len(list(store.iter_valid())), 1)

    def test_fenced_lease_cannot_publish_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            status = root / "status.json"
            status.write_text(
                json.dumps(
                    {
                        "job_id": "job",
                        "attempt": 2,
                        "lease_id": "new",
                        "fence_epoch": 2,
                    }
                )
            )
            source = root / "model.pt"
            source.write_bytes(b"stale")
            store = resilience.CheckpointStore(
                root / "checkpoints",
                lease=resilience.Lease("job", 1, "old", 1),
                lease_path=status,
            )
            with self.assertRaises(resilience.LeaseLostError):
                store.commit(source, sequence=99)
            self.assertIsNone(store.latest_valid())

    def test_cleanup_keeps_requested_valid_generations_and_removes_partials(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "model.pt"
            store = resilience.CheckpointStore(root / "checkpoints")
            for sequence in range(1, 6):
                source.write_bytes(str(sequence).encode())
                store.commit(source, sequence=sequence)
            partial = store.committed_dir / ".orphan.partial"
            partial.mkdir()
            (partial / "payload").write_bytes(b"orphan")

            removed = store.cleanup(keep=2)
            self.assertIn(".orphan.partial", removed)
            self.assertEqual(
                [item.sequence for item in store.iter_valid()],
                [5, 4],
            )
            self.assertEqual(store.latest_valid().sequence, 5)  # type: ignore[union-attr]


class WorkerAttemptGuardTests(unittest.TestCase):
    def test_bootstrap_marks_zero_exit_during_app_launcher_initialization(self) -> None:
        class FakeAppLauncher:
            def __init__(self) -> None:
                raise SystemExit(0)

        isaaclab = types.ModuleType("isaaclab")
        app = types.ModuleType("isaaclab.app")
        app.AppLauncher = FakeAppLauncher  # type: ignore[attr-defined]
        isaaclab.app = app  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "launcher.json"
            with (
                mock.patch.dict(
                    sys.modules,
                    {"isaaclab": isaaclab, "isaaclab.app": app},
                ),
                mock.patch.dict(
                    os.environ,
                    {isaac_bootstrap.APP_LAUNCHER_STATE_ENV: str(marker)},
                ),
            ):
                isaac_bootstrap.install_app_launcher_hook()
                with self.assertRaises(SystemExit):
                    FakeAppLauncher()
            self.assertEqual(
                json.loads(marker.read_text()),
                {"schema_version": 1, "state": "system_exit", "exit_code": 0},
            )

    def test_gpu_activity_watchdog_cannot_be_reported_as_success(self) -> None:
        self.assertEqual(
            worker_run.final_attempt_outcome(
                0,
                interrupted=False,
                activity_watchdog_fired=True,
            ),
            (1, "failed"),
        )
        self.assertEqual(
            worker_run.final_attempt_outcome(
                143,
                interrupted=False,
                activity_watchdog_fired=True,
            ),
            (143, "failed"),
        )

    def test_only_explicit_boolean_progress_completion_ends_teardown(self) -> None:
        self.assertTrue(
            worker_run.progress_declares_completion(
                {"finished": True, "completed_iterations": 10}
            )
        )
        for progress in (
            None,
            {},
            {"finished": False},
            {"finished": 1},
            {"finished": "true"},
            "finished",
        ):
            self.assertFalse(worker_run.progress_declares_completion(progress))

    def test_replacement_without_checkpoint_is_rejected_without_resume_arg(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "valid resumable training-state"):
            worker_run.build_attempt_command(
                {"command": ["python3", "train.py"], "resume_arg": ""},
                2,
                None,
                isaac_bootstrap=Path("/nonexistent"),
            )

    def test_replacement_uses_environment_checkpoint_without_resume_arg(self) -> None:
        command = worker_run.build_attempt_command(
            {"command": ["python3", "train.py"], "resume_arg": ""},
            2,
            "/durable/checkpoint.pt",
            isaac_bootstrap=Path("/nonexistent"),
        )
        self.assertEqual(command, ["python3", "train.py"])

    def test_app_launcher_prestart_retry_does_not_require_checkpoint(self) -> None:
        command = worker_run.build_attempt_command(
            {
                "command": ["python3", "train.py"],
                "resume_arg": "--checkpoint",
                "retry_reason": "app_launcher_initialization_failed",
                "last_progress": None,
                "last_checkpoint": None,
            },
            2,
            None,
            isaac_bootstrap=Path("/nonexistent"),
        )
        self.assertEqual(command, ["python3", "train.py"])

    def test_app_launcher_retry_still_requires_checkpoint_after_progress(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "valid resumable training-state"):
            worker_run.build_attempt_command(
                {
                    "command": ["python3", "train.py"],
                    "retry_reason": "app_launcher_initialization_failed",
                    "last_progress": {"iteration": 1},
                },
                2,
                None,
                isaac_bootstrap=Path("/nonexistent"),
            )

    def test_zero_exit_during_app_launcher_start_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "launcher.json"
            marker.write_text(json.dumps({"state": "starting"}))
            self.assertTrue(
                worker_run.retryable_app_launcher_failure(marker, exit_code=0)
            )
            self.assertEqual(
                worker_run.final_attempt_outcome(
                    0,
                    interrupted=False,
                    activity_watchdog_fired=False,
                    retryable_infrastructure_failure=True,
                ),
                (0, "interrupted"),
            )

    def test_completed_or_nonzero_app_launcher_is_not_provider_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "launcher.json"
            marker.write_text(json.dumps({"state": "completed"}))
            self.assertFalse(
                worker_run.retryable_app_launcher_failure(marker, exit_code=0)
            )
            marker.write_text(json.dumps({"state": "system_exit", "exit_code": 1}))
            self.assertFalse(
                worker_run.retryable_app_launcher_failure(marker, exit_code=1)
            )

    def test_gpu_activity_watchdog_requires_full_grace_and_sample_window(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            samples = Path(raw) / "samples.jsonl"
            rows = []
            for index in range(12):
                rows.append(
                    json.dumps(
                        {
                            "epoch_s": 100 + index * 5,
                            "nvidia_smi_ok": True,
                            "gpus": [{"util_gpu_pct": 0, "mem_used_mib": 1024}],
                        }
                    )
                )
            samples.write_text("\n".join(rows) + "\n")
            self.assertFalse(
                worker_run.gpu_activity_stalled(
                    samples, started_epoch_s=100, now_epoch_s=399
                )
            )
            self.assertTrue(
                worker_run.gpu_activity_stalled(
                    samples, started_epoch_s=100, now_epoch_s=400
                )
            )

    def test_gpu_activity_watchdog_accepts_any_sampled_accelerator_progress(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            samples = Path(raw) / "samples.jsonl"
            rows = []
            for index in range(12):
                rows.append(
                    json.dumps(
                        {
                            "epoch_s": 100 + index * 5,
                            "nvidia_smi_ok": True,
                            "gpus": [
                                {
                                    "util_gpu_pct": 35 if index == 7 else 0,
                                    "mem_used_mib": 1024,
                                }
                            ],
                        }
                    )
                )
            samples.write_text("\n".join(rows) + "\n")
            self.assertFalse(
                worker_run.gpu_activity_stalled(
                    samples, started_epoch_s=100, now_epoch_s=400
                )
            )

    def test_gpu_activity_watchdog_does_not_accept_stale_progress(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            samples = Path(raw) / "samples.jsonl"
            rows = []
            for index in range(72):
                rows.append(
                    json.dumps(
                        {
                            "epoch_s": 100 + index * 5,
                            "nvidia_smi_ok": True,
                            "gpus": [
                                {
                                    "util_gpu_pct": 35 if index == 7 else 0,
                                    "mem_used_mib": 1024,
                                }
                            ],
                        }
                    )
                )
            samples.write_text("\n".join(rows) + "\n")
            self.assertTrue(
                worker_run.gpu_activity_stalled(
                    samples, started_epoch_s=100, now_epoch_s=500
                )
            )

    def test_training_progress_watchdog_requires_cursor_advance(self) -> None:
        watchdog = worker_run.TrainingProgressWatchdog(grace_seconds=300)
        self.assertFalse(watchdog.observe({"iteration": 0}, now_epoch_s=100))
        self.assertFalse(watchdog.observe({"iteration": 0}, now_epoch_s=399))
        self.assertTrue(watchdog.observe({"iteration": 0}, now_epoch_s=400))

        watchdog = worker_run.TrainingProgressWatchdog(grace_seconds=300)
        self.assertFalse(watchdog.observe({"iteration": 0}, now_epoch_s=100))
        self.assertFalse(watchdog.observe({"iteration": 1}, now_epoch_s=350))
        self.assertFalse(watchdog.observe({"iteration": 1}, now_epoch_s=649))
        self.assertTrue(watchdog.observe({"iteration": 1}, now_epoch_s=650))

    def test_watchdogs_disable_after_training_reaches_terminal_cursor(self) -> None:
        job = {
            "command": [
                "python3",
                "train.py",
                "--max-iterations",
                "400",
                "--eval",
            ]
        }
        self.assertEqual(worker_run.infer_job_kind(job), "train")
        self.assertEqual(worker_run.watchdog_phase(job, {"iteration": 398}), "training")
        self.assertEqual(
            worker_run.watchdog_phase(job, {"iteration": 399}), "finalizing"
        )

    def test_evaluation_and_verifier_jobs_do_not_use_training_watchdogs(self) -> None:
        evaluation = {"command": ["python3", "/app/eval_policies.py"]}
        verifier = {
            "job_kind": "verify",
            "command": ["bash", "-lc", "exec bash /opt/event-verifier/test.sh"],
        }
        self.assertEqual(worker_run.infer_job_kind(evaluation), "evaluate")
        self.assertEqual(worker_run.watchdog_phase(evaluation, None), "evaluating")
        self.assertEqual(worker_run.infer_job_kind(verifier), "verify")
        self.assertEqual(worker_run.watchdog_phase(verifier, None), "verifying")

    def test_agent_authored_evaluation_names_do_not_use_training_watchdogs(self) -> None:
        for script in (
            "targeted_gait_eval.py",
            "gait_sweep.py",
            "probe_pretrained.py",
            "inspect_gpu_env.py",
            "policy_benchmark.py",
        ):
            with self.subTest(script=script):
                job = {"job_kind": "auto", "command": ["python3", script]}
                self.assertEqual(worker_run.infer_job_kind(job), "evaluate")
                self.assertEqual(worker_run.watchdog_phase(job, None), "evaluating")

    def test_progress_watchdog_cannot_be_reported_as_success(self) -> None:
        self.assertEqual(
            worker_run.final_attempt_outcome(
                0,
                interrupted=False,
                activity_watchdog_fired=False,
                progress_watchdog_fired=True,
            ),
            (1, "failed"),
        )


class ReplayAndResumeTests(unittest.TestCase):
    def test_completion_journal_skips_replayed_work(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = resilience.CheckpointStore(Path(raw) / "checkpoints")
            journal = resilience.CompletionJournal(store)
            calls: list[str] = []

            def compute() -> dict[str, int]:
                calls.append("called")
                return {"score": 7}

            first, executed_first = journal.run_once("eval:seed-17", compute)
            second, executed_second = journal.run_once("eval:seed-17", compute)
            self.assertTrue(executed_first)
            self.assertFalse(executed_second)
            self.assertEqual(first, second)
            self.assertEqual(calls, ["called"])

    def test_abrupt_loss_resumes_from_last_commit_not_local_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "state.json"
            store = resilience.CheckpointStore(root / "checkpoints")
            source.write_text(json.dumps({"completed": [0, 1, 2]}))
            store.commit(source, sequence=3, replay_cursor=3)

            # Attempt 1 computes unit 3 in memory and dies before publication.
            local_only = {"completed": [0, 1, 2, 3]}
            self.assertEqual(len(local_only["completed"]), 4)

            resumed = json.loads(store.latest_valid().path.read_text())  # type: ignore[union-attr]
            self.assertEqual(resumed, {"completed": [0, 1, 2]})
            resumed["completed"].append(3)
            source.write_text(json.dumps(resumed))
            store.commit(source, sequence=4, replay_cursor=4)
            self.assertEqual(
                json.loads(store.latest_valid().path.read_text())["completed"],  # type: ignore[union-attr]
                [0, 1, 2, 3],
            )

    def test_repeated_preemptions_make_monotonic_progress(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "state.json"
            store = resilience.CheckpointStore(root / "checkpoints")
            policy = resilience.RetryPolicy(
                max_attempts=4, initial_backoff_s=0, max_backoff_s=0
            )
            preempt_after = {1: 2, 2: 4}
            completed: list[int] = []
            used_attempts = 0
            for attempt in range(1, policy.max_attempts + 1):
                used_attempts = attempt
                checkpoint = store.latest_valid()
                if checkpoint:
                    completed = json.loads(checkpoint.path.read_text())["completed"]
                for unit in range(len(completed), 6):
                    completed.append(unit)
                    source.write_text(json.dumps({"completed": completed}))
                    store.commit(
                        source, sequence=len(completed), replay_cursor=len(completed)
                    )
                    if len(completed) == preempt_after.get(attempt):
                        break
                else:
                    break
                self.assertTrue(policy.allows_after(attempt))

            self.assertEqual(used_attempts, 3)
            self.assertEqual(completed, list(range(6)))
            self.assertEqual(store.latest_valid().sequence, 6)  # type: ignore[union-attr]


class InterruptionTests(unittest.TestCase):
    def test_graceful_preemption_forwards_signal_and_allows_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "checkpoint.txt"
            code = (
                "import pathlib,signal,sys,time\n"
                f"p=pathlib.Path({str(marker)!r})\n"
                "def stop(sig, frame):\n"
                " p.write_text('checkpointed')\n"
                " raise SystemExit(143)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "while True: time.sleep(0.05)\n"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", code], start_new_session=True
            )
            timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
            try:
                exit_code, interrupted = worker_run.supervise_child(
                    proc,
                    heartbeat_interval=0.05,
                    interruption_grace_sec=2,
                    on_heartbeat=lambda _status: None,
                )
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            self.assertTrue(interrupted)
            self.assertNotEqual(exit_code, 0)
            self.assertEqual(marker.read_text(), "checkpointed")

    def test_host_retries_graceful_interruption_without_heartbeat_timeout(self) -> None:
        job = {
            "job_id": "logical",
            "run_id": "unit",
            "status": "running",
            "attempt": 1,
            "max_attempts": 3,
            "lease_id": "lease-1",
            "sandbox_id": "sb",
            "fence_epoch": 1,
        }
        attempt = {
            "attempt": 1,
            "lease_id": "lease-1",
            "status": "interrupted",
            "exit_code": 143,
        }
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease-1",
            "updated_at_epoch_s": 100,
        }
        with mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt):
            with mock.patch.object(
                gpu_worker, "load_heartbeat", return_value=heartbeat
            ):
                with mock.patch.object(gpu_worker, "schedule_retry") as retry:
                    retry.return_value = {"status": "retry_wait", "next_attempt": 2}
                    out, detail = gpu_worker.reconcile_job(
                        {"run_id": "unit"}, job, now=101
                    )
        self.assertEqual(out["status"], "retry_wait")
        self.assertEqual(detail["reason"], "graceful_preemption")
        retry.assert_called_once()

    def test_host_preserves_app_launcher_retry_reason(self) -> None:
        job = {
            "job_id": "logical",
            "run_id": "unit",
            "status": "running",
            "attempt": 1,
            "max_attempts": 3,
            "lease_id": "lease-1",
            "sandbox_id": "sb",
        }
        attempt = {
            "attempt": 1,
            "lease_id": "lease-1",
            "status": "interrupted",
            "exit_code": 0,
            "retry_reason": "app_launcher_initialization_failed",
        }
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
            mock.patch.object(gpu_worker, "schedule_retry") as retry,
        ):
            retry.return_value = {"status": "retry_wait", "next_attempt": 2}
            _out, detail = gpu_worker.reconcile_job({"run_id": "unit"}, job, now=101)
        self.assertEqual(detail["reason"], "app_launcher_initialization_failed")
        self.assertEqual(
            retry.call_args.kwargs["reason"],
            "app_launcher_initialization_failed",
        )


class RetryPolicyTests(unittest.TestCase):
    def test_retry_exhaustion_is_bounded_and_backoff_is_capped(self) -> None:
        policy = resilience.RetryPolicy(
            max_attempts=3, initial_backoff_s=2, max_backoff_s=3
        )
        self.assertTrue(policy.allows_after(1))
        self.assertTrue(policy.allows_after(2))
        self.assertFalse(policy.allows_after(3))
        self.assertEqual(policy.delay_after(1), 2)
        self.assertEqual(policy.delay_after(2), 3)
        self.assertEqual(policy.delay_after(20), 3)

    def test_abrupt_worker_death_requires_stale_lease_and_two_observations(
        self,
    ) -> None:
        job = {
            "job_id": "job",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease",
            "claimed_at_epoch_s": 1,
        }
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease",
            "updated_at_epoch_s": 10,
        }
        first = gpu_claim.assess_worker_liveness(
            job,
            heartbeat,
            probe_state="exited",
            now=100,
            heartbeat_timeout_sec=5,
            startup_grace_sec=0,
            dead_grace_sec=10,
        )
        second = gpu_claim.assess_worker_liveness(
            {**job, "status": "death_observed", "death_observed_epoch_s": 100},
            heartbeat,
            probe_state="exited",
            now=111,
            heartbeat_timeout_sec=5,
            startup_grace_sec=0,
            dead_grace_sec=10,
        )
        self.assertEqual((first, second), ("observe", "dead"))


if __name__ == "__main__":
    unittest.main()
