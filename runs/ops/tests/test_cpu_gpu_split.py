"""Focused unit tests for CPU-agent / GPU-worker split ops."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "runs" / "ops"
ENV = ROOT / "challenge" / "g1-sprint-100m-lane" / "environment"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ENV))

import gpu_claim  # noqa: E402
import gpu_worker  # noqa: E402
import sprint_resilience as resilience  # noqa: E402
import start_lane_supervisor  # noqa: E402
import supervise_lane  # noqa: E402
import validate_agent_env  # noqa: E402

# Timeline module is named with hyphens on disk; load via importlib.
_spec = importlib.util.spec_from_file_location(
    "sprint_gpu_timeline", ENV / "sprint-gpu-timeline.py"
)
assert _spec and _spec.loader
timeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(timeline)

_worker_spec = importlib.util.spec_from_file_location(
    "sprint_gpu_worker_run", ENV / "sprint-gpu-worker-run.py"
)
assert _worker_spec and _worker_spec.loader
worker_run = importlib.util.module_from_spec(_worker_spec)
_worker_spec.loader.exec_module(worker_run)


class ClaimSelectionTests(unittest.TestCase):
    def test_pending_is_claimable(self) -> None:
        job = {"job_id": "a", "status": "pending"}
        self.assertEqual(gpu_claim.select_claim_action(job, claim_id="c1"), "claim")

    def test_terminal_skipped(self) -> None:
        for status in ("succeeded", "failed", "terminated"):
            job = {"job_id": "a", "status": status}
            self.assertEqual(gpu_claim.select_claim_action(job, claim_id="c1"), "skip")

    def test_foreign_fresh_claim_skipped(self) -> None:
        job = {
            "job_id": "a",
            "status": "claiming",
            "claim_id": "other",
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.assertEqual(
            gpu_claim.select_claim_action(job, claim_id="c1", stale_sec=900),
            "skip",
        )

    def test_retry_wait_claimable_when_due(self) -> None:
        job = {"job_id": "a", "status": "retry_wait", "retry_not_before_epoch_s": 10}
        self.assertEqual(
            gpu_claim.select_claim_action(job, claim_id="c1", now=11),
            "claim",
        )

    def test_running_with_sandbox_skipped(self) -> None:
        job = {
            "job_id": "a",
            "status": "running",
            "sandbox_id": "sb-x",
            "claim_id": "other",
        }
        self.assertEqual(gpu_claim.select_claim_action(job, claim_id="c1"), "skip")

    def test_normalize_python_to_python3(self) -> None:
        job = {"command": ["python", "-u", "x.py"]}
        out = gpu_worker.normalize_job_command(job)
        self.assertEqual(out["command"][0], "python3")


class LeaseLivenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = {
            "job_id": "job",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease-1",
            "claimed_at_epoch_s": 100,
        }

    def test_fresh_heartbeat_graces_exited_probe(self) -> None:
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease-1",
            "updated_at_epoch_s": 190,
        }
        decision = gpu_claim.assess_worker_liveness(
            self.job,
            heartbeat,
            probe_state="exited",
            now=200,
            heartbeat_timeout_sec=30,
            startup_grace_sec=0,
            dead_grace_sec=10,
        )
        self.assertEqual(decision, "grace")

    def test_stale_heartbeat_requires_two_observations(self) -> None:
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease-1",
            "updated_at_epoch_s": 120,
        }
        first = gpu_claim.assess_worker_liveness(
            self.job,
            heartbeat,
            probe_state="exited",
            now=200,
            heartbeat_timeout_sec=30,
            startup_grace_sec=0,
            dead_grace_sec=10,
        )
        self.assertEqual(first, "observe")
        observed = {
            **self.job,
            "status": "death_observed",
            "death_observed_epoch_s": 200,
        }
        second = gpu_claim.assess_worker_liveness(
            observed,
            heartbeat,
            probe_state="exited",
            now=211,
            heartbeat_timeout_sec=30,
            startup_grace_sec=0,
            dead_grace_sec=10,
        )
        self.assertEqual(second, "dead")

    def test_unknown_probe_gets_extra_lease_window(self) -> None:
        observed = {
            **self.job,
            "status": "death_observed",
            "death_observed_epoch_s": 195,
        }
        decision = gpu_claim.assess_worker_liveness(
            observed,
            None,
            probe_state="unknown",
            now=200,
            heartbeat_timeout_sec=100,
            startup_grace_sec=0,
            dead_grace_sec=1,
        )
        self.assertEqual(decision, "grace")


class ClaimLockTests(unittest.TestCase):
    def test_dispatch_lock_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            with gpu_claim.dispatch_lock(state, timeout_sec=1.0) as a:
                self.assertTrue(a)
                with gpu_claim.dispatch_lock(state, timeout_sec=0.2) as b:
                    self.assertFalse(b)


class TimelineAccountingTests(unittest.TestCase):
    def test_gpu_active_s_is_wall_minus_wait_startup(self) -> None:
        events = [
            {
                "event_id": "1",
                "epoch_s": 1000,
                "job_id": "j1",
                "phase": "gpu_queue_wait",
                "action": "enter",
            },
            {
                "event_id": "2",
                "epoch_s": 1010,
                "job_id": "j1",
                "phase": "gpu_queue_wait",
                "action": "exit",
            },
            {
                "event_id": "3",
                "epoch_s": 1010,
                "job_id": "j1",
                "phase": "gpu_worker_starting",
                "action": "enter",
            },
            {
                "event_id": "4",
                "epoch_s": 1015,
                "job_id": "j1",
                "phase": "gpu_worker_starting",
                "action": "exit",
            },
            {
                "event_id": "5",
                "epoch_s": 1015,
                "job_id": "j1",
                "phase": "isaac_starting",
                "action": "enter",
            },
            {
                "event_id": "6",
                "epoch_s": 1017,
                "job_id": "j1",
                "phase": "isaac_starting",
                "action": "exit",
            },
            {
                "event_id": "7",
                "epoch_s": 1017,
                "job_id": "j1",
                "phase": "gpu_active",
                "action": "enter",
            },
            {
                "event_id": "8",
                "epoch_s": 1062,
                "job_id": "j1",
                "phase": "gpu_active",
                "action": "exit",
            },
        ]
        summary = timeline.summarize_events(events)
        self.assertEqual(summary["wall_time_s"], 62.0)
        self.assertEqual(summary["gpu_wait_s"], 10.0)
        self.assertEqual(summary["gpu_startup_s"], 5.0)
        self.assertEqual(summary["isaac_startup_s"], 2.0)
        # Fairness: wall - wait - startup - isaac = 62 - 10 - 5 - 2 = 45
        self.assertEqual(summary["gpu_active_s"], 45.0)
        self.assertEqual(summary["wall_minus_wait_s"], 45.0)
        self.assertEqual(summary["gpu_active_interval_s"], 45.0)

    def test_duplicate_enter_not_double_counted(self) -> None:
        events = [
            {
                "event_id": "a",
                "epoch_s": 100,
                "job_id": "j",
                "phase": "gpu_worker_starting",
                "action": "enter",
            },
            {
                "event_id": "b",
                "epoch_s": 101,
                "job_id": "j",
                "phase": "gpu_worker_starting",
                "action": "enter",
            },
            {
                "event_id": "c",
                "epoch_s": 110,
                "job_id": "j",
                "phase": "gpu_worker_starting",
                "action": "exit",
            },
        ]
        summary = timeline.summarize_events(events)
        self.assertEqual(summary["gpu_startup_s"], 10.0)
        self.assertEqual(summary["ignored_duplicate_enters"], 1)

    def test_never_negative_active(self) -> None:
        events = [
            {
                "event_id": "1",
                "epoch_s": 50,
                "job_id": "j",
                "phase": "gpu_queue_wait",
                "action": "enter",
            },
            {
                "event_id": "2",
                "epoch_s": 80,
                "job_id": "j",
                "phase": "gpu_queue_wait",
                "action": "exit",
            },
        ]
        summary = timeline.summarize_events(events)
        self.assertEqual(summary["wall_time_s"], 30.0)
        self.assertEqual(summary["gpu_wait_s"], 30.0)
        self.assertEqual(summary["gpu_active_s"], 0.0)

    def test_attempts_close_without_orphan_inflation(self) -> None:
        events = [
            {
                "event_id": "a1",
                "epoch_s": 100,
                "job_id": "logical",
                "attempt": 1,
                "phase": "gpu_active",
                "action": "enter",
            },
            {
                "event_id": "a2",
                "epoch_s": 110,
                "job_id": "logical",
                "attempt": 1,
                "phase": "gpu_active",
                "action": "exit",
            },
            {
                "event_id": "b1",
                "epoch_s": 130,
                "job_id": "logical",
                "attempt": 2,
                "phase": "gpu_active",
                "action": "enter",
            },
            {
                "event_id": "b2",
                "epoch_s": 150,
                "job_id": "logical",
                "attempt": 2,
                "phase": "gpu_active",
                "action": "exit",
            },
        ]
        summary = timeline.summarize_events(events)
        self.assertEqual(summary["gpu_active_s"], 30.0)
        self.assertEqual(summary["gpu_active_interval_s"], 30.0)
        self.assertEqual(len(summary["per_attempt"]), 2)
        self.assertEqual(summary["per_job"][0]["attempts"], 2)


class SuperviseTests(unittest.TestCase):
    def test_operator_stop_blocks_relaunch(self) -> None:
        ok, reason = supervise_lane.should_relaunch(
            stop=True,
            harbor_alive=False,
            max_restarts=10,
            restarts=0,
            consecutive_failures=0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "operator_stop")

    def test_alive_skips_relaunch(self) -> None:
        ok, reason = supervise_lane.should_relaunch(
            stop=False,
            harbor_alive=True,
            max_restarts=10,
            restarts=0,
            consecutive_failures=0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "already_alive")

    def test_backoff_grows_and_caps(self) -> None:
        self.assertEqual(supervise_lane.next_backoff_s(1, min_s=30, max_s=600), 30)
        self.assertEqual(supervise_lane.next_backoff_s(2, min_s=30, max_s=600), 60)
        self.assertEqual(supervise_lane.next_backoff_s(10, min_s=30, max_s=600), 600)

    def test_stop_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "STOP_REQUESTED.json").write_text(
                json.dumps({"reason": "operator_stop"}) + "\n"
            )
            stop, reason = supervise_lane.stop_requested(state)
            self.assertTrue(stop)
            self.assertIn("STOP_REQUESTED", reason)

    def test_loop_relaunches_then_honors_stop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            launches = {"n": 0}
            alive = {"v": False}
            stopped = {"v": False}

            def launch_fn() -> int:
                launches["n"] += 1
                alive["v"] = True
                return 0

            def alive_fn() -> bool:
                return alive["v"]

            def stop_fn() -> tuple[bool, str]:
                return stopped["v"], "operator_stop" if stopped["v"] else ""

            sleeps: list[float] = []

            def sleep_fn(sec: float) -> None:
                sleeps.append(sec)
                # After first successful launch, kill process and request stop
                # on second decision cycle.
                if launches["n"] >= 1:
                    alive["v"] = False
                    stopped["v"] = True

            result = supervise_lane.run_loop(
                "unit-sup",
                launch_fn=launch_fn,
                alive_fn=alive_fn,
                stop_fn=stop_fn,
                sleep_fn=sleep_fn,
                max_restarts=5,
                min_backoff_s=1,
                max_backoff_s=4,
                max_iterations=5,
                state_dir=state,
            )
            self.assertGreaterEqual(launches["n"], 1)
            self.assertTrue(result["stopped"])
            lifecycle = [
                json.loads(line)
                for line in (state / "telemetry" / "cpu_lifecycle.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in lifecycle],
                ["cpu_launch_started", "cpu_launch_exited"],
            )
            self.assertEqual(lifecycle[-1]["exit_code"], 0)
            # After stop, must not keep launching.
            n_after_stop = launches["n"]
            result2 = supervise_lane.run_loop(
                "unit-sup",
                launch_fn=launch_fn,
                alive_fn=lambda: False,
                stop_fn=lambda: (True, "operator_stop"),
                sleep_fn=lambda _s: None,
                max_restarts=5,
                min_backoff_s=1,
                max_backoff_s=4,
                max_iterations=3,
                state_dir=state,
            )
            self.assertEqual(launches["n"], n_after_stop)
            self.assertEqual(result2["history"][0]["decision"], "operator_stop")

    def test_unrecoverable_launcher_exit_stops_and_writes_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            launches = {"n": 0}

            def launch_fn() -> int:
                launches["n"] += 1
                return 78

            result = supervise_lane.run_loop(
                "unit-unrecoverable",
                launch_fn=launch_fn,
                alive_fn=lambda: False,
                stop_fn=lambda: (False, ""),
                sleep_fn=lambda _s: None,
                max_restarts=50,
                state_dir=state,
            )
            self.assertEqual(launches["n"], 1)
            self.assertEqual(result["state"]["last_decision"], "unrecoverable_exit:78")
            failure = json.loads((state / "SUPERVISOR_FAILED.json").read_text())
            self.assertEqual(failure["reason"], "unrecoverable_exit:78")
            self.assertFalse(failure["recoverable"])

    def test_retry_exhaustion_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            result = supervise_lane.run_loop(
                "unit-exhausted",
                launch_fn=lambda: 1,
                alive_fn=lambda: False,
                stop_fn=lambda: (False, ""),
                sleep_fn=lambda _s: None,
                max_restarts=2,
                state_dir=state,
            )
            self.assertEqual(result["restarts"], 2)
            failure = json.loads((state / "SUPERVISOR_FAILED.json").read_text())
            self.assertIn(
                failure["reason"], {"max_restarts", "max_consecutive_failures"}
            )


class TryClaimPersistenceTests(unittest.TestCase):
    def test_try_claim_persists_before_spawn_semantics(self) -> None:
        """Claim write happens; spawn is not invoked when ownership lost."""
        run = {"run_id": "unit", "volume_name": "v", "app_name": "a"}
        job = {
            "job_id": "job1",
            "status": "pending",
            "command": ["python3", "-c", "pass"],
        }
        written: list[dict] = []

        def fake_persist(_run, payload):
            written.append(dict(payload))
            return payload

        with mock.patch.object(gpu_worker, "persist_job", side_effect=fake_persist):
            with mock.patch.object(
                gpu_worker,
                "load_job",
                side_effect=[
                    # confirm after write: stolen by other claim_id
                    {
                        "job_id": "job1",
                        "status": "claiming",
                        "claim_id": "other",
                        "claimed_at": gpu_claim.utc_now(),
                    }
                ],
            ):
                out = gpu_worker.try_claim_job(run, job, claim_id="mine")
        self.assertIsNone(out)
        self.assertEqual(written[0]["status"], "claiming")
        self.assertEqual(written[0]["claim_id"], "mine")


class GpuConcurrencyLimitTests(unittest.TestCase):
    def test_running_job_keeps_second_job_queued(self) -> None:
        jobs = {
            "active": {
                "job_id": "active",
                "status": "running",
                "attempt": 1,
                "lease_id": "lease-1",
            },
            "queued": {"job_id": "queued", "status": "pending"},
        }
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(Path(raw), run)
                ),
                mock.patch.object(
                    gpu_worker, "list_job_ids", return_value=["active", "queued"]
                ),
                mock.patch.object(
                    gpu_worker,
                    "load_job",
                    side_effect=lambda _run, job_id: dict(jobs[job_id]),
                ),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_job",
                    side_effect=lambda _run, job, now: (job, {"decision": "alive"}),
                ),
                mock.patch.object(gpu_worker.ModalSandboxProvider, "start") as start,
            ):
                result = gpu_worker.dispatch_once("unit")
        self.assertEqual(result["reason"], "training_concurrency_limit")
        self.assertEqual(result["active_training_jobs"], ["active"])
        self.assertEqual(result["pending"], ["queued"])
        self.assertEqual(result["max_active_training_jobs"], 1)
        start.assert_not_called()

    def test_retry_wait_does_not_consume_gpu_slot(self) -> None:
        run = {"run_id": "unit"}
        jobs = {
            "waiting": {"job_id": "waiting", "status": "retry_wait"},
            "done": {"job_id": "done", "status": "succeeded"},
        }
        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=list(jobs)),
            mock.patch.object(
                gpu_worker, "load_job", side_effect=lambda _run, job_id: jobs[job_id]
            ),
        ):
            self.assertEqual(gpu_worker.active_training_job_ids(run), [])


class AgentCredentialBoundaryTests(unittest.TestCase):
    def test_agent_env_allows_only_model_auth_and_endpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agent.env"
            path.write_text(
                "OPENAI_API_KEY=secret\n"
                "OPENAI_BASE_URL=https://api.example.test\n"
                "SPRINT_CODEX_PROVIDER=example\n"
            )
            names = validate_agent_env.validate(path, "OPENAI_API_KEY")
        self.assertEqual(
            names,
            {"OPENAI_API_KEY", "OPENAI_BASE_URL", "SPRINT_CODEX_PROVIDER"},
        )

    def test_agent_env_rejects_modal_control_plane_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agent.env"
            path.write_text("OPENAI_API_KEY=secret\nMODAL_TOKEN_SECRET=escape\n")
            with self.assertRaisesRegex(ValueError, "cloud control-plane"):
                validate_agent_env.validate(path, "OPENAI_API_KEY")

    def test_launcher_records_fixed_resource_budget(self) -> None:
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        worker = (ROOT / "runs" / "ops" / "gpu_worker.py").read_text()
        self.assertIn('"agent_cpu_instances": 1', launcher)
        self.assertIn('"training_max_concurrent_per_run": 1', launcher)
        self.assertIn("export SPRINT_CPU_LAUNCH_ATTEMPT=", launcher)
        self.assertGreaterEqual(
            launcher.count("KEEPALIVE_JSON=$(make_keepalive_json)"), 2
        )
        self.assertIn('"gpu_worker_gpus_per_job": 1', launcher)
        self.assertIn(
            '"agent_cloud_control_plane_credentials_injected": False', launcher
        )
        self.assertIn("MAX_ACTIVE_TRAINING_JOBS_PER_RUN = 1", worker)
        self.assertIn('gpu="A10G"', worker)


class NetworkIsolationTests(unittest.TestCase):
    def test_agent_and_verifier_start_offline(self) -> None:
        import tomllib

        task = tomllib.loads(
            (ROOT / "challenge" / "g1-sprint-100m-lane" / "task.toml").read_text()
        )
        self.assertEqual(task["environment"]["network_mode"], "no-network")
        self.assertEqual(task["verifier"]["environment"]["network_mode"], "no-network")

    def test_gpu_workers_block_all_network(self) -> None:
        worker = (ROOT / "runs" / "ops" / "gpu_worker.py").read_text()
        self.assertGreaterEqual(worker.count("block_network=True"), 2)

    def test_launcher_allows_only_one_audited_model_host(self) -> None:
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        self.assertIn('--allow-agent-host "$MODEL_API_HOST"', launcher)
        self.assertIn(
            "api.anthropic.com|api.deepseek.com|api.openai.com|openrouter.ai",
            launcher,
        )
        self.assertNotIn("modal.com|", launcher)
        self.assertNotIn("modal.run|", launcher)

    def test_legacy_modal_endpoint_launcher_is_removed(self) -> None:
        self.assertFalse((ROOT / "runs" / "run-kimi-k3.sh").exists())

    def test_arbitrary_harbor_arguments_fail_closed(self) -> None:
        import subprocess

        result = subprocess.run(
            [
                str(ROOT / "runs" / "run-lane-durable.sh"),
                "--dry-run",
                "--run-id",
                "unit-extra-args",
                "--",
                "--allow-agent-host",
                "api.modal.com",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("extra Harbor arguments are disabled", result.stderr)

    def test_resume_refreshes_network_policy_metadata(self) -> None:
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        update_start = launcher.index("    payload.update({")
        update_end = launcher.index("    })", update_start)
        resume_update = launcher[update_start:update_end]
        self.assertIn('"agent_network_policy": "model-api-only"', resume_update)
        self.assertIn('"agent_allowed_host": model_api_host', resume_update)
        self.assertIn('"gpu_worker_network_policy": "no-network"', resume_update)
        self.assertIn('"verifier_network_policy": "no-network"', resume_update)

    def test_unreviewed_endpoint_is_rejected_before_launch(self) -> None:
        import subprocess

        launcher = ROOT / "runs" / "run-lane-durable.sh"
        result = subprocess.run(
            [
                str(launcher),
                "--dry-run",
                "--run-id",
                "unit-network-policy",
                "--agent-kind",
                "codex",
                "--model",
                "openai/test",
                "--endpoint",
                "https://compute.example.test/v1",
            ],
            env={"OPENAI_API_KEY": "unit-test-secret-value"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("not in the audited egress allowlist", result.stderr)

    def test_agent_clis_are_baked_at_launcher_pins(self) -> None:
        dockerfile = (
            ROOT / "challenge" / "g1-sprint-100m-lane" / "environment" / "Dockerfile"
        ).read_text()
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        self.assertIn("ARG CODEX_VERSION=0.147.0", dockerfile)
        self.assertIn("ARG CLAUDE_CODE_VERSION=2.1.220", dockerfile)
        self.assertIn("BAKED_CODEX_VERSION=0.147.0", launcher)
        self.assertIn("BAKED_CLAUDE_VERSION=2.1.220", launcher)


class RetryAndFencingTests(unittest.TestCase):
    def test_retry_keeps_logical_job_and_fences_lease(self) -> None:
        job = {
            "job_id": "logical",
            "run_id": "unit",
            "status": "running",
            "attempt": 1,
            "max_attempts": 3,
            "lease_id": "old-lease",
            "sandbox_id": "sb-old",
            "fence_epoch": 1,
            "retry_backoff_sec": 2,
            "retry_backoff_max_sec": 10,
        }
        heartbeat = {
            "attempt": 1,
            "lease_id": "old-lease",
            "updated_at_epoch_s": 100,
            "progress": {"step": 7},
            "checkpoint": "/durable/model_7.pt",
        }
        run = {"run_id": "unit"}
        with mock.patch.object(gpu_worker, "persist_job", side_effect=lambda _r, x: x):
            with mock.patch.object(gpu_worker, "_close_attempt_timeline"):
                with mock.patch.object(gpu_worker, "_timeline_event"):
                    with mock.patch.object(
                        gpu_worker, "_terminate_sandbox", return_value=None
                    ):
                        out = gpu_worker.schedule_retry(
                            run,
                            job,
                            heartbeat=heartbeat,
                            exit_code=137,
                            reason="worker_lost",
                            now=200,
                        )
        self.assertEqual(out["job_id"], "logical")
        self.assertEqual(out["status"], "retry_wait")
        self.assertEqual(out["next_attempt"], 2)
        self.assertEqual(out["retry_not_before_epoch_s"], 202)
        self.assertEqual(out["fenced_lease_id"], "old-lease")
        self.assertNotIn("lease_id", out)
        self.assertEqual(out["last_progress"], {"step": 7})

    def test_max_attempts_exhausts_cleanly(self) -> None:
        job = {
            "job_id": "logical",
            "run_id": "unit",
            "status": "running",
            "attempt": 2,
            "max_attempts": 2,
            "lease_id": "lease-2",
        }
        with mock.patch.object(gpu_worker, "persist_job", side_effect=lambda _r, x: x):
            with mock.patch.object(gpu_worker, "_close_attempt_timeline"):
                with mock.patch.object(
                    gpu_worker, "_terminate_sandbox", return_value=None
                ):
                    out = gpu_worker.schedule_retry(
                        {"run_id": "unit"},
                        job,
                        heartbeat=None,
                        exit_code=137,
                        reason="worker_lost",
                        now=200,
                    )
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["failure_reason"], "max_attempts_exhausted")

    def test_old_worker_cannot_own_new_lease(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            status = Path(raw) / "status.json"
            status.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "attempt": 2,
                        "lease_id": "new",
                    }
                )
            )
            self.assertFalse(worker_run.lease_owned(status, 1, "old"))
            self.assertTrue(worker_run.lease_owned(status, 2, "new"))

    def test_stop_fences_before_terminate(self) -> None:
        run = {"run_id": "unit"}
        job = {
            "job_id": "logical",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease",
            "sandbox_id": "sb",
        }
        order: list[str] = []

        def persist(_run, payload):
            order.append(f"persist:{payload['status']}")
            return payload

        with mock.patch.object(gpu_worker, "list_job_ids", return_value=["logical"]):
            with mock.patch.object(gpu_worker, "load_job", return_value=job):
                with mock.patch.object(gpu_worker, "load_heartbeat", return_value=None):
                    with mock.patch.object(
                        gpu_worker, "persist_job", side_effect=persist
                    ):
                        with mock.patch.object(gpu_worker, "_close_attempt_timeline"):
                            with mock.patch.object(
                                gpu_worker,
                                "_terminate_sandbox",
                                side_effect=lambda _job: order.append("terminate"),
                            ):
                                out = gpu_worker._stop_all_locked(run)
        self.assertEqual(out[0]["status"], "terminated")
        self.assertEqual(order[:2], ["persist:terminated", "terminate"])

    def test_reconcile_automatically_retries_lost_running_worker(self) -> None:
        job = {
            "job_id": "logical",
            "run_id": "unit",
            "status": "death_observed",
            "attempt": 1,
            "max_attempts": 3,
            "lease_id": "lease",
            "sandbox_id": "sb",
            "claimed_at_epoch_s": 10,
            "death_observed_epoch_s": 100,
            "heartbeat_timeout_sec": 10,
            "startup_grace_sec": 0,
            "dead_grace_sec": 5,
            "retry_backoff_sec": 1,
        }
        attempt = {
            "attempt": 1,
            "lease_id": "lease",
            "status": "running",
            "started_at_epoch_s": 20,
        }
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease",
            "updated_at_epoch_s": 50,
            "progress": {"step": 1},
        }
        with mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt):
            with mock.patch.object(
                gpu_worker, "load_heartbeat", return_value=heartbeat
            ):
                with mock.patch.object(
                    gpu_worker, "persist_job", side_effect=lambda _r, x: x
                ):
                    with mock.patch.object(gpu_worker, "_close_attempt_timeline"):
                        with mock.patch.object(gpu_worker, "_timeline_event"):
                            with mock.patch.object(
                                gpu_worker, "_terminate_sandbox", return_value=None
                            ):
                                out, detail = gpu_worker.reconcile_job(
                                    {"run_id": "unit"},
                                    job,
                                    now=200,
                                    probe_fn=lambda _job: ("exited", 137, None),
                                )
        self.assertEqual(detail["decision"], "dead")
        self.assertEqual(out["status"], "retry_wait")
        self.assertEqual(out["next_attempt"], 2)


class CheckpointContinuationTests(unittest.TestCase):
    def test_replacement_gets_progress_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            progress = root / "progress.json"
            source = root / "model_7.pt"
            progress.write_text(json.dumps({"step": 7}))
            source.write_bytes(b"checkpoint")
            checkpoint = resilience.CheckpointStore(root).commit(source, sequence=7)
            state, latest = worker_run.progress_snapshot(progress, root)
            command = worker_run.build_attempt_command(
                {
                    "command": ["python3", "train.py"],
                    "resume_arg": "--checkpoint",
                },
                2,
                latest,
            )
        self.assertEqual(state, {"step": 7})
        self.assertEqual(latest, str(checkpoint.path))
        self.assertEqual(command[-2:], ["--checkpoint", str(checkpoint.path)])


class LauncherWiringTests(unittest.TestCase):
    def test_all_model_launchers_default_to_systemd_supervisor(self) -> None:
        for name in ("run-opus.sh", "run-terra.sh", "run-luna.sh", "run-deepseek.sh"):
            text = (ROOT / "runs" / name).read_text()
            self.assertIn("start_lane_supervisor.py", text)
            self.assertIn("--supervised-launch", text)
            self.assertIn("CPU_MAX_RESTARTS", text)
            self.assertIn("CPU_MAX_RESTARTS:-50", text)
        starter = (ROOT / "runs" / "ops" / "start_lane_supervisor.py").read_text()
        self.assertIn("Restart=on-failure", starter)
        self.assertIn("RestartPreventExitStatus=75 78", starter)

    def test_cpu_sandbox_has_full_modal_lifetime_and_no_gpu(self) -> None:
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        self.assertIn("SANDBOX_TIMEOUT_SECONDS=86400", launcher)
        self.assertIn("sandbox_timeout_secs=$SANDBOX_TIMEOUT_SECONDS", launcher)
        task = (ROOT / "challenge" / "g1-sprint-100m-lane" / "task.toml").read_text()
        self.assertIn("gpus = 0", task)
        self.assertIn("cpus = 4", task)
        self.assertIn("memory_mb = 16384", task)

    def test_supervisor_starter_uses_systemd_watchdog_without_secret_in_argv(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            launch = [
                str(ROOT / "runs/run-lane-durable.sh"),
                "--run-id",
                "unit-run",
            ]
            argv = [
                "start_lane_supervisor.py",
                "--run-id",
                "unit-run",
                "--launch-argv-json",
                json.dumps(launch),
                "--secret-env",
                "OPENAI_API_KEY",
            ]
            completed = mock.Mock(returncode=0)
            with (
                mock.patch.object(start_lane_supervisor, "OPS", Path(raw)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(
                    "os.environ", {"OPENAI_API_KEY": "secret-value"}, clear=True
                ),
                mock.patch.object(
                    start_lane_supervisor.subprocess, "run", return_value=completed
                ) as run,
            ):
                self.assertEqual(start_lane_supervisor.main(), 0)
            command = run.call_args.args[0]
            self.assertIn("--property=Restart=on-failure", command)
            self.assertIn("--property=RestartPreventExitStatus=75 78", command)
            self.assertIn("--setenv=OPENAI_API_KEY", command)
            self.assertNotIn("secret-value", command)
            metadata = json.loads(
                (Path(raw) / "unit-run" / "supervisor.json").read_text()
            )
            self.assertEqual(metadata["launch_argv"], launch)

    def test_goal_templates_are_launchable_and_synced(self) -> None:
        smoke = (ROOT / "runs" / "codex-recovery-smoke-goal.j2").read_text()
        self.assertIn("{{ instruction }}", smoke)
        source = (ROOT / "runs" / "codex-goal-slash.j2").read_text()
        live = (ROOT / "runs" / "codex-goal.j2").read_text()
        self.assertEqual(source, live)


if __name__ == "__main__":
    unittest.main()


class StandingWorkerTests(unittest.TestCase):
    """A standing sandbox outlives its jobs, so sandbox liveness != job liveness.

    Without this, a crashed job inside a still-running standing container would
    read as ``alive`` forever and the logical job would never be retried.
    """

    NOW = 1_000_000.0

    def _job(self, **kw):
        d = {
            "status": "running",
            "attempt": 1,
            "lease_id": "L1",
            "job_id": "J",
            "claimed_at_epoch_s": self.NOW - 5000,
            "dispatched_at_epoch_s": self.NOW - 5000,
        }
        d.update(kw)
        return d

    def _hb(self, age):
        return {"lease_id": "L1", "attempt": 1, "updated_at_epoch_s": self.NOW - age}

    def test_live_standing_sandbox_with_stale_heartbeat_is_not_alive(self):
        got = gpu_claim.assess_worker_liveness(
            self._job(), self._hb(600), probe_state="alive", now=self.NOW, standing=True
        )
        self.assertEqual(got, "observe")

    def test_live_standing_sandbox_with_fresh_heartbeat_is_alive(self):
        got = gpu_claim.assess_worker_liveness(
            self._job(), self._hb(5), probe_state="alive", now=self.NOW, standing=True
        )
        self.assertEqual(got, "alive")

    def test_standing_stale_heartbeat_reaches_dead_after_grace(self):
        got = gpu_claim.assess_worker_liveness(
            self._job(death_observed_epoch_s=self.NOW - 999),
            self._hb(600),
            probe_state="alive",
            now=self.NOW,
            standing=True,
        )
        self.assertEqual(got, "dead")

    def test_per_job_mode_is_unchanged(self):
        """Per-job mode short-circuits on a live sandbox."""
        got = gpu_claim.assess_worker_liveness(
            self._job(), self._hb(600), probe_state="alive", now=self.NOW
        )
        self.assertEqual(got, "alive")

    def test_standing_is_opt_in(self):
        self.assertFalse(gpu_worker.standing_enabled({}))
        self.assertTrue(gpu_worker.standing_enabled({"standing_gpu_worker": True}))
