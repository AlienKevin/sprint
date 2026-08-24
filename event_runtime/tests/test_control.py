from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))

from event_runtime.export import frontier as frontier_update  # noqa: E402
from event_runtime.control import run as sprintctl  # noqa: E402
from event_runtime.control import render_task  # noqa: E402


def write_policy(trial: Path, index: int, name: str, data: bytes) -> Path:
    path = (
        trial
        / "artifacts"
        / "continuous"
        / "attempts"
        / f"{index:04d}-{name}"
        / "artifacts"
        / "app"
        / "submission"
        / "policy.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def row(index: int, name: str, best: float | None) -> dict:
    if best is None:
        return {
            "index": index,
            "name": name,
            "submitted_at": "2026-08-02T00:00:00Z",
            "started_at": None,
            "finished_at": None,
            "rewards": None,
            "error": None,
        }
    return {
        "index": index,
        "name": name,
        "submitted_at": "2026-08-02T00:00:00Z",
        "started_at": "2026-08-02T00:00:01Z",
        "finished_at": "2026-08-02T00:01:00Z",
        "rewards": {"valid_run": 1, "best_100m_s": best},
        "error": None,
    }


class DurableOpsTests(unittest.TestCase):
    def test_modal_volume_list_commands_are_serialized_across_lane_threads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            active = 0
            max_active = 0
            guard = threading.Lock()

            def fake_run(*_args: object, **_kwargs: object):
                nonlocal active, max_active
                with guard:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.01)
                with guard:
                    active -= 1
                return subprocess.CompletedProcess([], 0, "[]", "")

            errors: list[Exception] = []

            def invoke(index: int) -> None:
                try:
                    sprintctl.run_command(
                        sprintctl.modal_command(
                            "volume", "ls", "--json", f"volume-{index}", "/queue"
                        ),
                        run={"run_id": f"run-{index}"},
                        check=False,
                    )
                except Exception as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            with (
                mock.patch.object(
                    sprintctl, "MODAL_VOLUME_LIST_LOCK", root / "volume.lock"
                ),
                mock.patch.object(
                    sprintctl, "MODAL_VOLUME_LIST_STATE", root / "volume-state.json"
                ),
                mock.patch.object(sprintctl, "MODAL_VOLUME_LIST_MIN_GAP_SECONDS", 0.0),
                mock.patch.object(sprintctl.subprocess, "run", side_effect=fake_run),
            ):
                threads = [threading.Thread(target=invoke, args=(i,)) for i in range(6)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(max_active, 1)

    def test_modal_volume_rate_limit_sets_shared_cooldown_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            responses = [
                subprocess.CompletedProcess(
                    [], 1, "", "VolumeListFiles rate limit exceeded. Please retry."
                ),
                subprocess.CompletedProcess([], 0, "[]", ""),
            ]
            with (
                mock.patch.object(
                    sprintctl, "MODAL_VOLUME_LIST_LOCK", root / "volume.lock"
                ),
                mock.patch.object(
                    sprintctl, "MODAL_VOLUME_LIST_STATE", root / "volume-state.json"
                ),
                mock.patch.object(sprintctl, "MODAL_VOLUME_LIST_MIN_GAP_SECONDS", 0.0),
                mock.patch.object(
                    sprintctl, "MODAL_VOLUME_LIST_BACKOFF_MAX_SECONDS", 0.0
                ),
                mock.patch.object(
                    sprintctl.subprocess, "run", side_effect=responses
                ) as runner,
            ):
                result = sprintctl.run_command(
                    sprintctl.modal_command(
                        "volume", "get", "volume", "remote", "local"
                    ),
                    run={"run_id": "rate-run"},
                    check=False,
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(runner.call_count, 2)
            state = json.loads((root / "volume-state.json").read_text())
            self.assertFalse(state["rate_limited"])
            self.assertEqual(state["attempt"], 2)

    def test_live_finalize_defers_recursive_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "live-run", "agent_kind": "codex"}
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(
                    sprintctl, "run_services_should_exit", return_value=False
                ),
                mock.patch.object(sprintctl, "monitor_once") as monitor,
                mock.patch.object(sprintctl, "sync_durable_api_usage") as api_sync,
                mock.patch.object(sprintctl, "sync_durable_trace") as trace_sync,
                mock.patch.object(sprintctl, "sync_durable_telemetry") as telemetry,
            ):
                complete, payload = sprintctl.finalize("live-run")

            self.assertFalse(complete)
            self.assertFalse(payload["conditions"]["run_terminal"])
            monitor.assert_called_once()
            api_sync.assert_not_called()
            trace_sync.assert_not_called()
            telemetry.assert_not_called()

    def test_live_api_usage_sync_reads_only_exact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "usage-run", "volume_name": "usage-volume"}
            summary = '{"schema_version":2,"run_id":"usage-run"}\n'
            with mock.patch.object(
                sprintctl, "volume_get_text", return_value=summary
            ) as reader:
                self.assertTrue(sprintctl.sync_durable_api_usage(state, run))

            reader.assert_called_once_with(
                run,
                "runs/usage-run/api-usage/summary.json",
                timeout_seconds=30,
            )
            self.assertEqual(
                (state / "provider-api-usage/api-usage/summary.json").read_text(),
                summary,
            )
            stamp = json.loads((state / "provider-api-usage-sync.json").read_text())
            self.assertEqual(stamp["mode"], "summary-live")

    def test_exact_volume_read_bypasses_modal_volume_get_cli(self) -> None:
        run = {"run_id": "exact-run", "volume_name": "exact-volume"}
        with mock.patch.object(
            sprintctl,
            "run_command",
            return_value=subprocess.CompletedProcess([], 0, "x", ""),
        ) as command:
            self.assertEqual(sprintctl.volume_get_text(run, "known/file.json"), "x")
        argv = command.call_args.args[0]
        self.assertEqual(
            argv,
            [
                sys.executable,
                "-m",
                "event_runtime.control.volume_read",
                "exact-volume",
                "known/file.json",
            ],
        )

    def test_codex_wrapper_waits_for_delayed_setsid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = root / "run"
            logs = root / "logs"
            codex_home = root / "codex-home"
            fake_bin = root / "bin"
            package_root = root / "codex-package"
            launcher = package_root / "bin/codex.js"
            native = (
                package_root
                / "node_modules/@openai/codex-linux-test/vendor/test/bin/codex"
            )
            for path in (
                runtime,
                logs,
                codex_home,
                fake_bin,
                launcher.parent,
                native.parent,
            ):
                path.mkdir(parents=True, exist_ok=True)
            launcher.write_text("#!/usr/bin/env node\n")
            launcher.chmod(0o755)
            native.write_text("#!/usr/bin/env bash\nsleep 0.5\nexit 0\n")
            native.chmod(0o755)
            delayed_setsid = fake_bin / "setsid"
            delayed_setsid.write_text(
                '#!/usr/bin/env bash\nsleep 0.1\nexec /usr/bin/setsid "$@"\n'
            )
            delayed_setsid.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "CODEX_HOME": str(codex_home),
                    "SPRINT_RUNTIME_DIR": str(runtime),
                    "SPRINT_AGENT_LOG_DIR": str(logs),
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "event_runtime/container/sprint-codex-exec-wrapper.sh"),
                    str(launcher),
                    "exec",
                    "--json",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(
                "failed to create an isolated Codex process group", result.stderr
            )

    def test_monitor_loop_self_registers_and_cleans_own_pid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            observed: list[str] = []

            @contextlib.contextmanager
            def owned_lock(_path: Path, *, blocking: bool = True):
                self.assertFalse(blocking)
                yield True

            def observe_finalize(_run_id: str) -> tuple[bool, dict]:
                observed.append((state / "monitor.pid").read_text().strip())
                return True, {"run_id": "monitor-run", "agent_kind": "codex"}

            with (
                mock.patch.object(
                    sprintctl,
                    "load_run",
                    return_value=(
                        state,
                        {"run_id": "monitor-run", "agent_kind": "codex"},
                    ),
                ),
                mock.patch.object(sprintctl, "file_lock", side_effect=owned_lock),
                mock.patch.object(sprintctl, "finalize", side_effect=observe_finalize),
            ):
                self.assertEqual(sprintctl.monitor_loop("monitor-run", 10), 0)

            self.assertEqual(observed, [str(os.getpid())])
            self.assertFalse((state / "monitor.pid").exists())

    def test_budget_pulse_uses_fresh_watchdog_and_mirrors_both_consumers(
        self,
    ) -> None:
        from event_runtime.compute import worker as gpu_worker

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "pulse-run", "agent_container_id": "ta-agent"}
            canonical = {
                "schema_version": 2,
                "run_id": "pulse-run",
                "checked_at_epoch_s": 1000.0,
                "total_usd": 1.0,
            }
            merged = {
                "schema_version": 2,
                "run_id": "pulse-run",
                "checked_at_epoch_s": 900.0,
                "as_of_epoch_ms": 900000,
                "as_of": "1970-01-01T00:15:00Z",
                "total_usd": 1.25,
                "budget_remaining_usd": 8.75,
                "status": "within_budget",
            }
            cost_lock_held = False

            @contextlib.contextmanager
            def serialized_cost_lock(path: Path, *, blocking: bool = True):
                nonlocal cost_lock_held
                self.assertTrue(blocking)
                self.assertEqual(path, state / "telemetry/agent-cost.lock")
                cost_lock_held = True
                try:
                    yield True
                finally:
                    cost_lock_held = False

            def mirror_gpu(_run: dict, _payload: dict) -> dict:
                self.assertTrue(cost_lock_held)
                return {"gpu_budget_mirror": "updated"}

            def mirror_agent(_run: dict, _payload: dict) -> dict:
                self.assertTrue(cost_lock_held)
                return {"agent_cost_mirror": "updated"}

            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(
                    sprintctl, "file_lock", side_effect=serialized_cost_lock
                ),
                mock.patch.object(
                    sprintctl, "fetch_budget_watchdog", return_value=canonical
                ) as fetch,
                mock.patch.object(
                    sprintctl, "build_unified_timeline", return_value={"events": []}
                ),
                mock.patch.object(
                    sprintctl.agent_cost, "build_snapshot", return_value=merged
                ),
                mock.patch.object(
                    gpu_worker,
                    "mirror_gpu_budget",
                    side_effect=mirror_gpu,
                ) as gpu_mirror,
                mock.patch.object(
                    gpu_worker,
                    "mirror_agent_cost",
                    side_effect=mirror_agent,
                ) as agent_mirror,
                mock.patch.object(sprintctl, "enforce_agent_cost_budget") as enforce,
            ):
                payload = sprintctl.budget_pulse_once("pulse-run", now=1010.0)

            fetch.assert_called_once()
            gpu_mirror.assert_called_once()
            agent_mirror.assert_called_once()
            enforce.assert_called_once()
            mirrored = gpu_mirror.call_args.args[1]
            self.assertEqual(mirrored["checked_at_epoch_s"], 1010.0)
            self.assertEqual(mirrored["as_of_epoch_ms"], 1010000)
            self.assertEqual(mirrored["as_of"], "1970-01-01T00:16:50Z")
            self.assertEqual(payload["total_usd"], 1.25)
            self.assertEqual(payload["upstream_watchdog_age_seconds"], 10.0)
            persisted = json.loads((state / "telemetry/budget-pulse.json").read_text())
            self.assertEqual(persisted["gpu_mirror"], "updated")

    def test_artifact_cost_refresh_releases_lock_before_fallback_mirrors(self) -> None:
        from event_runtime.compute import worker as gpu_worker

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "telemetry").mkdir()
            run = {"run_id": "serialized-run"}
            payload = {
                "schema_version": 2,
                "run_id": "serialized-run",
                "total_usd": 2.0,
            }
            held = False

            @contextlib.contextmanager
            def serialized_lock(path: Path, *, blocking: bool = True):
                nonlocal held
                self.assertTrue(blocking)
                self.assertEqual(path, state / "telemetry/agent-cost.lock")
                held = True
                try:
                    yield True
                finally:
                    held = False

            def assert_released(*_args, **_kwargs):
                self.assertFalse(held)
                return {"agent_cost_mirror": "updated"}

            def assert_gpu_released(*_args, **_kwargs):
                self.assertFalse(held)
                return {"gpu_budget_mirror": "updated"}

            with (
                mock.patch.object(sprintctl, "file_lock", side_effect=serialized_lock),
                mock.patch.object(
                    sprintctl.agent_cost, "build_snapshot", return_value=payload
                ),
                mock.patch.object(
                    gpu_worker, "mirror_agent_cost", side_effect=assert_released
                ),
                mock.patch.object(
                    gpu_worker, "mirror_gpu_budget", side_effect=assert_gpu_released
                ),
                mock.patch.object(sprintctl, "enforce_agent_cost_budget") as enforce,
            ):
                result = sprintctl.refresh_agent_cost_snapshot(
                    "serialized-run", state, run, {"events": []}
                )

            self.assertEqual(result, payload)
            enforce.assert_called_once_with("serialized-run", state, run, payload)
            self.assertEqual(
                json.loads((state / "telemetry/agent-cost.json").read_text()),
                payload,
            )

    def test_artifact_cost_refresh_delegates_when_budget_pulse_is_alive(
        self,
    ) -> None:
        from event_runtime.compute import worker as gpu_worker

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "telemetry").mkdir()
            run = {"run_id": "delegated-run"}
            payload = {
                "schema_version": 2,
                "run_id": "delegated-run",
                "total_usd": 2.0,
            }
            with (
                mock.patch.object(
                    sprintctl.agent_cost, "build_snapshot", return_value=payload
                ),
                mock.patch.object(sprintctl, "budget_pulse_alive", return_value=True),
                mock.patch.object(sprintctl, "enforce_agent_cost_budget"),
                mock.patch.object(gpu_worker, "mirror_agent_cost") as agent_mirror,
                mock.patch.object(gpu_worker, "mirror_gpu_budget") as gpu_mirror,
            ):
                result = sprintctl.refresh_agent_cost_snapshot(
                    "delegated-run", state, run, {"events": []}
                )

            self.assertEqual(result, payload)
            agent_mirror.assert_not_called()
            gpu_mirror.assert_not_called()
            self.assertEqual(
                json.loads((state / "telemetry/agent-cost-mirror.json").read_text())[
                    "agent_cost_mirror"
                ],
                "delegated_to_budget_pulse",
            )

    def test_live_trace_sync_timeout_is_best_effort_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "trace-run", "volume_name": "trace-volume"}
            timeout = subprocess.TimeoutExpired(["modal", "volume", "get"], 60)
            with mock.patch.object(
                sprintctl, "run_command", side_effect=timeout
            ) as command:
                self.assertFalse(sprintctl.sync_durable_trace(state, run))

            self.assertEqual(
                command.call_args.kwargs["timeout"],
                sprintctl.DURABLE_TRACE_LIVE_SYNC_TIMEOUT_SECONDS,
            )
            stamp = json.loads((state / "durable-trace-sync.json").read_text())
            self.assertFalse(stamp["ok"])
            self.assertIn("TimeoutExpired after 60s", stamp["error"])

    def test_budget_pulse_rejects_stale_upstream_watchdog(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            with (
                mock.patch.object(
                    sprintctl,
                    "load_run",
                    return_value=(state, {"run_id": "pulse-run"}),
                ),
                mock.patch.object(
                    sprintctl,
                    "fetch_budget_watchdog",
                    return_value={
                        "schema_version": 2,
                        "run_id": "pulse-run",
                        "checked_at_epoch_s": 900.0,
                        "total_usd": 1.0,
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "snapshot is stale"):
                    sprintctl.budget_pulse_once("pulse-run", now=1000.0)

    def test_budget_pulse_reports_bounded_watchdog_startup_without_alert(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "pulse-run",
                "created_at": "1970-01-01T00:16:30Z",
            }
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(
                    sprintctl, "fetch_budget_watchdog", return_value=None
                ),
            ):
                payload = sprintctl.budget_pulse_once("pulse-run", now=1000.0)

            self.assertEqual(payload["status"], "watchdog_starting")
            self.assertEqual(payload["startup_age_seconds"], 10.0)
            self.assertEqual(payload["gpu_mirror"], "not_started")
            self.assertEqual(
                json.loads((state / "telemetry/budget-pulse.json").read_text()),
                payload,
            )

    def test_budget_pulse_fails_closed_when_watchdog_misses_startup_deadline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "pulse-run",
                "created_at": "1970-01-01T00:15:00Z",
            }
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(
                    sprintctl, "fetch_budget_watchdog", return_value=None
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "no valid in-sandbox watchdog"
                ):
                    sprintctl.budget_pulse_once("pulse-run", now=1000.0)

    def test_budget_pulse_advances_host_mirror_during_supervised_retry_gap(
        self,
    ) -> None:
        from event_runtime.compute import worker as gpu_worker

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "retry-gap"}
            canonical = {
                "schema_version": 2,
                "run_id": "retry-gap",
                "checked_at_epoch_s": 800.0,
                "components": {"model_api": {"pending_request_count": 0}},
            }
            merged = {
                "schema_version": 2,
                "run_id": "retry-gap",
                "checked_at_epoch_s": 800.0,
                "as_of_epoch_ms": 800000,
                "as_of": "1970-01-01T00:13:20Z",
                "total_usd": 2.5,
                "budget_remaining_usd": 7.5,
                "status": "within_budget",
            }
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(
                    sprintctl, "fetch_budget_watchdog", return_value=canonical
                ),
                mock.patch.object(
                    sprintctl,
                    "_supervised_retry_gap_allows_host_pulse",
                    return_value=True,
                ) as retry_gap,
                mock.patch.object(
                    sprintctl, "build_unified_timeline", return_value={"events": []}
                ),
                mock.patch.object(
                    sprintctl.agent_cost, "build_snapshot", return_value=merged
                ),
                mock.patch.object(
                    gpu_worker,
                    "mirror_gpu_budget",
                    return_value={"gpu_budget_mirror": "updated"},
                ) as gpu_mirror,
                mock.patch.object(
                    gpu_worker,
                    "mirror_agent_cost",
                    return_value={"agent_cost_mirror": "agent_stopped"},
                ),
                mock.patch.object(sprintctl, "enforce_agent_cost_budget"),
            ):
                payload = sprintctl.budget_pulse_once("retry-gap", now=1000.0)

            retry_gap.assert_called_once_with(state, run, canonical)
            mirrored = gpu_mirror.call_args.args[1]
            self.assertEqual(mirrored["checked_at_epoch_s"], 1000.0)
            self.assertEqual(
                mirrored["budget_pulse"]["source"],
                "host_supervised_retry_gap",
            )
            self.assertEqual(payload["source"], "host_supervised_retry_gap")
            self.assertEqual(payload["upstream_watchdog_age_seconds"], 200.0)

    def test_retry_gap_host_pulse_requires_supervisor_and_no_pending_request(
        self,
    ) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
            canonical = {"components": {"model_api": {"pending_request_count": 0}}}
            lock_path = state / "supervise.lock"
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(sprintctl, "harbor_alive", return_value=False):
                    self.assertTrue(
                        sprintctl._supervised_retry_gap_allows_host_pulse(
                            state, {}, canonical
                        )
                    )
                    canonical["components"]["model_api"]["pending_request_count"] = 1
                    self.assertFalse(
                        sprintctl._supervised_retry_gap_allows_host_pulse(
                            state, {}, canonical
                        )
                    )
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

            canonical["components"]["model_api"]["pending_request_count"] = 0
            with mock.patch.object(sprintctl, "harbor_alive", return_value=False):
                self.assertFalse(
                    sprintctl._supervised_retry_gap_allows_host_pulse(
                        state, {}, canonical
                    )
                )

    def test_budget_watchdog_age_allows_bounded_cross_sandbox_clock_skew(
        self,
    ) -> None:
        self.assertEqual(sprintctl._budget_watchdog_age(1000.0, 1005.5), -5.5)

    def test_budget_watchdog_age_rejects_implausible_future_timestamp(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "snapshot is stale"):
            sprintctl._budget_watchdog_age(1000.0, 1060.001)

    def test_budget_pulse_exits_after_natural_harbor_completion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            job = state / "jobs" / "natural-run"
            trial = job / "task__abc"
            trial.mkdir(parents=True)
            (job / "result.json").write_text(
                json.dumps({"finished_at": "2026-08-19T11:16:15Z"})
            )
            (trial / "result.json").write_text(
                json.dumps({"finished_at": "2026-08-19T11:16:15Z"})
            )
            run = {
                "run_id": "natural-run",
                "jobs_root": str(state / "jobs"),
                "job_path": str(job),
                "trial_path": str(trial),
            }

            @contextlib.contextmanager
            def owned_lock(_path: Path, *, blocking: bool = True):
                self.assertFalse(blocking)
                yield True

            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(sprintctl, "file_lock", side_effect=owned_lock),
                mock.patch.object(sprintctl, "budget_pulse_once") as pulse,
                mock.patch.object(sprintctl, "update_run_fields"),
            ):
                self.assertEqual(sprintctl.budget_pulse_loop("natural-run", 15), 0)

            pulse.assert_not_called()
            self.assertFalse((state / "budget-pulse.pid").exists())

    def test_finished_attempt_does_not_close_supervisor_owned_services(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            job = state / "jobs" / "retry-run"
            trial = job / "task__abc"
            trial.mkdir(parents=True)
            for path in (job / "result.json", trial / "result.json"):
                path.write_text(json.dumps({"finished_at": "2026-08-20T19:00:00Z"}))
            run = {
                "run_id": "retry-run",
                "jobs_root": str(state / "jobs"),
                "job_path": str(job),
                "trial_path": str(trial),
            }
            (state / "run.json").write_text(json.dumps(run))
            lock_path = state / "supervise.lock"
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(sprintctl.run_services_should_exit(state, run))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

            self.assertTrue(sprintctl.run_services_should_exit(state, run))

    def test_recoverable_agent_exit_ack_does_not_close_run_services(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))

            self.assertFalse(sprintctl.terminal_stop_acknowledged(state))

            (state / "STOP_ACK.json").write_text(
                json.dumps({"reason": "agent_cost_budget_exhausted"})
            )
            self.assertTrue(sprintctl.terminal_stop_acknowledged(state))

    def test_budget_pulse_continues_after_recoverable_agent_exit_ack(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
            run = {"run_id": "retry-run"}

            @contextlib.contextmanager
            def owned_lock(_path: Path, *, blocking: bool = True):
                self.assertFalse(blocking)
                yield True

            def pulse(_run_id: str) -> dict:
                (state / "FINALIZED.json").write_text("{}\n")
                return {"run_id": "retry-run", "status": "within_budget"}

            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(sprintctl, "file_lock", side_effect=owned_lock),
                mock.patch.object(
                    sprintctl, "run_results_finished", return_value=False
                ),
                mock.patch.object(
                    sprintctl, "budget_pulse_once", side_effect=pulse
                ) as run_pulse,
                mock.patch.object(sprintctl.time, "sleep"),
            ):
                self.assertEqual(sprintctl.budget_pulse_loop("retry-run", 15), 0)

            run_pulse.assert_called_once_with("retry-run")

    def test_gpu_dispatch_loop_registers_and_dispatches_independently(self) -> None:
        from event_runtime.compute import worker as gpu_worker

        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "dispatch-run", "cpu_agent_gpu_worker": True}
            observed: list[str] = []

            @contextlib.contextmanager
            def owned_lock(_path: Path, *, blocking: bool = True):
                self.assertFalse(blocking)
                yield True

            def dispatch(_run_id: str) -> dict:
                observed.append((state / "gpu-dispatch-loop.pid").read_text().strip())
                (state / "FINALIZED.json").write_text("{}\n")
                return {"run_id": "dispatch-run"}

            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state, run)),
                mock.patch.object(sprintctl, "file_lock", side_effect=owned_lock),
                mock.patch.object(
                    sprintctl, "run_results_finished", return_value=False
                ),
                mock.patch.object(gpu_worker, "dispatch_once", side_effect=dispatch),
                mock.patch.object(sprintctl.time, "sleep"),
            ):
                self.assertEqual(sprintctl.gpu_dispatch_loop("dispatch-run", 5), 0)

            self.assertEqual(observed, [str(os.getpid())])
            self.assertFalse((state / "gpu-dispatch-loop.pid").exists())

    def test_budget_watchdog_prefers_live_agent_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "pulse-run",
                "agent_container_id": "ta-agent",
            }
            canonical = {
                "schema_version": 2,
                "run_id": "pulse-run",
                "checked_at_epoch_s": 1000.0,
            }
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(canonical), stderr=""
            )
            with (
                mock.patch.object(
                    sprintctl, "exec_container", return_value=completed
                ) as execute,
                mock.patch.object(sprintctl, "fetch_remote_json") as volume,
            ):
                observed = sprintctl.fetch_budget_watchdog(state, run)

            self.assertEqual(observed, canonical)
            execute.assert_called_once()
            volume.assert_not_called()
            self.assertEqual(
                json.loads((state / "telemetry/budget-watchdog.json").read_text()),
                canonical,
            )

    def test_budget_watchdog_uses_fresh_local_copy_after_remote_miss(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            local = state / "telemetry/budget-watchdog.json"
            local.parent.mkdir(parents=True)
            canonical = {
                "schema_version": 2,
                "run_id": "pulse-run",
                "checked_at_epoch_s": 1000.0,
            }
            local.write_text(json.dumps(canonical))
            with mock.patch.object(sprintctl, "fetch_remote_json", return_value=None):
                observed = sprintctl.fetch_budget_watchdog(
                    state, {"run_id": "pulse-run"}
                )
            self.assertEqual(observed, canonical)

    def test_final_sync_imports_authoritative_durable_gpu_telemetry_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "sync-run", "volume_name": "sync-volume"}
            prefix = "runs/sync-run/telemetry"
            remote = {
                f"{prefix}/samples.jsonl": '{"role":"cpu-agent"}\n',
                f"{prefix}/gpu-stream/samples.jsonl": ('{"role":"training-gpu"}\n'),
                f"{prefix}/gpu_timeline.jsonl": '{"event_id":"gpu-1"}\n',
            }
            with mock.patch.object(
                sprintctl,
                "volume_get_text",
                side_effect=lambda _run, path, **_kwargs: remote.get(path),
            ) as getter:
                self.assertTrue(sprintctl.sync_durable_telemetry(state, run))
                self.assertTrue(sprintctl.sync_durable_telemetry(state, run))
            self.assertEqual(getter.call_count, 3)
            telemetry = state / "telemetry"
            self.assertEqual(
                (telemetry / "durable-gpu-samples.jsonl").read_text(),
                remote[f"{prefix}/gpu-stream/samples.jsonl"],
            )
            self.assertEqual(
                (telemetry / "durable-gpu-timeline.jsonl").read_text(),
                remote[f"{prefix}/gpu_timeline.jsonl"],
            )
            stamp = json.loads((telemetry / "durable-sync.json").read_text())
            self.assertTrue(stamp["ok"])
            self.assertEqual(len(stamp["sources"]), 3)

    def test_live_durable_telemetry_refreshes_after_bounded_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "sync-run", "volume_name": "sync-volume"}
            with (
                mock.patch.object(
                    sprintctl,
                    "volume_get_text",
                    return_value='{"role":"training-gpu"}\n',
                ) as getter,
                mock.patch.object(
                    sprintctl.time,
                    "time",
                    side_effect=(1000.0, 1100.0, 1301.0),
                ),
            ):
                self.assertTrue(
                    sprintctl.sync_durable_telemetry(state, run, max_age_seconds=300)
                )
                self.assertTrue(
                    sprintctl.sync_durable_telemetry(state, run, max_age_seconds=300)
                )
                self.assertTrue(
                    sprintctl.sync_durable_telemetry(state, run, max_age_seconds=300)
                )
            self.assertEqual(getter.call_count, 6)

    def test_live_durable_telemetry_isolates_a_slow_volume_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "sync-run", "volume_name": "sync-volume"}
            prefix = "runs/sync-run/telemetry"

            def fetch(_run: dict, path: str, **kwargs: object) -> str | None:
                self.assertEqual(kwargs["timeout_seconds"], 15)
                if path.endswith("/samples.jsonl") and "/gpu-stream/" not in path:
                    raise subprocess.TimeoutExpired(["modal", "volume", "get"], 15)
                if path.endswith("/gpu-stream/samples.jsonl"):
                    return '{"role":"training-gpu"}\n'
                return None

            with mock.patch.object(sprintctl, "volume_get_text", side_effect=fetch):
                self.assertFalse(
                    sprintctl.sync_durable_telemetry(state, run, max_age_seconds=300)
                )

            telemetry = state / "telemetry"
            self.assertTrue((telemetry / "durable-gpu-samples.jsonl").is_file())
            stamp = json.loads((telemetry / "durable-sync.json").read_text())
            self.assertFalse(stamp["ok"])
            self.assertIn(f"{prefix}/samples.jsonl", stamp["errors"])
            self.assertIn(f"{prefix}/gpu-stream/samples.jsonl", stamp["sources"])

    def test_final_sync_recovers_per_job_stream_missing_from_merge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {"run_id": "sync-run", "volume_name": "sync-volume"}
            prefix = "runs/sync-run/telemetry"
            remote = {
                f"{prefix}/samples.jsonl": '{"role":"cpu-agent"}\n',
                f"{prefix}/gpu-stream/samples.jsonl": (
                    '{"role":"training-gpu","job_id":"present"}\n'
                ),
                f"{prefix}/gpu_timeline.jsonl": (
                    '{"job_id":"present","attempt":1,"event_id":"a"}\n'
                    '{"job_id":"missing","attempt":1,"event_id":"b"}\n'
                ),
                f"{prefix}/by-job/missing/samples.jsonl": (
                    '{"role":"training-gpu","job_id":"missing"}\n'
                ),
                f"{prefix}/by-job/present/samples.jsonl": (
                    '{"role":"training-gpu","job_id":"present","sample_index":1}\n'
                ),
                "runs/sync-run/gpu-jobs/attempts/missing/1.json": (
                    '{"job_id":"missing","attempt":1,"status":"succeeded"}\n'
                ),
            }
            with mock.patch.object(
                sprintctl,
                "volume_get_text",
                side_effect=lambda _run, path, **_kwargs: remote.get(path),
            ) as getter:
                self.assertTrue(
                    sprintctl.sync_durable_telemetry(state, run, force=True)
                )
            self.assertEqual(getter.call_count, 7)
            recovered = (
                state / "telemetry" / "durable-by-job" / "missing" / "samples.jsonl"
            )
            self.assertEqual(
                recovered.read_text(), remote[f"{prefix}/by-job/missing/samples.jsonl"]
            )
            stamp = json.loads((state / "telemetry" / "durable-sync.json").read_text())
            self.assertIn(f"{prefix}/by-job/missing/samples.jsonl", stamp["sources"])

    def test_terra_usage_audit_requires_checksums_and_matching_atif_cost(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            trial = Path(raw)
            sessions = trial / "agent" / "sessions" / "2026" / "08" / "08"
            sessions.mkdir(parents=True)
            source = sessions / "rollout.jsonl"
            source.write_text('{"type":"event_msg"}\n')
            trajectory = trial / "agent" / "trajectory.json"
            trajectory.write_text(
                json.dumps({"final_metrics": {"total_cost_usd": 0.25}})
            )
            provenance = trial / "agent" / "usage-provenance"
            provenance.mkdir()
            source_snapshot = provenance / "source-session.jsonl"
            trajectory_snapshot = provenance / "trajectory.json"
            source_snapshot.write_bytes(source.read_bytes())
            trajectory_snapshot.write_bytes(trajectory.read_bytes())
            (trial / "result.json").write_text(
                json.dumps({"agent_result": {"cost_usd": 0.25}})
            )
            audit = {
                "schema_version": 1,
                "cost_reconstruction_complete": True,
                "request_count": 1,
                "requests": [
                    {
                        "model": "gpt-5.6-terra",
                        "service_tier": "default",
                        "cost_reconstruction_status": "complete",
                    }
                ],
                "reconciliation_mismatches": {},
                "pricing_snapshots": [
                    {
                        "model": "gpt-5.6-terra",
                        "captured_at": "2026-08-08",
                        "source_url": "https://developers.openai.com/api/docs/models/gpt-5.6-terra",
                    }
                ],
                "calculated_api_usage_usd": 0.25,
                "selected_total_cost_usd": 0.25,
                "provenance": {
                    "source_session_path": "usage-provenance/source-session.jsonl",
                    "source_session_sha256": sprintctl.sha256_file(source_snapshot),
                    "trajectory_path": "usage-provenance/trajectory.json",
                    "trajectory_sha256": sprintctl.sha256_file(trajectory_snapshot),
                },
            }
            (trial / "agent" / "usage-audit.json").write_text(json.dumps(audit))

            duplicate = trial / "agent" / "codex-state" / "sessions"
            duplicate.mkdir(parents=True)
            (duplicate / source.name).write_bytes(source.read_bytes())

            ready, details = sprintctl.usage_audit_ready(
                trial, {"model": "openai/gpt-5.6-terra"}
            )

            self.assertTrue(ready)
            self.assertEqual(details, [])

    def test_terra_usage_audit_rejects_tampered_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            trial = Path(raw)
            sessions = trial / "agent" / "sessions"
            sessions.mkdir(parents=True)
            source = sessions / "rollout.jsonl"
            source.write_text("{}\n")
            trajectory = trial / "agent" / "trajectory.json"
            trajectory.write_text(
                json.dumps({"final_metrics": {"total_cost_usd": 0.25}})
            )
            (trial / "result.json").write_text(
                json.dumps({"agent_result": {"cost_usd": 0.25}})
            )
            original_hash = sprintctl.sha256_file(trajectory)
            provenance = trial / "agent" / "usage-provenance"
            provenance.mkdir()
            source_snapshot = provenance / "source-session.jsonl"
            trajectory_snapshot = provenance / "trajectory.json"
            source_snapshot.write_bytes(source.read_bytes())
            trajectory_snapshot.write_bytes(trajectory.read_bytes())
            audit = {
                "schema_version": 1,
                "cost_reconstruction_complete": True,
                "request_count": 1,
                "requests": [
                    {
                        "model": "gpt-5.6-terra",
                        "service_tier": "default",
                        "cost_reconstruction_status": "complete",
                    }
                ],
                "reconciliation_mismatches": {},
                "calculated_api_usage_usd": 0.25,
                "selected_total_cost_usd": 0.25,
                "provenance": {
                    "source_session_path": "usage-provenance/source-session.jsonl",
                    "source_session_sha256": sprintctl.sha256_file(source_snapshot),
                    "trajectory_path": "usage-provenance/trajectory.json",
                    "trajectory_sha256": original_hash,
                },
            }
            (trial / "agent" / "usage-audit.json").write_text(json.dumps(audit))
            trajectory.write_text(
                json.dumps({"final_metrics": {"total_cost_usd": 0.10}})
            )

            ready, details = sprintctl.usage_audit_ready(
                trial, {"model": "gpt-5.6-terra"}
            )

            self.assertFalse(ready)
            self.assertIn("ATIF total cost differs from usage audit", details)
            self.assertNotIn(
                "usage audit trajectory snapshot checksum mismatch", details
            )

    def test_unified_timeline_readiness_requires_current_schema(self) -> None:
        payload = {
            "schema_version": 6,
            "run": {"run_id": "timeline-current"},
            "coverage": {"ready": True},
        }
        self.assertTrue(sprintctl.unified_timeline_ready(payload, "timeline-current"))
        payload["schema_version"] = 1
        self.assertFalse(sprintctl.unified_timeline_ready(payload, "timeline-current"))
        payload["schema_version"] = 6
        self.assertFalse(sprintctl.unified_timeline_ready(payload, "other-run"))

    def test_natural_completion_does_not_require_stop_ack(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            job = state / "jobs" / "natural-run"
            trial = job / "task__abc"
            (trial / "artifacts" / "continuous").mkdir(parents=True)
            (trial / "artifacts" / "continuous" / "ledger.jsonl").write_text("")
            (job / "result.json").write_text(
                json.dumps({"finished_at": "2026-08-19T11:16:15Z"})
            )
            (trial / "result.json").write_text(
                json.dumps({"finished_at": "2026-08-19T11:16:15Z"})
            )
            run = {
                "run_id": "natural-run",
                "state_dir": str(state),
                "jobs_root": str(state / "jobs"),
                "job_path": str(job),
                "trial_path": str(trial),
                "evaluation_result_policy": "all_blind_archival_submissions",
            }
            with (
                mock.patch.object(sprintctl, "update_run_fields"),
                mock.patch.object(sprintctl, "harbor_alive", return_value=False),
                mock.patch.object(sprintctl, "worker_alive", return_value=False),
            ):
                _complete, conditions, _details = sprintctl.final_conditions(state, run)
                self.assertTrue(conditions["stop_ack"])

                (state / "STOP_REQUESTED.json").write_text("{}\n")
                _complete, conditions, _details = sprintctl.final_conditions(state, run)
                self.assertTrue(conditions["stop_ack"])

                (job / "result.json").write_text("{}\n")
                _complete, conditions, _details = sprintctl.final_conditions(state, run)
                self.assertFalse(conditions["stop_ack"])

    def test_finalization_rejects_forwarded_gpu_submission_missing_from_ledger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            job = state / "jobs" / "bridge-run"
            trial = job / "task__abc"
            continuous = trial / "artifacts" / "continuous"
            continuous.mkdir(parents=True)
            ledger = continuous / "ledger.jsonl"
            ledger.write_text("")
            (job / "result.json").write_text('{"finished_at":"now"}\n')
            (trial / "result.json").write_text(
                '{"finished_at":"now","continuous_verification":{"submissions":[]}}\n'
            )
            bridge = state / "submission-bridge"
            bridge.mkdir()
            (bridge / "123456-abcd.json").write_text(
                json.dumps(
                    {
                        "submission_id": "123456-abcd",
                        "queue_name": "123456-abcd.pt",
                        "state": "forwarded",
                    }
                )
            )
            run = {
                "run_id": "bridge-run",
                "state_dir": str(state),
                "jobs_root": str(state / "jobs"),
                "job_path": str(job),
                "trial_path": str(trial),
                "evaluation_result_policy": "all_blind_archival_submissions",
            }
            with (
                mock.patch.object(sprintctl, "update_run_fields"),
                mock.patch.object(sprintctl, "harbor_alive", return_value=False),
                mock.patch.object(sprintctl, "worker_alive", return_value=False),
            ):
                _complete, conditions, details = sprintctl.final_conditions(state, run)
                self.assertFalse(conditions["submission_bridge_drained"])
                self.assertTrue(any("absent from Harbor ledger" in d for d in details))

                ledger.write_text(json.dumps(row(1, "123456-abcd.pt", 0.0)) + "\n")
                _complete, conditions, _details = sprintctl.final_conditions(state, run)
                self.assertTrue(conditions["submission_bridge_drained"])

    def test_stale_finalized_file_is_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {"run_id": "stale-final", "agent_kind": "codex"}
            (state_dir / "FINALIZED.json").write_text(
                json.dumps({"complete": True, "schema_version": 1})
            )
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(sprintctl, "monitor_once") as monitor,
                mock.patch.object(
                    sprintctl, "run_services_should_exit", return_value=True
                ),
                mock.patch.object(
                    sprintctl,
                    "final_conditions",
                    return_value=(
                        False,
                        {"unified_timeline_ready": False},
                        ["unified_timeline_ready"],
                    ),
                ),
            ):
                complete, payload = sprintctl.finalize(
                    "stale-final", upload=False, include_remote=False
                )
            self.assertFalse(complete)
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["timeline_schema_version"], 6)
            monitor.assert_called_once()

    def test_provider_finalized_marker_without_settlement_gate_is_rechecked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "provider-final",
                "agent_kind": "codex",
                "provider_usage_ledger_required": True,
            }
            (state_dir / "FINALIZED.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "timeline_schema_version": 6,
                        "conditions": {},
                    }
                )
            )
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(sprintctl, "monitor_once") as monitor,
                mock.patch.object(sprintctl, "sync_durable_api_usage"),
                mock.patch.object(
                    sprintctl,
                    "final_conditions",
                    return_value=(
                        False,
                        {"provider_usage_ledger_settled": False},
                        ["provider_usage_ledger_settled"],
                    ),
                ),
            ):
                complete, _payload = sprintctl.finalize(
                    "provider-final", upload=False, include_remote=False
                )
            self.assertFalse(complete)
            monitor.assert_called_once()

    def test_verifier_cannot_replace_a_known_agent_in_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "known-agent",
                "agent_kind": "codex",
                "app_id": "ap-test",
                "agent_container_id": "ta-old-agent",
                "cpu_agent_gpu_worker": True,
            }
            (state_dir / "run.json").write_text(json.dumps(run))
            with (
                mock.patch.object(sprintctl, "discover_app_id", return_value="ap-test"),
                mock.patch.object(
                    sprintctl, "containers_for_app", return_value=["ta-verifier"]
                ),
                mock.patch.object(sprintctl, "is_agent_container", return_value=False),
            ):
                self.assertIsNone(sprintctl.discover_agent_container(state_dir, run))
            saved = json.loads((state_dir / "run.json").read_text())
            self.assertEqual(saved["agent_container_id"], "ta-old-agent")

    def test_single_unverified_build_container_is_not_agent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "startup-agent",
                "agent_kind": "codex",
                "app_id": "ap-test",
                "cpu_agent_gpu_worker": True,
            }
            (state_dir / "run.json").write_text(json.dumps(run))
            with (
                mock.patch.object(sprintctl, "discover_app_id", return_value="ap-test"),
                mock.patch.object(
                    sprintctl, "containers_for_app", return_value=["ta-starting"]
                ),
                mock.patch.object(sprintctl, "is_agent_container", return_value=False),
            ):
                self.assertIsNone(sprintctl.discover_agent_container(state_dir, run))

    def test_redacted_dry_run_does_not_create_state(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        token = "fake-oauth-value-that-must-never-print-123456789"
        state = OPS / run_id
        env = os.environ.copy()
        env.pop("AGENT_COST_BUDGET_USD", None)
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.assertNotIn(token, completed.stdout + completed.stderr)
        self.assertIn("CLAUDE_CODE_OAUTH_TOKEN=[configured]", completed.stdout)
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_kind"], "claude-code")
        self.assertTrue(config["automatic_stop"])
        self.assertEqual(config["automatic_stop_reason"], "agent_cost_budget_exhausted")
        self.assertEqual(config["agent_cost_budget_usd"], 10.0)
        self.assertNotIn("stop_after_seconds", config)
        self.assertFalse(state.exists())

    def test_codex_dry_run_is_redacted_and_manual_only(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        key = "fake-openai-key-that-must-never-print-123456789"
        state = OPS / run_id
        env = os.environ.copy()
        env.pop("AGENT_COST_BUDGET_USD", None)
        env["OPENAI_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "openai/test-codex-model",
                "--endpoint",
                "https://api.openai.com/v1",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.assertNotIn(key, completed.stdout + completed.stderr)
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_kind"], "codex")
        self.assertEqual(config["model"], "openai/test-codex-model")
        self.assertEqual(config["agent_allowed_host"], "api.openai.com")
        self.assertEqual(config["hosted_model_tools_policy"], "disabled")
        self.assertIsNone(config["service_tier"])
        self.assertFalse(config["usage_audit_required"])
        self.assertTrue(config["automatic_stop"])
        self.assertEqual(config["automatic_stop_reason"], "agent_cost_budget_exhausted")
        self.assertEqual(config["agent_cost_budget_usd"], 10.0)
        self.assertNotIn("stop_after_seconds", config)
        self.assertFalse(state.exists())

    def test_any_codex_model_on_openrouter_uses_exact_shared_ledger(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        key = "fake-openrouter-key-that-must-never-print-123456789"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "vendor/future-model",
                "--endpoint",
                "https://openrouter.ai/api/v1",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        self.assertNotIn(key, completed.stdout + completed.stderr)
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_allowed_host"], "openrouter.ai")
        self.assertTrue(config["usage_audit_required"])
        self.assertEqual(
            config["budget_enforcement"]["api_cost_source"],
            "openrouter_reported_per_request",
        )

    def test_dry_run_accepts_one_global_budget_override(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["CLAUDE_CODE_OAUTH_TOKEN"] = (
            "fake-oauth-value-that-must-never-print-123456789"
        )
        env["AGENT_COST_BUDGET_USD"] = "12.5"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_cost_budget_usd"], 12.5)

    def test_terra_dry_run_pins_reconstructible_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-openai-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "openai/gpt-5.6-terra",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["service_tier"], "default")
        self.assertEqual(config["hosted_model_tools_policy"], "disabled")
        self.assertTrue(config["usage_audit_required"])

    def test_luna_dry_run_pins_reconstructible_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        key = "fake-openrouter-key-that-must-never-print-123456789"
        env["OPENROUTER_API_KEY"] = key
        env["OPENAI_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "openai/gpt-5.6-luna",
                "--endpoint",
                "https://openrouter.ai/api/v1",
                "--reasoning-effort",
                "max",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["service_tier"], "default")
        self.assertEqual(config["reasoning_effort"], "max")
        self.assertEqual(config["agent_allowed_host"], "openrouter.ai")
        self.assertEqual(
            config["openrouter_route"],
            {
                "only": ["openai"],
                "order": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": [],
            },
        )
        self.assertEqual(
            config["openrouter_request_contract"],
            {
                "model": "openai/gpt-5.6-luna",
                "max_output_tokens": 128_000,
                "reasoning": {"effort": "max"},
                "service_tier": "default",
            },
        )
        self.assertEqual(
            config["budget_enforcement"]["api_cost_source"],
            "openrouter_reported_per_request",
        )
        self.assertEqual(config["agent_cost_budget_usd"], 10.0)
        self.assertEqual(config["budget_enforcement"]["shutdown_reserve_usd"], 0.0)
        self.assertEqual(
            config["budget_enforcement"]["minimum_safe_shutdown_reserve_usd"],
            0.0,
        )
        self.assertTrue(config["usage_audit_required"])

    def test_deepseek_vision_codex_dry_run_seals_benchmark_contract(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        key = "fake-openrouter-key-that-must-never-print-123456789"
        env["OPENROUTER_API_KEY"] = key
        env["OPENAI_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "deepseek/deepseek-v4-flash-vision-exp",
                "--endpoint",
                "https://openrouter.ai/api/v1",
                "--reasoning-effort",
                "max",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["openrouter_route"]["only"], ["deepseek"])
        self.assertEqual(
            config["openrouter_request_contract"],
            {
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "temperature": 1.0,
                "top_p": 0.95,
                "max_output_tokens": 384_000,
                "reasoning": {"effort": "max"},
            },
        )

    def test_sol_dry_run_pins_reconstructible_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-openai-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "openai/gpt-5.6-sol",
                "--reasoning-effort",
                "high",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["service_tier"], "default")
        self.assertEqual(config["reasoning_effort"], "high")
        self.assertTrue(config["usage_audit_required"])

    def test_deepseek_dry_run_pins_baidu_and_peak_normalized_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-deepseek-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "deepseek/deepseek-v4-flash",
                "--endpoint",
                "https://openrouter.ai/api/v1",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_allowed_host"], "openrouter.ai")
        self.assertEqual(config["model"], "deepseek/deepseek-v4-flash-0731")
        self.assertEqual(
            config["openrouter_route"],
            {
                "only": ["baidu/fp8"],
                "order": ["baidu/fp8"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
        )
        self.assertEqual(
            config["budget_enforcement"]["api_budget_cost_basis"],
            "openrouter_list_price_with_deepseek_peak_floor",
        )
        self.assertTrue(config["usage_audit_required"])

    def test_deepseek_dry_run_rejects_provider_override(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-deepseek-key-that-must-never-print-123456789"
        env["SPRINT_OPENROUTER_PROVIDER_ENDPOINT"] = "deepinfra"

        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "deepseek/deepseek-v4-flash-0731",
                "--endpoint",
                "https://openrouter.ai/api/v1",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("locked to baidu/fp8", completed.stderr)

    def test_deepseek_harness_dry_run_seals_official_vision_contract(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        key = "fake-openrouter-key-that-must-never-print-123456789"
        env["OPENROUTER_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "event_runtime/control/launch.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "deepseek-harness",
                "--model",
                "deepseek/deepseek-v4-flash-vision-exp",
                "--reasoning-effort",
                "max",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        self.assertNotIn(key, completed.stdout + completed.stderr)
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_kind"], "deepseek-harness")
        self.assertEqual(config["agent_allowed_host"], "openrouter.ai")
        self.assertEqual(config["deepseek_harness_version"], "0.1.1-rc.2")
        self.assertEqual(config["deepseek_harness_sdk_version"], "0.1.1rc1")
        self.assertTrue(config["provider_usage_ledger_required"])
        self.assertTrue(config["usage_audit_required"])
        self.assertEqual(
            config["openrouter_route"],
            {
                "only": ["deepseek"],
                "order": ["deepseek"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": [],
            },
        )
        self.assertEqual(
            config["openrouter_request_contract"],
            {
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "stream": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 384_000,
                "reasoning_effort": "max",
            },
        )

    def test_stop_watcher_signals_only_dummy_claude_and_acks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            durable = root / "durable"
            runtime = root / "run"
            app = root / "app"
            agent_logs = root / "agent"
            artifact_logs = root / "artifacts"
            fake_root = root / "root"
            for path in (durable, runtime, app, agent_logs, artifact_logs, fake_root):
                path.mkdir()
            password = durable / "runs/test-watch/secrets/restic-password"
            password.parent.mkdir(parents=True)
            password.write_text("test-password-with-enough-entropy\n")
            password.chmod(0o600)

            fake_restic = root / "fake-restic"
            fake_restic.write_text(
                """#!/usr/bin/env bash
set -e
repo=""
for ((i=1;i<=$#;i++)); do
  if [[ "${!i}" == "-r" ]]; then j=$((i+1)); repo="${!j}"; fi
done
if [[ " $* " == *" init "* ]]; then
  mkdir -p "$repo"; printf config > "$repo/config"; exit 0
fi
if [[ " $* " == *" backup "* ]]; then
  printf '%s\n' '{"message_type":"summary","snapshot_id":"deadbeef"}'; exit 0
fi
exit 0
"""
            )
            fake_restic.chmod(0o755)

            signal_file = root / "dummy-signal"
            dummy = root / "sprint-dummy-claude.sh"
            dummy.write_text(
                f"""#!/usr/bin/env bash
trap 'printf INT > "{signal_file}"; exit 0' INT
while true; do sleep 1; done
"""
            )
            dummy.chmod(0o755)
            dummy_process = subprocess.Popen([str(dummy)])
            watcher = subprocess.Popen(
                [
                    "bash",
                    str(ROOT / "event_runtime/container/sprint-snapshot-loop.sh"),
                    "--run-id",
                    "test-watch",
                    "--agent-kind",
                    "claude-code",
                    "--snapshot-seconds",
                    "300",
                    "--poll-seconds",
                    "1",
                    "--term-grace-seconds",
                    "3",
                    "--durable-dir",
                    str(durable),
                    "--runtime-dir",
                    str(runtime),
                    "--app-dir",
                    str(app),
                    "--agent-log-dir",
                    str(agent_logs),
                    "--artifact-log-dir",
                    str(artifact_logs),
                    "--root-dir",
                    str(fake_root),
                    "--restic-bin",
                    str(fake_restic),
                    "--budget-watchdog-bin",
                    "/bin/true",
                    "--claude-pattern",
                    str(dummy).replace(".", r"\."),
                    "--exit-after-ack",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                first_seen = (
                    durable / "runs/test-watch/snapshot/first-claude-seen.attempt-001"
                )
                deadline = time.time() + 15
                while not first_seen.exists() and time.time() < deadline:
                    time.sleep(0.1)
                self.assertTrue(first_seen.exists())
                (runtime / "sprint-stop").touch()
                watcher.wait(timeout=20)
                dummy_process.wait(timeout=10)
                ack = json.loads((durable / "runs/test-watch/STOP_ACK").read_text())
                self.assertEqual(ack["reason"], "operator_stop")
                self.assertEqual(ack["final_snapshot_id"], "deadbeef")
                self.assertEqual(signal_file.read_text(), "INT")
            finally:
                watcher.kill()
                dummy_process.kill()
                watcher.communicate()
                dummy_process.wait()

    def test_codex_group_escalation_snapshot_and_ack_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            durable = root / "durable"
            runtime = root / "run"
            app = root / "app"
            agent_logs = root / "agent"
            artifact_logs = root / "artifacts"
            fake_root = root / "root"
            codex_home = root / "codex-home"
            for path in (
                durable,
                runtime,
                app,
                agent_logs,
                artifact_logs,
                fake_root,
                codex_home,
            ):
                path.mkdir()
            (codex_home / "sessions/2026/08/02").mkdir(parents=True)
            event = codex_home / "sessions/2026/08/02/rollout.jsonl"
            event.write_text('{"type":"event_msg","payload":{"type":"turn_started"}}\n')
            (codex_home / "history.jsonl").write_text('{"session_id":"trace"}\n')
            (codex_home / "config.toml").write_text('model = "test-model"\n')
            (codex_home / "auth.json").write_text('{"OPENAI_API_KEY":"raw-secret"}\n')
            (codex_home / "cache").mkdir()
            (codex_home / "cache/large.bin").write_bytes(b"cache")
            (fake_root / ".ssh").mkdir()
            (fake_root / ".ssh/id_test").write_text("raw-private-key\n")

            password = durable / "runs/test-codex/secrets/restic-password"
            password.parent.mkdir(parents=True)
            password.write_text("test-password-with-enough-entropy\n")
            password.chmod(0o600)

            args_log = root / "restic-args.log"
            failed_final = root / "failed-final"
            fake_restic = root / "fake-restic"
            fake_restic.write_text(
                f"""#!/usr/bin/env bash
set -e
repo=""
for ((i=1;i<=$#;i++)); do
  printf 'ARG:%s\\n' "${{!i}}" >> "{args_log}"
  if [[ "${{!i}}" == "-r" ]]; then j=$((i+1)); repo="${{!j}}"; fi
done
printf 'CALL-END\\n' >> "{args_log}"
if [[ " $* " == *" init "* ]]; then
  mkdir -p "$repo"; printf config > "$repo/config"; exit 0
fi
if [[ " $* " == *" backup "* ]]; then
  if [[ " $* " == *" reason:operator_stop "* && ! -e "{failed_final}" ]]; then
    : > "{failed_final}"
    exit 1
  fi
  printf '%s\\n' '{{"message_type":"summary","snapshot_id":"codex-snapshot"}}'
fi
"""
            )
            fake_restic.chmod(0o755)

            signal_file = root / "codex-signals"
            package_root = root / "codex-package"
            launcher = package_root / "bin/codex.js"
            dummy = (
                package_root
                / "node_modules/@openai/codex-linux-test/vendor/test/bin/codex"
            )
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/usr/bin/env node\n")
            launcher.chmod(0o755)
            dummy.parent.mkdir(parents=True)
            dummy.write_text(
                f"""#!/usr/bin/env python3
import signal
import time

path = {str(signal_file)!r}

def record(name):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(name + "\\n")

signal.signal(signal.SIGINT, lambda *_: record("INT"))
def terminate(*_):
    record("TERM")
    # Codex may surface an interrupted unified_exec as exit 1. The
    # wrapper must classify it from the trusted stop handshake, not this code.
    raise SystemExit(1)
signal.signal(signal.SIGTERM, terminate)
while True:
    time.sleep(1)
"""
            )
            dummy.chmod(0o755)

            wrapper_env = os.environ.copy()
            wrapper_env.update(
                {
                    "CODEX_HOME": str(codex_home),
                    "SPRINT_RUNTIME_DIR": str(runtime),
                    "SPRINT_AGENT_LOG_DIR": str(agent_logs),
                    "SPRINT_DURABLE_DIR": str(durable),
                    "SPRINT_RUN_ID": "test-codex",
                    "SPRINT_STOP_ACK_TIMEOUT_SECONDS": "30",
                }
            )
            wrapper = subprocess.Popen(
                [
                    "bash",
                    str(ROOT / "event_runtime/container/sprint-codex-exec-wrapper.sh"),
                    str(launcher),
                    "exec",
                    "--json",
                    "--",
                    "test prompt",
                ],
                env=wrapper_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            monitor = subprocess.Popen(
                ["bash", "-c", "exec -a harbor-codex-monitor sleep 60"]
            )
            watcher = subprocess.Popen(
                [
                    "bash",
                    str(ROOT / "event_runtime/container/sprint-snapshot-loop.sh"),
                    "--run-id",
                    "test-codex",
                    "--agent-kind",
                    "codex",
                    "--snapshot-seconds",
                    "300",
                    "--poll-seconds",
                    "1",
                    "--term-grace-seconds",
                    "2",
                    "--durable-dir",
                    str(durable),
                    "--runtime-dir",
                    str(runtime),
                    "--app-dir",
                    str(app),
                    "--agent-log-dir",
                    str(agent_logs),
                    "--artifact-log-dir",
                    str(artifact_logs),
                    "--root-dir",
                    str(fake_root),
                    "--codex-home-dir",
                    str(codex_home),
                    "--restic-bin",
                    str(fake_restic),
                    "--budget-watchdog-bin",
                    "/bin/true",
                    "--codex-pattern",
                    str(dummy).replace(".", r"\."),
                    "--exit-after-ack",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                first_seen = (
                    durable / "runs/test-codex/snapshot/first-codex-seen.attempt-001"
                )
                deadline = time.time() + 15
                while not first_seen.exists() and time.time() < deadline:
                    time.sleep(0.1)
                self.assertTrue(first_seen.exists())
                (runtime / "sprint-stop").touch()

                deadline = time.time() + 15
                while not failed_final.exists() and time.time() < deadline:
                    time.sleep(0.1)
                self.assertTrue(failed_final.exists())
                self.assertFalse((durable / "runs/test-codex/STOP_ACK").exists())

                watcher.wait(timeout=25)
                wrapper.wait(timeout=10)
                self.assertEqual(wrapper.returncode, 0)
                self.assertIsNone(monitor.poll())
                self.assertEqual(signal_file.read_text().splitlines(), ["INT", "TERM"])

                ack = json.loads((durable / "runs/test-codex/STOP_ACK").read_text())
                self.assertEqual(ack["agent_kind"], "codex")
                self.assertEqual(ack["reason"], "operator_stop")
                self.assertEqual(ack["final_snapshot_id"], "codex-snapshot")

                copied = agent_logs / "codex-state"
                self.assertEqual(
                    (copied / "sessions/2026/08/02/rollout.jsonl").read_text(),
                    event.read_text(),
                )
                self.assertTrue((copied / "history.jsonl").is_file())
                self.assertTrue((copied / "config.toml").is_file())
                self.assertFalse((copied / "auth.json").exists())
                self.assertFalse((copied / "cache").exists())

                arguments = [
                    line.removeprefix("ARG:")
                    for line in args_log.read_text().splitlines()
                    if line.startswith("ARG:")
                ]
                excluded = {
                    arguments[index + 1]
                    for index, value in enumerate(arguments[:-1])
                    if value == "--exclude"
                }
                self.assertIn(str(codex_home / "sessions"), arguments)
                self.assertIn(str(codex_home / "history.jsonl"), arguments)
                self.assertIn(str(codex_home / "config.toml"), arguments)
                self.assertIn(str(codex_home / "auth.json"), excluded)
                self.assertIn(str(codex_home / "cache"), excluded)
                self.assertNotIn(str(fake_root), arguments)
            finally:
                watcher.kill()
                monitor.kill()
                wrapper.kill()
                process_file = runtime / "sprint-agent/codex-process"
                if process_file.exists():
                    try:
                        _, pgid, _ = process_file.read_text().split()
                        os.killpg(int(pgid), signal.SIGKILL)
                    except (OSError, ValueError):
                        pass
                watcher.communicate()
                monitor.wait()
                wrapper.communicate()

    def test_atomic_attempt_archive_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            attempt = root / "0001-policy.pt"
            attempt.mkdir()
            (attempt / "result.json").write_text('{"finished_at":"now"}\n')
            (attempt / "policy.pt").write_bytes(b"policy")
            archive, digest, checksum = sprintctl.tar_attempt_atomic(
                attempt, root / "archives"
            )
            again, again_digest, _ = sprintctl.tar_attempt_atomic(
                attempt, root / "archives"
            )
            self.assertEqual(archive, again)
            self.assertEqual(digest, again_digest)
            self.assertEqual(sprintctl.sha256_file(archive), digest)
            self.assertEqual(checksum.read_text().split()[0], digest)
            self.assertFalse(archive.stat().st_mode & stat.S_IWUSR)

    def test_attempt_archive_refreshes_after_recovery_changes_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            attempt = root / "0001-policy.pt"
            attempt.mkdir()
            result = attempt / "result.json"
            result.write_text('{"error":"transport loss"}\n')
            first, first_digest, _ = sprintctl.tar_attempt_atomic(
                attempt, root / "archives"
            )
            result.write_text('{"rewards":{"reward":1}}\n')
            second, second_digest, _ = sprintctl.tar_attempt_atomic(
                attempt, root / "archives", refresh=True
            )
            self.assertNotEqual(first, second)
            self.assertNotEqual(first_digest, second_digest)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

    def test_completed_attempt_manifest_refreshes_when_ledger_row_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw) / "state"
            trial = Path(raw) / "trial"
            state.mkdir()
            policy = write_policy(trial, 1, "policy.pt", b"policy")
            attempt = policy.parents[3]
            ledger = trial / "artifacts/continuous/ledger.jsonl"
            first_row = {
                **row(1, "policy.pt", 9.0),
                "artifact_sha256": sprintctl.sha256_file(policy),
            }
            ledger.write_text(json.dumps(first_row) + "\n")
            first = sprintctl.archive_completed_attempts(
                state, {"run_id": "run"}, trial, upload=False
            )
            first_record = first["attempts"][attempt.name]

            recovered_row = {
                **first_row,
                "verification_attempts": 2,
                "verification_recovery_events": [{"resume_attempt": 2}],
            }
            ledger.write_text(json.dumps(recovered_row) + "\n")
            (attempt / "result.json").write_text(json.dumps(recovered_row) + "\n")
            second = sprintctl.archive_completed_attempts(
                state, {"run_id": "run"}, trial, upload=False
            )
            second_record = second["attempts"][attempt.name]
            self.assertNotEqual(
                first_record["ledger_row_sha256"],
                second_record["ledger_row_sha256"],
            )
            self.assertNotEqual(first_record["archive"], second_record["archive"])

    def test_malformed_partial_ledger_keeps_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = Path(raw) / "ledger.jsonl"
            ledger.write_text(json.dumps(row(1, "one.pt", 10.0)) + '\n{"index":')
            read = frontier_update.read_ledger(ledger)
            self.assertEqual(len(read.rows), 1)
            self.assertEqual(len(read.errors), 1)
            self.assertIn("partial JSON", read.errors[0])
            self.assertEqual(frontier_update.ledger_counts(read)["malformed"], 1)

    def test_out_of_order_grading_recomputes_full_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            job = root / "job"
            trial = job / "task__trial"
            web = root / "web"
            ledger = trial / "artifacts/continuous/ledger.jsonl"
            ledger.parent.mkdir(parents=True)
            web.mkdir()
            first_policy = write_policy(trial, 1, "one.pt", b"faster")
            second_policy = write_policy(trial, 2, "two.pt", b"slower")
            state_path = root / "frontier.json"

            ledger.write_text(
                "\n".join(
                    [
                        json.dumps(row(1, "one.pt", None)),
                        json.dumps(row(2, "two.pt", 9.0)),
                    ]
                )
                + "\n"
            )
            first = frontier_update.scan_frontier(
                job=job, trial=trial, state_path=state_path, web=web
            )
            second_hash = frontier_update.sha256_file(second_policy)
            self.assertEqual(first["frontier"], [second_hash])

            ledger.write_text(
                "\n".join(
                    [
                        json.dumps(row(1, "one.pt", 8.0)),
                        json.dumps(row(2, "two.pt", 9.0)),
                    ]
                )
                + "\n"
            )
            second = frontier_update.scan_frontier(
                job=job, trial=trial, state_path=state_path, web=web
            )
            first_hash = frontier_update.sha256_file(first_policy)
            self.assertEqual(second["frontier"], [first_hash])
            old = next(
                item
                for item in second["capture_queue"]
                if item["policy_hash"] == second_hash
            )
            self.assertEqual(old["status"], "skipped_dominated")

    def test_frontier_uses_only_valid_time_with_tolerance(self) -> None:
        candidate = frontier_update.Candidate
        speed_only = [
            candidate(1, "one", 10.0, "a", "/a"),
            candidate(2, "two", 9.9995, "b", "/b"),
        ]
        frontier, has_secondary_objective = frontier_update.compute_frontier(speed_only)
        self.assertFalse(has_secondary_objective)
        self.assertEqual([item.index for item in frontier], [1])

        faster = [
            candidate(1, "one", 9.0, "a", "/a"),
            candidate(2, "two", 10.0, "b", "/b"),
        ]
        frontier, has_secondary_objective = frontier_update.compute_frontier(faster)
        self.assertFalse(has_secondary_objective)
        self.assertEqual([item.index for item in frontier], [1])

    def test_stop_and_finalize_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "test-idempotent",
                "agent_kind": "codex",
                "agent_container_id": "ta-test",
            }
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(sprintctl, "fetch_remote_json", return_value=None),
                mock.patch.object(
                    sprintctl, "discover_agent_container", return_value="ta-test"
                ),
                mock.patch.object(sprintctl, "exec_container") as remote_exec,
            ):
                first = sprintctl.request_stop("test-idempotent")
                marker = (state_dir / "STOP_REQUESTED.json").read_bytes()
                second = sprintctl.request_stop("test-idempotent")
                self.assertEqual(first["agent_kind"], "codex")
                self.assertEqual(first["requested_at"], second["requested_at"])
                self.assertEqual(
                    marker, (state_dir / "STOP_REQUESTED.json").read_bytes()
                )
                self.assertEqual(remote_exec.call_count, 2)

            stale_crash_ack = {
                "reason": "agent_exit",
                "acknowledged_at": "2026-08-08T00:00:00Z",
            }
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(
                    sprintctl, "fetch_remote_json", return_value=stale_crash_ack
                ),
                mock.patch.object(
                    sprintctl, "discover_agent_container", return_value="ta-test"
                ),
                mock.patch.object(sprintctl, "exec_container") as remote_exec,
            ):
                resumed_stop = sprintctl.request_stop("test-idempotent")
                self.assertEqual(resumed_stop["status"], "requested")
                remote_exec.assert_called_once()

            run_path = state_dir / "run.json"
            run_path.write_text(
                json.dumps(
                    {
                        "run_id": "test-idempotent",
                        "agent_kind": "codex",
                        "cpu_launch_attempt": 2,
                        "cpu_launch_history": [{"attempt": 1}, {"attempt": 2}],
                        "jobs_root": "/attempt-2",
                    }
                )
            )
            updated = sprintctl.update_run_fields(
                state_dir, agent_container_id="ta-new"
            )
            self.assertEqual(updated["cpu_launch_attempt"], 2)
            self.assertEqual(updated["jobs_root"], "/attempt-2")
            self.assertEqual(updated["agent_container_id"], "ta-new")

            (state_dir / "archive-manifest.json").write_text(
                '{"schema_version":1,"attempts":{}}\n'
            )
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(sprintctl, "monitor_once") as monitor,
                mock.patch.object(
                    sprintctl, "run_services_should_exit", return_value=True
                ),
                mock.patch.object(
                    sprintctl,
                    "final_conditions",
                    return_value=(True, {"all": True}, []),
                ),
                mock.patch.object(sprintctl, "volume_upload"),
            ):
                complete_one, payload_one = sprintctl.finalize("test-idempotent")
                final_bytes = (state_dir / "FINALIZED.json").read_bytes()
                complete_two, payload_two = sprintctl.finalize("test-idempotent")
                self.assertTrue(complete_one and complete_two)
                self.assertEqual(payload_one, payload_two)
                self.assertEqual(
                    final_bytes, (state_dir / "FINALIZED.json").read_bytes()
                )
                self.assertEqual(monitor.call_count, 1)

    def test_stop_signals_cpu_before_waiting_for_gpu_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "test-stop-order",
                "agent_kind": "codex",
                "agent_container_id": "ta-test",
            }
            events: list[str] = []

            def signal_cpu(*_args, **_kwargs) -> None:
                events.append("cpu-signalled")

            def stop_gpu(*_args, **_kwargs) -> list[dict]:
                self.assertEqual(events, ["cpu-signalled"])
                events.append("gpu-cleanup")
                return []

            def fetch_ack(*_args, **_kwargs) -> None:
                self.assertEqual(events, ["cpu-signalled", "gpu-cleanup"])
                events.append("ack-fetched")
                return None

            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(
                    sprintctl, "fetch_remote_json", side_effect=fetch_ack
                ),
                mock.patch.object(
                    sprintctl, "discover_agent_container", return_value="ta-test"
                ),
                mock.patch.object(sprintctl, "exec_container", side_effect=signal_cpu),
                mock.patch(
                    "event_runtime.compute.worker.stop_all", side_effect=stop_gpu
                ),
            ):
                result = sprintctl.request_stop(
                    "test-stop-order", reason="agent_cost_budget_exhausted"
                )

            self.assertEqual(events, ["cpu-signalled", "gpu-cleanup", "ack-fetched"])
            self.assertEqual(result["status"], "requested")

    def test_monitor_reapplies_stop_requested_before_sandbox_existed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "stop-during-build",
                "agent_kind": "codex",
                "cpu_agent_gpu_worker": False,
            }
            (state_dir / "STOP_REQUESTED.json").write_text(
                json.dumps({"reason": "operator_batch_stop"})
            )
            expected_status = {"run_id": "stop-during-build"}
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(sprintctl, "request_stop") as request_stop,
                mock.patch("event_runtime.telemetry.host.poll_once"),
                mock.patch.object(
                    sprintctl, "discover_job_and_trial", return_value=(None, None)
                ),
                mock.patch.object(
                    sprintctl, "status_snapshot", return_value=expected_status
                ),
            ):
                status = sprintctl.monitor_once(
                    "stop-during-build", upload=False, include_remote=False
                )

            self.assertEqual(status, expected_status)
            request_stop.assert_called_once_with(
                "stop-during-build", reason="operator_batch_stop"
            )

    def test_monitor_never_trusts_agent_writable_cloud_budget_stop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "cloud-budget-stop",
                "agent_kind": "codex",
                "cpu_agent_gpu_worker": True,
                "budget_enforcement": {"in_sandbox_watchdog": True},
            }
            expected_status = {"run_id": "cloud-budget-stop"}
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(sprintctl, "fetch_remote_json") as fetch_remote,
                mock.patch.object(sprintctl, "request_stop") as request_stop,
                mock.patch("event_runtime.compute.worker.dispatch_once") as dispatch,
                mock.patch("event_runtime.telemetry.host.poll_once"),
                mock.patch.object(
                    sprintctl, "discover_job_and_trial", return_value=(None, None)
                ),
                mock.patch.object(
                    sprintctl, "status_snapshot", return_value=expected_status
                ),
            ):
                status = sprintctl.monitor_once(
                    "cloud-budget-stop", upload=False, include_remote=False
                )

            self.assertEqual(status, expected_status)
            self.assertFalse((state_dir / "STOP_REQUESTED.json").exists())
            request_stop.assert_not_called()
            fetch_remote.assert_not_called()
            dispatch.assert_called_once_with("cloud-budget-stop")

    def test_agent_cost_budget_stops_at_complete_ten_dollars(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {"agent_cost_budget_usd": 10.0}
            with mock.patch.object(sprintctl, "request_stop") as request_stop:
                stopped = sprintctl.enforce_agent_cost_budget(
                    "budget-run",
                    state_dir,
                    run,
                    {"status": "complete", "total_usd": 10.0},
                )
            self.assertTrue(stopped)
            request_stop.assert_called_once_with(
                "budget-run", reason="agent_cost_budget_exhausted"
            )

    def test_agent_cost_budget_does_not_stop_below_full_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {"agent_cost_budget_usd": 10.0}
            with mock.patch.object(sprintctl, "request_stop") as request_stop:
                stopped = sprintctl.enforce_agent_cost_budget(
                    "budget-run",
                    state_dir,
                    run,
                    {
                        "status": "within_budget",
                        "total_usd": 9.9,
                        "stop_threshold_usd": 10.0,
                    },
                )
            self.assertFalse(stopped)
            request_stop.assert_not_called()

    def test_agent_cost_budget_waits_for_complete_snapshot_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {"agent_cost_budget_usd": 10.0}
            with mock.patch.object(sprintctl, "request_stop") as request_stop:
                incomplete = sprintctl.enforce_agent_cost_budget(
                    "budget-run",
                    state_dir,
                    run,
                    {"status": "incomplete_api_usage", "total_usd": None},
                )
                below = sprintctl.enforce_agent_cost_budget(
                    "budget-run",
                    state_dir,
                    run,
                    {
                        "status": "within_budget",
                        "total_usd": 9.899,
                        "stop_threshold_usd": 9.9,
                    },
                )
            self.assertFalse(incomplete)
            self.assertFalse(below)
            request_stop.assert_not_called()

    def test_agent_cost_budget_rejects_invalid_stop_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "invalid stop_threshold_usd"):
                sprintctl.enforce_agent_cost_budget(
                    "budget-run",
                    Path(raw),
                    {"agent_cost_budget_usd": 10.0},
                    {
                        "status": "within_budget",
                        "total_usd": 1.0,
                        "stop_threshold_usd": 10.1,
                    },
                )

    def test_task_instruction_renders_from_global_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "instruction.md").write_text(
                "Work within ${{ agent_cost_budget_usd }}.\n"
            )
            (source / "task.toml").write_text("schema_version = '1.3'\n")
            destination = root / "rendered"
            rendered = render_task.render_task(source, destination, "12.5")
            self.assertEqual(
                (rendered / "instruction.md").read_text(),
                "Work within $12.5.\n",
            )
            self.assertTrue((rendered / "task.toml").is_file())
            with self.assertRaisesRegex(ValueError, "does not match"):
                render_task.render_task(source, destination, "10")

    def test_monitor_dispatches_gpu_recovery_before_slow_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "dispatch-first",
                "agent_kind": "codex",
                "cpu_agent_gpu_worker": True,
            }
            order: list[str] = []
            expected_status = {"run_id": "dispatch-first"}
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch(
                    "event_runtime.compute.worker.dispatch_once",
                    side_effect=lambda _run_id: order.append("dispatch"),
                ),
                mock.patch(
                    "event_runtime.telemetry.host.poll_once",
                    side_effect=lambda _run_id: order.append("telemetry"),
                ),
                mock.patch.object(
                    sprintctl, "discover_job_and_trial", return_value=(None, None)
                ),
                mock.patch.object(
                    sprintctl, "status_snapshot", return_value=expected_status
                ),
            ):
                status = sprintctl.monitor_once(
                    "dispatch-first", upload=False, include_remote=False
                )

            self.assertEqual(status, expected_status)
            self.assertEqual(order, ["dispatch", "telemetry"])

    def test_monitor_leaves_dispatch_to_healthy_dedicated_loop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "dedicated-dispatch",
                "agent_kind": "codex",
                "cpu_agent_gpu_worker": True,
            }
            expected_status = {"run_id": "dedicated-dispatch"}
            with (
                mock.patch.object(sprintctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(
                    sprintctl, "gpu_dispatch_loop_alive", return_value=True
                ),
                mock.patch("event_runtime.compute.worker.dispatch_once") as dispatch,
                mock.patch("event_runtime.telemetry.host.poll_once"),
                mock.patch.object(
                    sprintctl, "discover_job_and_trial", return_value=(None, None)
                ),
                mock.patch.object(
                    sprintctl, "status_snapshot", return_value=expected_status
                ),
            ):
                status = sprintctl.monitor_once(
                    "dedicated-dispatch", upload=False, include_remote=False
                )

            self.assertEqual(status, expected_status)
            dispatch.assert_not_called()

    def test_status_reports_explicit_agent_kind(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "test-status",
                "agent_kind": "codex",
                "state_dir": str(state_dir),
                "jobs_root": str(state_dir / "jobs"),
                "app_name": "sprint-test-status",
                "volume_name": "sprint-test-status",
            }
            payload = sprintctl.status_snapshot(state_dir, run, include_remote=False)
            self.assertEqual(payload["agent_kind"], "codex")
            self.assertTrue(payload["snapshot_agent_kind_matches"])

    def test_noop_vercel_update_never_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            web = Path(raw)
            (web / "index.html").write_text("same")
            digest = frontier_update.site_tree_hash(web)
            state = {"last_deployed_site_hash": digest}

            def fail_runner(command, cwd):
                raise AssertionError(f"runner called: {command} in {cwd}")

            deployed, message = frontier_update.deploy_if_needed(
                state, web=web, runner=fail_runner
            )
            self.assertFalse(deployed)
            self.assertEqual(message, "no site file changes")
            self.assertEqual(state["site_status"], "noop")

    def test_site_hash_retries_live_atomic_replacement_race(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            web = Path(raw)
            (web / "index.html").write_text("stable")
            real_sha256_file = frontier_update.sha256_file
            calls = 0

            def transient_missing(path: Path) -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise FileNotFoundError(path)
                return real_sha256_file(path)

            with (
                mock.patch.object(
                    frontier_update, "sha256_file", side_effect=transient_missing
                ),
                mock.patch.object(frontier_update.time, "sleep"),
            ):
                digest = frontier_update.site_tree_hash(web)

            self.assertEqual(digest, frontier_update.site_tree_hash(web))
            self.assertGreaterEqual(calls, 2)

    def test_site_hash_ignores_atomic_writer_staging_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            web = Path(raw)
            (web / "index.html").write_text("stable")
            baseline = frontier_update.site_tree_hash(web)
            (web / ".current.json.123.tmp").write_text("partial")

            self.assertEqual(frontier_update.site_tree_hash(web), baseline)

    def test_public_artifact_hashes_retries_live_replacement_race(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            web = Path(raw)
            artifact = web / "data/policies/run.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"policies": []}\n')
            real_sha256_file = frontier_update.sha256_file
            calls = 0

            def transient_missing(path: Path) -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise FileNotFoundError(path)
                return real_sha256_file(path)

            with (
                mock.patch.object(
                    frontier_update, "sha256_file", side_effect=transient_missing
                ),
                mock.patch.object(frontier_update.time, "sleep"),
            ):
                hashes = frontier_update.public_artifact_hashes(web)

            self.assertEqual(
                hashes,
                {"data/policies/run.json": real_sha256_file(artifact)},
            )
            self.assertGreaterEqual(calls, 2)

    def test_vercel_debounce_does_not_starve_on_continuous_site_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            web = Path(raw)
            page = web / "index.html"
            page.write_text("first")
            (web / ".vercel").mkdir()
            (web / ".vercel/project.json").write_text(
                json.dumps(
                    {
                        "projectId": frontier_update.PROJECT_ID,
                        "orgId": frontier_update.ORG_ID,
                        "projectName": "sprint",
                    }
                )
            )
            state = {"last_deployed_site_hash": "old"}

            deployed, _ = frontier_update.deploy_if_needed(
                state,
                web=web,
                now=1000,
                debounce_seconds=60,
                runner=lambda *_args: "unused",
            )
            self.assertFalse(deployed)
            first_seen = state["site_change_first_seen_at"]
            first_hash = state["pending_site_hash"]

            page.write_text("second")
            deployed, _ = frontier_update.deploy_if_needed(
                state,
                web=web,
                now=1030,
                debounce_seconds=60,
                runner=lambda *_args: "unused",
            )
            self.assertFalse(deployed)
            self.assertEqual(state["site_change_first_seen_at"], first_seen)
            self.assertNotEqual(state["pending_site_hash"], first_hash)

            commands = []

            def runner(command, cwd):
                commands.append((command, cwd))
                if command[1] == "deploy":
                    return "https://sprint-live-alienkevins-projects.vercel.app"
                return "Success"

            deployed, _ = frontier_update.deploy_if_needed(
                state,
                web=web,
                now=1060,
                debounce_seconds=60,
                runner=runner,
            )
            self.assertTrue(deployed)
            self.assertEqual(len(commands), 2)

    def test_vercel_deploy_reassigns_exact_public_alias(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            web = Path(raw)
            (web / "index.html").write_text("new")
            (web / ".vercel").mkdir()
            (web / ".vercel/project.json").write_text(
                json.dumps(
                    {
                        "projectId": frontier_update.PROJECT_ID,
                        "orgId": frontier_update.ORG_ID,
                        "projectName": "sprint",
                    }
                )
            )
            state = {
                "last_deployed_site_hash": "old",
                "pending_site_hash": frontier_update.site_tree_hash(web),
                "site_change_first_seen_at": "2026-08-08T00:00:00Z",
            }
            commands = []

            def runner(command, cwd):
                commands.append((command, cwd))
                if command[1] == "deploy":
                    return (
                        "Production: https://sprint-new-alienkevins-projects.vercel.app"
                    )
                return "Success"

            deployed, _ = frontier_update.deploy_if_needed(
                state,
                web=web,
                now=1786224000,
                debounce_seconds=0,
                runner=runner,
            )
            self.assertTrue(deployed)
            self.assertEqual(len(commands), 2)
            self.assertEqual(
                commands[1][0],
                [
                    "vercel",
                    "alias",
                    "set",
                    "https://sprint-new-alienkevins-projects.vercel.app",
                    "g1-sprint.vercel.app",
                    "--scope",
                    frontier_update.VERCEL_SCOPE,
                ],
            )
            self.assertEqual(state["production_alias"], "https://g1-sprint.vercel.app")


if __name__ == "__main__":
    unittest.main()
