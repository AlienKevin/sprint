from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
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
    def test_monitor_loop_self_registers_and_cleans_own_pid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            observed: list[str] = []

            @contextlib.contextmanager
            def owned_lock(_path: Path, *, blocking: bool = True):
                self.assertFalse(blocking)
                yield True

            def observe_monitor(_run_id: str) -> dict:
                observed.append((state / "monitor.pid").read_text().strip())
                return {"run_id": "monitor-run", "agent_kind": "codex"}

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
                mock.patch.object(
                    sprintctl, "monitor_once", side_effect=observe_monitor
                ),
                mock.patch.object(sprintctl, "finalize", return_value=(True, {})),
            ):
                self.assertEqual(sprintctl.monitor_loop("monitor-run", 10), 0)

            self.assertEqual(observed, [str(os.getpid())])
            self.assertFalse((state / "monitor.pid").exists())

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
                side_effect=lambda _run, path: remote.get(path),
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
                side_effect=lambda _run, path: remote.get(path),
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
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/run-lane-durable.sh"),
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
        self.assertFalse(config["automatic_stop"])
        self.assertNotIn("stop_after_seconds", config)
        self.assertFalse(state.exists())

    def test_codex_dry_run_is_redacted_and_manual_only(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        key = "fake-openai-key-that-must-never-print-123456789"
        state = OPS / run_id
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/run-lane-durable.sh"),
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
        self.assertFalse(config["automatic_stop"])
        self.assertNotIn("stop_after_seconds", config)
        self.assertFalse(state.exists())

    def test_terra_dry_run_pins_reconstructible_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-openai-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/run-lane-durable.sh"),
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
        env["OPENAI_API_KEY"] = "fake-openai-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/run-lane-durable.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "openai/gpt-5.6-luna",
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
        self.assertTrue(config["usage_audit_required"])

    def test_sol_dry_run_pins_reconstructible_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-openai-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/run-lane-durable.sh"),
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

    def test_deepseek_dry_run_requires_reconstructible_cost_policy(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "fake-deepseek-key-that-must-never-print-123456789"
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/run-lane-durable.sh"),
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "deepseek/deepseek-v4-flash",
                "--endpoint",
                "https://api.deepseek.com",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        self.assertEqual(config["agent_allowed_host"], "api.deepseek.com")
        self.assertTrue(config["usage_audit_required"])

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
                    str(
                        ROOT
                        / "events/g1-100-metres/environment/sprint-snapshot-loop.sh"
                    ),
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
    # Codex 0.147.0 may surface an interrupted unified_exec as exit 1. The
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
                    str(
                        ROOT
                        / "events/g1-100-metres/environment/sprint-codex-exec-wrapper.sh"
                    ),
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
                    str(
                        ROOT
                        / "events/g1-100-metres/environment/sprint-snapshot-loop.sh"
                    ),
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
