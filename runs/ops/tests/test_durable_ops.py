from __future__ import annotations

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

OPS = Path("/data/qwop-bench/runs/ops")
sys.path.insert(0, str(OPS))

import frontier_update  # noqa: E402
import qwopctl  # noqa: E402


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
    def test_redacted_dry_run_does_not_create_state(self) -> None:
        run_id = f"dry-{uuid.uuid4().hex[:12]}"
        token = "fake-oauth-value-that-must-never-print-123456789"
        state = Path("/data/qwop-bench/runs/ops") / run_id
        env = os.environ.copy()
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        completed = subprocess.run(
            [
                "bash",
                "/data/qwop-bench/runs/run-lane-durable.sh",
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
        state = Path("/data/qwop-bench/runs/ops") / run_id
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = key
        completed = subprocess.run(
            [
                "bash",
                "/data/qwop-bench/runs/run-lane-durable.sh",
                "--dry-run",
                "--run-id",
                run_id,
                "--agent-kind",
                "codex",
                "--model",
                "openai/test-codex-model",
                "--endpoint",
                "https://example.invalid/v1",
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
        self.assertFalse(config["automatic_stop"])
        self.assertNotIn("stop_after_seconds", config)
        self.assertFalse(state.exists())

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
            dummy = root / "qwop-dummy-claude.sh"
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
                    "/data/qwop-bench/challenge/g1-sprint-100m-lane/environment/qwop-snapshot-loop.sh",
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
                first_seen = durable / "runs/test-watch/snapshot/first-claude-seen"
                deadline = time.time() + 15
                while not first_seen.exists() and time.time() < deadline:
                    time.sleep(0.1)
                self.assertTrue(first_seen.exists())
                (runtime / "qwop-stop").touch()
                watcher.wait(timeout=20)
                dummy_process.wait(timeout=10)
                ack = json.loads(
                    (durable / "runs/test-watch/STOP_ACK").read_text()
                )
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
    raise SystemExit(143)
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
                    "QWOP_RUNTIME_DIR": str(runtime),
                    "QWOP_AGENT_LOG_DIR": str(agent_logs),
                }
            )
            wrapper = subprocess.Popen(
                [
                    "bash",
                    "/data/qwop-bench/challenge/g1-sprint-100m-lane/environment/qwop-codex-exec-wrapper.sh",
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
                    "/data/qwop-bench/challenge/g1-sprint-100m-lane/environment/qwop-snapshot-loop.sh",
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
                first_seen = durable / "runs/test-codex/snapshot/first-codex-seen"
                deadline = time.time() + 15
                while not first_seen.exists() and time.time() < deadline:
                    time.sleep(0.1)
                self.assertTrue(first_seen.exists())
                (runtime / "qwop-stop").touch()

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

                ack = json.loads(
                    (durable / "runs/test-codex/STOP_ACK").read_text()
                )
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
                process_file = runtime / "qwop-agent/codex-process"
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
            archive, digest, checksum = qwopctl.tar_attempt_atomic(
                attempt, root / "archives"
            )
            again, again_digest, _ = qwopctl.tar_attempt_atomic(
                attempt, root / "archives"
            )
            self.assertEqual(archive, again)
            self.assertEqual(digest, again_digest)
            self.assertEqual(qwopctl.sha256_file(archive), digest)
            self.assertEqual(checksum.read_text().split()[0], digest)
            self.assertFalse(archive.stat().st_mode & stat.S_IWUSR)

    def test_malformed_partial_ledger_keeps_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = Path(raw) / "ledger.jsonl"
            ledger.write_text(json.dumps(row(1, "one.pt", 10.0)) + "\n{\"index\":")
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
                    [json.dumps(row(1, "one.pt", None)), json.dumps(row(2, "two.pt", 9.0))]
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
                    [json.dumps(row(1, "one.pt", 8.0)), json.dumps(row(2, "two.pt", 9.0))]
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

    def test_frontier_tolerance_and_completed_robustness_gate(self) -> None:
        candidate = frontier_update.Candidate
        speed_only = [
            candidate(1, "one", 10.0, 0.5, False, "a", "/a"),
            candidate(2, "two", 9.9995, 0.9, True, "b", "/b"),
        ]
        frontier, uses_robustness = frontier_update.compute_frontier(speed_only)
        self.assertFalse(uses_robustness)
        self.assertEqual([item.index for item in frontier], [1])

        complete = [
            candidate(1, "one", 9.0, 0.5, True, "a", "/a"),
            candidate(2, "two", 10.0, 0.9, True, "b", "/b"),
        ]
        frontier, uses_robustness = frontier_update.compute_frontier(complete)
        self.assertTrue(uses_robustness)
        self.assertEqual({item.index for item in frontier}, {1, 2})

    def test_stop_and_finalize_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "test-idempotent",
                "agent_kind": "codex",
                "agent_container_id": "ta-test",
            }
            with (
                mock.patch.object(qwopctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(qwopctl, "fetch_remote_json", return_value=None),
                mock.patch.object(
                    qwopctl, "discover_agent_container", return_value="ta-test"
                ),
                mock.patch.object(qwopctl, "exec_container") as remote_exec,
            ):
                first = qwopctl.request_stop("test-idempotent")
                marker = (state_dir / "STOP_REQUESTED.json").read_bytes()
                second = qwopctl.request_stop("test-idempotent")
                self.assertEqual(first["agent_kind"], "codex")
                self.assertEqual(first["requested_at"], second["requested_at"])
                self.assertEqual(marker, (state_dir / "STOP_REQUESTED.json").read_bytes())
                self.assertEqual(remote_exec.call_count, 2)

            (state_dir / "archive-manifest.json").write_text(
                '{"schema_version":1,"attempts":{}}\n'
            )
            with (
                mock.patch.object(qwopctl, "load_run", return_value=(state_dir, run)),
                mock.patch.object(qwopctl, "monitor_once") as monitor,
                mock.patch.object(
                    qwopctl,
                    "final_conditions",
                    return_value=(True, {"all": True}, []),
                ),
                mock.patch.object(qwopctl, "volume_upload"),
            ):
                complete_one, payload_one = qwopctl.finalize("test-idempotent")
                final_bytes = (state_dir / "FINALIZED.json").read_bytes()
                complete_two, payload_two = qwopctl.finalize("test-idempotent")
                self.assertTrue(complete_one and complete_two)
                self.assertEqual(payload_one, payload_two)
                self.assertEqual(final_bytes, (state_dir / "FINALIZED.json").read_bytes())
                self.assertEqual(monitor.call_count, 1)

    def test_status_reports_explicit_agent_kind(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            run = {
                "run_id": "test-status",
                "agent_kind": "codex",
                "state_dir": str(state_dir),
                "jobs_root": str(state_dir / "jobs"),
                "app_name": "qwop-test-status",
                "volume_name": "qwop-test-status",
            }
            payload = qwopctl.status_snapshot(
                state_dir, run, include_remote=False
            )
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


if __name__ == "__main__":
    unittest.main()
