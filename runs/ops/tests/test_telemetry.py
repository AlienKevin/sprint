from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path("/data/qwop-bench")
TELEMETRY_PY = ROOT / "challenge/g1-sprint-100m-lane/environment/qwop-telemetry.py"
TELEMETRY_SH = ROOT / "challenge/g1-sprint-100m-lane/environment/qwop-telemetry.sh"
KEEPALIVE_PY = ROOT / "runs/ops/telemetry_keepalive.py"
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))

import telemetry_keepalive  # noqa: E402
import telemetry_host  # noqa: E402


class TelemetrySamplerTests(unittest.TestCase):
    def test_once_writes_snapshot_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TELEMETRY_PY),
                    "--once",
                    "--role",
                    "host",
                    "--run-id",
                    "unit-telem",
                    "--out-dir",
                    str(out),
                    "--force",
                    "--pidfile",
                    str(out / "telem.pid"),
                    "--interval-seconds",
                    "5",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            snapshot = json.loads((out / "snapshot.json").read_text())
            self.assertEqual(snapshot["schema_version"], 1)
            self.assertIn("nvidia_smi", snapshot)
            latest = json.loads((out / "latest.json").read_text())
            self.assertEqual(latest["role"], "host")
            self.assertEqual(latest["run_id"], "unit-telem")
            self.assertIn("cpu_util_pct", latest)
            self.assertIn("mem_total_kib", latest)
            self.assertTrue((out / "samples.jsonl").exists())
            self.assertTrue((out / "samples.csv").exists())
            with (out / "samples.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("util_gpu_pct", rows[0])
            self.assertIn("load1", rows[0])

    def test_shell_wrapper_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            completed = subprocess.run(
                [
                    "bash",
                    str(TELEMETRY_SH),
                    "--once",
                    "--role",
                    "agent",
                    "--run-id",
                    "shell-telem",
                    "--out-dir",
                    str(out),
                    "--pidfile",
                    str(out / "telem.pid"),
                    "--force",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((out / "latest.json").exists())

    def test_no_secret_env_leak_in_outputs(self) -> None:
        secret = "sk-test-secret-value-must-never-appear-123456789"
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = secret
            env["CLAUDE_CODE_OAUTH_TOKEN"] = secret
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TELEMETRY_PY),
                    "--once",
                    "--out-dir",
                    str(out),
                    "--force",
                    "--pidfile",
                    str(out / "telem.pid"),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            blob = ""
            for path in out.rglob("*"):
                if path.is_file():
                    blob += path.read_text(encoding="utf-8", errors="replace")
            blob += completed.stdout + completed.stderr
            self.assertNotIn(secret, blob)

    def test_keepalive_json_starts_telemetry(self) -> None:
        argv = telemetry_keepalive.keepalive_argv(run_id="lane-smoke")
        self.assertEqual(argv[0], "sh")
        self.assertEqual(argv[1], "-c")
        self.assertIn("/opt/qwop-telemetry.sh", argv[2])
        self.assertIn("/logs/artifacts/telemetry", argv[2])
        self.assertIn("lane-smoke", argv[2])
        completed = subprocess.run(
            [sys.executable, str(KEEPALIVE_PY), "--run-id", "x"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        parsed = json.loads(completed.stdout)
        self.assertIsInstance(parsed, list)

    def test_durable_dry_run_keepalive_includes_telemetry(self) -> None:
        run_id = "dry-telem-check"
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
                "openai/test-model",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        config = json.loads(completed.stdout)
        keepalive = config["keepalive"]
        self.assertEqual(keepalive[0], "sh")
        command = keepalive[2]
        self.assertIn("/opt/qwop-telemetry.sh", command)
        self.assertIn("/logs/artifacts/telemetry", command)
        self.assertIn("/opt/qwop-snapshot-loop.sh", command)
        self.assertNotIn(env["OPENAI_API_KEY"], completed.stdout)

    def test_host_redaction(self) -> None:
        text = "Authorization: Bearer sk-ant-oat-abcdefghij OPENAI_API_KEY=sk-xyzABC12345"
        redacted = telemetry_host._redact(text)
        self.assertNotIn("sk-ant-oat-abcdefghij", redacted)
        self.assertNotIn("sk-xyzABC12345", redacted)
        self.assertIn("[REDACTED]", redacted)


if __name__ == "__main__":
    unittest.main()
