from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PY = ROOT / "challenge/g1-sprint-100m-lane/environment/sprint-telemetry.py"
TELEMETRY_SH = ROOT / "challenge/g1-sprint-100m-lane/environment/sprint-telemetry.sh"
KEEPALIVE_PY = ROOT / "runs/ops/telemetry_keepalive.py"
VERIFIER_TELEMETRY_PY = (
    ROOT / "challenge/g1-sprint-100m-lane/tests/verifier_telemetry.py"
)
OPS = ROOT / "runs/ops"
sys.path.insert(0, str(OPS))

import telemetry_keepalive  # noqa: E402
import telemetry_host  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
                    "host-controller",
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
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertIn("nvidia_smi", snapshot)
            latest = json.loads((out / "latest.json").read_text())
            self.assertEqual(latest["role"], "host-controller")
            self.assertEqual(latest["run_id"], "unit-telem")
            self.assertIn("cpu_util_pct", latest)
            self.assertIn("mem_total_kib", latest)
            self.assertIn(
                latest["resource_accounting_scope"],
                {"cgroup-v1", "cgroup-v2", "host-proc-fallback"},
            )
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
                    "cpu-agent",
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

    def test_cgroup_v2_metrics_are_container_scoped(self) -> None:
        sampler = load_module("sprint_telemetry", TELEMETRY_PY)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "cpu.stat").write_text(
                "usage_usec 1200000\nuser_usec 900000\nsystem_usec 300000\n"
                "nr_throttled 2\nthrottled_usec 4000\n"
            )
            (root / "cpu.max").write_text("400000 100000\n")
            (root / "memory.current").write_text(str(3 * 1024**3))
            (root / "memory.max").write_text(str(16 * 1024**3))
            (root / "memory.peak").write_text(str(5 * 1024**3))
            (root / "memory.events").write_text("oom 1\noom_kill 0\n")
            (root / "memory.swap.current").write_text(str(256 * 1024**2))
            (root / "memory.swap.max").write_text(str(1024**3))
            old_cpu = os.environ.get("SPRINT_REQUESTED_CPU_CORES")
            old_memory = os.environ.get("SPRINT_REQUESTED_MEMORY_MIB")
            os.environ["SPRINT_REQUESTED_CPU_CORES"] = "4"
            os.environ["SPRINT_REQUESTED_MEMORY_MIB"] = "16384"
            try:
                metrics, snapshot = sampler.read_cgroup_v2(
                    root, captured_ns=2_000_000_000
                )
            finally:
                if old_cpu is None:
                    os.environ.pop("SPRINT_REQUESTED_CPU_CORES", None)
                else:
                    os.environ["SPRINT_REQUESTED_CPU_CORES"] = old_cpu
                if old_memory is None:
                    os.environ.pop("SPRINT_REQUESTED_MEMORY_MIB", None)
                else:
                    os.environ["SPRINT_REQUESTED_MEMORY_MIB"] = old_memory
            self.assertEqual(metrics["resource_accounting_scope"], "cgroup-v2")
            self.assertEqual(metrics["cpu_requested_cores"], 4.0)
            self.assertEqual(metrics["cpu_limit_cores"], 4.0)
            self.assertEqual(metrics["mem_used_kib"], 3 * 1024**2)
            self.assertEqual(metrics["mem_total_kib"], 16 * 1024**2)
            self.assertEqual(metrics["mem_requested_kib"], 16 * 1024**2)
            self.assertEqual(metrics["mem_limit_kib"], 16 * 1024**2)
            self.assertEqual(metrics["memory_peak_kib"], 5 * 1024**2)
            self.assertEqual(metrics["memory_oom_events"], 1)
            used, percent = sampler.cgroup_cpu_delta(
                {"captured_ns": 1_000_000_000, "usage_usec": 1_000_000},
                snapshot,
            )
            self.assertEqual(used, 0.2)
            self.assertEqual(percent, 5.0)

    def test_cgroup_v1_metrics_are_container_scoped(self) -> None:
        sampler = load_module("sprint_telemetry_v1", TELEMETRY_PY)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for controller in ("cpu", "cpuacct", "memory"):
                (root / controller).mkdir()
            (root / "cpuacct/cpuacct.usage").write_text("1200000000\n")
            (root / "cpuacct/cpuacct.stat").write_text("user 90\nsystem 30\n")
            (root / "cpu/cpu.cfs_quota_us").write_text("400000\n")
            (root / "cpu/cpu.cfs_period_us").write_text("100000\n")
            (root / "cpu/cpu.stat").write_text(
                "nr_throttled 2\nthrottled_time 4000000\n"
            )
            (root / "memory/memory.usage_in_bytes").write_text(str(3 * 1024**3))
            (root / "memory/memory.limit_in_bytes").write_text(str(16 * 1024**3))
            (root / "memory/memory.max_usage_in_bytes").write_text(str(5 * 1024**3))
            (root / "memory/memory.failcnt").write_text("1\n")
            old_cpu = os.environ.get("SPRINT_REQUESTED_CPU_CORES")
            old_memory = os.environ.get("SPRINT_REQUESTED_MEMORY_MIB")
            os.environ["SPRINT_REQUESTED_CPU_CORES"] = "4"
            os.environ["SPRINT_REQUESTED_MEMORY_MIB"] = "16384"
            try:
                metrics, snapshot = sampler.read_cgroup_v1(
                    root, captured_ns=2_000_000_000
                )
            finally:
                if old_cpu is None:
                    os.environ.pop("SPRINT_REQUESTED_CPU_CORES", None)
                else:
                    os.environ["SPRINT_REQUESTED_CPU_CORES"] = old_cpu
                if old_memory is None:
                    os.environ.pop("SPRINT_REQUESTED_MEMORY_MIB", None)
                else:
                    os.environ["SPRINT_REQUESTED_MEMORY_MIB"] = old_memory
            self.assertEqual(metrics["resource_accounting_scope"], "cgroup-v1")
            self.assertEqual(metrics["cpu_requested_cores"], 4.0)
            self.assertEqual(metrics["cpu_limit_cores"], 4.0)
            self.assertEqual(metrics["mem_used_kib"], 3 * 1024**2)
            self.assertEqual(metrics["mem_total_kib"], 16 * 1024**2)
            self.assertEqual(metrics["mem_requested_kib"], 16 * 1024**2)
            self.assertEqual(metrics["mem_limit_kib"], 16 * 1024**2)
            self.assertEqual(metrics["memory_peak_kib"], 5 * 1024**2)
            self.assertEqual(metrics["memory_oom_events"], 1)
            self.assertEqual(metrics["cpu_throttled_usec"], 4000)
            used, percent = sampler.cgroup_cpu_delta(
                {"captured_ns": 1_000_000_000, "usage_usec": 1_000_000},
                snapshot,
            )
            self.assertEqual(used, 0.2)
            self.assertEqual(percent, 5.0)

    def test_cpu_attempt_is_preserved_in_jsonl_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            env = os.environ.copy()
            env["SPRINT_CPU_LAUNCH_ATTEMPT"] = "7"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TELEMETRY_PY),
                    "--once",
                    "--role",
                    "cpu-agent",
                    "--run-id",
                    "cpu-attempt-test",
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
            self.assertEqual(
                json.loads((out / "latest.json").read_text())["cpu_attempt"], 7
            )
            with (out / "samples.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["cpu_attempt"], "7")

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
        self.assertIn("/opt/sprint-telemetry.sh", argv[2])
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
        self.assertEqual(
            config["cpu_agent"],
            {
                "physical_cpu_cores": 4,
                "vcpus_equivalent": 8,
                "memory_mb": 16384,
                "gpus": 0,
            },
        )
        self.assertTrue(config["cgroup_telemetry_required"])
        keepalive = config["keepalive"]
        self.assertEqual(keepalive[0], "sh")
        command = keepalive[2]
        self.assertIn("/opt/sprint-telemetry.sh", command)
        self.assertIn("/logs/artifacts/telemetry", command)
        self.assertIn("/opt/sprint-snapshot-loop.sh", command)
        self.assertNotIn(env["OPENAI_API_KEY"], completed.stdout)

    def test_host_redaction(self) -> None:
        text = (
            "Authorization: Bearer sk-ant-oat-abcdefghij OPENAI_API_KEY=sk-xyzABC12345"
        )
        redacted = telemetry_host._redact(text)
        self.assertNotIn("sk-ant-oat-abcdefghij", redacted)
        self.assertNotIn("sk-xyzABC12345", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_host_exec_uses_sandbox_sdk_for_training_worker_ids(self) -> None:
        calls: list[tuple] = []

        class Stream:
            def __init__(self, value: str) -> None:
                self.value = value

            def read(self) -> str:
                return self.value

        class Process:
            stdout = Stream('{"ok":true}\n')
            stderr = Stream("")

            @staticmethod
            def wait() -> int:
                return 0

        class SandboxInstance:
            def exec(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Process()

        class Sandbox:
            @staticmethod
            def from_id(sandbox_id: str):
                self.assertEqual(sandbox_id, "sb-training")
                return SandboxInstance()

        fake_modal = types.SimpleNamespace(Sandbox=Sandbox)
        with mock.patch.dict(sys.modules, {"modal": fake_modal}):
            result = telemetry_host._exec_target(
                {"modal_profile": "test-profile"},
                "sb-training",
                "printf ok",
                timeout=17,
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '{"ok":true}\n')
        self.assertEqual(calls, [(("sh", "-c", "printf ok"), {"timeout": 17})])

    def test_host_fallback_collects_cpu_and_memory(self) -> None:
        completed = subprocess.run(
            ["bash", "-c", telemetry_host._shell_oneshot("verifier-gpu", "fallback")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        sample = telemetry_host._parse_json_payload(completed.stdout)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["role"], "verifier-gpu")
        self.assertIsInstance(sample.get("cpu_util_pct"), float)
        self.assertIn(
            sample.get("resource_accounting_scope"),
            {"cgroup-v1", "cgroup-v2", "host-proc-fallback"},
        )
        self.assertGreater(int(sample.get("mem_total_kib") or 0), 0)
        self.assertIn("gpus", sample)
        self.assertEqual(sample.get("gpu_count"), len(sample["gpus"]))

    def test_host_poll_normalizes_to_requested_resource_contract(self) -> None:
        run = {
            "resource_contract": {
                "cpu_agent": {"physical_cpu_cores": 4, "memory_mb": 16384},
                "training_worker": {
                    "physical_cpu_cores": 8,
                    "memory_mb": 32768,
                },
                "verifier": {"physical_cpu_cores": 8, "memory_mb": 32768},
            }
        }
        sample = {
            "resource_accounting_scope": "cgroup-v1",
            "cpu_limit_cores": 24.0,
            "cpu_usage_cores": 4.8,
            "cpu_util_pct": 20.0,
            "mem_limit_kib": 1_055_904_376,
            "mem_total_kib": 1_055_904_376,
            "mem_used_kib": 9_000_000,
        }

        telemetry_host.normalize_resource_contract(run, "training-gpu", sample)

        self.assertEqual(sample["cpu_requested_cores"], 8.0)
        self.assertEqual(sample["cpu_limit_cores"], 24.0)
        self.assertEqual(sample["cpu_util_pct"], 60.0)
        self.assertEqual(sample["mem_requested_kib"], 32768 * 1024)
        self.assertEqual(sample["mem_total_kib"], 32768 * 1024)
        self.assertEqual(sample["mem_available_kib"], 32768 * 1024 - 9_000_000)
        self.assertEqual(sample["mem_limit_kib"], 1_055_904_376)

    def test_sealed_verifier_sampler_stops_with_complete_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            fake_bin = out / "bin"
            fake_bin.mkdir()
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' '0, NVIDIA A10, 53, 12, 2749, 23028, 64.47, 150, 42, 1230, 5001'\n"
            )
            nvidia_smi.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(VERIFIER_TELEMETRY_PY),
                    "--out-dir",
                    str(out),
                    "--interval-seconds",
                    "0.2",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            deadline = time.monotonic() + 5
            while not (out / "latest.json").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            latest = json.loads((out / "latest.json").read_text())
            lifecycle = json.loads((out / "lifecycle.json").read_text())
            self.assertEqual(latest["role"], "verifier-gpu")
            self.assertEqual(latest["schema_version"], 2)
            self.assertIn(
                latest["resource_accounting_scope"],
                {"cgroup-v1", "cgroup-v2", "host-proc-fallback"},
            )
            self.assertIn("cpu_util_pct", latest)
            self.assertTrue(latest["nvidia_smi_ok"])
            self.assertEqual(latest["gpu_count"], 1)
            self.assertEqual(latest["gpus"][0]["gpu_name"], "NVIDIA A10")
            self.assertEqual(latest["gpus"][0]["util_gpu_pct"], 53.0)
            self.assertEqual(latest["gpus"][0]["mem_used_mib"], 2749.0)
            self.assertEqual(lifecycle["role"], "verifier-gpu")
            self.assertTrue(lifecycle["complete"])
            self.assertIsNotNone(lifecycle["finished_at"])
            self.assertGreaterEqual(lifecycle["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
