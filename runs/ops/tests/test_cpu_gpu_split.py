"""Focused unit tests for CPU-agent / GPU-worker split ops."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import hashlib
import io
import json
import subprocess
import sys
import tarfile
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

_train_spec = importlib.util.spec_from_loader(
    "sprint_gpu_train",
    importlib.machinery.SourceFileLoader(
        "sprint_gpu_train", str(ENV / "bin" / "sprint-gpu-train")
    ),
)
assert _train_spec and _train_spec.loader
train_cli = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(train_cli)


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

    def test_normalize_isaaclab_python_launcher_to_python3(self) -> None:
        job = {
            "command": [
                "/opt/IsaacLab/isaaclab.sh",
                "-p",
                "/app/train.py",
                "--headless",
            ]
        }
        out = gpu_worker.normalize_job_command(job)
        self.assertEqual(
            out["command"],
            ["python3", "/app/train.py", "--headless"],
        )

    def test_host_pins_and_restores_exact_claimed_workspace(self) -> None:
        archive_bytes = b"agent workspace bytes"
        digest = hashlib.sha256(archive_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "run-a",
                "state_dir": raw,
                "volume_name": "volume-a",
            }
            job = {
                "job_id": "job-a",
                "work_archive": "runs/run-a/gpu-jobs/work/job-a/app.tar.gz",
                "submitted_work_archive_sha256": digest,
            }

            def download(command, **_kwargs):
                Path(command[-1]).write_bytes(archive_bytes)
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(gpu_worker.sprintctl, "run_command", download),
                mock.patch.object(gpu_worker.sprintctl, "volume_upload") as upload,
            ):
                pinned = gpu_worker.pin_work_archive(run, job)
                gpu_worker.restore_pinned_work_archive(run, pinned)

            self.assertEqual(pinned["work_archive_sha256"], digest)
            self.assertEqual(pinned["work_archive_size_bytes"], len(archive_bytes))
            self.assertEqual(
                pinned["work_archive_provenance"], "host-pinned-at-lease-claim"
            )
            self.assertNotIn("work_archive_changed_before_claim", pinned)
            canonical = gpu_worker.host_work_archive_path(run, "job-a")
            self.assertIsNotNone(canonical)
            assert canonical is not None
            self.assertEqual(canonical.read_bytes(), archive_bytes)
            upload.assert_called_once_with(
                run, canonical, "runs/run-a/gpu-jobs/work/job-a/app.tar.gz"
            )

    def test_host_records_workspace_changed_before_claim(self) -> None:
        archive_bytes = b"changed after enqueue"
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "run-a",
                "state_dir": raw,
                "volume_name": "volume-a",
            }
            job = {
                "job_id": "job-a",
                "work_archive": "runs/run-a/gpu-jobs/work/job-a/app.tar.gz",
                "submitted_work_archive_sha256": hashlib.sha256(b"old").hexdigest(),
            }

            def download(command, **_kwargs):
                Path(command[-1]).write_bytes(archive_bytes)
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(gpu_worker.sprintctl, "run_command", download):
                pinned = gpu_worker.pin_work_archive(run, job)

            self.assertTrue(pinned["work_archive_changed_before_claim"])
            self.assertEqual(
                pinned["work_archive_sha256"],
                hashlib.sha256(archive_bytes).hexdigest(),
            )

    def test_retry_rejects_corrupt_host_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "run-a",
                "state_dir": raw,
                "volume_name": "volume-a",
            }
            job = {
                "job_id": "job-a",
                "work_archive": "runs/run-a/gpu-jobs/work/job-a/app.tar.gz",
                "work_archive_sha256": hashlib.sha256(b"expected").hexdigest(),
            }
            canonical = gpu_worker.host_work_archive_path(run, "job-a")
            assert canonical is not None
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                gpu_worker.restore_pinned_work_archive(run, job)

    def test_worker_extracts_only_regular_app_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "work.tar.gz"
            content = b"print('ok')\n"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("app/train.py")
                info.size = len(content)
                info.mode = 0o755
                tar.addfile(info, io.BytesIO(content))
            destination = root / "out"
            worker_run.safe_extract_work_archive(archive, destination)
            extracted = destination / "app" / "train.py"
            self.assertEqual(extracted.read_bytes(), content)
            self.assertTrue(extracted.stat().st_mode & 0o100)

    def test_worker_rejects_tar_traversal_and_links(self) -> None:
        for name, kind in (("../../opt/pwn", "file"), ("app/link", "link")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive = root / "work.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    info = tarfile.TarInfo(name)
                    if kind == "link":
                        info.type = tarfile.SYMTYPE
                        info.linkname = "/opt/sprint-gpu-worker-run.py"
                    else:
                        info.size = 1
                    tar.addfile(info, None if kind == "link" else io.BytesIO(b"x"))
                with self.assertRaisesRegex(RuntimeError, "unsafe|unsupported"):
                    worker_run.safe_extract_work_archive(archive, root / "out")

    def test_does_not_rewrite_other_isaaclab_actions(self) -> None:
        job = {"command": ["/opt/IsaacLab/isaaclab.sh", "-s"]}
        self.assertIs(gpu_worker.normalize_job_command(job), job)

    def test_worker_wraps_agent_isaac_python_with_local_asset_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "train.py"
            script.write_text("from isaaclab.app import AppLauncher\n")
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text("# trusted bootstrap\n")
            command = worker_run.build_attempt_command(
                {"command": ["python3", "-u", str(script)]},
                1,
                None,
                isaac_bootstrap=bootstrap,
            )

        self.assertEqual(
            command,
            ["python3", "-u", str(bootstrap), str(script)],
        )

    def test_worker_does_not_wrap_pure_torch_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "train.py"
            script.write_text("import torch\n")
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text("# trusted bootstrap\n")
            command = worker_run.build_attempt_command(
                {"command": ["python3", str(script)]},
                1,
                None,
                isaac_bootstrap=bootstrap,
            )

        self.assertEqual(command, ["python3", str(script)])

    def test_worker_wraps_workspace_entrypoint_with_delegated_isaac_imports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "train" / "main.py"
            script.parent.mkdir()
            script.write_text("from train.environment import build_env\n")
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text("# trusted bootstrap\n")
            command = worker_run.build_attempt_command(
                {
                    "command": ["python3", "-u", str(script)],
                    "workdir": str(root),
                },
                1,
                None,
                isaac_bootstrap=bootstrap,
            )

        self.assertEqual(
            command,
            ["python3", "-u", str(bootstrap), str(script)],
        )

    def test_archives_terminal_modal_streams_to_agent_visible_log(self) -> None:
        uploaded: dict[str, object] = {}

        def upload(_run: dict, source: Path, remote: str) -> None:
            uploaded["remote"] = remote
            uploaded["content"] = source.read_bytes()

        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 2,
            "status": "failed",
            "sandbox_id": "sb-1",
        }
        with mock.patch.object(
            gpu_worker.sprintctl, "volume_upload", side_effect=upload
        ):
            archived, detail = gpu_worker.archive_provider_logs(
                {"run_id": "run-1"},
                job,
                read_output=lambda _sandbox_id: ("stdout line\n", "stderr line\n"),
            )

        expected = b"== Modal stdout ==\nstdout line\n== Modal stderr ==\nstderr line\n"
        self.assertEqual(
            uploaded["remote"],
            "runs/run-1/gpu-jobs/out/job-1/attempt-2/worker.log",
        )
        self.assertEqual(uploaded["content"], expected)
        self.assertEqual(detail["provider_logs"], "archived")
        self.assertEqual(archived["provider_logs_size_bytes"], len(expected))
        self.assertEqual(
            archived["provider_logs_sha256"], hashlib.sha256(expected).hexdigest()
        )

    def test_provider_log_archive_failure_is_retryable(self) -> None:
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 1,
            "status": "failed",
            "sandbox_id": "sb-1",
        }

        def fail(_sandbox_id: str) -> tuple[str, str]:
            raise RuntimeError("not readable yet")

        archived, detail = gpu_worker.archive_provider_logs(
            {"run_id": "run-1"}, job, read_output=fail
        )
        self.assertNotIn("provider_logs_archived_at", archived)
        self.assertEqual(detail["provider_logs"], "error")
        self.assertIn("not readable yet", archived["provider_logs_archive_error"])
        self.assertEqual(archived["provider_logs_archive_attempts"], 1)
        self.assertGreater(
            archived["provider_logs_archive_retry_after_epoch_s"], time.time()
        )

    def test_provider_traceback_overrides_false_zero_exit(self) -> None:
        uploaded: dict[str, bytes] = {}
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 1,
            "status": "succeeded",
            "exit_code": 0,
            "sandbox_id": "sb-1",
        }
        stderr = (
            "recoverable warning\nTraceback (most recent call last):\n"
            "  File 'train.py', line 1\nFileNotFoundError: robot.usd\n"
        )

        def upload(_run: dict, source: Path, _remote: str) -> None:
            uploaded["content"] = source.read_bytes()

        with mock.patch.object(
            gpu_worker.sprintctl, "volume_upload", side_effect=upload
        ):
            archived, detail = gpu_worker.archive_provider_logs(
                {"run_id": "run-1"},
                job,
                read_output=lambda _sandbox_id: ("stdout\n", stderr),
            )

        self.assertIn(b"FileNotFoundError", uploaded["content"])
        self.assertEqual(archived["status"], "failed")
        self.assertEqual(archived["exit_code"], 1)
        self.assertEqual(archived["provider_reported_exit_code"], 0)
        self.assertEqual(
            archived["failure_reason"],
            "provider_stream_terminal_error",
        )
        self.assertEqual(
            detail["provider_terminal_error"], "FileNotFoundError: robot.usd"
        )

    def test_recoverable_isaac_gpu_warning_does_not_override_success(self) -> None:
        self.assertIsNone(
            gpu_worker.provider_terminal_error(
                "[Error] [gpu.foundation.plugin] No device could be created"
            )
        )

    def test_physx_software_fallback_overrides_false_zero_exit(self) -> None:
        output = "PhysX warning: GPU solver pipeline failed, switching to software"
        self.assertEqual(gpu_worker.provider_terminal_error(output), output)

    def test_kit_semantic_startup_failures_override_false_zero_exit(self) -> None:
        for output in (
            "Failed to resolve extension dependencies",
            "Failed to startup python app",
            "ModuleNotFoundError: No module named 'train'",
        ):
            self.assertEqual(gpu_worker.provider_terminal_error(output), output)

    def test_audits_preexisting_archived_log_and_repairs_status(self) -> None:
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 1,
            "status": "succeeded",
            "exit_code": 0,
            "provider_logs_archived_at": "earlier",
            "provider_logs_path": "runs/run-1/gpu-jobs/out/job-1/worker.log",
        }
        text = (
            "== Modal stderr ==\nTraceback (most recent call last):\n"
            "FileNotFoundError: robot.usd\n"
        )
        with (
            mock.patch.object(
                gpu_worker.sprintctl, "volume_get_text", return_value=text
            ),
            mock.patch.object(
                gpu_worker, "mirror_agent_job", return_value={"agent_mirror": "updated"}
            ),
        ):
            audited, detail = gpu_worker.audit_archived_provider_logs(
                {"run_id": "run-1"}, job
            )

        self.assertEqual(audited["status"], "failed")
        self.assertTrue(audited["provider_terminal_error_detected"])
        self.assertEqual(detail["provider_logs"], "audited")

    def test_pushes_fresh_status_and_log_to_cpu_agent_mirror(self) -> None:
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 2,
            "status": "failed",
        }

        def local_exec(
            _run: dict,
            container_id: str,
            shell_command: str,
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertEqual(container_id, "ta-agent")
            return subprocess.run(
                shell_command,
                shell=True,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        with tempfile.TemporaryDirectory() as tmp:
            mirror = Path(tmp) / "mirror"
            cli_path = Path(tmp) / "bin" / "sprint-gpu-train"
            cli_path.parent.mkdir()
            with (
                mock.patch.object(gpu_worker, "AGENT_GPU_MIRROR_ROOT", str(mirror)),
                mock.patch.object(gpu_worker, "AGENT_GPU_CLI_PATH", str(cli_path)),
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "exec_container",
                    side_effect=local_exec,
                ),
            ):
                detail = gpu_worker.mirror_agent_job(
                    {"agent_container_id": "ta-agent"},
                    job,
                    log_content=b"complete child output\n",
                    artifact_name="policy.pt",
                    artifact_content=b"policy bytes",
                )

            self.assertEqual(detail["agent_mirror"], "updated")
            self.assertEqual(
                (mirror / "artifacts" / "job-1" / "policy.pt").read_bytes(),
                b"policy bytes",
            )
            self.assertEqual(detail["agent_mirror_files"], 3)
            self.assertEqual(
                hashlib.sha256(cli_path.read_bytes()).hexdigest(),
                detail["agent_cli_sha256"],
            )
            self.assertEqual(
                json.loads((mirror / "status" / "job-1.json").read_text()),
                job,
            )
            self.assertEqual(
                (mirror / "out" / "job-1" / "attempt-2" / "worker.log").read_bytes(),
                b"complete child output\n",
            )

    def test_streams_large_agent_mirror_payload_over_sandbox_stdin(self) -> None:
        captured = bytearray()

        class Writer:
            def write(self, content: bytes) -> None:
                captured.extend(content)

            def drain(self) -> None:
                return None

            def write_eof(self) -> None:
                return None

        class Reader:
            def read(self) -> bytes:
                return b""

        class Process:
            stdin = Writer()
            stdout = Reader()
            stderr = Reader()

            def wait(self) -> int:
                return 0

        class Sandbox:
            def exec(self, *args: str, **kwargs: object) -> Process:
                self_args = args
                self_kwargs = kwargs
                self.assertEqual(self_args[0:2], ("python3", "-c"))
                self.assertEqual(self_args[-3], "-")
                self.assertFalse(self_kwargs["text"])
                return Process()

            def assertEqual(self, left: object, right: object) -> None:
                unittest.TestCase().assertEqual(left, right)

            def assertFalse(self, value: object) -> None:
                unittest.TestCase().assertFalse(value)

        identity = subprocess.CompletedProcess([], 0, "sb-agent", "")
        with (
            mock.patch.object(gpu_worker, "AGENT_GPU_MIRROR_ARG_BYTES", 1),
            mock.patch.object(
                gpu_worker.sprintctl, "exec_container", return_value=identity
            ),
            mock.patch.object(
                gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox()
            ) as from_id,
        ):
            detail = gpu_worker.mirror_agent_job(
                {"agent_container_id": "ta-agent"},
                {"job_id": "job-1", "attempt": 1, "status": "succeeded"},
                artifact_name="policy.pt",
                artifact_content=b"policy bytes",
            )

        from_id.assert_called_once_with("sb-agent")
        envelope = json.loads(gpu_worker.gzip.decompress(bytes(captured)))
        self.assertEqual(
            gpu_worker.base64.b64decode(envelope["files"]["artifacts/job-1/policy.pt"]),
            b"policy bytes",
        )
        self.assertEqual(detail["agent_mirror"], "updated")

    def test_fetches_only_reported_scoped_policy_for_agent_mirror(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {"policy_path": "/durable/runs/run-1/policies/policy_7.pt"},
        }

        def fake_get(command, **_kwargs):
            self.assertIn("runs/run-1/policies/policy_7.pt", command)
            Path(command[-1]).write_bytes(b"trusted policy")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            gpu_worker.sprintctl, "run_command", side_effect=fake_get
        ):
            payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
                run, job
            )

        self.assertEqual(name, "policy_7.pt")
        self.assertEqual(content, b"trusted policy")
        self.assertEqual(detail["policy_mirror"], "fetched")
        self.assertEqual(
            payload["agent_policy_mirror_path"],
            "/run/sprint-gpu-mirror/artifacts/job-1/policy_7.pt",
        )

    def test_fetches_policy_from_its_job_checkpoint_directory(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {
                "policy_path": (
                    "/durable/runs/run-1/gpu-jobs/checkpoints/job-1/policy_2.pt"
                )
            },
        }

        def fake_get(command, **_kwargs):
            self.assertIn("runs/run-1/gpu-jobs/checkpoints/job-1/policy_2.pt", command)
            Path(command[-1]).write_bytes(b"trusted checkpoint policy")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            gpu_worker.sprintctl, "run_command", side_effect=fake_get
        ):
            payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
                run, job
            )

        self.assertEqual(name, "policy_2.pt")
        self.assertEqual(content, b"trusted checkpoint policy")
        self.assertEqual(detail["policy_mirror"], "fetched")
        self.assertEqual(
            payload["agent_policy_mirror_path"],
            "/run/sprint-gpu-mirror/artifacts/job-1/policy_2.pt",
        )

    def test_fetches_policy_from_custom_run_gpu_job_directory(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {
                "policy_path": "/durable/runs/run-1/gpu-jobs/sprint-long/policy.pt"
            },
        }

        def fake_get(command, **_kwargs):
            Path(command[-1]).write_bytes(b"custom job policy")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            gpu_worker.sprintctl, "run_command", side_effect=fake_get
        ):
            payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
                run, job
            )

        self.assertEqual(name, "policy.pt")
        self.assertEqual(content, b"custom job policy")
        self.assertEqual(detail["policy_mirror"], "fetched")
        self.assertIn("agent_policy_mirror_path", payload)

    def test_fetches_policy_from_run_candidate_directory(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {"policy_path": "/durable/runs/run-1/candidates/policy_599.pt"},
        }

        def fake_get(command, **_kwargs):
            self.assertIn("runs/run-1/candidates/policy_599.pt", command)
            Path(command[-1]).write_bytes(b"candidate policy")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            gpu_worker.sprintctl, "run_command", side_effect=fake_get
        ):
            payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
                run, job
            )

        self.assertEqual(name, "policy_599.pt")
        self.assertEqual(content, b"candidate policy")
        self.assertEqual(detail["policy_mirror"], "fetched")
        self.assertEqual(
            payload["agent_policy_mirror_path"],
            "/run/sprint-gpu-mirror/artifacts/job-1/policy_599.pt",
        )

    def test_rejects_policy_outside_run_gpu_job_and_policy_directories(self) -> None:
        payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
            {"run_id": "run-1", "volume_name": "volume-1"},
            {
                "job_id": "job-1",
                "progress": {
                    "policy_path": ("/durable/runs/run-1/private/checkpoints/policy.pt")
                },
            },
        )

        self.assertIsNone(name)
        self.assertIsNone(content)
        self.assertEqual(detail["policy_mirror"], "rejected_scope")
        self.assertNotIn("agent_policy_mirror_path", payload)

    def test_rejects_policy_outside_run_policy_directory(self) -> None:
        payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
            {"run_id": "run-1", "volume_name": "volume-1"},
            {
                "job_id": "job-1",
                "progress": {"policy_path": "/durable/runs/other/secrets.pt"},
            },
        )

        self.assertIsNone(name)
        self.assertIsNone(content)
        self.assertEqual(detail["policy_mirror"], "rejected_scope")
        self.assertNotIn("agent_policy_mirror_path", payload)

    def test_agent_cli_prefers_host_mirror_over_stale_volume_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "durable" / "gpu-jobs"
            mirror = Path(tmp) / "mirror"
            (root / "status").mkdir(parents=True)
            (mirror / "status").mkdir(parents=True)
            (root / "status" / "job-1.json").write_text(
                json.dumps({"job_id": "job-1", "status": "pending"})
            )
            (mirror / "status" / "job-1.json").write_text(
                json.dumps({"job_id": "job-1", "status": "succeeded"})
            )

            with mock.patch.object(train_cli, "AGENT_MIRROR_ROOT", mirror):
                status = train_cli.read_status(root, "job-1")
                latest = train_cli.latest_job_id(root)

            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(latest, "job-1")


class HostJobRegistryTests(unittest.TestCase):
    def test_persist_records_job_outside_agent_volume_first(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "run-1", "state_dir": raw}
            job = {"job_id": "job-1", "status": "running", "attempt": 1}
            writes: list[str] = []

            def put(_run: dict, remote: str, _payload: dict) -> None:
                writes.append(remote)

            with (
                mock.patch.object(gpu_worker, "put_json", side_effect=put),
                mock.patch.object(gpu_worker, "mirror_agent_job"),
            ):
                gpu_worker.persist_job(run, job)

            local = Path(raw) / "gpu-job-registry" / "job-1.json"
            self.assertEqual(json.loads(local.read_text()), job)
            self.assertEqual(
                writes,
                [
                    "runs/run-1/gpu-jobs/status/job-1.json",
                    "runs/run-1/gpu-jobs/queue/job-1.json",
                ],
            )

    def test_load_survives_agent_removing_volume_mirrors(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "run-1", "state_dir": raw}
            path = Path(raw) / "gpu-job-registry" / "job-1.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"job_id": "job-1", "status": "retry_wait"}))
            with mock.patch.object(
                gpu_worker.sprintctl,
                "volume_get_text",
                side_effect=AssertionError("host registry must win"),
            ):
                payload = gpu_worker.load_job(run, "job-1")

            self.assertEqual(payload["status"], "retry_wait")

    def test_list_retains_host_job_after_agent_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "run-1", "state_dir": raw}
            path = Path(raw) / "gpu-job-registry" / "job-1.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"job_id": "job-1", "status": "running"}))
            with mock.patch.object(gpu_worker, "volume_ls_json_names", return_value=[]):
                self.assertEqual(gpu_worker.list_job_ids(run), ["job-1"])


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
        self.assertEqual(written[0]["claim_restore"], {"status": "pending"})
        self.assertEqual(
            gpu_claim.process_identity_state(written[0]["claim_owner_process"]),
            "alive",
        )


class GpuConcurrencyLimitTests(unittest.TestCase):
    def test_orphan_audit_keeps_registered_and_terminates_unknown_sandbox(self) -> None:
        run = {"run_id": "unit", "training_app_name": "unit-training"}
        jobs = {
            "known": {
                "job_id": "known",
                "status": "running",
                "sandbox_id": "sb-known",
            }
        }
        known = mock.Mock(object_id="sb-known")
        orphan = mock.Mock(object_id="sb-orphan")
        app = mock.Mock(app_id="ap-unit")
        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=["known"]),
            mock.patch.object(
                gpu_worker, "load_job", side_effect=lambda _run, job_id: jobs[job_id]
            ),
            mock.patch("modal.App.lookup", return_value=app) as lookup,
            mock.patch("modal.Sandbox.list", return_value=iter([known, orphan])),
            mock.patch.object(
                gpu_worker.ModalSandboxProvider, "terminate", return_value=None
            ) as terminate,
        ):
            actions = gpu_worker.cleanup_orphaned_training_sandboxes(run)

        lookup.assert_called_once_with("unit-training", create_if_missing=False)
        self.assertEqual(
            actions,
            [
                {
                    "action": "orphan_terminated",
                    "sandbox_id": "sb-orphan",
                    "error": None,
                }
            ],
        )
        self.assertEqual(terminate.call_count, 1)
        self.assertEqual(terminate.call_args.args[0].attempt_id, "sb-orphan")

    def test_orphan_audit_is_skipped_without_separate_training_app(self) -> None:
        self.assertEqual(
            gpu_worker.cleanup_orphaned_training_sandboxes({"run_id": "unit"}), []
        )

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

    def test_agent_cancelled_host_job_is_fenced_before_dispatch(self) -> None:
        job = {
            "job_id": "cancelled",
            "run_id": "unit",
            "status": "claiming",
            "attempt": 2,
            "lease_id": "old-lease",
        }
        persisted: list[dict] = []
        with (
            mock.patch.object(
                gpu_worker, "list_agent_cancelled_job_ids", return_value=["cancelled"]
            ),
            mock.patch.object(gpu_worker, "load_host_job", return_value=job),
            mock.patch.object(
                gpu_worker,
                "persist_job",
                side_effect=lambda _run, payload: (
                    persisted.append(dict(payload)) or payload
                ),
            ),
        ):
            result = gpu_worker.reconcile_agent_cancelled_jobs({"run_id": "unit"})

        self.assertEqual(result[0]["decision"], "agent_cancelled_before_dispatch")
        self.assertEqual(persisted[0]["status"], "terminated")
        self.assertEqual(
            persisted[0]["termination_reason"], "agent_cancelled_before_dispatch"
        )
        self.assertEqual(persisted[0]["fenced_lease_id"], "old-lease")
        self.assertEqual(persisted[0]["fence_epoch"], 1)

    def test_agent_cancel_markers_are_discovered_from_delivery_directories(
        self,
    ) -> None:
        with mock.patch.object(
            gpu_worker,
            "volume_ls_json_names",
            side_effect=[
                [".cancelled-job-a.json", "live.json"],
                [".cancelled-job-a.json", ".cancelled-job-b.json"],
            ],
        ):
            self.assertEqual(
                gpu_worker.list_agent_cancelled_job_ids(
                    {"run_id": "unit", "volume_name": "unit-volume"}
                ),
                ["job-a", "job-b"],
            )

    def test_hidden_cancel_markers_are_not_enumerated_as_jobs(self) -> None:
        run = {"run_id": "unit", "state_dir": "/tmp/unit"}
        with (
            mock.patch.object(gpu_worker, "list_host_job_ids", return_value=[]),
            mock.patch.object(
                gpu_worker,
                "volume_ls_json_names",
                side_effect=[
                    ["live.json", ".cancelled-dead.json"],
                    ["live.json", ".cancelled-dead.json"],
                ],
            ),
        ):
            self.assertEqual(gpu_worker.list_job_ids(run), ["live"])

        with mock.patch.object(
            gpu_worker,
            "volume_ls_json_names",
            return_value=["live.json", ".cancelled-dead.json"],
        ):
            self.assertEqual(gpu_worker.list_pending_job_ids(run), ["live"])

    def test_agent_cancel_marker_does_not_terminate_allocated_worker(self) -> None:
        job = {
            "job_id": "running",
            "run_id": "unit",
            "status": "running",
            "sandbox_id": "sb-live",
        }
        with (
            mock.patch.object(
                gpu_worker, "list_agent_cancelled_job_ids", return_value=["running"]
            ),
            mock.patch.object(gpu_worker, "load_host_job", return_value=job),
            mock.patch.object(gpu_worker, "persist_job") as persist,
        ):
            self.assertEqual(
                gpu_worker.reconcile_agent_cancelled_jobs({"run_id": "unit"}), []
            )
        persist.assert_not_called()

    def test_retry_backoff_reserves_slot_ahead_of_new_job(self) -> None:
        jobs = {
            "recovering": {
                "job_id": "recovering",
                "status": "retry_wait",
                "attempt": 1,
                "retry_not_before_epoch_s": time.time() + 60,
            },
            "new": {
                "job_id": "new",
                "status": "pending",
                "attempt": 0,
                "created_at_epoch_s": 2,
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "load_run",
                    return_value=(Path(raw), run),
                ),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=list(jobs)),
                mock.patch.object(
                    gpu_worker,
                    "load_job",
                    side_effect=lambda _run, job_id: dict(jobs[job_id]),
                ),
                mock.patch.object(gpu_worker.ModalSandboxProvider, "start") as start,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(result["reason"], "retry_backoff_reserved")
        self.assertEqual(result["pending"], [])
        start.assert_not_called()


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

    def test_gpu_workers_force_headless_isaac_runtime(self) -> None:
        worker = (ROOT / "runs" / "ops" / "gpu_worker.py").read_text()
        self.assertGreaterEqual(worker.count('env={"HEADLESS": "1"}'), 2)

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
    def test_live_claim_owner_keeps_startup_grace(self) -> None:
        identity = gpu_claim.process_identity()
        self.assertIsNotNone(identity)
        job = gpu_claim.build_claim_payload(
            {"job_id": "logical", "status": "pending"},
            claim_id="lease",
            owner_process=identity,
        )
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=None),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
        ):
            out, detail = gpu_worker.reconcile_job(
                {"run_id": "unit"},
                job,
                now=float(job["claimed_at_epoch_s"]) + 1,
                probe_fn=lambda _job: ("unknown", None, None),
            )
        self.assertIs(out, job)
        self.assertEqual(detail["decision"], "grace")

    def test_dead_claim_owner_restores_pending_without_consuming_attempt(self) -> None:
        original = {"job_id": "logical", "status": "pending", "retry_count": 2}
        job = gpu_claim.build_claim_payload(
            original,
            claim_id="abandoned",
            owner_process={"pid": 999_999_999, "start_ticks": 1, "boot_id": "old"},
        )
        with (
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
            mock.patch.object(gpu_worker, "_timeline_event") as timeline_event,
        ):
            out, detail = gpu_worker.reconcile_job(
                {"run_id": "unit"},
                job,
                now=200,
                probe_fn=mock.Mock(side_effect=AssertionError("must not probe")),
            )
        self.assertEqual(detail["decision"], "abandoned_claim_released")
        self.assertEqual(out["status"], "pending")
        self.assertEqual(out["retry_count"], 2)
        self.assertNotIn("attempt", out)
        self.assertNotIn("lease_id", out)
        self.assertEqual(out["fenced_lease_id"], "abandoned")
        self.assertEqual(len(out["abandoned_claim_history"]), 1)
        self.assertEqual(timeline_event.call_count, 2)

    def test_dead_retry_claim_restores_same_retry_attempt_and_due_time(self) -> None:
        original = {
            "job_id": "logical",
            "status": "retry_wait",
            "attempt": 1,
            "next_attempt": 2,
            "retry_count": 1,
            "retry_not_before": "1970-01-01T00:01:40Z",
            "retry_not_before_epoch_s": 100,
            "retry_reason": "worker_lost",
        }
        job = gpu_claim.build_claim_payload(
            original,
            claim_id="retry-claim",
            owner_process={"pid": 999_999_999, "start_ticks": 1, "boot_id": "old"},
        )
        self.assertEqual(job["attempt"], 2)
        with (
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
            mock.patch.object(gpu_worker, "_timeline_event"),
        ):
            out, detail = gpu_worker.reconcile_job({"run_id": "unit"}, job, now=200)
        self.assertEqual(detail["decision"], "abandoned_claim_released")
        self.assertEqual(out["status"], "retry_wait")
        self.assertEqual(out["attempt"], 1)
        self.assertEqual(out["next_attempt"], 2)
        self.assertEqual(out["retry_count"], 1)
        self.assertEqual(out["retry_not_before_epoch_s"], 100)

    def test_mismatched_reused_pid_is_dead(self) -> None:
        identity = gpu_claim.process_identity()
        self.assertIsNotNone(identity)
        mismatched = dict(identity or {})
        mismatched["start_ticks"] = int(mismatched["start_ticks"]) + 1
        self.assertEqual(gpu_claim.process_identity_state(mismatched), "dead")

    def test_operator_stop_waits_for_inflight_dispatch_lock(self) -> None:
        run = {"run_id": "unit", "state_dir": "/tmp/unit-stop-lock"}
        lock = mock.MagicMock()
        lock.return_value.__enter__.return_value = True
        lock.return_value.__exit__.return_value = False
        with (
            mock.patch.object(gpu_claim, "dispatch_lock", lock),
            mock.patch.object(gpu_worker, "list_job_ids", return_value=[]),
        ):
            self.assertEqual(gpu_worker.stop_all(run), [])
        lock.assert_called_once_with(
            Path(run["state_dir"]),
            timeout_sec=gpu_worker.STOP_DISPATCH_LOCK_TIMEOUT_SEC,
        )

    def test_stop_arriving_during_spawn_fences_new_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            jobs = {
                "queued": {
                    "job_id": "queued",
                    "run_id": "unit",
                    "status": "pending",
                    "command": ["python3", "train.py"],
                }
            }

            def load_job(_run, job_id):
                return dict(jobs[job_id])

            def persist_job(_run, payload):
                jobs[str(payload["job_id"])] = dict(payload)
                return dict(payload)

            handle = mock.Mock(attempt_id="sb-new")
            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "load_run",
                    return_value=(Path(raw), run),
                ),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["queued"]),
                mock.patch.object(gpu_worker, "load_job", side_effect=load_job),
                mock.patch.object(gpu_worker, "persist_job", side_effect=persist_job),
                mock.patch.object(
                    gpu_worker, "pin_work_archive", side_effect=lambda _run, job: job
                ),
                mock.patch.object(gpu_worker, "restore_pinned_work_archive"),
                mock.patch.object(
                    gpu_worker,
                    "operator_stop_requested",
                    side_effect=[False, True],
                ),
                mock.patch.object(
                    gpu_worker.ModalSandboxProvider,
                    "start",
                    return_value=handle,
                ),
                mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
                mock.patch.object(gpu_worker, "_close_attempt_timeline"),
                mock.patch.object(gpu_worker, "_timeline_event"),
                mock.patch.object(
                    gpu_worker, "_terminate_sandbox", return_value=None
                ) as terminate,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(jobs["queued"]["status"], "terminated")
        self.assertEqual(jobs["queued"]["sandbox_id"], "sb-new")
        self.assertEqual(result["actions"][0]["action"], "stop_after_spawn")
        terminate.assert_called_once()

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

    def test_inference_policy_is_not_selected_over_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trainer = root / "model_7.pt"
            policy = root / "policy_8.pt"
            trainer.write_bytes(b"optimizer and model")
            policy.write_bytes(b"torchscript")
            store = resilience.CheckpointStore(root)
            expected = store.commit(
                trainer,
                sequence=7,
                metadata={"kind": "training_state"},
            )
            store.commit(
                policy,
                sequence=8,
                metadata={"kind": "torchscript_policy"},
            )

            latest = worker_run.latest_checkpoint(root)

        self.assertEqual(latest, str(expected.path))

    def test_replacement_fails_closed_without_resumable_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "resumable training-state"):
            worker_run.build_attempt_command(
                {
                    "command": ["python3", "train.py"],
                    "resume_arg": "--checkpoint",
                },
                2,
                None,
            )


class LauncherWiringTests(unittest.TestCase):
    def test_training_image_uses_pinned_warmup_id(self) -> None:
        pinned = object()
        run = {
            "evaluation_provenance": {"agent_training_image_id": "im-ExactTraining123"}
        }
        with mock.patch("modal.Image.from_id", return_value=pinned) as from_id:
            self.assertIs(gpu_worker.training_image(run), pinned)
        from_id.assert_called_once_with("im-ExactTraining123")

    def test_training_image_fails_closed_without_valid_id(self) -> None:
        for run in (
            {},
            {"evaluation_provenance": {}},
            {"evaluation_provenance": {"agent_training_image_id": "latest"}},
        ):
            with self.subTest(run=run):
                with self.assertRaisesRegex(RuntimeError, "warmed.*image ID"):
                    gpu_worker.training_image(run)

    def test_launcher_passes_pinned_agent_and_verifier_images(self) -> None:
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        self.assertIn('--ek "modal_image_id=$AGENT_TRAINING_IMAGE_ID"', launcher)
        self.assertIn('--ek "verifier_image_id=$VERIFIER_IMAGE_ID"', launcher)

    def test_cpu_resume_uses_launch_source_and_recorded_images(self) -> None:
        launcher = (ROOT / "runs" / "run-lane-durable.sh").read_text()
        self.assertIn('"sprint_source_commit": sprint_source_commit', launcher)
        self.assertIn('payload.get("sprint_source_commit")', launcher)
        self.assertIn(
            'SOURCE_ROOT="/data/sprint-run-sources/$RUN_ID-$SPRINT_SOURCE_COMMIT"',
            launcher,
        )
        self.assertIn('git -C "$ROOT" worktree add --detach "$SOURCE_ROOT"', launcher)
        self.assertIn('provenance.get("agent_training_image_id")', launcher)
        self.assertIn('provenance.get("verifier_image_id")', launcher)
        self.assertGreater(
            launcher.index('python3 "$ROOT/runs/ops/check_modal_image_warmup.py"'),
            launcher.index("if (( ! RESUMING )); then"),
        )

    def test_all_model_launchers_default_to_systemd_supervisor(self) -> None:
        for name in ("run-luna.sh", "run-deepseek.sh"):
            text = (ROOT / "runs" / name).read_text()
            self.assertIn("start_lane_supervisor.py", text)
            self.assertIn("--supervised-launch", text)
            self.assertIn("CPU_MAX_RESTARTS", text)
            self.assertIn("CPU_MAX_RESTARTS:-50", text)
        for name in ("run-opus.sh", "run-terra.sh", "run-lane.sh"):
            self.assertFalse((ROOT / "runs" / name).exists())
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
                "--batch-id",
                "eval-batch",
            ]
            completed = mock.Mock(returncode=0)
            with (
                mock.patch.object(start_lane_supervisor, "OPS", Path(raw)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "secret-value",
                        "UV": "/test/bin/uv",
                        "PATH": "/usr/bin",
                    },
                    clear=True,
                ),
                mock.patch.object(Path, "is_file", return_value=True),
                mock.patch("os.access", return_value=True),
                mock.patch.object(
                    start_lane_supervisor.shutil, "which", return_value=None
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
            self.assertIn("--setenv=UV=/test/bin/uv", command)
            self.assertIn(
                f"--setenv=PATH={ROOT / 'harbor/.venv/bin'}:/test/bin:/usr/bin",
                command,
            )
            self.assertIn("--setenv=SPRINT_BATCH_ID=eval-batch", command)
            self.assertNotIn("secret-value", command)
            metadata = json.loads(
                (Path(raw) / "unit-run" / "supervisor.json").read_text()
            )
            self.assertEqual(metadata["launch_argv"], launch)
            self.assertEqual(metadata["batch_id"], "eval-batch")
            self.assertEqual(
                metadata["controller_python"],
                str(ROOT / "harbor/.venv/bin/python3"),
            )

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
