"""Focused unit tests for CPU-agent / GPU-worker split ops."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
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
from event_runtime.control import start_trial as start_cpu_trial  # noqa: E402
from event_runtime.control import credentials as validate_agent_env  # noqa: E402
from event_runtime.container import sprint_resilience as resilience  # noqa: E402

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
    "event_gpu",
    importlib.machinery.SourceFileLoader(
        "event_gpu", str(ROOT / "event_runtime/agent/gpu.py")
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


class WorkerRuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_limit_is_enforced_inside_started_sandbox(self) -> None:
        with mock.patch.object(
            worker_run.shutil, "which", return_value="/usr/bin/timeout"
        ):
            command = worker_run.runtime_bounded_command(["python3", "train.py"], 120)

        self.assertEqual(
            command,
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=20s",
                "120s",
                "python3",
                "train.py",
            ],
        )

    def test_runtime_limit_is_clamped_to_supported_bounds(self) -> None:
        with mock.patch.object(
            worker_run.shutil, "which", return_value="/usr/bin/timeout"
        ):
            self.assertEqual(worker_run.runtime_bounded_command(["true"], 1)[3], "60s")
            self.assertEqual(
                worker_run.runtime_bounded_command(["true"], 100_000)[3], "86400s"
            )

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

    def test_pending_candidates_are_fifo(self) -> None:
        jobs = {
            "later": {
                "job_id": "later",
                "status": "pending",
                "created_at_epoch_s": 20,
            },
            "earlier": {
                "job_id": "earlier",
                "status": "pending",
                "created_at_epoch_s": 10,
            },
        }
        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=list(jobs)),
            mock.patch.object(
                gpu_worker, "load_job", side_effect=lambda _run, job_id: jobs[job_id]
            ),
        ):
            candidates = gpu_worker._candidate_job_ids({}, now=30)
        self.assertEqual(candidates, ["earlier", "later"])

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

            def download(_run, _remote, destination, **_kwargs):
                destination.write_bytes(archive_bytes)
                return True

            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_download_exact",
                    side_effect=download,
                ),
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

            def download(_run, _remote, destination, **_kwargs):
                destination.write_bytes(archive_bytes)
                return True

            with mock.patch.object(
                gpu_worker.sprintctl,
                "volume_download_exact",
                side_effect=download,
            ):
                pinned = gpu_worker.pin_work_archive(run, job)

            self.assertTrue(pinned["work_archive_changed_before_claim"])
            self.assertEqual(
                pinned["work_archive_sha256"],
                hashlib.sha256(archive_bytes).hexdigest(),
            )

    def test_host_archive_transfer_retries_without_consuming_gpu_attempt(self) -> None:
        archive_bytes = b"eventual Modal Volume response"
        with tempfile.TemporaryDirectory() as raw:
            run = {
                "run_id": "run-a",
                "state_dir": raw,
                "volume_name": "volume-a",
            }
            job = {
                "job_id": "job-a",
                "work_archive": "runs/run-a/gpu-jobs/work/job-a/app.tar.gz",
            }
            calls = 0

            def download(_run, _remote, destination, **_kwargs):
                nonlocal calls
                calls += 1
                if calls < 3:
                    return False
                destination.write_bytes(archive_bytes)
                return True

            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_download_exact",
                    side_effect=download,
                ),
                mock.patch.object(gpu_worker.time, "sleep") as sleep,
            ):
                pinned = gpu_worker.pin_work_archive(run, job)

            self.assertEqual(calls, 3)
            self.assertEqual(
                pinned["work_archive_sha256"], hashlib.sha256(archive_bytes).hexdigest()
            )
            self.assertEqual(
                [call.args[0] for call in sleep.call_args_list], [1.0, 2.0]
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

    def test_worker_wraps_env_prefixed_isaac_python_with_local_asset_bootstrap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "train.py"
            script.write_text("from isaaclab.app import AppLauncher\n")
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text("# trusted bootstrap\n")
            command = worker_run.build_attempt_command(
                {
                    "command": [
                        "/usr/bin/env",
                        "POP=64",
                        "MAXS=12",
                        "python3",
                        "-u",
                        str(script),
                    ]
                },
                1,
                None,
                isaac_bootstrap=bootstrap,
            )

        self.assertEqual(
            command,
            [
                "/usr/bin/env",
                "POP=64",
                "MAXS=12",
                "python3",
                "-u",
                str(bootstrap),
                str(script),
            ],
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

    def test_mirrors_bounded_live_modal_streams_before_exit(self) -> None:
        mirrored: dict[str, object] = {}

        def mirror(_run: dict, payload: dict, **kwargs: object) -> dict:
            mirrored["payload"] = payload
            mirrored.update(kwargs)
            return {"agent_mirror": "updated", "agent_mirror_files": 2}

        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 1,
            "status": "running",
            "sandbox_id": "sb-1",
        }
        with mock.patch.object(gpu_worker, "mirror_agent_job", side_effect=mirror):
            refreshed, detail = gpu_worker.refresh_live_provider_logs(
                {"run_id": "run-1"},
                job,
                now=100.0,
                read_output=lambda _run, _sandbox_id: "iteration 12/100\n",
            )

        self.assertEqual(detail["live_provider_logs"], "mirrored")
        self.assertEqual(
            mirrored["log_content"],
            b"== Modal live log tail ==\niteration 12/100\n",
        )
        self.assertEqual(refreshed["provider_live_logs_checked_at_epoch_s"], 100.0)
        self.assertEqual(refreshed["provider_live_logs_source"], "modal-sandbox-tail")

    def test_live_modal_stream_mirror_is_rate_limited(self) -> None:
        job = {
            "job_id": "job-1",
            "attempt": 1,
            "sandbox_id": "sb-1",
            "provider_live_logs_checked_at_epoch_s": 90.0,
        }
        refreshed, detail = gpu_worker.refresh_live_provider_logs(
            {"run_id": "run-1"},
            job,
            now=100.0,
            read_output=lambda _run, _sandbox_id: self.fail("unexpected read"),
        )

        self.assertIs(refreshed, job)
        self.assertEqual(detail["live_provider_logs"], "fresh")

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

    def test_worker_error_overrides_false_success_without_traceback(self) -> None:
        audited = gpu_worker.apply_provider_terminal_error(
            {
                "status": "succeeded",
                "exit_code": 0,
                "error": "GPU activity watchdog: no accelerator progress",
            },
            None,
        )
        self.assertEqual(audited["status"], "failed")
        self.assertEqual(audited["exit_code"], 1)
        self.assertEqual(audited["failure_reason"], "worker_reported_error")
        self.assertEqual(audited["worker_reported_status"], "succeeded")

    def test_recoverable_isaac_gpu_warning_does_not_override_success(self) -> None:
        self.assertIsNone(
            gpu_worker.provider_terminal_error(
                "[Error] [gpu.foundation.plugin] No device could be created"
            )
        )

    def test_missing_outputs_plus_vulkan_warning_is_not_provider_failure(self) -> None:
        stderr = "\n".join(
            (
                "VkResult: ERROR_INITIALIZATION_FAILED",
                "vkCreateDevice failed",
                "No device could be created",
            )
        )
        self.assertIsNone(
            gpu_worker.provider_terminal_error_for_job(
                {
                    "status": "failed",
                    "error": "required GPU output missing or invalid: /app/policy.pt",
                },
                "",
                stderr,
            )
        )

    def test_vulkan_warning_does_not_mask_agent_traceback(self) -> None:
        stderr = "\n".join(
            (
                "VkResult: ERROR_INITIALIZATION_FAILED",
                "vkCreateDevice failed",
                "No device could be created",
                "Traceback (most recent call last):",
                "AttributeError: bad reward term",
            )
        )
        self.assertEqual(
            gpu_worker.provider_terminal_error_for_job(
                {
                    "status": "failed",
                    "error": "required GPU output missing or invalid: /app/policy.pt",
                },
                "",
                stderr,
            ),
            "AttributeError: bad reward term",
        )

    def test_physx_software_fallback_overrides_false_zero_exit(self) -> None:
        output = "PhysX warning: GPU solver pipeline failed, switching to software"
        self.assertEqual(gpu_worker.provider_terminal_error(output), output)

    def test_agent_stdout_cannot_forge_a_provider_terminal_error(self) -> None:
        archived = (
            "== Modal stdout ==\n"
            "source = 'GPU solver pipeline failed|GPU Bp pipeline failed|"
            "switching to software'\n"
            "== Modal stderr ==\n"
            "recoverable Vulkan warning\n"
        )
        self.assertIsNone(gpu_worker.provider_terminal_error(archived))

    def test_archive_classifies_only_provider_stderr(self) -> None:
        uploaded: dict[str, bytes] = {}
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 1,
            "status": "succeeded",
            "exit_code": 0,
            "sandbox_id": "sb-1",
        }

        def upload(_run: dict, source: Path, _remote: str) -> None:
            uploaded["content"] = source.read_bytes()

        stdout = "print('GPU solver pipeline failed|switching to software')\n"
        with (
            mock.patch.object(
                gpu_worker.sprintctl, "volume_upload", side_effect=upload
            ),
            mock.patch.object(
                gpu_worker, "mirror_agent_job", return_value={"agent_mirror": "updated"}
            ),
        ):
            archived, detail = gpu_worker.archive_provider_logs(
                {"run_id": "run-1"},
                job,
                read_output=lambda _sandbox_id: (stdout, "recoverable warning\n"),
            )

        self.assertIn(stdout.encode(), uploaded["content"])
        self.assertEqual(archived["status"], "succeeded")
        self.assertFalse(archived["provider_terminal_error_detected"])
        self.assertIsNone(detail["provider_terminal_error"])

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

    def test_audit_does_not_misclassify_archived_vulkan_output_loss(self) -> None:
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "attempt": 1,
            "status": "failed",
            "exit_code": 2,
            "error": "required GPU output missing or invalid: /app/policy.pt",
            "provider_logs_archived_at": "earlier",
            "provider_logs_path": "runs/run-1/gpu-jobs/out/job-1/worker.log",
        }
        text = (
            "== Modal stdout ==\nlauncher output\n"
            "== Modal stderr ==\n"
            "VkResult: ERROR_INITIALIZATION_FAILED\n"
            "vkCreateDevice failed\n"
            "No device could be created\n"
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

        self.assertFalse(audited["provider_terminal_error_detected"])
        self.assertEqual(
            audited["provider_terminal_error_classifier_version"],
            gpu_worker.PROVIDER_TERMINAL_ERROR_CLASSIFIER_VERSION,
        )
        self.assertNotIn("provider_terminal_error", audited)
        self.assertIsNone(detail["provider_terminal_error"])

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
            cli_path = Path(tmp) / "event_runtime" / "agent" / "gpu.py"
            cli_path.parent.mkdir(parents=True)
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

    def test_terminal_run_skips_dead_cpu_agent_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "STOP_ACK.json").write_text("{}\n")
            with mock.patch.object(gpu_worker.sprintctl, "exec_container") as execute:
                detail = gpu_worker.mirror_agent_job(
                    {
                        "state_dir": str(state),
                        "agent_container_id": "ta-agent",
                    },
                    {"job_id": "job-1", "attempt": 1, "status": "succeeded"},
                    log_content=b"complete child output\n",
                )
        self.assertEqual(detail["agent_mirror"], "terminal_cpu_unavailable")
        execute.assert_not_called()

    def test_terminal_run_persists_job_without_remote_delivery_mirrors(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            (state / "STOP_ACK.json").write_text("{}\n")
            run = {"run_id": "run-1", "state_dir": str(state)}
            job = {"job_id": "job-1", "status": "succeeded"}
            with (
                mock.patch.object(gpu_worker, "persist_host_job") as persist_host,
                mock.patch.object(gpu_worker, "put_json") as put_remote,
                mock.patch.object(gpu_worker, "mirror_agent_job") as mirror_agent,
            ):
                result = gpu_worker.persist_job(run, job)
        self.assertEqual(result, job)
        persist_host.assert_called_once_with(run, job)
        put_remote.assert_not_called()
        mirror_agent.assert_not_called()

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

    def test_pushes_canonical_budget_snapshot_into_active_gpu_sandbox(self) -> None:
        captured: dict[str, object] = {}

        class Reader:
            def read(self) -> str:
                return ""

        class Process:
            stdout = Reader()
            stderr = Reader()

            def wait(self) -> int:
                return 0

        class Sandbox:
            def exec(self, *args: str, **kwargs: object) -> Process:
                captured["args"] = args
                captured["kwargs"] = kwargs
                return Process()

        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 9.9,
            "status": "within_budget",
        }
        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox()
        ) as from_id:
            detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-gpu", "status": "running"}],
            )

        from_id.assert_called_once_with("sb-gpu")
        args = captured["args"]
        self.assertIsInstance(args, tuple)
        assert isinstance(args, tuple)
        self.assertEqual(args[0:2], ("python3", "-c"))
        self.assertEqual(args[3], gpu_worker.GPU_BUDGET_MIRROR_PATH)
        self.assertEqual(json.loads(gpu_worker.base64.b64decode(args[4])), payload)
        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["updated_sandbox_ids"], ["sb-gpu"])
        self.assertEqual(detail["finished_sandbox_ids"], [])

    def test_budget_mirror_discovers_targets_without_remote_volume_reads(self) -> None:
        class Reader:
            def read(self) -> str:
                return ""

        class Process:
            stdout = Reader()
            stderr = Reader()

            def wait(self) -> int:
                return 0

        class Sandbox:
            def exec(self, *_args: str, **_kwargs: object) -> Process:
                return Process()

        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 9.9,
            "status": "within_budget",
        }
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            registry = state_dir / "gpu-job-registry"
            registry.mkdir()
            (registry / "active.json").write_text(
                json.dumps(
                    {
                        "job_id": "active",
                        "status": "running",
                        "sandbox_id": "sb-gpu",
                    }
                )
            )
            # A locally known but unallocated job is not a mirror target.
            (registry / "pending.json").write_text(
                json.dumps({"job_id": "pending", "status": "pending"})
            )
            run = {"run_id": "run-1", "state_dir": str(state_dir)}
            with (
                mock.patch.object(
                    gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox()
                ) as from_id,
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_get_text",
                    side_effect=AssertionError("remote job fetch must not run"),
                ),
            ):
                detail = gpu_worker.mirror_gpu_budget(run, payload)

        from_id.assert_called_once_with("sb-gpu")
        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["sandbox_ids"], ["sb-gpu"])

    def test_budget_mirror_accepts_sandbox_that_finished_after_scan(self) -> None:
        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 9.9,
            "status": "within_budget",
        }
        with mock.patch.object(
            gpu_worker.modal.Sandbox,
            "from_id",
            side_effect=gpu_worker.modal.exception.NotFoundError(
                "Task has already finished with status success"
            ),
        ) as from_id:
            detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-finished", "status": "running"}],
            )

        from_id.assert_called_once_with("sb-finished")
        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["updated_sandbox_ids"], [])
        self.assertEqual(detail["finished_sandbox_ids"], ["sb-finished"])
        self.assertEqual(detail["errors"], {})

    def test_budget_mirror_accepts_exec_killed_as_sandbox_finishes(self) -> None:
        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 9.9,
            "status": "within_budget",
        }

        class Reader:
            def read(self) -> str:
                return ""

        class Process:
            stdout = Reader()
            stderr = Reader()

            def wait(self) -> int:
                return 137

        class Sandbox:
            def __init__(self, poll_result: int | None) -> None:
                self.poll_result = poll_result

            def exec(self, *_args: str, **_kwargs: object) -> Process:
                return Process()

            def poll(self) -> int | None:
                return self.poll_result

        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox(2)
        ):
            detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-finished", "status": "running"}],
            )

        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["updated_sandbox_ids"], [])
        self.assertEqual(detail["finished_sandbox_ids"], ["sb-finished"])
        self.assertEqual(detail["errors"], {})

        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox(None)
        ):
            live_detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-live", "status": "running"}],
            )

        self.assertEqual(live_detail["gpu_budget_mirror"], "error")
        self.assertEqual(live_detail["finished_sandbox_ids"], [])
        self.assertIn("exit 137", live_detail["errors"]["sb-live"])

    def test_budget_mirror_accepts_fetchspec_race_after_sandbox_finishes(self) -> None:
        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 10.0,
            "status": "within_budget",
        }

        class Sandbox:
            def exec(self, *_args: str, **_kwargs: object) -> None:
                raise RuntimeError(
                    "FetchSpec failed: loading container: file does not exist"
                )

        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox()
        ):
            detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-finished", "status": "running"}],
            )

        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["updated_sandbox_ids"], [])
        self.assertEqual(detail["finished_sandbox_ids"], ["sb-finished"])
        self.assertEqual(detail["errors"], {})

    def test_gpu_budget_mirror_refuses_to_replace_newer_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "cost.json"
            current = {
                "schema_version": 2,
                "run_id": "run-1",
                "checked_at_epoch_s": 200.0,
                "total_usd": 3.0,
                "stop_threshold_usd": 9.9,
                "status": "within_budget",
            }
            target.write_text(json.dumps(current))

            class Sandbox:
                def exec(self, *args: str, **_kwargs: object) -> subprocess.Popen:
                    return subprocess.Popen(
                        args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

            incoming = {
                "schema_version": 2,
                "run_id": "run-1",
                "checked_at_epoch_s": 100.0,
                "total_usd": 9.0,
                "stop_threshold_usd": 9.9,
                "status": "within_budget",
            }
            with (
                mock.patch.object(
                    gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox()
                ),
                mock.patch.object(gpu_worker, "GPU_BUDGET_MIRROR_PATH", str(target)),
            ):
                detail = gpu_worker.mirror_gpu_budget(
                    {"run_id": "run-1"},
                    incoming,
                    jobs=[{"sandbox_id": "sb-gpu", "status": "running"}],
                )

            self.assertEqual(detail["updated_sandbox_ids"], [])
            self.assertEqual(detail["stale_ignored_sandbox_ids"], ["sb-gpu"])
            self.assertEqual(json.loads(target.read_text()), current)

    def test_budget_mirror_accepts_sandbox_that_is_shutting_down(self) -> None:
        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 9.9,
            "status": "within_budget",
        }
        with mock.patch.object(
            gpu_worker.modal.Sandbox,
            "from_id",
            side_effect=gpu_worker.modal.exception.ConflictError(
                "Modal Sandbox is shutting down."
            ),
        ):
            detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-stopping", "status": "running"}],
            )

        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["finished_sandbox_ids"], ["sb-stopping"])
        self.assertEqual(detail["errors"], {})

    def test_budget_mirror_accepts_exec_race_with_stopped_sandbox(self) -> None:
        payload = {
            "schema_version": 2,
            "run_id": "run-1",
            "checked_at_epoch_s": 1234.5,
            "total_usd": 2.25,
            "stop_threshold_usd": 9.9,
            "status": "within_budget",
        }

        class Sandbox:
            def exec(self, *_args: str, **_kwargs: object) -> object:
                raise RuntimeError(
                    "executing processes for container: cannot execute in "
                    'container "ta-gpu" in state stopped'
                )

        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=Sandbox()
        ):
            detail = gpu_worker.mirror_gpu_budget(
                {"run_id": "run-1"},
                payload,
                jobs=[{"sandbox_id": "sb-stopped", "status": "running"}],
            )

        self.assertEqual(detail["gpu_budget_mirror"], "updated")
        self.assertEqual(detail["finished_sandbox_ids"], ["sb-stopped"])
        self.assertEqual(detail["errors"], {})

    def test_dispatch_budget_snapshot_uses_fresh_canonical_agent_cost(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            telemetry = state_dir / "telemetry"
            telemetry.mkdir()
            canonical = {
                "schema_version": 2,
                "run_id": "run-1",
                "checked_at_epoch_s": 100.0,
                "total_usd": 7.69,
                "stop_threshold_usd": 9.9,
                "status": "within_budget",
            }
            (telemetry / "agent-cost.json").write_text(json.dumps(canonical))
            # A lower raw watchdog must never be the dispatch source.
            (telemetry / "budget-watchdog.json").write_text(
                json.dumps({**canonical, "total_usd": 2.63})
            )

            loaded = gpu_worker.fresh_dispatch_budget_snapshot(
                state_dir, "run-1", now=120.0
            )

            self.assertEqual(loaded, canonical)

    def test_dispatch_budget_snapshot_rejects_stale_canonical_cost(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_dir = Path(raw)
            telemetry = state_dir / "telemetry"
            telemetry.mkdir()
            (telemetry / "agent-cost.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "run_id": "run-1",
                        "checked_at_epoch_s": 100.0,
                        "total_usd": 7.69,
                        "stop_threshold_usd": 9.9,
                        "status": "within_budget",
                    }
                )
            )

            self.assertIsNone(
                gpu_worker.fresh_dispatch_budget_snapshot(state_dir, "run-1", now=161.0)
            )

    def test_fetches_only_reported_scoped_policy_for_agent_mirror(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {"policy_path": "/durable/runs/run-1/policies/policy_7.pt"},
        }

        def fake_get(_run, remote, **_kwargs):
            self.assertEqual(remote, "runs/run-1/policies/policy_7.pt")
            return b"trusted policy"

        with mock.patch.object(
            gpu_worker.sprintctl, "volume_get_bytes", side_effect=fake_get
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
        self.assertEqual(
            payload["agent_policy_source_path"],
            "/durable/runs/run-1/policies/policy_7.pt",
        )

    def test_terminal_output_fetch_retries_after_volume_lag(self) -> None:
        job = {
            "job_id": "job-1",
            "status": "succeeded",
            "progress": {
                "output_artifacts": [
                    {
                        "source_path": "/app/diagnostics.json",
                        "path": (
                            "/durable/runs/run-1/gpu-jobs/artifacts/job-1/"
                            "attempt-1/diagnostics.json"
                        ),
                    }
                ]
            },
        }
        with mock.patch.object(
            gpu_worker,
            "fetch_agent_output_artifact",
            return_value=(job, None, None, {"artifact_mirror": "fetch_retry"}),
        ):
            payload, detail = gpu_worker.retry_terminal_artifact_mirror(
                {"run_id": "run-1"}, job
            )
        self.assertEqual(detail["artifact_mirror"], "fetch_retry")
        self.assertEqual(payload["artifact_mirror_attempts"], 1)
        self.assertGreater(payload["artifact_mirror_retry_after_epoch_s"], time.time())

    def test_fetches_declared_non_policy_output_for_agent_mirror(self) -> None:
        content = b'{"loss": 0.5}\n'
        digest = hashlib.sha256(content).hexdigest()
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {
                "output_artifacts": [
                    {
                        "source_path": "/app/diagnostics.json",
                        "path": (
                            "/durable/runs/run-1/gpu-jobs/artifacts/job-1/"
                            "attempt-1/diagnostics.json"
                        ),
                        "size_bytes": len(content),
                        "sha256": digest,
                    }
                ]
            },
        }

        with mock.patch.object(
            gpu_worker.sprintctl, "volume_get_bytes", return_value=content
        ):
            payload, name, mirrored, detail = gpu_worker.fetch_agent_output_artifact(
                run, job
            )

        self.assertEqual(name, "diagnostics.json")
        self.assertEqual(mirrored, content)
        self.assertEqual(detail["artifact_mirror"], "fetched")
        self.assertEqual(
            payload["agent_artifact_mirrors"]["/app/diagnostics.json"]["mirror_path"],
            "/run/sprint-gpu-mirror/artifacts/job-1/diagnostics.json",
        )

    def test_live_heartbeat_policy_is_mirrored_once_while_training(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease-1",
        }
        progress = {
            "iteration": 25,
            "policy_path": (
                "/durable/runs/run-1/gpu-jobs/checkpoints/job-1/policy_25.pt"
            ),
        }
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease-1",
            "progress": progress,
            "checkpoint": "/durable/runs/run-1/checkpoint.pt",
        }
        fetched = {
            **job,
            "progress": progress,
            "checkpoint": heartbeat["checkpoint"],
            "agent_policy_mirror_path": (
                "/run/sprint-gpu-mirror/artifacts/job-1/policy_25.pt"
            ),
            "agent_policy_source_path": progress["policy_path"],
            "agent_policy_sha256": "abc",
            "agent_policy_size_bytes": 6,
        }
        with (
            mock.patch.object(
                gpu_worker,
                "fetch_agent_policy_artifact",
                return_value=(
                    fetched,
                    "policy_25.pt",
                    b"policy",
                    {"policy_mirror": "fetched"},
                ),
            ) as fetch,
            mock.patch.object(
                gpu_worker,
                "mirror_agent_job",
                return_value={"agent_mirror": "updated"},
            ) as mirror,
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ) as persist,
        ):
            payload, detail = gpu_worker.refresh_live_policy_mirror(run, job, heartbeat)
            repeated, repeated_detail = gpu_worker.refresh_live_policy_mirror(
                run, payload, heartbeat
            )

        self.assertEqual(detail["live_policy_mirror"], "updated")
        self.assertEqual(payload["progress"], progress)
        self.assertEqual(payload["checkpoint"], heartbeat["checkpoint"])
        self.assertIn("agent_policy_source_progress_sha256", payload)
        self.assertEqual(repeated_detail["live_policy_mirror"], "already_mirrored")
        self.assertEqual(repeated, payload)
        fetch.assert_called_once()
        mirror.assert_called_once()
        persist.assert_called_once()

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

        def fake_get(_run, remote, **_kwargs):
            self.assertEqual(
                remote, "runs/run-1/gpu-jobs/checkpoints/job-1/policy_2.pt"
            )
            return b"trusted checkpoint policy"

        with mock.patch.object(
            gpu_worker.sprintctl, "volume_get_bytes", side_effect=fake_get
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

    def test_fetches_policy_alias_from_progress_record(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {
                "policy": "/durable/runs/run-1/gpu-jobs/checkpoints/job-1/policy.pt"
            },
        }

        with mock.patch.object(
            gpu_worker.sprintctl,
            "volume_get_bytes",
            return_value=b"aliased policy",
        ):
            payload, name, content, detail = gpu_worker.fetch_agent_policy_artifact(
                run, job
            )

        self.assertEqual(name, "policy.pt")
        self.assertEqual(content, b"aliased policy")
        self.assertEqual(detail["policy_mirror"], "fetched")
        self.assertEqual(payload["agent_policy_source_path"], job["progress"]["policy"])

    def test_fetches_policy_from_custom_run_gpu_job_directory(self) -> None:
        run = {"run_id": "run-1", "volume_name": "volume-1"}
        job = {
            "job_id": "job-1",
            "progress": {
                "policy_path": "/durable/runs/run-1/gpu-jobs/sprint-long/policy.pt"
            },
        }

        with mock.patch.object(
            gpu_worker.sprintctl,
            "volume_get_bytes",
            return_value=b"custom job policy",
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

        def fake_get(_run, remote, **_kwargs):
            self.assertEqual(remote, "runs/run-1/candidates/policy_599.pt")
            return b"candidate policy"

        with mock.patch.object(
            gpu_worker.sprintctl, "volume_get_bytes", side_effect=fake_get
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

    def test_agent_cli_prefers_completed_attempt_over_later_batch_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "durable" / "gpu-jobs"
            mirror = Path(tmp) / "mirror"
            (root / "status").mkdir(parents=True)
            (root / "attempts" / "job-1").mkdir(parents=True)
            (mirror / "status").mkdir(parents=True)
            (mirror / "status" / "job-1.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "attempt": 1,
                        "lease_id": "lease-1",
                        "status": "terminated",
                        "terminated_at_epoch_s": 200,
                    }
                )
            )
            (root / "attempts" / "job-1" / "1.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "attempt": 1,
                        "lease_id": "lease-1",
                        "status": "succeeded",
                        "finished_at_epoch_s": 100,
                        "checkpoint": "/durable/model.pt",
                    }
                )
            )

            with mock.patch.object(train_cli, "AGENT_MIRROR_ROOT", mirror):
                status = train_cli.read_status(root, "job-1")

            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["checkpoint"], "/durable/model.pt")
            self.assertEqual(
                status["status_reconciliation"]["selected_source"],
                "durable_attempt",
            )

    def test_agent_cli_preserves_fence_that_predates_worker_completion(self) -> None:
        stopped = {
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "status": "terminated",
            "terminated_at_epoch_s": 90,
        }
        late_worker = {
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "status": "succeeded",
            "finished_at_epoch_s": 100,
        }
        status = train_cli.reconcile_status_candidates(
            [(stopped, "host_mirror"), (late_worker, "durable_attempt")]
        )
        self.assertEqual(status["status"], "terminated")

    def test_agent_workspace_archive_omits_links_rejected_by_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app"
            app.mkdir()
            (app / "train.py").write_text("print('train')\n")
            (app / "linked-verifier").symlink_to("/opt/event/verifier")
            (app / "verifier").mkdir()
            (app / "verifier" / "mutable.py").write_text("MUTABLE = True\n")
            archive = root / "app.tar.gz"

            train_cli.pack_sync_dirs(archive, [app])

            with tarfile.open(archive, "r:gz") as handle:
                members = handle.getmembers()
            names = {member.name for member in members}
            self.assertIn("app/train.py", names)
            self.assertNotIn("app/linked-verifier", names)
            self.assertNotIn("app/verifier", names)
            self.assertNotIn("app/verifier/mutable.py", names)
            self.assertTrue(
                all(member.isfile() or member.isdir() for member in members)
            )

    def test_agent_workspace_archive_preserves_safe_app_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "agent-app"
            train = app / "train"
            train.mkdir(parents=True)
            (train / "train.py").write_text("print('train')\n")
            staged = root / "staged-workspace"
            (staged / "train").mkdir(parents=True)
            (staged / "train" / "staged.py").write_text("print('staged')\n")
            course = root / "course"
            course.mkdir()
            (course / "rules.py").write_text("RULES = True\n")
            archive = root / "app.tar.gz"

            with mock.patch.object(train_cli, "AGENT_WORKSPACE_ROOT", app):
                train_cli.pack_sync_dirs(archive, [train, staged, course])

            with tarfile.open(archive, "r:gz") as handle:
                names = {member.name for member in handle.getmembers()}
            self.assertIn("app/train/train.py", names)
            self.assertIn("app/train/staged.py", names)
            self.assertIn("app/course/rules.py", names)


class SubmissionBridgeTests(unittest.TestCase):
    def request(self, *, attempt: int = 1, submission_id: str = "123456-abcd") -> dict:
        content = b"immutable-policy"
        digest = hashlib.sha256(content).hexdigest()
        return {
            "receipt": {
                "schema_version": 2,
                "bridge": "host_owned_gpu_submission_v1",
                "submission_id": submission_id,
                "queue_name": f"{submission_id}.pt",
                "run_id": "run-1",
                "gpu_job_id": "job-1",
                "gpu_attempt": attempt,
                "gpu_lease_id": "lease-1",
                "policy_sha256": digest,
                "policy_size_bytes": len(content),
                "note": "candidate",
            },
            "policy_sha256": digest,
            "policy_size_bytes": len(content),
            "policy_base64": base64.b64encode(content).decode(),
        }

    def run_and_job(self, root: Path) -> tuple[dict, dict]:
        return (
            {"run_id": "run-1", "state_dir": str(root)},
            {
                "run_id": "run-1",
                "job_id": "job-1",
                "attempt": 1,
                "lease_id": "lease-1",
                "sandbox_id": "sb-worker",
                "submission_bridge_enabled": True,
            },
        )

    def enqueue_snapshot_job(
        self, root: Path, *, include_policy: bool = True
    ) -> tuple[dict, dict, bytes]:
        run = {"run_id": "run-1", "state_dir": str(root)}
        content = b"enqueue-snapshot-policy"
        archive = root / "gpu-job-work/job-1/app.tar.gz"
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, "w:gz") as handle:
            directory = tarfile.TarInfo("app")
            directory.type = tarfile.DIRTYPE
            handle.addfile(directory)
            if include_policy:
                member = tarfile.TarInfo("app/policy.pt")
                member.size = len(content)
                handle.addfile(member, io.BytesIO(content))
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        job = {
            "run_id": "run-1",
            "job_id": "job-1",
            "status": "pending",
            "attempt": 0,
            "created_at_epoch_s": 1_787_789_554,
            "submission_bridge_enabled": True,
            "submission_paths": ["/app/policy.pt"],
            "submitted_work_archive_sha256": archive_digest,
            "work_archive": "runs/run-1/gpu-jobs/work/job-1/app.tar.gz",
        }
        return run, job, content

    def test_request_identity_and_bytes_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            receipt, content = gpu_worker.validate_worker_submission_request(
                run, job, self.request()
            )
            self.assertEqual(receipt["submission_id"], "123456-abcd")
            self.assertEqual(content, b"immutable-policy")
            with self.assertRaisesRegex(ValueError, "gpu_attempt identity mismatch"):
                gpu_worker.validate_worker_submission_request(
                    run, job, self.request(attempt=2)
                )

    def test_host_queue_remains_authoritative_when_cpu_mirror_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trial = root / "jobs/job/trial__model"
            trial.mkdir(parents=True)
            run, job = self.run_and_job(root)
            run["trial_path"] = str(trial)
            receipt, content = gpu_worker.validate_worker_submission_request(
                run, job, self.request()
            )

            with mock.patch.object(
                gpu_worker,
                "_mirror_worker_policy_to_cpu_agent",
                side_effect=RuntimeError("Modal Sandbox is shutting down"),
            ):
                result = gpu_worker.submit_worker_policy_to_cpu_agent(
                    run, receipt, content
                )

            queued = trial / "artifacts/continuous/incoming/123456-abcd.pt"
            self.assertEqual(queued.read_bytes(), content)
            self.assertEqual(result["returncode"], 0)
            self.assertIn(
                "Modal Sandbox is shutting down", result["agent_mirror_error"]
            )

    def test_drain_forwards_once_and_persists_host_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            request = self.request()
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_worker_submission_outbox",
                    return_value=[request],
                ),
                mock.patch.object(
                    gpu_worker,
                    "submit_worker_policy_to_cpu_agent",
                    return_value={"returncode": 0, "stdout": "staged", "stderr": ""},
                ) as submit,
                mock.patch.object(
                    gpu_worker, "acknowledge_worker_submission"
                ) as acknowledge,
            ):
                updated, detail = gpu_worker.drain_worker_submission_outbox(run, job)
                self.assertEqual(detail["forwarded"], 1)
                self.assertNotIn("submission_bridge_error", updated)
                record = json.loads(
                    (Path(raw) / "submission-bridge/123456-abcd.json").read_text()
                )
                self.assertEqual(record["state"], "forwarded")
                submit.assert_called_once()
                acknowledge.assert_called_once()

                gpu_worker.drain_worker_submission_outbox(run, job)
                submit.assert_called_once()
                self.assertEqual(acknowledge.call_count, 2)

    def test_drain_recovers_request_after_gpu_sandbox_exit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            with (
                mock.patch.object(
                    gpu_worker, "read_worker_submission_outbox", return_value=[]
                ),
                mock.patch.object(
                    gpu_worker,
                    "read_durable_submission_outbox",
                    return_value=[self.request()],
                ),
                mock.patch.object(
                    gpu_worker,
                    "submit_worker_policy_to_cpu_agent",
                    return_value={"returncode": 0, "stdout": "staged", "stderr": ""},
                ) as submit,
                mock.patch.object(gpu_worker, "acknowledge_worker_submission"),
            ):
                _updated, detail = gpu_worker.drain_worker_submission_outbox(run, job)
            self.assertEqual(detail["forwarded"], 1)
            submit.assert_called_once()

    def test_durable_recovery_uses_exact_manifest_and_binary_reads(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            request = self.request()
            receipt = request["receipt"]
            index = {
                "schema_version": 1,
                "run_id": run["run_id"],
                "gpu_job_id": job["job_id"],
                "gpu_attempt": job["attempt"],
                "gpu_lease_id": job["lease_id"],
                "submissions": {receipt["submission_id"]: receipt},
            }
            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_get_text",
                    return_value=json.dumps(index),
                ) as text_read,
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_get_bytes",
                    return_value=b"immutable-policy",
                ) as binary_read,
                mock.patch.object(gpu_worker.sprintctl, "run_command") as legacy,
            ):
                recovered = gpu_worker.read_durable_submission_outbox(run, job)

            self.assertEqual(recovered, [request])
            text_read.assert_called_once_with(
                run,
                "runs/run-1/submission-bridge/indexes/job-1.json",
                timeout_seconds=15,
            )
            binary_read.assert_called_once_with(
                run,
                "runs/run-1/submission-bridge/outbox/123456-abcd.pt",
                timeout_seconds=120,
                max_bytes=gpu_worker.GPU_SUBMISSION_BRIDGE_MAX_POLICY_BYTES,
            )
            legacy.assert_not_called()

    def test_fifteen_durable_recovery_readers_do_not_serialize_on_volume_lists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            receipt = self.request()["receipt"]
            index_text = json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run["run_id"],
                    "gpu_job_id": job["job_id"],
                    "gpu_attempt": job["attempt"],
                    "gpu_lease_id": job["lease_id"],
                    "submissions": {receipt["submission_id"]: receipt},
                }
            )
            barrier = threading.Barrier(15, timeout=5)

            def exact_index(*_args, **_kwargs):
                barrier.wait()
                return index_text

            results: list[list[dict]] = []
            errors: list[Exception] = []

            def recover() -> None:
                try:
                    results.append(gpu_worker.read_durable_submission_outbox(run, job))
                except Exception as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_get_text",
                    side_effect=exact_index,
                ),
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "volume_get_bytes",
                    return_value=b"immutable-policy",
                ),
                mock.patch.object(gpu_worker.sprintctl, "run_command") as legacy,
            ):
                threads = [threading.Thread(target=recover) for _ in range(15)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 15)
            legacy.assert_not_called()

    def test_drain_pages_until_every_explicit_submission_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            first = self.request(submission_id="123456-abcd")
            second = self.request(submission_id="123457-abce")
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_worker_submission_outbox",
                    side_effect=[[first], [second], []],
                ),
                mock.patch.object(
                    gpu_worker,
                    "submit_worker_policy_to_cpu_agent",
                    return_value={"returncode": 0, "stdout": "staged", "stderr": ""},
                ) as submit,
                mock.patch.object(gpu_worker, "acknowledge_worker_submission"),
            ):
                _updated, detail = gpu_worker.drain_worker_submission_outbox(run, job)

            self.assertEqual(detail["observed"], 2)
            self.assertEqual(detail["forwarded"], 2)
            self.assertEqual(submit.call_count, 2)

    def test_drain_fails_closed_when_cpu_archive_omits_returncode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, job = self.run_and_job(Path(raw))
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_worker_submission_outbox",
                    return_value=[self.request()],
                ),
                mock.patch.object(
                    gpu_worker,
                    "submit_worker_policy_to_cpu_agent",
                    return_value={"stdout": "ambiguous response", "stderr": ""},
                ),
                mock.patch.object(
                    gpu_worker, "acknowledge_worker_submission"
                ) as acknowledge,
            ):
                updated, detail = gpu_worker.drain_worker_submission_outbox(run, job)

            self.assertEqual(detail["forwarded"], 0)
            self.assertEqual(detail["retry_wait"], 1)
            self.assertIn("awaiting retry", updated["submission_bridge_error"])
            record = json.loads(
                (Path(raw) / "submission-bridge/123456-abcd.json").read_text()
            )
            self.assertEqual(record["state"], "retry_wait")
            self.assertIn("integer returncode", record["error"])
            acknowledge.assert_not_called()

    def test_terminal_bridge_requires_results_and_forwarded_staged_policies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job = self.run_and_job(root)
            job.update(
                {
                    "submission_paths": ["/app/policy.pt"],
                    "lease_id": "lease-1",
                }
            )
            detail = {"error": 0, "retry_wait": 0}
            attempt = {
                "attempt": 1,
                "lease_id": "lease-1",
                "status": "terminated",
                "progress": {
                    "submission_results": [
                        {"path": "/app/policy.pt", "state": "staged"}
                    ]
                },
            }
            with mock.patch.object(
                gpu_worker, "load_attempt_record", return_value=attempt
            ):
                self.assertFalse(
                    gpu_worker.terminal_submission_bridge_complete(run, job, detail)
                )
                bridge = root / "submission-bridge"
                bridge.mkdir(exist_ok=True)
                (bridge / "123456-abcd.json").write_text(
                    json.dumps(
                        {
                            "gpu_job_id": "job-1",
                            "state": "forwarded",
                        }
                    )
                )
                self.assertTrue(
                    gpu_worker.terminal_submission_bridge_complete(run, job, detail)
                )

                attempt["progress"]["submission_results"][0]["state"] = "rejected"
                (bridge / "123456-abcd.json").unlink()
                self.assertTrue(
                    gpu_worker.terminal_submission_bridge_complete(run, job, detail)
                )

    def test_budget_stop_forwards_existing_attempt_zero_submission_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job, content = self.enqueue_snapshot_job(root)
            persisted: list[dict] = []
            with (
                mock.patch.object(
                    gpu_worker,
                    "pin_work_archive",
                    wraps=gpu_worker.pin_work_archive,
                ) as pin,
                mock.patch.object(
                    gpu_worker,
                    "submit_worker_policy_to_cpu_agent",
                    return_value={"returncode": 0, "stdout": "staged", "stderr": ""},
                ) as submit,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: (
                        persisted.append(dict(payload)) or dict(payload)
                    ),
                ),
                mock.patch.object(
                    gpu_worker, "owned_terminal_attempt", return_value=None
                ),
            ):
                recovered, detail = gpu_worker.recover_pending_submission_snapshots(
                    run, job
                )

            self.assertEqual(detail["forwarded"], 1)
            self.assertEqual(
                recovered["progress"]["submission_results"][0]["state"], "staged"
            )
            self.assertFalse(recovered["submission_enqueue_snapshot_recovery_pending"])
            pin.assert_called_once()
            self.assertEqual(pin.call_args.args[0], run)
            self.assertTrue(
                pin.call_args.args[1]["submission_enqueue_snapshot_recovery_pending"]
            )
            self.assertEqual(pin.call_args.kwargs["transfer_attempts"], 2)
            self.assertEqual(pin.call_args.kwargs["transfer_timeout_seconds"], 45)
            submit.assert_called_once()
            self.assertEqual(submit.call_args.args[2], content)
            self.assertTrue(
                gpu_worker.terminal_submission_bridge_complete(
                    run, recovered, {"error": 0, "retry_wait": 0}
                )
            )
            self.assertTrue(persisted)

    def test_budget_stop_rejects_missing_attempt_zero_output_without_gpu_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job, _content = self.enqueue_snapshot_job(root, include_policy=False)
            with (
                mock.patch.object(
                    gpu_worker, "submit_worker_policy_to_cpu_agent"
                ) as submit,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: dict(payload),
                ),
                mock.patch.object(
                    gpu_worker, "owned_terminal_attempt", return_value=None
                ),
            ):
                recovered, detail = gpu_worker.recover_pending_submission_snapshots(
                    run, job
                )

            self.assertEqual(detail["rejected"], 1)
            self.assertEqual(
                recovered["progress"]["submission_results"][0]["state"], "rejected"
            )
            submit.assert_not_called()
            self.assertTrue(
                gpu_worker.terminal_submission_bridge_complete(
                    run, recovered, {"error": 0, "retry_wait": 0}
                )
            )

    def test_budget_stop_recovers_dispatched_submission_when_worker_manifest_is_lost(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job, content = self.enqueue_snapshot_job(root)
            job.update(
                {
                    "status": "terminated",
                    "attempt": 1,
                    "lease_id": "lease-1",
                    "sandbox_id": "sb-worker",
                    "started_at": "2026-08-27T06:12:08Z",
                    "termination_reason": "agent_cost_budget_exhausted",
                }
            )
            terminal_attempt = {
                "attempt": 1,
                "lease_id": "lease-1",
                "status": "terminated",
                "progress": None,
            }
            with (
                mock.patch.object(
                    gpu_worker,
                    "owned_terminal_attempt",
                    return_value=terminal_attempt,
                ),
                mock.patch.object(
                    gpu_worker,
                    "submit_worker_policy_to_cpu_agent",
                    return_value={"returncode": 0, "stdout": "staged", "stderr": ""},
                ) as submit,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: dict(payload),
                ),
            ):
                recovered, detail = gpu_worker.recover_pending_submission_snapshots(
                    run, job
                )

            self.assertEqual(detail["forwarded"], 1)
            self.assertEqual(submit.call_args.args[2], content)
            receipt = submit.call_args.args[1]
            self.assertEqual(receipt["gpu_attempt"], 1)
            self.assertEqual(receipt["gpu_lease_id"], "lease-1")
            self.assertTrue(
                gpu_worker.terminal_submission_bridge_complete(
                    run, recovered, {"error": 0, "retry_wait": 0}
                )
            )

    def test_budget_stop_does_not_replace_worker_submission_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job, _content = self.enqueue_snapshot_job(root)
            job.update(
                {
                    "status": "terminated",
                    "attempt": 1,
                    "lease_id": "lease-1",
                    "sandbox_id": "sb-worker",
                    "started_at": "2026-08-27T06:12:08Z",
                    "termination_reason": "agent_cost_budget_exhausted",
                }
            )
            terminal_attempt = {
                "attempt": 1,
                "lease_id": "lease-1",
                "status": "terminated",
                "progress": {
                    "submission_results": [
                        {"path": "/app/policy.pt", "state": "staged"}
                    ]
                },
            }
            with (
                mock.patch.object(
                    gpu_worker,
                    "owned_terminal_attempt",
                    return_value=terminal_attempt,
                ),
                mock.patch.object(
                    gpu_worker, "submit_worker_policy_to_cpu_agent"
                ) as submit,
            ):
                recovered, detail = gpu_worker.recover_pending_submission_snapshots(
                    run, job
                )

            self.assertFalse(detail["eligible"])
            self.assertEqual(recovered, job)
            submit.assert_not_called()

    def test_budget_stop_repairs_attempt_identity_on_forwarded_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job, content = self.enqueue_snapshot_job(root)
            job.update(
                {
                    "status": "terminated",
                    "attempt": 1,
                    "lease_id": "lease-1",
                    "sandbox_id": "sb-worker",
                    "started_at": "2026-08-27T06:12:08Z",
                    "termination_reason": "agent_cost_budget_exhausted",
                }
            )
            digest = hashlib.sha256(content).hexdigest()
            submission_id = gpu_worker._enqueue_snapshot_submission_id(
                run, job, "/app/policy.pt", digest
            )
            bridge = root / "submission-bridge"
            bridge.mkdir()
            record_path = bridge / f"{submission_id}.json"
            record_path.write_text(
                json.dumps(
                    {
                        "state": "forwarded",
                        "submission_id": submission_id,
                        "gpu_job_id": "job-1",
                        "gpu_attempt": 0,
                        "policy_sha256": digest,
                    }
                )
            )
            terminal_attempt = {
                "attempt": 1,
                "lease_id": "lease-1",
                "status": "terminated",
                "progress": None,
            }
            with (
                mock.patch.object(
                    gpu_worker,
                    "owned_terminal_attempt",
                    return_value=terminal_attempt,
                ),
                mock.patch.object(
                    gpu_worker, "submit_worker_policy_to_cpu_agent"
                ) as submit,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: dict(payload),
                ),
            ):
                recovered, detail = gpu_worker.recover_pending_submission_snapshots(
                    run, job
                )

            submit.assert_not_called()
            self.assertEqual(detail["forwarded"], 1)
            repaired = json.loads(record_path.read_text())
            self.assertEqual(repaired["gpu_attempt"], 1)
            self.assertEqual(repaired["gpu_lease_id"], "lease-1")
            self.assertEqual(
                recovered["progress"]["submission_results"][0]["state"], "staged"
            )

    def test_budget_stop_fails_closed_on_enqueue_archive_digest_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run, job, _content = self.enqueue_snapshot_job(root)
            job["submitted_work_archive_sha256"] = "0" * 64
            with (
                mock.patch.object(
                    gpu_worker, "submit_worker_policy_to_cpu_agent"
                ) as submit,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: dict(payload),
                ),
            ):
                recovered, detail = gpu_worker.recover_pending_submission_snapshots(
                    run, job
                )

            self.assertEqual(detail["retry_wait"], 1)
            self.assertTrue(recovered["submission_enqueue_snapshot_recovery_pending"])
            self.assertIn(
                "digest", recovered["submission_enqueue_snapshot_recovery_error"]
            )
            submit.assert_not_called()

    def test_terminal_submission_evidence_is_attached_without_undoing_fence(
        self,
    ) -> None:
        run = {"run_id": "run-1", "state_dir": "/tmp/unused"}
        job = {
            "job_id": "job-1",
            "attempt": 1,
            "lease_id": "lease-1",
            "status": "terminated",
            "termination_reason": "agent_cost_budget_exhausted",
            "submission_paths": ["/app/policy.pt"],
        }
        attempt = {
            "attempt": 1,
            "lease_id": "lease-1",
            "status": "succeeded",
            "progress": {
                "submission_results": [{"path": "/app/policy.pt", "state": "staged"}]
            },
        }
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
        ):
            attached = gpu_worker.attach_terminal_submission_evidence(run, job)

        self.assertEqual(attached["status"], "terminated")
        self.assertEqual(attached["termination_reason"], "agent_cost_budget_exhausted")
        self.assertEqual(
            attached["progress"]["submission_results"],
            attempt["progress"]["submission_results"],
        )


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
                ["runs/run-1/gpu-jobs/status/job-1.json"],
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
            self.assertEqual(gpu_worker.list_job_ids(run), ["job-1"])

    def test_exact_job_index_replaces_queue_and_status_directory_scans(self) -> None:
        run = {
            "run_id": "run-1",
            "volume_name": "run-volume",
        }
        index = json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "jobs": {
                    "queued": {},
                    "cancelled": {"cancel_state": "cancelled_before_dispatch"},
                },
            }
        )
        with (
            mock.patch.object(
                gpu_worker.sprintctl, "volume_get_text", return_value=index
            ) as reader,
            mock.patch.object(gpu_worker, "list_host_job_ids", return_value=["hosted"]),
        ):
            self.assertEqual(gpu_worker.list_job_ids(run), ["hosted", "queued"])
            self.assertEqual(
                gpu_worker.list_agent_cancelled_job_ids(run), ["cancelled"]
            )

        self.assertEqual(reader.call_count, 2)
        self.assertTrue(
            all(
                call.args[1].endswith("/gpu-jobs/index.json")
                for call in reader.call_args_list
            )
        )

    def test_index_embeds_atomic_unclaimed_job_and_cancel_request(self) -> None:
        request = {
            "schema_version": 1,
            "request_id": "a" * 32,
            "run_id": "run-1",
            "job_id": "queued",
            "reason": "agent_cancelled",
        }
        job = {
            "schema_version": 3,
            "run_id": "run-1",
            "job_id": "queued",
            "status": "pending",
        }
        index = json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "jobs": {
                    "queued": {
                        "job": job,
                        "cancel_state": "requested",
                        "cancel_request": request,
                    }
                },
            }
        )
        run = {
            "run_id": "run-1",
            "volume_name": "run-volume",
        }
        with mock.patch.object(
            gpu_worker.sprintctl, "volume_get_text", return_value=index
        ) as reader:
            self.assertEqual(gpu_worker.load_job(run, "queued"), job)
            self.assertEqual(
                gpu_worker.read_durable_agent_cancel_requests(run), [request]
            )
        self.assertEqual(reader.call_count, 2)
        self.assertTrue(
            all(
                call.args[1].endswith("/gpu-jobs/index.json")
                for call in reader.call_args_list
            )
        )

    def test_missing_index_is_empty(self) -> None:
        run = {
            "run_id": "run-1",
            "volume_name": "run-volume",
        }
        with (
            mock.patch.object(
                gpu_worker.sprintctl, "volume_get_text", return_value=None
            ),
            mock.patch.object(gpu_worker, "list_host_job_ids", return_value=[]),
        ):
            self.assertEqual(gpu_worker.list_job_ids(run), [])


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

    def test_exited_startup_bypasses_cold_start_grace(self) -> None:
        job = {
            **self.job,
            "status": "dispatched",
            "claimed_at_epoch_s": 195,
        }
        decision = gpu_claim.assess_worker_liveness(
            job,
            None,
            probe_state="exited",
            now=200,
            heartbeat_timeout_sec=30,
            startup_grace_sec=600,
            dead_grace_sec=10,
        )
        self.assertEqual(decision, "observe")

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


class SingleCpuTrialTests(unittest.TestCase):
    def test_success_records_one_complete_process_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(start_cpu_trial, "OPS", Path(raw)):
                code = start_cpu_trial.run_once(
                    "unit-single-success", ["/bin/sh", "-c", "exit 0"]
                )
            self.assertEqual(code, 0)
            state = Path(raw) / "unit-single-success"
            lifecycle = [
                json.loads(line)
                for line in (state / "telemetry/cpu_lifecycle.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in lifecycle],
                ["cpu_launch_started", "cpu_launch_exited"],
            )
            exit_record = json.loads((state / "CPU_TRIAL_EXIT.json").read_text())
            self.assertEqual(exit_record["attempt"], 1)
            self.assertEqual(exit_record["reason"], "agent_process_completed")
            self.assertFalse(exit_record["recoverable_in_place"])

    def test_nonzero_exit_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(start_cpu_trial, "OPS", Path(raw)):
                code = start_cpu_trial.run_once(
                    "unit-single-failure", ["/bin/sh", "-c", "exit 17"]
                )
            self.assertEqual(code, 17)
            state = Path(raw) / "unit-single-failure"
            lifecycle = (
                (state / "telemetry/cpu_lifecycle.jsonl").read_text().splitlines()
            )
            self.assertEqual(len(lifecycle), 2)
            exit_record = json.loads((state / "CPU_TRIAL_EXIT.json").read_text())
            self.assertEqual(exit_record["raw_exit_code"], 17)
            self.assertEqual(exit_record["reason"], "agent_process_failed")

    def test_existing_stop_marker_attributes_exit_to_requested_stop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw) / "unit-single-stop"
            state.mkdir(parents=True)
            (state / "STOP_REQUESTED.json").write_text('{"reason":"operator_stop"}\n')
            with mock.patch.object(start_cpu_trial, "OPS", Path(raw)):
                start_cpu_trial.run_once(
                    "unit-single-stop", ["/bin/sh", "-c", "exit 143"]
                )
            exit_record = json.loads((state / "CPU_TRIAL_EXIT.json").read_text())
            self.assertEqual(exit_record["reason"], "requested_stop")
            self.assertTrue(exit_record["stop_requested"])


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
    def test_fifteen_trials_have_independent_single_gpu_slots(self) -> None:
        jobs = {
            f"run-{index}": {
                "job_id": f"job-{index}",
                "status": "running",
            }
            for index in range(15)
        }

        def list_ids(run, **_kwargs):
            return [jobs[str(run["run_id"])]["job_id"]]

        def load(run, _job_id, **_kwargs):
            return jobs[str(run["run_id"])]

        with (
            mock.patch.object(gpu_worker, "list_job_ids", side_effect=list_ids),
            mock.patch.object(gpu_worker, "load_job", side_effect=load),
        ):
            active = [
                gpu_worker.active_training_job_ids({"run_id": f"run-{index}"})
                for index in range(15)
            ]

        self.assertTrue(all(len(job_ids) == 1 for job_ids in active))
        self.assertEqual(len({job_ids[0] for job_ids in active}), 15)
        self.assertEqual(gpu_worker.MAX_ACTIVE_TRAINING_JOBS_PER_RUN, 1)

    def test_six_idle_indexed_lanes_do_one_exact_index_read_and_no_lists(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            def load_run(run_id: str):
                state = root / run_id
                state.mkdir()
                return state, {
                    "run_id": run_id,
                    "state_dir": str(state),
                    "volume_name": f"volume-{run_id}",
                    "cpu_agent_gpu_worker": True,
                }

            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", side_effect=load_run
                ),
                mock.patch.object(
                    gpu_worker, "indexed_agent_jobs", return_value={}
                ) as index_read,
                mock.patch.object(
                    gpu_worker, "reconcile_live_agent_cancel_requests", return_value=[]
                ),
            ):
                results = [
                    gpu_worker.dispatch_once(f"run-{index}") for index in range(6)
                ]

        self.assertEqual(index_read.call_count, 6)
        self.assertTrue(all(result["pending"] == [] for result in results))

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

    def test_orphan_audit_treats_missing_lazy_training_app_as_empty(self) -> None:
        run = {"run_id": "unit", "training_app_name": "unit-training"}
        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=[]),
            mock.patch(
                "modal.App.lookup",
                side_effect=gpu_worker.modal.exception.NotFoundError(
                    "training app not created yet"
                ),
            ),
        ):
            actions = gpu_worker.cleanup_orphaned_training_sandboxes(run)

        self.assertEqual(actions, [])

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
                    side_effect=lambda _run, job_id, **_kwargs: dict(jobs[job_id]),
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

    def test_terminal_job_submission_bridge_is_drained_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            job = {
                "job_id": "done",
                "status": "succeeded",
                "attempt": 1,
                "submission_bridge_enabled": True,
                "provider_logs_archived_at": "now",
                "provider_terminal_error_checked_at": "now",
            }
            persisted: list[dict] = []
            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "load_run",
                    return_value=(state, run),
                ),
                mock.patch.object(
                    gpu_worker, "indexed_agent_jobs", return_value={"done": job}
                ),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
                mock.patch.object(
                    gpu_worker,
                    "load_job_from_index_snapshot",
                    return_value=dict(job),
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(
                        {**job, "submission_bridge_counts": {"forwarded": 4}},
                        {
                            "submission_bridge": "drained",
                            "observed": 4,
                            "forwarded": 4,
                            "retry_wait": 0,
                            "error": 0,
                        },
                    ),
                ) as drain,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: (
                        persisted.append(dict(payload)) or payload
                    ),
                ),
                mock.patch.object(
                    gpu_worker, "cleanup_orphaned_training_sandboxes", return_value=[]
                ),
                mock.patch.object(
                    gpu_worker, "reconcile_live_agent_cancel_requests", return_value=[]
                ),
                mock.patch.object(
                    gpu_worker, "reconcile_agent_cancelled_jobs", return_value=[]
                ),
                mock.patch.object(gpu_worker, "_candidate_job_ids", return_value=[]),
                mock.patch.object(
                    gpu_worker, "active_training_job_ids", return_value=[]
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=False
                ),
            ):
                result = gpu_worker.dispatch_once("unit")

        drain.assert_called_once()
        self.assertEqual(result["submission_bridge_pending_jobs"], [])
        self.assertTrue(persisted[-1]["submission_bridge_terminal_drained_at"])

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

    def test_cancel_during_spawn_corrects_worker_lost_preemption(self) -> None:
        job = {
            "job_id": "cancelled",
            "run_id": "unit",
            "status": "preempted",
            "attempt": 1,
            "failure_reason": "worker_lost",
            "retry_policy": "agent_decides_new_job",
            "provider_exit_observed_epoch_s": 200,
        }
        indexed = {
            "cancelled": {
                "cancel_state": "cancelled_before_dispatch",
                "job": {
                    "terminated_at": "1970-01-01T00:02:30Z",
                    "terminated_at_epoch_s": 150,
                },
            }
        }
        persisted: list[dict] = []
        with (
            mock.patch.object(
                gpu_worker,
                "list_agent_cancelled_job_ids",
                return_value=["cancelled"],
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
            result = gpu_worker.reconcile_agent_cancelled_jobs(
                {"run_id": "unit"}, indexed=indexed
            )

        self.assertEqual(
            result[0]["decision"], "agent_cancelled_during_spawn_reconciled"
        )
        self.assertEqual(persisted[0]["status"], "terminated")
        self.assertEqual(
            persisted[0]["termination_reason"], "agent_cancelled_during_spawn"
        )
        self.assertEqual(persisted[0]["terminated_at_epoch_s"], 150)
        self.assertNotIn("failure_reason", persisted[0])
        self.assertNotIn("retry_policy", persisted[0])

    def test_late_cancel_does_not_mask_real_worker_loss(self) -> None:
        job = {
            "job_id": "lost",
            "run_id": "unit",
            "status": "preempted",
            "failure_reason": "worker_lost",
            "provider_exit_observed_epoch_s": 100,
        }
        indexed = {
            "lost": {
                "cancel_state": "cancelled_before_dispatch",
                "job": {"terminated_at_epoch_s": 150},
            }
        }
        with (
            mock.patch.object(
                gpu_worker,
                "list_agent_cancelled_job_ids",
                return_value=["lost"],
            ),
            mock.patch.object(gpu_worker, "load_host_job", return_value=job),
            mock.patch.object(gpu_worker, "persist_job") as persist,
        ):
            result = gpu_worker.reconcile_agent_cancelled_jobs(
                {"run_id": "unit"}, indexed=indexed
            )

        self.assertEqual(result, [])
        persist.assert_not_called()

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

    def test_live_cancel_is_delivered_without_killing_worker_or_acknowledging(
        self,
    ) -> None:
        request = {
            "schema_version": 1,
            "request_id": "a" * 32,
            "run_id": "unit",
            "job_id": "running",
            "reason": "agent_cancelled",
        }
        job = {
            "job_id": "running",
            "run_id": "unit",
            "status": "running",
            "sandbox_id": "sb-live",
        }
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "unit", "state_dir": raw}
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_durable_agent_cancel_requests",
                    return_value=[],
                ),
                mock.patch.object(
                    gpu_worker,
                    "read_live_agent_cancel_requests",
                    return_value=[request],
                ),
                mock.patch.object(gpu_worker, "load_host_job", return_value=job),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    return_value=job,
                ),
                mock.patch.object(
                    gpu_worker, "deliver_agent_cancel_to_gpu_worker"
                ) as deliver,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(gpu_worker, "_terminate_job_locked") as terminate,
            ):
                result = gpu_worker.reconcile_live_agent_cancel_requests(run)

            self.assertEqual(result[0]["decision"], "agent_cancel_delivered")
            deliver.assert_called_once_with(mock.ANY, request)
            terminate.assert_not_called()
            self.assertFalse((Path(raw) / "control-acks" / f"{'a' * 32}.json").exists())
            events = [
                json.loads(line)
                for line in (Path(raw) / "control-events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["cancel_requested", "cancel_delivered"],
            )
            self.assertEqual([event["sequence"] for event in events], [1, 2])

    def test_live_cancel_is_acknowledged_after_worker_commits_terminal_record(
        self,
    ) -> None:
        request = {
            "schema_version": 1,
            "request_id": "d" * 32,
            "run_id": "unit",
            "job_id": "running",
            "reason": "agent_cancelled",
        }
        terminal = {
            "job_id": "running",
            "run_id": "unit",
            "status": "terminated",
            "termination_reason": "agent_cancelled",
            "progress": {
                "output_artifacts": [{"name": "policy.pt"}],
                "policy_path": "/durable/runs/unit/gpu-jobs/artifacts/running/policy.pt",
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "unit", "state_dir": raw}
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_durable_agent_cancel_requests",
                    return_value=[request],
                ),
                mock.patch.object(
                    gpu_worker, "read_live_agent_cancel_requests", return_value=[]
                ),
                mock.patch.object(gpu_worker, "load_host_job", return_value=terminal),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    return_value=terminal,
                ),
                mock.patch.object(
                    gpu_worker, "deliver_agent_cancel_to_gpu_worker"
                ) as deliver,
                mock.patch.object(gpu_worker, "_terminate_job_locked") as terminate,
            ):
                result = gpu_worker.reconcile_live_agent_cancel_requests(run)

            self.assertEqual(result[0]["decision"], "agent_cancel_already_terminal")
            deliver.assert_not_called()
            terminate.assert_not_called()
            ack = json.loads(
                (Path(raw) / "control-acks" / f"{'d' * 32}.json").read_text()
            )
            self.assertEqual(ack["outcome"], "already_terminal")
            self.assertEqual(ack["status"], "terminated")

    def test_durable_cancel_replays_when_cpu_control_channel_is_down(self) -> None:
        request = {
            "schema_version": 1,
            "request_id": "b" * 32,
            "run_id": "unit",
            "job_id": "running",
            "reason": "agent_cancelled",
        }
        job = {
            "job_id": "running",
            "run_id": "unit",
            "status": "running",
            "sandbox_id": "sb-live",
        }
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "unit", "state_dir": raw}
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_durable_agent_cancel_requests",
                    return_value=[request],
                ),
                mock.patch.object(
                    gpu_worker,
                    "read_live_agent_cancel_requests",
                    side_effect=RuntimeError("CPU sandbox restarting"),
                ),
                mock.patch.object(gpu_worker, "load_host_job", return_value=job),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    return_value=job,
                ),
                mock.patch.object(
                    gpu_worker, "deliver_agent_cancel_to_gpu_worker"
                ) as deliver,
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: payload,
                ),
            ):
                result = gpu_worker.reconcile_live_agent_cancel_requests(run)

            self.assertEqual(result[0]["decision"], "agent_cancel_delivered")
            deliver.assert_called_once()
            self.assertFalse((Path(raw) / "control-acks" / f"{'b' * 32}.json").exists())

    def test_unconfirmed_cancel_is_not_acknowledged(self) -> None:
        request = {
            "schema_version": 1,
            "request_id": "c" * 32,
            "run_id": "unit",
            "job_id": "running",
            "reason": "agent_cancelled",
        }
        job = {
            "job_id": "running",
            "run_id": "unit",
            "status": "running",
            "sandbox_id": "sb-live",
        }
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "unit", "state_dir": raw}
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_durable_agent_cancel_requests",
                    return_value=[request],
                ),
                mock.patch.object(
                    gpu_worker, "read_live_agent_cancel_requests", return_value=[]
                ),
                mock.patch.object(gpu_worker, "load_host_job", return_value=job),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    return_value=job,
                ),
                mock.patch.object(
                    gpu_worker,
                    "deliver_agent_cancel_to_gpu_worker",
                    side_effect=RuntimeError("provider control channel unavailable"),
                ),
                mock.patch.object(
                    gpu_worker,
                    "persist_job",
                    side_effect=lambda _run, payload: payload,
                ),
            ):
                result = gpu_worker.reconcile_live_agent_cancel_requests(run)

            self.assertEqual(result[0]["decision"], "cancel_retry_required")
            self.assertFalse((Path(raw) / "control-acks" / f"{'c' * 32}.json").exists())

    def test_cancel_force_terminates_only_after_cooperative_grace(self) -> None:
        request_id = "e" * 32
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "run_id": "unit",
            "job_id": "running",
            "reason": "agent_cancelled",
        }
        job = {
            "job_id": "running",
            "run_id": "unit",
            "status": "running",
            "sandbox_id": "sb-live",
            "cancel_request_id": request_id,
            "cancel_signal_delivered_at_epoch_s": 100.0,
            "cancel_force_after_epoch_s": 160.0,
        }
        terminated = {**job, "status": "terminated"}
        with tempfile.TemporaryDirectory() as raw:
            run = {"run_id": "unit", "state_dir": raw}
            with (
                mock.patch.object(
                    gpu_worker,
                    "read_durable_agent_cancel_requests",
                    return_value=[request],
                ),
                mock.patch.object(
                    gpu_worker, "read_live_agent_cancel_requests", return_value=[]
                ),
                mock.patch.object(gpu_worker, "load_host_job", return_value=job),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    return_value=job,
                ),
                mock.patch.object(
                    gpu_worker,
                    "_terminate_job_locked",
                    return_value=terminated,
                ) as terminate,
                mock.patch.object(gpu_worker.time, "time", return_value=161.0),
            ):
                result = gpu_worker.reconcile_live_agent_cancel_requests(run)

            terminate.assert_called_once_with(
                run, "running", reason="agent_cancelled_forced"
            )
            self.assertEqual(result[0]["decision"], "agent_cancel_forced_terminated")
            ack = json.loads(
                (Path(raw) / "control-acks" / f"{request_id}.json").read_text()
            )
            self.assertEqual(ack["outcome"], "forced_terminated")


class AgentCredentialBoundaryTests(unittest.TestCase):
    def test_agent_env_allows_only_model_auth_and_endpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agent.env"
            path.write_text(
                "OPENAI_API_KEY=secret\nOPENAI_BASE_URL=https://api.example.test\n"
            )
            names = validate_agent_env.validate(path, "OPENAI_API_KEY")
        self.assertEqual(
            names,
            {"OPENAI_API_KEY", "OPENAI_BASE_URL"},
        )

    def test_agent_env_rejects_modal_control_plane_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agent.env"
            path.write_text("OPENAI_API_KEY=secret\nMODAL_TOKEN_SECRET=escape\n")
            with self.assertRaisesRegex(ValueError, "cloud control-plane"):
                validate_agent_env.validate(path, "OPENAI_API_KEY")

    def test_launcher_records_fixed_resource_budget(self) -> None:
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        worker = (ROOT / "event_runtime/compute/worker.py").read_text()
        self.assertIn('"agent_cpu_instances": 1', launcher)
        self.assertIn('"training_max_concurrent_per_run": 1', launcher)
        self.assertNotIn("SPRINT_CPU_LAUNCH_ATTEMPT", launcher)
        self.assertGreaterEqual(
            launcher.count("KEEPALIVE_JSON=$(make_keepalive_json)"), 2
        )
        self.assertIn('"gpu_worker_gpus_per_job": 1', launcher)
        # Both the dry-run contract and durable run contract must record the
        # tier pinned in the Harbor agent arguments. The in-sandbox budget
        # watchdog reads the latter to price live OpenAI requests.
        self.assertEqual(launcher.count('"service_tier": ('), 2)
        self.assertIn(
            '"agent_cloud_control_plane_credentials_injected": False', launcher
        )
        self.assertIn("MAX_ACTIVE_TRAINING_JOBS_PER_RUN = 1", worker)
        self.assertIn('gpu="A10G"', worker)


class NetworkIsolationTests(unittest.TestCase):
    def test_agent_and_verifier_start_offline(self) -> None:
        import tomllib

        task = tomllib.loads(
            (ROOT / "events" / "g1-100-metres" / "task.toml").read_text()
        )
        self.assertEqual(task["environment"]["network_mode"], "no-network")
        self.assertEqual(task["verifier"]["environment"]["network_mode"], "no-network")

    def test_gpu_workers_block_all_network(self) -> None:
        worker = (ROOT / "event_runtime/compute/worker.py").read_text()
        self.assertGreaterEqual(worker.count("block_network=True"), 2)

    def test_gpu_workers_force_headless_isaac_runtime(self) -> None:
        worker = (ROOT / "event_runtime/compute/worker.py").read_text()
        self.assertGreaterEqual(worker.count('env={"HEADLESS": "1"}'), 2)

    def test_launcher_allows_only_one_audited_model_host(self) -> None:
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        self.assertIn('--allow-agent-host "$MODEL_API_HOST"', launcher)
        self.assertIn("openrouter.ai) ;;", launcher)
        self.assertNotIn("api.anthropic.com", launcher)
        self.assertNotIn("api.deepseek.com", launcher)
        self.assertNotIn("api.openai.com", launcher)
        self.assertNotIn("modal.com|", launcher)
        self.assertNotIn("modal.run|", launcher)

    def test_legacy_modal_endpoint_launcher_is_removed(self) -> None:
        self.assertFalse((ROOT / "runs" / "run-kimi-k3.sh").exists())

    def test_arbitrary_harbor_arguments_fail_closed(self) -> None:
        import subprocess

        result = subprocess.run(
            [
                str(ROOT / "event_runtime/control/launch.sh"),
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

    def test_single_cpu_policy_preserves_network_metadata(self) -> None:
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        self.assertIn('"cpu_execution_policy": "single_process_no_resume"', launcher)
        self.assertIn('"agent_network_policy": "model-api-only"', launcher)
        self.assertIn('"agent_allowed_host": model_api_host', launcher)
        self.assertIn('"gpu_worker_network_policy": "no-network"', launcher)
        self.assertIn('"verifier_network_policy": "no-network"', launcher)

    def test_unreviewed_endpoint_is_rejected_before_launch(self) -> None:
        import subprocess

        launcher = ROOT / "event_runtime/control/launch.sh"
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
        image_source = (ROOT / "event_runtime" / "image.py").read_text()
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        self.assertIn('CODEX_VERSION = "0.149.1"', image_source)
        self.assertIn("BAKED_CODEX_VERSION=0.149.1", launcher)


class RetryAndFencingTests(unittest.TestCase):
    def test_terminal_reconciliation_preserves_worker_termination_reason(self) -> None:
        job = {
            "job_id": "logical",
            "run_id": "unit",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease",
        }
        attempt = {
            "attempt": 1,
            "lease_id": "lease",
            "status": "terminated",
            "finished_at_epoch_s": 100,
            "exit_code": -6,
            "error": "budget telemetry unavailable",
            "termination_reason": "budget_telemetry_unavailable",
        }
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
            mock.patch.object(
                gpu_worker,
                "drain_worker_submission_outbox",
                side_effect=lambda _run, payload: (payload, {}),
            ),
            mock.patch.object(
                gpu_worker,
                "archive_provider_logs",
                side_effect=lambda _run, payload: (payload, {}),
            ),
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
            mock.patch.object(gpu_worker, "_timeline_event"),
        ):
            reconciled, detail = gpu_worker.reconcile_job(
                {"run_id": "unit"}, job, now=101
            )

        self.assertEqual(detail["decision"], "terminal")
        self.assertEqual(reconciled["status"], "terminated")
        self.assertEqual(
            reconciled["termination_reason"], "budget_telemetry_unavailable"
        )

    def test_exited_provider_closes_billing_before_terminal_volume_commit(
        self,
    ) -> None:
        job = {
            "job_id": "logical",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease",
            "sandbox_id": "sb",
            "claimed_at_epoch_s": 100,
        }
        attempt = {
            "attempt": 1,
            "lease_id": "lease",
            "status": "running",
            "started_at_epoch_s": 110,
        }
        heartbeat = {
            "attempt": 1,
            "lease_id": "lease",
            "updated_at_epoch_s": 195,
        }
        persisted: list[dict] = []
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=heartbeat),
            mock.patch.object(
                gpu_worker,
                "drain_worker_submission_outbox",
                side_effect=lambda _run, payload: (payload, {}),
            ),
            mock.patch.object(
                gpu_worker,
                "refresh_live_policy_mirror",
                side_effect=lambda _run, payload, _heartbeat: (payload, {}),
            ),
            mock.patch.object(
                gpu_worker,
                "refresh_live_provider_logs",
                side_effect=lambda _run, payload, now: (payload, {}),
            ),
            mock.patch.object(
                gpu_worker,
                "persist_job",
                side_effect=lambda _run, payload: (
                    persisted.append(dict(payload)) or payload
                ),
            ),
            mock.patch.object(gpu_worker, "_timeline_event") as timeline_event,
        ):
            out, detail = gpu_worker.reconcile_job(
                {"run_id": "unit"},
                job,
                now=200,
                probe_fn=lambda _job: ("exited", 0, None),
            )

        self.assertEqual(detail["decision"], "grace")
        self.assertEqual(out["provider_exit_observed_epoch_s"], 200)
        self.assertEqual(out["provider_exit_code"], 0)
        self.assertEqual(persisted[-1]["provider_exit_observed_epoch_s"], 200)
        self.assertEqual(
            timeline_event.call_args.kwargs["event"],
            "gpu_provider_exit_observed",
        )

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

    def test_zombie_claim_owner_is_dead(self) -> None:
        identity = gpu_claim.process_identity()
        self.assertIsNotNone(identity)
        current = dict(identity or {})
        current["state"] = "Z"
        with mock.patch.object(gpu_claim, "process_identity", return_value=current):
            self.assertEqual(gpu_claim.process_identity_state(identity), "dead")

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

    def test_gpu_terminality_audit_requires_every_registered_job(self) -> None:
        run = {"run_id": "unit"}
        with (
            mock.patch.object(
                gpu_worker, "list_job_ids", return_value=["done", "active"]
            ),
            mock.patch.object(
                gpu_worker,
                "load_job",
                side_effect=[
                    {"job_id": "done", "status": "succeeded"},
                    {"job_id": "active", "status": "running"},
                ],
            ),
        ):
            self.assertFalse(gpu_worker.all_jobs_terminal(run))

        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
            mock.patch.object(
                gpu_worker,
                "load_job",
                return_value={"job_id": "done", "status": "succeeded"},
            ),
        ):
            self.assertTrue(gpu_worker.all_jobs_terminal(run))

    def test_operator_stop_releases_cpu_only_after_terminal_bridge_proof(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            job = {
                "job_id": "done",
                "status": "terminated",
                "submission_bridge_enabled": True,
                "submission_paths": ["/app/policy.pt"],
            }
            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(state, run)
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=True
                ),
                mock.patch.object(gpu_worker, "_stop_all_locked", return_value=[]),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
                mock.patch.object(gpu_worker, "load_job", return_value=dict(job)),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "attach_terminal_submission_evidence",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(dict(job), {"error": 0, "retry_wait": 0}),
                ),
                mock.patch.object(
                    gpu_worker, "terminal_submission_bridge_complete", return_value=True
                ),
                mock.patch.object(gpu_worker, "persist_job"),
                mock.patch.object(
                    gpu_worker.sprintctl, "seal_host_submission_bridge_complete"
                ) as seal,
                mock.patch.object(
                    gpu_worker, "signal_gpu_submission_drain_complete"
                ) as release,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(result["submission_bridge_pending_jobs"], [])
        seal.assert_called_once()
        release.assert_called_once_with(run)

    def test_operator_stop_fences_gpu_before_recovering_queued_submission(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            pending = {
                "job_id": "queued",
                "status": "pending",
                "attempt": 0,
                "submission_bridge_enabled": True,
                "submission_paths": ["/app/policy.pt"],
            }
            terminal = {
                **pending,
                "status": "terminated",
                "progress": {
                    "submission_results": [
                        {"path": "/app/policy.pt", "state": "staged"}
                    ]
                },
                "submission_enqueue_snapshot_recovered_at": "now",
                "submission_enqueue_snapshot_recovery_pending": False,
            }
            calls: list[str] = []

            def recover(_run, payload):
                calls.append("recover")
                return dict(payload), {
                    "eligible": True,
                    "forwarded": 1,
                    "error": 0,
                    "retry_wait": 0,
                }

            def stop(_run, *, reason):
                self.assertEqual(reason, "operator_stop")
                calls.append("stop")
                return [terminal]

            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(state, run)
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=True
                ),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["queued"]),
                mock.patch.object(
                    gpu_worker,
                    "load_job",
                    side_effect=[dict(pending), dict(terminal), dict(terminal)],
                ),
                mock.patch.object(
                    gpu_worker,
                    "recover_pending_submission_snapshots",
                    side_effect=recover,
                ),
                mock.patch.object(gpu_worker, "_stop_all_locked", side_effect=stop),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "attach_terminal_submission_evidence",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(dict(terminal), {"error": 0, "retry_wait": 0}),
                ),
                mock.patch.object(
                    gpu_worker, "terminal_submission_bridge_complete", return_value=True
                ),
                mock.patch.object(gpu_worker, "persist_job"),
                mock.patch.object(
                    gpu_worker.sprintctl, "seal_host_submission_bridge_complete"
                ),
                mock.patch.object(
                    gpu_worker, "signal_gpu_submission_drain_complete"
                ) as release,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(calls, ["stop", "recover"])
        self.assertEqual(
            result["enqueue_snapshot_submission_recoveries"]["queued"]["forwarded"],
            1,
        )
        self.assertEqual(result["submission_bridge_pending_jobs"], [])
        release.assert_called_once_with(run)

    def test_operator_stop_keeps_post_recovery_submission_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            terminal = {
                "job_id": "done",
                "status": "terminated",
                "attempt": 1,
                "submission_bridge_enabled": True,
                "submission_paths": ["/app/policy.pt"],
            }
            recovered = {
                **terminal,
                "progress": {
                    "submission_results": [
                        {"path": "/app/policy.pt", "state": "staged"}
                    ]
                },
                "submission_enqueue_snapshot_recovered_at": "now",
                "submission_enqueue_snapshot_recovery_pending": False,
            }
            stale_reload = dict(terminal)

            def reconcile(_run, payload):
                self.assertEqual(payload, recovered)
                return dict(payload)

            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(state, run)
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=True
                ),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
                mock.patch.object(
                    gpu_worker,
                    "load_job",
                    side_effect=[dict(terminal), dict(terminal), stale_reload],
                ),
                mock.patch.object(gpu_worker, "_stop_all_locked", return_value=[]),
                mock.patch.object(
                    gpu_worker,
                    "recover_pending_submission_snapshots",
                    return_value=(dict(recovered), {"eligible": True, "forwarded": 1}),
                ),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    side_effect=reconcile,
                ),
                mock.patch.object(
                    gpu_worker,
                    "attach_terminal_submission_evidence",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(dict(recovered), {"error": 0, "retry_wait": 0}),
                ),
                mock.patch.object(
                    gpu_worker, "terminal_submission_bridge_complete", return_value=True
                ),
                mock.patch.object(gpu_worker, "persist_job"),
                mock.patch.object(
                    gpu_worker.sprintctl, "seal_host_submission_bridge_complete"
                ),
                mock.patch.object(gpu_worker, "signal_gpu_submission_drain_complete"),
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(result["submission_bridge_pending_jobs"], [])

    def test_operator_stop_without_submissions_releases_cpu_before_gpu_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            job = {
                "job_id": "done",
                "status": "terminated",
                "submission_bridge_enabled": True,
                "submission_paths": [],
            }
            calls: list[str] = []

            def release(_run):
                calls.append("release")

            def stop(_run, *, reason):
                self.assertEqual(reason, "operator_stop")
                calls.append("stop")
                return []

            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(state, run)
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=True
                ),
                mock.patch.object(gpu_worker, "_stop_all_locked", side_effect=stop),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
                mock.patch.object(gpu_worker, "load_job", return_value=dict(job)),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "attach_terminal_submission_evidence",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(dict(job), {"error": 0, "retry_wait": 0}),
                ),
                mock.patch.object(
                    gpu_worker, "terminal_submission_bridge_complete", return_value=True
                ),
                mock.patch.object(gpu_worker, "persist_job"),
                mock.patch.object(
                    gpu_worker.sprintctl, "seal_host_submission_bridge_complete"
                ),
                mock.patch.object(
                    gpu_worker,
                    "signal_gpu_submission_drain_complete",
                    side_effect=release,
                ),
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(result["submission_bridge_pending_jobs"], [])
        self.assertEqual(calls, ["release", "stop"])

    def test_operator_stop_does_not_resignal_after_cpu_ack(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            job = {
                "job_id": "done",
                "status": "terminated",
                "submission_bridge_enabled": True,
                "submission_paths": ["/app/policy.pt"],
            }
            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(state, run)
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=True
                ),
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "terminal_stop_acknowledged",
                    return_value=True,
                ),
                mock.patch.object(gpu_worker, "_stop_all_locked", return_value=[]),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
                mock.patch.object(gpu_worker, "load_job", return_value=dict(job)),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "attach_terminal_submission_evidence",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(dict(job), {"error": 0, "retry_wait": 0}),
                ),
                mock.patch.object(
                    gpu_worker, "terminal_submission_bridge_complete", return_value=True
                ),
                mock.patch.object(gpu_worker, "persist_job"),
                mock.patch.object(
                    gpu_worker.sprintctl, "seal_host_submission_bridge_complete"
                ) as seal,
                mock.patch.object(
                    gpu_worker, "signal_gpu_submission_drain_complete"
                ) as release,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(result["submission_bridge_pending_jobs"], [])
        seal.assert_called_once()
        release.assert_not_called()

    def test_operator_stop_keeps_cpu_alive_while_submission_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw)
            run = {
                "run_id": "unit",
                "state_dir": raw,
                "cpu_agent_gpu_worker": True,
            }
            job = {
                "job_id": "done",
                "status": "terminated",
                "submission_bridge_enabled": True,
                "submission_paths": ["/app/policy.pt"],
            }
            with (
                mock.patch.object(
                    gpu_worker.sprintctl, "load_run", return_value=(state, run)
                ),
                mock.patch.object(
                    gpu_worker, "operator_stop_requested", return_value=True
                ),
                mock.patch.object(gpu_worker, "_stop_all_locked", return_value=[]),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["done"]),
                mock.patch.object(gpu_worker, "load_job", return_value=dict(job)),
                mock.patch.object(
                    gpu_worker,
                    "reconcile_terminal_attempt_before_stop",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "attach_terminal_submission_evidence",
                    side_effect=lambda _run, payload: payload,
                ),
                mock.patch.object(
                    gpu_worker,
                    "drain_worker_submission_outbox",
                    return_value=(dict(job), {"error": 0, "retry_wait": 0}),
                ),
                mock.patch.object(
                    gpu_worker,
                    "terminal_submission_bridge_complete",
                    return_value=False,
                ),
                mock.patch.object(gpu_worker, "persist_job"),
                mock.patch.object(
                    gpu_worker, "signal_gpu_submission_drain_complete"
                ) as release,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(result["submission_bridge_pending_jobs"], ["done"])
        release.assert_not_called()

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

            def load_job(_run, job_id, **_kwargs):
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
                    gpu_worker, "persist_host_job", side_effect=persist_job
                ),
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
                mock.patch.object(gpu_worker, "_timeline_event") as timeline_event,
                mock.patch.object(
                    gpu_worker, "_terminate_sandbox", return_value=None
                ) as terminate,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(jobs["queued"]["status"], "terminated")
        self.assertEqual(jobs["queued"]["sandbox_id"], "sb-new")
        self.assertEqual(result["actions"][0]["action"], "stop_after_spawn")
        terminate.assert_called_once()
        self.assertTrue(
            any(
                call.kwargs.get("phase") == "gpu_sandbox_create"
                and call.kwargs.get("action") == "enter"
                and call.kwargs.get("lifecycle_boundary") == "before_sandbox_create"
                for call in timeline_event.call_args_list
            )
        )

    def test_agent_cancel_during_spawn_fences_new_sandbox(self) -> None:
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

            def load_job(_run, job_id, **_kwargs):
                return dict(jobs[job_id])

            def persist_job(_run, payload):
                jobs[str(payload["job_id"])] = dict(payload)
                return dict(payload)

            handle = mock.Mock(attempt_id="sb-cancelled")
            with (
                mock.patch.object(
                    gpu_worker.sprintctl,
                    "load_run",
                    return_value=(Path(raw), run),
                ),
                mock.patch.object(
                    gpu_worker,
                    "indexed_agent_jobs",
                    side_effect=[
                        {"queued": {"job": dict(jobs["queued"])}},
                        {
                            "queued": {
                                "job": dict(jobs["queued"]),
                                "cancel_state": "cancelled_before_dispatch",
                            }
                        },
                    ],
                ),
                mock.patch.object(gpu_worker, "list_job_ids", return_value=["queued"]),
                mock.patch.object(gpu_worker, "load_job", side_effect=load_job),
                mock.patch.object(gpu_worker, "persist_job", side_effect=persist_job),
                mock.patch.object(
                    gpu_worker, "persist_host_job", side_effect=persist_job
                ),
                mock.patch.object(
                    gpu_worker, "pin_work_archive", side_effect=lambda _run, job: job
                ),
                mock.patch.object(gpu_worker, "restore_pinned_work_archive"),
                mock.patch.object(
                    gpu_worker.ModalSandboxProvider,
                    "start",
                    return_value=handle,
                ),
                mock.patch.object(gpu_worker, "_close_attempt_timeline"),
                mock.patch.object(gpu_worker, "_timeline_event"),
                mock.patch.object(
                    gpu_worker, "_terminate_sandbox", return_value=None
                ) as terminate,
            ):
                result = gpu_worker.dispatch_once("unit")

        self.assertEqual(jobs["queued"]["status"], "terminated")
        self.assertEqual(
            jobs["queued"]["termination_reason"],
            "agent_cancelled_during_spawn",
        )
        self.assertEqual(jobs["queued"]["sandbox_id"], "sb-cancelled")
        self.assertEqual(result["actions"][0]["action"], "agent_cancelled_during_spawn")
        terminate.assert_called_once()

    def test_preemption_is_terminal_and_fences_lease(self) -> None:
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
                        out = gpu_worker.finalize_lost_job(
                            run,
                            job,
                            heartbeat=heartbeat,
                            exit_code=137,
                            reason="worker_lost",
                            now=200,
                        )
        self.assertEqual(out["job_id"], "logical")
        self.assertEqual(out["status"], "preempted")
        self.assertEqual(out["failure_reason"], "worker_lost")
        self.assertEqual(out["retry_policy"], "agent_decides_new_job")
        self.assertEqual(out["fenced_lease_id"], "old-lease")
        self.assertNotIn("lease_id", out)
        self.assertEqual(out["last_progress"], {"step": 7})

    def test_spawn_failure_is_terminal_without_hidden_retry(self) -> None:
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
                    out = gpu_worker.finalize_lost_job(
                        {"run_id": "unit"},
                        job,
                        heartbeat=None,
                        exit_code=137,
                        reason="spawn_failed",
                        now=200,
                    )
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["failure_reason"], "spawn_failed")

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
            self.assertFalse(worker_run.lease_owned(status, 1, "old", 0))
            self.assertTrue(worker_run.lease_owned(status, 2, "new", 0))

    def test_worker_waits_for_newer_lease_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            status = Path(raw) / "status.json"
            status.write_text(
                json.dumps(
                    {
                        "job_id": "logical",
                        "status": "pending",
                        "attempt": 0,
                        "fence_epoch": 0,
                    }
                )
            )

            def publish() -> None:
                time.sleep(0.03)
                status.write_text(
                    json.dumps(
                        {
                            "job_id": "logical",
                            "status": "claiming",
                            "attempt": 1,
                            "lease_id": "lease",
                            "fence_epoch": 1,
                        }
                    )
                )

            thread = threading.Thread(target=publish)
            thread.start()
            try:
                self.assertTrue(
                    worker_run.wait_for_lease(
                        status,
                        1,
                        "lease",
                        1,
                        timeout_sec=0.5,
                        poll_sec=0.01,
                    )
                )
            finally:
                thread.join()

    def test_worker_rejects_newer_fence_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            status = Path(raw) / "status.json"
            status.write_text(
                json.dumps(
                    {
                        "job_id": "logical",
                        "status": "claiming",
                        "attempt": 2,
                        "lease_id": "new",
                        "fence_epoch": 2,
                    }
                )
            )
            self.assertFalse(
                worker_run.wait_for_lease(
                    status,
                    1,
                    "old",
                    1,
                    timeout_sec=10,
                    poll_sec=0.01,
                )
            )

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
                        gpu_worker, "persist_host_job", side_effect=persist
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

    def test_stop_fences_queued_submission_before_archive_recovery(self) -> None:
        run = {"run_id": "unit"}
        job = {
            "job_id": "queued",
            "status": "pending",
            "attempt": 0,
            "submission_bridge_enabled": True,
            "submission_paths": ["/app/policy.pt"],
        }
        persisted: list[dict] = []
        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=["queued"]),
            mock.patch.object(gpu_worker, "load_job", return_value=job),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
            mock.patch.object(
                gpu_worker,
                "persist_host_job",
                side_effect=lambda _run, payload: (
                    persisted.append(dict(payload)) or payload
                ),
            ),
            mock.patch.object(gpu_worker, "_close_attempt_timeline"),
            mock.patch.object(gpu_worker, "_timeline_event"),
            mock.patch.object(gpu_worker, "_terminate_sandbox") as terminate,
            mock.patch.object(gpu_worker, "drain_worker_submission_outbox") as drain,
        ):
            stopped = gpu_worker._stop_all_locked(run)

        self.assertEqual(stopped[0]["status"], "terminated")
        self.assertTrue(persisted[0]["submission_enqueue_snapshot_recovery_pending"])
        drain.assert_not_called()
        terminate.assert_called_once_with(job)

    def test_stop_fences_all_jobs_before_parallel_provider_termination(self) -> None:
        run = {"run_id": "unit"}
        jobs = {
            job_id: {
                "job_id": job_id,
                "status": "running",
                "attempt": 1,
                "lease_id": f"lease-{job_id}",
                "sandbox_id": f"sb-{job_id}",
            }
            for job_id in ("one", "two")
        }
        fenced: list[str] = []
        rendezvous = threading.Barrier(2)

        def persist(_run, payload):
            fenced.append(payload["job_id"])
            return payload

        def terminate(_job):
            self.assertEqual(set(fenced), {"one", "two"})
            rendezvous.wait(timeout=2)
            return None

        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=list(jobs)),
            mock.patch.object(
                gpu_worker,
                "load_job",
                side_effect=lambda _run, job_id, indexed=None: jobs[job_id],
            ),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
            mock.patch.object(gpu_worker, "persist_host_job", side_effect=persist),
            mock.patch.object(gpu_worker, "_close_attempt_timeline"),
            mock.patch.object(gpu_worker, "_timeline_event"),
            mock.patch.object(gpu_worker, "_terminate_sandbox", side_effect=terminate),
        ):
            out = gpu_worker._stop_all_locked(run)

        self.assertEqual([item["status"] for item in out], ["terminated", "terminated"])

    def test_stop_queue_depth_does_not_use_remote_job_or_timeline_mirrors(self) -> None:
        run = {"run_id": "unit"}
        jobs = {
            str(index): {
                "job_id": str(index),
                "status": "pending",
                "attempt": 0,
            }
            for index in range(32)
        }
        persisted: list[str] = []
        timeline_uploads: list[bool] = []

        def timeline(_run, _job, **kwargs):
            timeline_uploads.append(kwargs["upload"])

        with (
            mock.patch.object(
                gpu_worker,
                "indexed_agent_jobs",
                return_value={job_id: {"job": job} for job_id, job in jobs.items()},
            ) as read_index,
            mock.patch.object(gpu_worker, "list_job_ids", return_value=list(jobs)),
            mock.patch.object(
                gpu_worker,
                "load_job",
                side_effect=lambda _run, job_id, indexed=None: jobs[job_id],
            ),
            mock.patch.object(gpu_worker, "load_heartbeat", return_value=None),
            mock.patch.object(
                gpu_worker,
                "persist_host_job",
                side_effect=lambda _run, payload: (
                    persisted.append(payload["job_id"]) or payload
                ),
            ),
            mock.patch.object(gpu_worker, "persist_job") as remote_persist,
            mock.patch.object(
                gpu_worker, "_close_attempt_timeline", side_effect=timeline
            ),
            mock.patch.object(gpu_worker, "_timeline_event", side_effect=timeline),
            mock.patch.object(gpu_worker, "_terminate_sandbox", return_value=None),
        ):
            stopped = gpu_worker._stop_all_locked(run)

        self.assertEqual(len(stopped), 32)
        self.assertEqual(persisted, list(jobs))
        self.assertEqual(timeline_uploads, [False] * 64)
        read_index.assert_called_once_with(run)
        remote_persist.assert_not_called()

    def test_stop_preserves_completed_attempt_and_does_not_terminate(self) -> None:
        run = {"run_id": "unit"}
        job = {
            "job_id": "logical",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease",
            "sandbox_id": "sb",
        }
        attempt = {
            "job_id": "logical",
            "status": "succeeded",
            "attempt": 1,
            "lease_id": "lease",
            "exit_code": 0,
            "finished_at_epoch_s": 100,
            "checkpoint": "/durable/model.pt",
        }
        persisted: list[dict] = []
        with (
            mock.patch.object(gpu_worker, "list_job_ids", return_value=["logical"]),
            mock.patch.object(gpu_worker, "load_job", return_value=job),
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(
                gpu_worker,
                "persist_job",
                side_effect=lambda _run, payload: (
                    persisted.append(dict(payload)) or payload
                ),
            ),
            mock.patch.object(gpu_worker, "_terminate_sandbox") as terminate,
        ):
            stopped = gpu_worker._stop_all_locked(run)
        self.assertEqual(stopped, [])
        self.assertEqual(persisted[-1]["status"], "succeeded")
        self.assertEqual(persisted[-1]["checkpoint"], "/durable/model.pt")
        terminate.assert_not_called()

    def test_stop_repairs_previously_overwritten_terminal_registry(self) -> None:
        run = {"run_id": "unit"}
        job = {
            "job_id": "logical",
            "status": "terminated",
            "attempt": 1,
            "lease_id": "lease",
            "terminated_at_epoch_s": 200,
        }
        attempt = {
            "job_id": "logical",
            "status": "succeeded",
            "attempt": 1,
            "lease_id": "lease",
            "finished_at_epoch_s": 100,
            "exit_code": 0,
        }
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
        ):
            repaired = gpu_worker.reconcile_terminal_attempt_before_stop(run, job)
        self.assertEqual(repaired["status"], "succeeded")
        self.assertEqual(repaired["finished_at_epoch_s"], 100)

    def test_stop_repair_preserves_worker_termination_reason(self) -> None:
        run = {"run_id": "unit"}
        job = {
            "job_id": "logical",
            "status": "running",
            "attempt": 1,
            "lease_id": "lease",
        }
        attempt = {
            "job_id": "logical",
            "status": "terminated",
            "attempt": 1,
            "lease_id": "lease",
            "finished_at_epoch_s": 100,
            "termination_reason": "agent_cost_budget_exhausted",
        }
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
        ):
            repaired = gpu_worker.reconcile_terminal_attempt_before_stop(run, job)
        self.assertEqual(repaired["status"], "terminated")
        self.assertEqual(repaired["termination_reason"], "agent_cost_budget_exhausted")

    def test_stop_repair_preserves_newer_host_submission_recovery(self) -> None:
        run = {"run_id": "unit"}
        recovered_results = [
            {
                "path": "/app/policy.pt",
                "state": "staged",
                "submission_id": "recovered",
            }
        ]
        job = {
            "job_id": "logical",
            "status": "terminated",
            "attempt": 1,
            "lease_id": "lease",
            "finished_at_epoch_s": 100,
            "progress": {"submission_results": recovered_results},
            "submission_enqueue_snapshot_recovered_at": "now",
        }
        attempt = {
            "job_id": "logical",
            "status": "terminated",
            "attempt": 1,
            "lease_id": "lease",
            "finished_at_epoch_s": 100,
            "termination_reason": "agent_cost_budget_exhausted",
            "progress": None,
        }
        with (
            mock.patch.object(gpu_worker, "load_attempt_record", return_value=attempt),
            mock.patch.object(
                gpu_worker, "persist_job", side_effect=lambda _run, payload: payload
            ),
        ):
            repaired = gpu_worker.reconcile_terminal_attempt_before_stop(run, job)
        self.assertEqual(repaired["progress"]["submission_results"], recovered_results)

    def test_reconcile_reports_lost_running_worker_as_preempted(self) -> None:
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
        self.assertEqual(detail["decision"], "preempted")
        self.assertEqual(out["status"], "preempted")
        self.assertEqual(out["retry_policy"], "agent_decides_new_job")


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

    def test_spawn_failure_retries_from_scratch_without_checkpoint(self) -> None:
        command = worker_run.build_attempt_command(
            {
                "command": ["python3", "train.py"],
                "retry_reason": "spawn_failed",
            },
            2,
            None,
        )
        self.assertEqual(command, ["python3", "train.py"])

    def test_stateless_evaluation_retries_without_checkpoint(self) -> None:
        command = worker_run.build_attempt_command(
            {
                "command": ["python3", "evaluate_policy.py"],
                "job_kind": "evaluate",
                "retry_reason": "graceful_preemption",
            },
            2,
            None,
        )
        self.assertEqual(command, ["python3", "evaluate_policy.py"])

    def test_declared_policy_output_is_committed_with_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "app"
            durable = root / "durable"
            workspace.mkdir()
            (workspace / "policy.pt").write_bytes(b"torchscript")
            records, missing = worker_run.collect_output_artifacts(
                {"output_paths": ["/app/policy.pt"]},
                run_id="run-1",
                job_id="job-1",
                attempt=2,
                durable_dir=durable,
                workspace_dir=workspace,
            )
        self.assertEqual(missing, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_path"], "/app/policy.pt")
        self.assertEqual(
            records[0]["sha256"], hashlib.sha256(b"torchscript").hexdigest()
        )


class AgentGpuCliTests(unittest.TestCase):
    def test_dispatch_index_preserves_multiple_submissions_and_cancellation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "gpu-jobs"
            job_a = {"run_id": "run-1", "job_id": "job-a", "status": "pending"}
            train_cli.update_dispatch_index(root, "run-1", "job-a", job=job_a)
            train_cli.update_dispatch_index(root, "run-1", "job-b")
            train_cli.update_dispatch_index(
                root,
                "run-1",
                "job-a",
                cancel_state="cancelled_before_dispatch",
            )
            index = json.loads((root / "index.json").read_text())

        self.assertEqual(index["schema_version"], 1)
        self.assertEqual(index["run_id"], "run-1")
        self.assertEqual(sorted(index["jobs"]), ["job-a", "job-b"])
        self.assertEqual(
            index["jobs"]["job-a"]["cancel_state"],
            "cancelled_before_dispatch",
        )
        self.assertEqual(index["generation"], 3)
        self.assertEqual(index["jobs"]["job-a"]["job"], job_a)

    def test_running_job_cancel_writes_scoped_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "gpu-jobs"
            control_root = Path(raw) / "control"
            (root / "status").mkdir(parents=True)
            (root / "status" / "job-1.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "running",
                        "sandbox_id": "sb-1",
                    }
                )
            )
            with (
                mock.patch.object(train_cli, "jobs_root", return_value=root),
                mock.patch.object(train_cli, "AGENT_CONTROL_ROOT", control_root),
                mock.patch.object(train_cli, "flush_durable"),
                mock.patch.dict(os.environ, {"SPRINT_RUN_ID": "run-1"}),
                mock.patch("sys.stdout", io.StringIO()),
            ):
                self.assertEqual(
                    train_cli.cmd_cancel(type("Args", (), {"job_id": "job-1"})()),
                    0,
                )

            request = json.loads((root / "cancel" / "job-1.json").read_text())
            self.assertEqual(request["run_id"], "run-1")
            self.assertEqual(request["job_id"], "job-1")
            self.assertEqual(request["reason"], "agent_cancelled")
            self.assertRegex(request["request_id"], r"^[0-9a-f]{32}$")
            self.assertEqual(
                json.loads((control_root / "cancel" / "job-1.json").read_text()),
                request,
            )
            index = json.loads((root / "index.json").read_text())
            self.assertEqual(index["run_id"], "run-1")
            self.assertEqual(index["jobs"]["job-1"]["cancel_state"], "requested")

    def test_worker_recognizes_scoped_running_job_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "runs/run-1/gpu-jobs/cancel/job-1.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run-1",
                        "job_id": "job-1",
                        "reason": "agent_cancelled",
                    }
                )
            )
            self.assertTrue(worker_run.job_cancel_requested("run-1", "job-1", raw))

    def test_worker_recognizes_private_runtime_cancel_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runtime_marker = Path(raw) / "cancel.json"
            runtime_marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run-1",
                        "job_id": "job-1",
                        "reason": "agent_cancelled",
                    }
                )
            )
            self.assertTrue(
                worker_run.job_cancel_requested(
                    "run-1",
                    "job-1",
                    str(Path(raw) / "missing-durable"),
                    runtime_marker=runtime_marker,
                )
            )

    def test_output_paths_are_scoped_and_have_unique_names(self) -> None:
        self.assertEqual(
            train_cli.validate_output_paths(["/app/results/policy.pt"]),
            ["/app/results/policy.pt"],
        )
        with self.assertRaisesRegex(SystemExit, "under /app"):
            train_cli.validate_output_paths(["/tmp/policy.pt"])
        with self.assertRaisesRegex(SystemExit, "basenames must be unique"):
            train_cli.validate_output_paths(["/app/a/p.pt", "/app/b/p.pt"])

    def test_submission_paths_require_explicit_torchscript_files(self) -> None:
        self.assertEqual(
            train_cli.validate_submission_paths(["/app/results/policy.pt"]),
            ["/app/results/policy.pt"],
        )
        with self.assertRaisesRegex(SystemExit, "TorchScript .pt"):
            train_cli.validate_submission_paths(["/app/results/checkpoint.pth"])

    def test_declared_submission_is_archived_when_worker_finalizes(self) -> None:
        completed = mock.Mock(returncode=0, stdout="staged locally abc\n", stderr="")
        with mock.patch.object(
            worker_run.subprocess, "run", return_value=completed
        ) as run:
            results = worker_run.submit_declared_policies(
                {"job_id": "job-1", "submission_paths": ["/app/policy.pt"]}
            )

        self.assertEqual(results[0]["state"], "staged")
        command = run.call_args.args[0]
        self.assertEqual(
            command[:3], ["/usr/local/bin/event", "archive", "/app/policy.pt"]
        )

    def test_terminal_status_requires_explicit_artifact_retrieval(self) -> None:
        payload = {
            "status": "succeeded",
            "output_paths": ["/app/results/policy.pt"],
            "agent_policy_mirror_path": (
                "/run/sprint-gpu-mirror/artifacts/job-1/policy.pt"
            ),
        }

        enriched = train_cli.status_with_artifact_retrieval(payload, "job-1")

        self.assertEqual(
            enriched["artifact_retrieval"]["command"],
            "event gpu get job-1 /app/results/policy.pt",
        )
        self.assertIn("may be stale", enriched["artifact_retrieval"]["warning"])
        self.assertNotIn("artifact_retrieval", payload)

    def test_terminal_status_identifies_artifact_mirror_sync_lag(self) -> None:
        payload = {
            "status": "succeeded",
            "output_paths": ["/app/policy.pt"],
        }

        enriched = train_cli.status_with_artifact_retrieval(payload, "job-1")

        self.assertEqual(enriched["artifact_retrieval"]["status"], "syncing")
        self.assertEqual(
            enriched["artifact_retrieval"]["retry_commands"],
            ["event gpu get job-1 /app/policy.pt"],
        )

    def test_non_policy_output_reports_artifact_mirror_syncing(self) -> None:
        payload = {
            "status": "succeeded",
            "output_paths": ["/app/diagnostics.json"],
            "progress": {
                "output_artifacts": [
                    {"source_path": "/app/diagnostics.json", "size_bytes": 42}
                ]
            },
        }

        enriched = train_cli.status_with_artifact_retrieval(payload, "job-1")

        self.assertEqual(enriched["artifact_retrieval"]["status"], "syncing")
        self.assertEqual(
            enriched["artifact_retrieval"]["retry_commands"],
            ["event gpu get job-1 /app/diagnostics.json"],
        )

    def test_cancelled_job_does_not_claim_missing_artifact_is_syncing(self) -> None:
        payload = {
            "status": "terminated",
            "termination_reason": "agent_cancelled_during_spawn",
            "output_paths": ["/app/diagnostics.json"],
        }

        enriched = train_cli.status_with_artifact_retrieval(payload, "job-1")

        self.assertNotIn("artifact_retrieval", enriched)

    def test_get_retrieves_non_policy_declared_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "gpu-jobs"
            mirror = Path(raw) / "mirror"
            source = mirror / "artifacts" / "job-1" / "diagnostics.json"
            source.parent.mkdir(parents=True)
            content = b'{"loss": 0.5}\n'
            source.write_bytes(content)
            destination = Path(raw) / "app" / "diagnostics.json"
            payload = {
                "job_id": "job-1",
                "status": "succeeded",
                "output_paths": [str(destination)],
                "agent_artifact_mirrors": {
                    str(destination): {
                        "mirror_path": str(source),
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                },
            }
            (root / "status").mkdir(parents=True)
            (root / "status" / "job-1.json").write_text(json.dumps(payload))

            with (
                mock.patch.object(train_cli, "jobs_root", return_value=root),
                mock.patch.object(train_cli, "AGENT_MIRROR_ROOT", mirror),
                mock.patch.object(train_cli, "AGENT_WORKSPACE_ROOT", Path(raw) / "app"),
            ):
                self.assertEqual(
                    train_cli.cmd_get(
                        type(
                            "Args",
                            (),
                            {"job_id": "job-1", "destination": str(destination)},
                        )()
                    ),
                    0,
                )

            self.assertEqual(destination.read_bytes(), content)

    def test_logs_end_with_artifact_retrieval_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "gpu-jobs"
            mirror = Path(raw) / "mirror"
            (root / "status").mkdir(parents=True)
            log = root / "out" / "job-1" / "attempt-1" / "worker.log"
            log.parent.mkdir(parents=True)
            log.write_text("training complete\n")
            (root / "status" / "job-1.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "succeeded",
                        "output_paths": ["/app/policy.pt"],
                        "agent_policy_mirror_path": (
                            f"{mirror}/artifacts/job-1/policy.pt"
                        ),
                    }
                )
            )
            stdout = io.StringIO()
            with (
                mock.patch.object(train_cli, "jobs_root", return_value=root),
                mock.patch.object(train_cli, "AGENT_MIRROR_ROOT", mirror),
                mock.patch.object(train_cli, "AGENT_WORKSPACE_ROOT", Path("/app")),
                mock.patch("sys.stdout", stdout),
            ):
                self.assertEqual(
                    train_cli.cmd_logs(type("Args", (), {"job_id": "job-1"})()),
                    0,
                )

            output = stdout.getvalue()
            self.assertIn("training complete", output)
            self.assertTrue(
                output.rstrip().endswith("event gpu get job-1 /app/policy.pt")
            )

    def test_get_reports_sync_lag_instead_of_missing_output_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "gpu-jobs"
            (root / "status").mkdir(parents=True)
            (root / "status" / "job-1.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "succeeded",
                        "output_paths": ["/app/policy.pt"],
                    }
                )
            )
            with mock.patch.object(train_cli, "jobs_root", return_value=root):
                with self.assertRaisesRegex(SystemExit, "still syncing"):
                    train_cli.cmd_get(
                        type(
                            "Args",
                            (),
                            {"job_id": "job-1", "destination": None},
                        )()
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
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        self.assertIn('--ek "modal_image_id=$AGENT_TRAINING_IMAGE_ID"', launcher)
        self.assertIn('--ek "verifier_image_id=$VERIFIER_IMAGE_ID"', launcher)

    def test_cpu_trial_is_fresh_and_non_resumable(self) -> None:
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        self.assertIn('"sprint_source_commit": sprint_source_commit', launcher)
        self.assertIn("CPU-agent resume is forbidden", launcher)
        self.assertNotIn("--supervised-launch", launcher)
        self.assertNotIn("RESUMING", launcher)
        self.assertNotIn("cpu_supervised", launcher)

    def test_all_model_launchers_use_non_restarting_trial_unit(self) -> None:
        for name in ("openai.sh", "deepseek_harness.sh"):
            text = (ROOT / "event_runtime/control/providers" / name).read_text()
            self.assertIn("start_trial.py", text)
            self.assertNotIn("--supervised-launch", text)
            self.assertNotIn("CPU_MAX_RESTARTS", text)
        deepseek_harness = (
            ROOT / "event_runtime/control/providers/deepseek_harness.sh"
        ).read_text()
        self.assertIn("--secret-env OPENROUTER_API_KEY", deepseek_harness)
        self.assertIn("--launch-env OPENROUTER_MODEL", deepseek_harness)
        self.assertIn('export OPENROUTER_MODEL="$MODEL"', deepseek_harness)
        self.assertIn(
            "--launch-env SPRINT_OPENROUTER_PROVIDER_ENDPOINT",
            deepseek_harness,
        )
        for name in ("run-opus.sh", "run-terra.sh", "run-lane.sh"):
            self.assertFalse((ROOT / "runs" / name).exists())
        starter = (ROOT / "event_runtime/control/start_trial.py").read_text()
        self.assertIn("--property=Restart=no", starter)
        self.assertNotIn("RestartPreventExitStatus", starter)

    def test_cpu_sandbox_has_full_modal_lifetime_and_no_gpu(self) -> None:
        launcher = (ROOT / "event_runtime/control/launch.sh").read_text()
        self.assertIn("SANDBOX_TIMEOUT_SECONDS=86400", launcher)
        self.assertIn("sandbox_timeout_secs=$SANDBOX_TIMEOUT_SECONDS", launcher)
        task = (ROOT / "events" / "g1-100-metres" / "task.toml").read_text()
        self.assertIn("gpus = 0", task)
        self.assertIn("cpus = 2", task)
        self.assertIn("memory_mb = 8192", task)
        verifier = task.split("[verifier.environment]", 1)[1].split("[environment]", 1)[
            0
        ]
        self.assertIn("cpus = 4", verifier)
        self.assertIn("memory_mb = 10240", verifier)

    def test_trial_starter_uses_non_restarting_systemd_unit_without_secret_in_argv(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            launch = [
                str(ROOT / "event_runtime/control/launch.sh"),
                "--run-id",
                "unit-run",
            ]
            argv = [
                "start_trial.py",
                "--run-id",
                "unit-run",
                "--launch-argv-json",
                json.dumps(launch),
                "--secret-env",
                "OPENROUTER_API_KEY",
                "--launch-env",
                "SPRINT_OPENROUTER_PROVIDER_ENDPOINT",
                "--batch-id",
                "eval-batch",
            ]
            completed = mock.Mock(returncode=0)
            with (
                mock.patch.object(start_cpu_trial, "OPS", Path(raw)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(
                    "os.environ",
                    {
                        "OPENROUTER_API_KEY": "openrouter-secret-value",
                        "SPRINT_OPENROUTER_PROVIDER_ENDPOINT": "baidu/fp8",
                        "UV": "/test/bin/uv",
                        "PATH": "/usr/bin",
                    },
                    clear=True,
                ),
                mock.patch.object(Path, "is_file", return_value=True),
                mock.patch("os.access", return_value=True),
                mock.patch.object(start_cpu_trial.shutil, "which", return_value=None),
                mock.patch.object(
                    start_cpu_trial.subprocess, "run", return_value=completed
                ) as run,
            ):
                self.assertEqual(start_cpu_trial.main(), 0)
            command = run.call_args.args[0]
            self.assertIn("--property=Restart=no", command)
            self.assertIn("--property=KillMode=control-group", command)
            self.assertIn("--setenv=OPENROUTER_API_KEY", command)
            self.assertIn("--setenv=SPRINT_OPENROUTER_PROVIDER_ENDPOINT", command)
            self.assertIn("--setenv=UV=/test/bin/uv", command)
            self.assertIn(
                f"--setenv=PATH={ROOT / 'harbor/.venv/bin'}:/test/bin:/usr/bin",
                command,
            )
            self.assertIn("--setenv=SPRINT_BATCH_ID=eval-batch", command)
            self.assertNotIn("openrouter-secret-value", command)
            self.assertNotIn("baidu/fp8", command)
            metadata = json.loads(
                (Path(raw) / "unit-run" / "trial-launch.json").read_text()
            )
            self.assertEqual(metadata["launch_argv"], launch)
            self.assertEqual(metadata["batch_id"], "eval-batch")
            self.assertEqual(
                metadata["launch_env_names"],
                ["SPRINT_OPENROUTER_PROVIDER_ENDPOINT"],
            )
            self.assertEqual(
                metadata["controller_python"],
                str(ROOT / "harbor/.venv/bin/python3"),
            )
            self.assertEqual(metadata["process_manager_restart"], "no")
            self.assertEqual(
                metadata["cpu_execution_policy"], "single_process_no_resume"
            )

    def test_goal_template_is_launchable(self) -> None:
        live = (ROOT / "event_runtime/control/templates/codex.j2").read_text()
        self.assertIn("{{ instruction }}", live)
        self.assertNotIn("event history", live)
        self.assertNotIn("event wait", live)


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


class ModalProviderTerminationTests(unittest.TestCase):
    def test_termination_waits_for_provider_acknowledgement(self) -> None:
        sandbox = mock.Mock()
        sandbox.poll.return_value = 0
        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=sandbox
        ):
            error = gpu_worker.ModalSandboxProvider({}).terminate(
                resilience.ProviderHandle(
                    provider="modal-sandbox", attempt_id="sb-test"
                )
            )
        self.assertIsNone(error)
        sandbox.terminate.assert_called_once_with(wait=False)
        sandbox.poll.assert_called_once_with()

    def test_termination_confirmation_is_bounded(self) -> None:
        sandbox = mock.Mock()
        sandbox.poll.return_value = None
        with (
            mock.patch.object(
                gpu_worker.modal.Sandbox, "from_id", return_value=sandbox
            ),
            mock.patch.object(gpu_worker.time, "monotonic", side_effect=[0.0, 31.0]),
        ):
            error = gpu_worker.ModalSandboxProvider({}).terminate(
                resilience.ProviderHandle(
                    provider="modal-sandbox", attempt_id="sb-test"
                )
            )
        self.assertIn("did not confirm", str(error))
        sandbox.terminate.assert_called_once_with(wait=False)

    def test_already_stopped_termination_is_idempotent(self) -> None:
        sandbox = mock.Mock()
        sandbox.terminate.side_effect = RuntimeError("container is not running")
        with mock.patch.object(
            gpu_worker.modal.Sandbox, "from_id", return_value=sandbox
        ):
            error = gpu_worker.ModalSandboxProvider({}).terminate(
                resilience.ProviderHandle(
                    provider="modal-sandbox", attempt_id="sb-test"
                )
            )
        self.assertIsNone(error)
