"""Verify submissions while the agent is still working.

The default lifecycle is agent, then verifier, once.  That fits a task whose
answer is the final state of the filesystem.  It fits badly when the work is
iterative and each attempt is expensive to evaluate: the agent gets no signal
until it can no longer act on it, and whatever interim check it builds for
itself is its own approximation of the grader rather than the grader.

This runs alongside the agent. A submission appears in the watch directory and
gets verified in a container of its own. Tasks may opt into blind operation, in
which results remain only in the trusted archive until the agent phase ends.
The agent never blocks: submitting is a local file write.

Three properties this has to preserve, in order of how badly it breaks if it
does not:

1. **Isolation.**  Every verification goes through the same separate-container
   path the final verifier uses.  The only thing crossing from the agent is the
   submitted file, placed at ``submission_path``; everything else the verifier
   needs comes from its own image.
2. **Completeness.**  A submission is only picked up once its size has settled,
   so a file still being written is not scored half-formed.
3. **Provenance.**  Both the submitted bytes and the result are archived under
   the trial's artifacts, so a run leaves behind the whole sequence of attempts
   and any one of them can be re-scored later.
"""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from harbor.models.task.config import ContinuousVerificationConfig, TaskOS
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import (
    ContinuousSubmission,
    ContinuousVerificationSummary,
)
from harbor.models.verifier.result import VerifierResult
from harbor.utils.scripts import quote_shell_arg

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

# (key, per-attempt paths) -> verifier result.  ``paths.artifacts_dir`` holds
# the staged submission, mirroring the verifier container's filesystem, so the
# caller re-materializes it with the same artifact upload the final verifier
# uses; ``paths.verifier_dir`` is where that verification writes its own output.
RunVerifier = Callable[[str, TrialPaths], Awaitable[VerifierResult]]
RetryableVerifierError = Callable[[BaseException], bool]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _append_scheduler_event(raw_path: str, event: str, payload: dict[str, Any]) -> None:
    path = Path(raw_path).with_name("scheduler-events.jsonl")
    record = {"schema_version": 1, "at": _now().isoformat(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.asynccontextmanager
async def shared_verifier_slot(raw_path: str | None, *, event_payload: dict[str, Any]):
    """Acquire an optional crash-safe, cross-process verifier lease."""
    if raw_path is None:
        yield
        return

    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                await asyncio.sleep(0.1)
        _append_scheduler_event(raw_path, "acquired", event_payload)
        yield
    finally:
        if acquired:
            _append_scheduler_event(raw_path, "released", event_payload)
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class ContinuousVerificationService:
    """Polls the agent environment for submissions and verifies them."""

    def __init__(
        self,
        *,
        config: ContinuousVerificationConfig,
        environment: "BaseEnvironment",
        artifacts_dir: Path,
        run_verifier: RunVerifier,
        verification_timeout_sec: float,
        verification_context: str,
        queue_key: str,
        logger: logging.Logger,
        retryable_verifier_error: RetryableVerifierError | None = None,
    ) -> None:
        self._config = config
        self._env = environment
        self._run_verifier = run_verifier
        # Legacy feedback mode bounds the drain to one verification's worth.
        # Blind mode can instead drain every accepted submission after the
        # agent exits because queue latency cannot influence agent behavior.
        self._verification_timeout_sec = verification_timeout_sec
        self._verification_context = verification_context
        self._queue_key = queue_key
        self._logger = logger
        self._retryable_verifier_error = retryable_verifier_error

        self._quoted_watch_dir = quote_shell_arg(config.watch_dir, TaskOS.LINUX)
        self._artifacts_dir = artifacts_dir
        self._root = artifacts_dir / config.artifact_name
        self._ledger = self._root / "ledger.jsonl"

        self._summary = ContinuousVerificationSummary()
        self._seen: set[str] = set()
        self._sizes: dict[str, int] = {}
        self._accepted = 0
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._running: set[asyncio.Task[Any]] = set()
        self._draining = False
        self._watcher: asyncio.Task[Any] | None = None

    @property
    def summary(self) -> ContinuousVerificationSummary:
        return self._summary

    # -- lifecycle -----------------------------------------------------------

    @contextlib.asynccontextmanager
    async def running(self):
        """Watch for submissions for the duration of the block."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def start(self) -> None:
        # Windows is refused rather than half-supported: the watcher is a POSIX
        # `find`, and a task author would otherwise discover the gap only when
        # submissions silently never appeared.
        if self._env.os != TaskOS.LINUX:
            raise ValueError(
                "[verifier.continuous] supports Linux environments only, "
                f"got {self._env.os.value}."
            )

        self._root.mkdir(parents=True, exist_ok=True)
        directories = [self._quoted_watch_dir]
        if self._config.return_results_to_agent:
            directories.append(quote_shell_arg(self._config.results_dir, TaskOS.LINUX))
        await self._env.exec(f"mkdir -p {' '.join(directories)}")
        self._watcher = asyncio.create_task(self._watch())
        self._logger.debug(f"Continuous verification watching {self._config.watch_dir}")

    async def stop(self) -> None:
        """Stop watching and finish according to the configured drain policy.

        A submission being scored when the agent stopped is as legitimate as one
        from an hour earlier, and abandoning it would make the deadline part of
        the score. A submission still *queued* is different: verifications are
        serialized and each takes minutes, so waiting for a backlog would hold
        the trial open for hours after the agent had nothing left to do. An
        agent that writes its whole queue at the end would otherwise stall the
        run rather than lose a result.

        Blind evaluation drains all accepted submissions for a complete
        retrospective trajectory. Feedback mode retains the bounded legacy
        drain, recording any unscored queue entries explicitly.
        """
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None

        # One last pass, so a submission written between the final poll and the
        # agent exiting is not lost to timing.
        with contextlib.suppress(Exception):
            await self._poll(settle=False)

        self._draining = True
        if self._tasks:
            self._logger.info(
                f"Draining {len(self._tasks)} continuous verification(s), "
                f"up to {self._verification_timeout_sec:.0f}s"
            )
        pending = asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self._config.drain_pending_on_stop:
            await pending
            self._write_ledger()
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(pending), timeout=self._verification_timeout_sec
            )
        except asyncio.TimeoutError:
            # Whatever is left is a backlog, not a buzzer-beater. Verifications
            # are serialized and each takes minutes, so waiting it out would
            # hold the trial open for hours after the agent had nothing left to
            # do; an agent that dumps its queue at the end would stall the run
            # rather than lose a result.
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._write_ledger()

    # -- watching ------------------------------------------------------------

    async def _watch(self) -> None:
        while True:
            try:
                await self._poll()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # A polling failure must not take down the agent phase: the
                # trial is still valid without continuous feedback.
                self._logger.warning(f"Continuous verification poll failed: {exc}")
            await asyncio.sleep(self._config.poll_interval_sec)

    async def _poll(self, *, settle: bool = True) -> None:
        entries = await self._list_watch_dir()
        for name, size in sorted(entries.items()):
            if name in self._seen:
                continue
            # A dotfile is the conventional "not finished yet" marker, and it is
            # what lets an agent write-then-rename into the watch dir: without
            # this, a slow copy could settle under its temporary name and be
            # scored twice, once as `.job.pt` and once as `job.pt`.
            if name.startswith("."):
                continue
            # Only take a file whose size has stopped changing.  The agent
            # submits with an ordinary file write, and catching one mid-flight
            # would score a truncated archive.  On the final pass there is no
            # next poll to wait for, so take it as it stands.
            if settle and self._sizes.get(name) != size:
                self._sizes[name] = size
                continue
            self._seen.add(name)
            self._sizes.pop(name, None)
            if self._draining:
                continue
            self._spawn(name)

    async def _list_watch_dir(self) -> dict[str, int]:
        result = await self._env.exec(
            f"find {self._quoted_watch_dir} -maxdepth 1 -type f "
            "-printf '%f\\t%s\\n' 2>/dev/null || true"
        )
        entries: dict[str, int] = {}
        for line in (result.stdout or "").splitlines():
            name, tab, size = line.partition("\t")
            if not tab:
                continue
            with contextlib.suppress(ValueError):
                entries[name] = int(size)
        return entries

    def _spawn(self, name: str) -> None:
        task = asyncio.create_task(self._verify(name))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- verifying -----------------------------------------------------------

    def _attempt_paths(self, index: int, name: str) -> TrialPaths:
        """A directory of its own for one attempt.

        ``artifacts/`` under it is what gets uploaded: Harbor places a host file
        at its original absolute source path inside the target container, so
        mirroring ``submission_path`` there puts the submission exactly where
        the verifier expects it and nothing else alongside it.  ``verifier/`` is
        where that verification writes its reward file and stdout, kept apart
        from every other attempt and from the trial's own final verification.
        """
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
        return TrialPaths(trial_dir=self._root / "attempts" / f"{index:04d}-{safe}")

    async def _verify(self, name: str) -> None:
        self._accepted += 1
        index = self._accepted
        record = ContinuousSubmission(
            name=name,
            index=index,
            submitted_at=_now(),
            queue_key=self._queue_key,
        )
        self._summary.submissions.append(record)

        cap = self._config.max_submissions
        if cap is not None and index > cap:
            # Always retained in the trusted ledger. Feedback mode also hands
            # the rejection back; blind mode intentionally exposes no status.
            record.error = f"submission limit of {cap} reached"
            await self._publish(name, record)
            self._logger.info(
                f"Continuous verification rejected {name}: {record.error}"
            )
            return

        try:
            await self._semaphore.acquire()
        except asyncio.CancelledError:
            # Cancelled while queued: the drain budget ran out before this one's
            # turn came round.
            record.error = "not scored: still queued when the agent phase ended"
            self._write_ledger()
            self._logger.info(f"Continuous verification dropped {name}: queued at stop")
            raise

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("continuous verification has no current asyncio task")
        self._running.add(current_task)
        try:
            record.started_at = _now()
            paths = self._attempt_paths(index, name)
            # Not paths.mkdir(): an attempt has no agent of its own, and an
            # empty agent/ dir in every archive is misleading.
            paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
            relative = PurePosixPath(self._config.submission_path).relative_to("/")
            local = paths.artifacts_dir / Path(*relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Pull the bytes out first.  The agent keeps working and may
                # overwrite or delete its own file while this runs; the copy
                # under artifacts is what gets scored and what gets archived, so
                # the record and the score always describe the same bytes.
                await self._env.download_file(f"{self._config.watch_dir}/{name}", local)
                record.artifact_path = str(local.relative_to(self._artifacts_dir))
                record.artifact_sha256 = hashlib.sha256(local.read_bytes()).hexdigest()
                record.evaluation_fingerprint = hashlib.sha256(
                    (
                        self._verification_context + "\0" + record.artifact_sha256
                    ).encode()
                ).hexdigest()
                self._logger.info(
                    f"Continuous verification {index}: {name} "
                    f"({local.stat().st_size} bytes)"
                )
                wait_started = _now()
                async with self._shared_verifier_slot(record):
                    record.scheduler_acquired_at = _now()
                    record.scheduler_wait_sec = round(
                        (record.scheduler_acquired_at - wait_started).total_seconds(), 3
                    )
                    cached = self._read_cached_result(record)
                    if cached is not None:
                        record.rewards = cached["rewards"]
                        record.cache_hit = True
                        record.source_evaluation_id = cached["evaluation_id"]
                    else:
                        record.verification_started_at = _now()
                        result = await self._run_verifier_with_retries(
                            f"continuous-{index:04d}", paths, record
                        )
                        record.rewards = result.rewards or {}
                        record.source_evaluation_id = self._evaluation_id(record)
                        self._write_cached_result(record)
            except asyncio.CancelledError:
                # Cancelled mid-verification: the drain budget ran out while
                # this one was running.  Recorded rather than left as a row that
                # simply stops, which reads like the service lost it.
                record.error = "not scored: cut short when the agent phase ended"
                record.finished_at = _now()
                self._write_ledger()
                raise
            except asyncio.TimeoutError:
                record.error = (
                    f"verification exceeded {self._config.timeout_sec} s"
                    if self._config.timeout_sec
                    else "verification timed out"
                )
            except Exception as exc:  # noqa: BLE001
                record.error = f"{type(exc).__name__}: {exc}"

            record.finished_at = _now()
            record.duration_sec = round(
                (record.finished_at - record.started_at).total_seconds(), 2
            )
        finally:
            self._running.discard(current_task)
            self._semaphore.release()

        await self._publish(name, record)
        self._logger.info(
            f"Continuous verification {index}: {name} -> "
            + (record.error or f"reward {record.reward}")
        )

    async def _run_verifier_with_retries(
        self,
        key: str,
        paths: TrialPaths,
        record: ContinuousSubmission,
    ) -> VerifierResult:
        """Retry only provider-classified loss, always in a fresh sandbox."""
        max_attempts = self._config.max_verifier_attempts
        for attempt in range(1, max_attempts + 1):
            record.verification_attempts = attempt
            attempt_key = key if attempt == 1 else f"{key}-retry-{attempt}"
            try:
                return await self._run_verifier(attempt_key, paths)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                retryable = bool(
                    self._retryable_verifier_error
                    and self._retryable_verifier_error(exc)
                )
                if not retryable or attempt >= max_attempts:
                    raise
                detail = f"{type(exc).__name__}: {exc}"
                record.verification_retry_events.append(
                    {
                        "attempt": attempt,
                        "failed_at": _now().isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                self._write_ledger()
                self._logger.warning(
                    "Continuous verifier infrastructure loss for %s "
                    "(attempt %d/%d): %s; retrying in a fresh sandbox",
                    record.name,
                    attempt,
                    max_attempts,
                    detail,
                )
                delay = self._config.verifier_retry_backoff_sec * (2 ** (attempt - 1))
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable verifier retry loop")

    # -- reporting -----------------------------------------------------------

    async def _publish(self, name: str, record: ContinuousSubmission) -> None:
        """Archive the result and optionally hand a copy back to the agent."""
        local = self._attempt_paths(record.index, name).trial_dir / "result.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        # Set before serializing, so the copy the agent reads carries the same
        # fields as the one in the ledger.
        record.result_path = str(local.relative_to(self._artifacts_dir))
        local.write_text(record.model_dump_json(indent=2) + "\n")
        self._write_ledger()

        if not self._config.return_results_to_agent:
            return
        try:
            await self._env.upload_file(
                local, f"{self._config.results_dir}/{name}.json"
            )
        except Exception as exc:  # noqa: BLE001
            # The archived copy is the record; failing to hand it back costs the
            # agent its feedback but not the result.
            self._logger.warning(f"Could not return result for {name}: {exc}")

    def _evaluation_id(self, record: ContinuousSubmission) -> str:
        fingerprint = record.evaluation_fingerprint or "unknown"
        return f"{self._queue_key}:{record.index}:{fingerprint[:16]}"

    @contextlib.asynccontextmanager
    async def _shared_verifier_slot(self, record: ContinuousSubmission):
        """Acquire the optional host-wide verifier lease without blocking asyncio."""
        raw_path = self._config.shared_scheduler_lock_path
        async with shared_verifier_slot(
            raw_path,
            event_payload={
                "queue_key": self._queue_key,
                "submission_index": record.index,
                "submission_name": record.name,
                "evaluation_fingerprint": record.evaluation_fingerprint,
            },
        ):
            yield

    @staticmethod
    def _cache_checksum(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _cache_path(self, record: ContinuousSubmission) -> Path | None:
        root = self._config.shared_result_cache_dir
        fingerprint = record.evaluation_fingerprint
        if root is None or fingerprint is None:
            return None
        return Path(root) / fingerprint[:2] / f"{fingerprint}.json"

    def _read_cached_result(
        self, record: ContinuousSubmission
    ) -> dict[str, Any] | None:
        path = self._cache_path(record)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        checksum = payload.pop("checksum", None)
        if checksum != self._cache_checksum(payload):
            return None
        if (
            payload.get("schema_version") != 1
            or payload.get("evaluation_fingerprint") != record.evaluation_fingerprint
            or payload.get("artifact_sha256") != record.artifact_sha256
            or payload.get("verification_context") != self._verification_context
            or not isinstance(payload.get("rewards"), dict)
            or not isinstance(payload.get("evaluation_id"), str)
        ):
            return None
        return payload

    def _write_cached_result(self, record: ContinuousSubmission) -> None:
        path = self._cache_path(record)
        if path is None or record.rewards is None:
            return
        payload = {
            "schema_version": 1,
            "evaluation_fingerprint": record.evaluation_fingerprint,
            "artifact_sha256": record.artifact_sha256,
            "verification_context": self._verification_context,
            "evaluation_id": self._evaluation_id(record),
            "rewards": record.rewards,
            "created_at": _now().isoformat(),
        }
        payload["checksum"] = self._cache_checksum(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(raw, 0o600)
            os.replace(raw, path)
        finally:
            Path(raw).unlink(missing_ok=True)

    def _write_ledger(self) -> None:
        self._ledger.parent.mkdir(parents=True, exist_ok=True)
        lines = [s.model_dump_json() for s in self._summary.submissions]
        self._ledger.write_text("\n".join(lines) + ("\n" if lines else ""))
