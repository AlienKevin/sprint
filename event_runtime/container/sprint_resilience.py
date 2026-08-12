"""Provider-neutral resilience primitives for preemptible accelerator work.

The control plane owns a logical job.  Providers only own fenced attempts of
that job.  Checkpoints are immutable, checksummed commits; an attempt may write
anything it likes while computing, but only a committed checkpoint is eligible
for resume.

This module deliberately has no Modal import.  The host-side Modal adapter is
the host compute worker and CPU simulations can exercise this contract
without cloud credentials or accelerator hardware.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterator, Mapping, Protocol, TypeVar

SCHEMA_VERSION = 1
STATE_DIR_NAME = ".sprint-resilience"
CHECKPOINTS_DIR_NAME = "committed"
LATEST_FILE_NAME = "latest.json"
COMPLETIONS_DIR_NAME = "completed"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON document atomically and fsync its containing directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Lease:
    """A fencing token for one attempt of a durable logical job."""

    job_id: str
    attempt: int
    lease_id: str
    fence_epoch: int = 0

    def owns(self, record: Mapping[str, Any] | None) -> bool:
        if not record:
            return False
        try:
            attempt = int(record.get("attempt") or 0)
            fence_epoch = int(record.get("fence_epoch") or 0)
        except (TypeError, ValueError):
            return False
        return (
            str(record.get("job_id") or self.job_id) == self.job_id
            and attempt == self.attempt
            and str(record.get("lease_id") or "") == self.lease_id
            and fence_epoch == self.fence_epoch
        )


class LeaseLostError(RuntimeError):
    """Raised when a fenced attempt tries to publish durable state."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry policy; ``max_attempts`` includes attempt 1."""

    max_attempts: int = 3
    initial_backoff_s: float = 10.0
    max_backoff_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_s < 0 or self.max_backoff_s < 0:
            raise ValueError("retry backoff must be non-negative")

    def allows_after(self, completed_attempt: int) -> bool:
        return int(completed_attempt) < self.max_attempts

    def delay_after(self, completed_attempt: int) -> float:
        exponent = max(0, int(completed_attempt) - 1)
        return min(self.max_backoff_s, self.initial_backoff_s * (2**exponent))

    @classmethod
    def from_job(cls, job: Mapping[str, Any]) -> "RetryPolicy":
        return cls(
            max_attempts=int(job.get("max_attempts") or 3),
            initial_backoff_s=float(job.get("retry_backoff_sec") or 10),
            max_backoff_s=float(job.get("retry_backoff_max_sec") or 120),
        )


class ProbeState(str, Enum):
    ALIVE = "alive"
    EXITED = "exited"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderHandle:
    provider: str
    attempt_id: str


@dataclass(frozen=True)
class ProbeResult:
    state: ProbeState
    exit_code: int | None = None
    error: str | None = None


class ExecutionProvider(Protocol):
    """Minimal backend boundary used by a durable logical-job controller."""

    name: str

    def start(
        self, job: Mapping[str, Any], lease: Lease
    ) -> ProviderHandle: ...

    def probe(self, handle: ProviderHandle) -> ProbeResult: ...

    def terminate(self, handle: ProviderHandle) -> str | None: ...


@dataclass(frozen=True)
class Checkpoint:
    """A verified, committed checkpoint generation."""

    checkpoint_id: str
    sequence: int
    payload_path: str
    payload_name: str
    size_bytes: int
    sha256: str
    created_at: str
    created_at_epoch_s: float
    replay_cursor: str | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def path(self) -> Path:
        return Path(self.payload_path)

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        # Persist a generation-relative name.  Absolute mount paths may differ
        # between the CPU caller and a replacement GPU worker.
        payload.pop("payload_path", None)
        return payload


class CheckpointStore:
    """Atomic, immutable checkpoints rooted in the job's durable directory.

    Publishing is copy -> fsync -> checksum -> manifest -> directory rename ->
    latest-pointer replace.  A kill at any earlier point leaves only a hidden
    partial directory, which readers ignore.  Readers validate payload size and
    hash and scan older generations if the newest pointer/payload is damaged.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        lease: Lease | None = None,
        lease_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.state_dir = self.root / STATE_DIR_NAME
        self.committed_dir = self.state_dir / CHECKPOINTS_DIR_NAME
        self.latest_path = self.state_dir / LATEST_FILE_NAME
        self.completions_dir = self.state_dir / COMPLETIONS_DIR_NAME
        self.lease = lease
        self.lease_path = Path(lease_path) if lease_path else None

    @classmethod
    def from_env(cls) -> "CheckpointStore":
        def env(name: str) -> str:
            return os.environ.get(name, "").strip()

        root = env("SPRINT_GPU_CHECKPOINT_DIR")
        if not root:
            raise RuntimeError("SPRINT_GPU_CHECKPOINT_DIR is not set")
        attempt = env("SPRINT_GPU_ATTEMPT")
        lease_id = env("SPRINT_GPU_LEASE_ID")
        lease = None
        if attempt and lease_id:
            lease = Lease(
                job_id=env("SPRINT_GPU_JOB_ID"),
                attempt=int(attempt),
                lease_id=lease_id,
                fence_epoch=int(env("SPRINT_GPU_FENCE_EPOCH") or 0),
            )
        return cls(
            root,
            lease=lease,
            lease_path=env("SPRINT_GPU_STATUS_FILE") or None,
        )

    def _assert_lease(self) -> None:
        if self.lease is None or self.lease_path is None:
            return
        try:
            record = json.loads(self.lease_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LeaseLostError(f"cannot verify checkpoint lease: {exc}") from exc
        if not isinstance(record, dict) or not self.lease.owns(record):
            raise LeaseLostError(
                f"checkpoint lease fenced for job={self.lease.job_id} "
                f"attempt={self.lease.attempt}"
            )

    def commit(
        self,
        source: str | os.PathLike[str],
        *,
        sequence: int,
        replay_cursor: str | int | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        _after_payload: Callable[[Path], None] | None = None,
    ) -> Checkpoint:
        """Publish ``source`` as the checkpoint for ``sequence``.

        ``_after_payload`` is a fault-injection seam used by CPU tests.  Callers
        should not use it.
        """
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"checkpoint source is not a file: {source_path}")
        sequence = int(sequence)
        if sequence < 0:
            raise ValueError("checkpoint sequence must be non-negative")

        self._assert_lease()

        # A trainer cursor is a logical commit sequence, not merely a label.
        # Replaying the exact same bytes is idempotent; publishing different
        # trainer state at the same cursor is ambiguous and, in practice, is a
        # common symptom of a loop whose iteration counter never advances.
        current = self.latest_valid(verify_hash=False)
        if current is not None and sequence == current.sequence:
            if (
                source_path.stat().st_size == current.size_bytes
                and sha256_file(source_path) == current.sha256
            ):
                return current
            raise ValueError(
                f"checkpoint sequence {sequence} was already committed with "
                "different trainer state; sequences must advance monotonically"
            )

        self.committed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = f"{sequence:020d}-{uuid.uuid4().hex}"
        partial = self.committed_dir / f".{checkpoint_id}.partial"
        final = self.committed_dir / checkpoint_id
        payload_name = source_path.name or "checkpoint.bin"
        payload = partial / payload_name
        try:
            partial.mkdir(mode=0o700)
            with source_path.open("rb") as src, payload.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if _after_payload is not None:
                _after_payload(payload)
            self._assert_lease()
            checkpoint_metadata = dict(metadata or {})
            if self.lease is not None:
                checkpoint_metadata.setdefault("lease", asdict(self.lease))
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                sequence=sequence,
                payload_path=str(final / payload_name),
                payload_name=payload_name,
                size_bytes=payload.stat().st_size,
                sha256=sha256_file(payload),
                created_at=utc_now(),
                created_at_epoch_s=time.time(),
                replay_cursor=(
                    None if replay_cursor is None else str(replay_cursor)
                ),
                idempotency_key=idempotency_key,
                metadata=checkpoint_metadata,
            )
            atomic_write_json(partial / "manifest.json", checkpoint.to_manifest())
            _fsync_dir(partial)
            os.replace(partial, final)
            _fsync_dir(self.committed_dir)
            self._assert_lease()
            # Pointer monotonicity only needs an atomically committed manifest
            # and payload size here.  Full hashing happens on resume, avoiding
            # an extra multi-GB read on every checkpoint publication.
            current = self.latest_valid(verify_hash=False)
            # A delayed/fenced writer must never move the resume pointer
            # backwards.  The immutable generation is harmless and cleanup
            # will remove it later.
            if current is None or sequence > current.sequence:
                atomic_write_json(
                    self.latest_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "checkpoint_id": checkpoint_id,
                        "sequence": sequence,
                    },
                )
            return checkpoint
        except BaseException:
            # Best effort only.  Even if cleanup itself is interrupted, hidden
            # partial directories are never considered resumable.
            shutil.rmtree(partial, ignore_errors=True)
            raise

    def latest_valid(self, *, verify_hash: bool = True) -> Checkpoint | None:
        candidates: list[Path] = []
        pointer_id = ""
        try:
            pointer = json.loads(self.latest_path.read_text())
            if isinstance(pointer, dict):
                pointer_id = str(pointer.get("checkpoint_id") or "")
        except (OSError, json.JSONDecodeError):
            pass
        if pointer_id:
            candidates.append(self.committed_dir / pointer_id)
        if self.committed_dir.is_dir():
            candidates.extend(
                sorted(
                    (
                        path
                        for path in self.committed_dir.iterdir()
                        if path.is_dir() and not path.name.startswith(".")
                    ),
                    key=lambda path: path.name,
                    reverse=True,
                )
            )
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            checkpoint = self._validate(candidate, verify_hash=verify_hash)
            if checkpoint is not None:
                if checkpoint.checkpoint_id != pointer_id:
                    atomic_write_json(
                        self.latest_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "sequence": checkpoint.sequence,
                        },
                    )
                return checkpoint
        return None

    def iter_valid(self, *, verify_hash: bool = True) -> Iterator[Checkpoint]:
        if not self.committed_dir.is_dir():
            return
        for path in sorted(self.committed_dir.iterdir(), reverse=True):
            if path.is_dir() and not path.name.startswith("."):
                checkpoint = self._validate(path, verify_hash=verify_hash)
                if checkpoint is not None:
                    yield checkpoint

    def _validate(
        self, generation: Path, *, verify_hash: bool = True
    ) -> Checkpoint | None:
        try:
            raw = json.loads((generation / "manifest.json").read_text())
            if not isinstance(raw, dict):
                return None
            if int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
                return None
            checkpoint_id = str(raw["checkpoint_id"])
            if checkpoint_id != generation.name:
                return None
            payload_name = str(raw["payload_name"])
            if Path(payload_name).name != payload_name:
                return None
            payload = generation / payload_name
            size = int(raw["size_bytes"])
            if not payload.is_file() or payload.stat().st_size != size:
                return None
            if verify_hash and sha256_file(payload) != str(raw["sha256"]):
                return None
            metadata = raw.get("metadata")
            if not isinstance(metadata, dict):
                return None
            return Checkpoint(
                checkpoint_id=checkpoint_id,
                sequence=int(raw["sequence"]),
                payload_path=str(payload),
                payload_name=payload_name,
                size_bytes=size,
                sha256=str(raw["sha256"]),
                created_at=str(raw["created_at"]),
                created_at_epoch_s=float(raw["created_at_epoch_s"]),
                replay_cursor=(
                    None
                    if raw.get("replay_cursor") is None
                    else str(raw["replay_cursor"])
                ),
                idempotency_key=(
                    None
                    if raw.get("idempotency_key") is None
                    else str(raw["idempotency_key"])
                ),
                metadata=metadata,
            )
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def cleanup(self, *, keep: int = 3) -> list[str]:
        """Remove partials and old generations, retaining newest valid commits."""
        if keep < 1:
            raise ValueError("keep must be at least 1")
        removed: list[str] = []
        if not self.committed_dir.is_dir():
            return removed
        valid = list(self.iter_valid())
        retain = {item.checkpoint_id for item in valid[:keep]}
        for path in self.committed_dir.iterdir():
            if path.name in retain:
                continue
            if path.name.startswith(".") or path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                if not path.exists():
                    removed.append(path.name)
        latest = self.latest_valid()
        if latest is None:
            self.latest_path.unlink(missing_ok=True)
        return sorted(removed)


T = TypeVar("T")


class CompletionJournal:
    """Durable result markers for deterministic units of work.

    ``run_once`` skips keys whose completion record was atomically published.
    This gives exactly-once publication for results written *through this
    journal*.  An arbitrary external side effect remains at-least-once across a
    crash between the side effect and marker; use ``key`` as the downstream
    idempotency key in that case.
    """

    def __init__(self, store: CheckpointStore) -> None:
        self.store = store

    @staticmethod
    def _safe_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.store.completions_dir / f"{self._safe_key(key)}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("key") != key:
            return None
        return payload

    def complete(self, key: str, result: Any) -> dict[str, Any]:
        self.store._assert_lease()
        existing = self.get(key)
        if existing is not None:
            return existing
        payload = {
            "schema_version": SCHEMA_VERSION,
            "key": key,
            "completed_at": utc_now(),
            "completed_at_epoch_s": time.time(),
            "result": result,
        }
        atomic_write_json(self.path_for(key), payload)
        return payload

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        lock_path = self.path_for(key).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

    def run_once(self, key: str, fn: Callable[[], T]) -> tuple[T | Any, bool]:
        with self._key_lock(key):
            existing = self.get(key)
            if existing is not None:
                return existing.get("result"), False
            self.store._assert_lease()
            result = fn()
            self.store._assert_lease()
            return self.complete(key, result).get("result"), True


class Interruption:
    """Signal-to-event bridge for checkpoint-aware training loops."""

    def __init__(self, signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)):
        self.signals = signals
        self._event = threading.Event()
        self.signum: int | None = None
        self._previous: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, signum: int = signal.SIGTERM) -> None:
        self.signum = signum
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        self.request(signum)

    def __enter__(self) -> "Interruption":
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be installed on the main thread")
        for signum in self.signals:
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()
