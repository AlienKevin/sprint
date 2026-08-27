#!/usr/bin/env python3
"""Build a local, sanitized Harbor job bundle for later manual upload.

The source job is never modified.  This module copies a deliberately small
allowlist into an atomic staging directory, redacts structured and free-form
text, omits transient agent state, and scans both the staged tree and a Harbor-
shaped tar archive.  A non-empty finding list is fatal: the destination is not
published and the command exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from event_runtime.export import trajectory

SCHEMA_VERSION = 1
KINGFISHER_VERSION = "2.0.0"
OMITTED_FIELDS = frozenset({"internal_chat_message_metadata_passthrough"})

JOB_FILES = ("config.json", "lock.json", "result.json", "analysis.md", "job.log")
TRIAL_FILES = (
    "config.json",
    "lock.json",
    "result.json",
    "analysis.md",
    "trial.log",
    "exception.txt",
)
AGENT_FILES = (
    "trajectory.json",
    "deepseek-harness-events.jsonl",
    "deepseek-harness.txt",
    "codex.txt",
    "goal-bootstrap.json",
    "goal-lifecycle.json",
    "usage-audit.json",
    "openrouter-ledger-proxy.log",
)
AGENT_TREES = ("usage-provenance",)
RUNTIME_TREES = ("verifier", "artifacts")
TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".log", ".md", ".txt"})

_SENSITIVE_ENV_KEY = re.compile(
    r"(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH|COOKIE)", re.IGNORECASE
)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)[ .-]?\d{3}[ .-]\d{4}|"
    r"\d{3}[ .-]\d{3}[ .-]\d{4})(?!\w)"
)
_IPV4_RE = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\."
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
)
_HOME_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:home|Users)/[^/\s'\"]+")
_URL_USERINFO_RE = re.compile(r"(?P<scheme>https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)
_HUGGINGFACE_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_MODAL_TOKEN_RE = re.compile(r"\b(?:ak|as)-[A-Za-z0-9_-]{20,}\b")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

_FINDING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openrouter_key", re.compile(r"sk-or-v1-[A-Za-z0-9_-]+")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", _GITHUB_TOKEN_RE),
    ("huggingface_token", _HUGGINGFACE_TOKEN_RE),
    ("modal_token", _MODAL_TOKEN_RE),
    ("aws_access_key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.I)),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\b"),
    ),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("email", _EMAIL_RE),
    ("phone", _PHONE_RE),
    ("ip_address", _IPV4_RE),
    ("home_path", _HOME_PATH_RE),
    ("url_userinfo", _URL_USERINFO_RE),
)


@dataclass
class Audit:
    redactions: Counter[str] = field(default_factory=Counter)
    included: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)

    def include(self, root: Path, path: Path) -> None:
        self.included.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    def exclude(self, root: Path, path: Path, reason: str) -> None:
        self.excluded.append(
            {"path": path.relative_to(root).as_posix(), "reason": reason}
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[:1] == value[-1:] and value[0] in "'\"":
            value = value[1:-1]
        if _SENSITIVE_ENV_KEY.search(name) and len(value) >= 8:
            values[name] = value
    return values


def collect_known_secrets(env_files: Iterable[Path]) -> dict[str, str]:
    """Collect values for replacement without ever serializing the values."""
    secrets = {
        name: value
        for name, value in os.environ.items()
        if _SENSITIVE_ENV_KEY.search(name) and len(value) >= 8
    }
    for path in env_files:
        secrets.update(_parse_env_file(path))
    return secrets


def _inferred_secret_files(source_job: Path) -> list[Path]:
    state_dir = source_job.parent.parent
    try:
        run = json.loads((state_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    secret_dir = run.get("secret_dir") if isinstance(run, dict) else None
    if not isinstance(secret_dir, str) or not secret_dir:
        return []
    candidate = Path(secret_dir) / "harbor.env"
    return [candidate] if candidate.is_file() else []


class Sanitizer:
    def __init__(self, secrets: dict[str, str], audit: Audit) -> None:
        self._secrets = sorted(set(secrets.values()), key=len, reverse=True)
        self.audit = audit

    def text(self, value: str) -> str:
        # The public viewer intentionally truncates unusually large strings.
        # Harbor archival staging must instead preserve the complete trace.
        result = trajectory._redact_string(value, max_chars=None)
        for secret in self._secrets:
            count = result.count(secret)
            if count:
                result = result.replace(secret, "[REDACTED]")
                self.audit.redactions["known_secret"] += count
        result = self._replace(
            _GITHUB_TOKEN_RE, result, "[REDACTED]", "github_token"
        )
        result = self._replace(
            _HUGGINGFACE_TOKEN_RE, result, "[REDACTED]", "huggingface_token"
        )
        result = self._replace(_MODAL_TOKEN_RE, result, "[REDACTED]", "modal_token")
        result = self._replace(_PRIVATE_KEY_RE, result, "[REDACTED]", "private_key")
        result = self._replace(_EMAIL_RE, result, "[EMAIL REDACTED]", "email")
        result = self._replace(_PHONE_RE, result, "[PHONE REDACTED]", "phone")
        result = self._replace(_IPV4_RE, result, "[IP REDACTED]", "ip_address")
        result = self._replace(
            _HOME_PATH_RE, result, "[HOME PATH REDACTED]", "home_path"
        )
        result = self._replace_url_userinfo(result)
        return result

    def value(self, value: Any, *, key: str | None = None) -> Any:
        if key and trajectory._SENSITIVE_KEY_RE.search(key):
            self.audit.redactions["sensitive_field"] += 1
            return "[REDACTED]"
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, dict):
            kind = str(value.get("type") or "").lower()
            if kind in {"image", "input_image", "image_url"}:
                self.audit.redactions["image_payload"] += 1
                return {"type": kind, "content": "[IMAGE OMITTED FROM ARCHIVE]"}
            result = {}
            for name, item in value.items():
                name = str(name)
                if name in OMITTED_FIELDS:
                    self.audit.redactions["internal_metadata"] += 1
                    continue
                result[name] = self.value(item, key=name)
            return result
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self.text(str(value))

    def _replace(
        self, pattern: re.Pattern[str], value: str, replacement: str, label: str
    ) -> str:
        result, count = pattern.subn(replacement, value)
        self.audit.redactions[label] += count
        return result

    def _replace_url_userinfo(self, value: str) -> str:
        def replacement(match: re.Match[str]) -> str:
            self.audit.redactions["url_userinfo"] += 1
            return f"{match.group('scheme')}[REDACTED]@"

        return _URL_USERINFO_RE.sub(replacement, value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sanitize_file(source: Path, target: Path, sanitizer: Sanitizer) -> None:
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        _write_json(target, sanitizer.value(payload))
        return
    if source.suffix == ".jsonl":
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(encoding="utf-8") as read, target.open(
            "w", encoding="utf-8"
        ) as write:
            for line_number, raw in enumerate(read, start=1):
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {source} at line {line_number}"
                    ) from exc
                write.write(
                    json.dumps(
                        sanitizer.value(payload),
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(sanitizer.text(source.read_text(encoding="utf-8")), encoding="utf-8")


def _record_include(audit: Audit, stage_root: Path, path: Path) -> None:
    audit.include(stage_root, path)


def _copy_runtime_tree(
    source: Path,
    target: Path,
    *,
    source_job: Path,
    stage_job: Path,
    sanitizer: Sanitizer,
) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            sanitizer.audit.exclude(source_job, path, "symlink excluded")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            sanitizer.audit.exclude(source_job, path, "non-text artifact excluded")
            continue
        relative = path.relative_to(source)
        destination = target / relative
        _sanitize_file(path, destination, sanitizer)
        _record_include(sanitizer.audit, stage_job, destination)


def _copy_agent(
    source: Path,
    target: Path,
    *,
    source_job: Path,
    stage_job: Path,
    sanitizer: Sanitizer,
) -> None:
    if not source.exists():
        return
    for name in AGENT_FILES:
        path = source / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            sanitizer.audit.exclude(source_job, path, "not a regular file")
            continue
        destination = target / name
        _sanitize_file(path, destination, sanitizer)
        _record_include(sanitizer.audit, stage_job, destination)
    for name in AGENT_TREES:
        _copy_runtime_tree(
            source / name,
            target / name,
            source_job=source_job,
            stage_job=stage_job,
            sanitizer=sanitizer,
        )
    for path in sorted(source.iterdir()):
        if path.name in AGENT_FILES or path.name in AGENT_TREES:
            continue
        sanitizer.audit.exclude(source_job, path, "agent file not allowlisted")


def _combined_trajectory(
    state_dir: Path, source_agent: Path, sanitizer: Sanitizer
) -> dict[str, Any] | None:
    sources = [
        (trajectory._attempt_number(path), path)
        for path in sorted(
            state_dir.glob("trace/reconstructed/cpu-attempt-*/*/trajectory.json")
        )
    ]
    if not sources:
        raw_deepseek = source_agent / "deepseek-harness-events.jsonl"
        if raw_deepseek.exists():
            payload = trajectory.deepseek_harness_trajectory(
                [raw_deepseek], model=None
            )
            result = sanitizer.value(payload)
            return result if isinstance(result, dict) else None
    if not sources:
        return None
    steps: list[dict[str, Any]] = []
    for attempt, path in sources:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for step in payload.get("steps") or []:
            if not isinstance(step, dict):
                continue
            item = dict(step)
            item.setdefault("attempt", attempt)
            steps.append(item)
    steps.sort(key=lambda item: (str(item.get("timestamp") or ""), str(item.get("step_id"))))
    result = sanitizer.value({"schema_version": "1.0", "steps": steps})
    return result if isinstance(result, dict) else None


def _scan_text(
    text: str,
    *,
    path: str,
    secrets: Iterable[str],
    findings: list[dict[str, Any]],
) -> None:
    for secret in secrets:
        if secret and secret in text:
            findings.append({"path": path, "kind": "known_secret"})
            break
    for name, pattern in _FINDING_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append({"path": path, "kind": name, "offset": match.start()})


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield str(name)
            yield from _walk_strings(item)


def _scan_content(
    content: str,
    *,
    suffix: str,
    path: str,
    secrets: Iterable[str],
    findings: list[dict[str, Any]],
) -> None:
    if suffix == ".json":
        try:
            values = _walk_strings(json.loads(content))
        except json.JSONDecodeError:
            findings.append({"path": path, "kind": "invalid_json"})
            return
    elif suffix == ".jsonl":
        parsed: list[Any] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                findings.append(
                    {"path": path, "kind": "invalid_jsonl", "line": line_number}
                )
        values = (text for item in parsed for text in _walk_strings(item))
    else:
        values = (content,)
    for index, value in enumerate(values):
        _scan_text(
            value,
            path=f"{path}#string-{index}" if suffix in {".json", ".jsonl"} else path,
            secrets=secrets,
            findings=findings,
        )


def scan_tree(root: Path, secrets: Iterable[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            findings.append({"path": path.relative_to(root).as_posix(), "kind": "symlink"})
            continue
        if not path.is_file() or path.name == "SANITIZATION_REPORT.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(
                {"path": path.relative_to(root).as_posix(), "kind": "binary_file"}
            )
            continue
        _scan_content(
            text,
            suffix=path.suffix.lower(),
            path=path.relative_to(root).as_posix(),
            secrets=secrets,
            findings=findings,
        )
    return findings


def _create_audit_archive(job_dir: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(job_dir, arcname=job_dir.name)


def _scan_with_kingfisher(target: Path, report_path: Path) -> dict[str, Any]:
    executable = shutil.which("kingfisher")
    if executable is None:
        raise RuntimeError(
            "kingfisher is required; install with "
            f"`uv tool install kingfisher-bin=={KINGFISHER_VERSION}`"
        )
    command = [
        executable,
        "--no-update-check",
        "--quiet",
        "scan",
        str(target),
        "--no-validate",
        "--git-history",
        "none",
        "--format",
        "json",
        "--redact",
        "--output",
        str(report_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if completed.returncode not in {0, 200}:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"kingfisher exited {completed.returncode}: {error[:500]}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    findings = payload.get("findings")
    metadata = payload.get("metadata")
    if not isinstance(findings, list) or not isinstance(metadata, dict):
        raise RuntimeError("kingfisher returned an invalid JSON report")
    version = metadata.get("kingfisher_version")
    if version != KINGFISHER_VERSION:
        raise RuntimeError(
            f"kingfisher version {version!r} does not match {KINGFISHER_VERSION}"
        )
    return {"version": version, "findings": len(findings)}


def _validate_harbor_compatibility(job_dir: Path) -> dict[str, Any]:
    """Exercise the same parsers and trial archiver used by ``harbor upload``."""
    from harbor.upload.uploader import (
        _create_trial_archive,
        _load_job_from_disk,
        _read_trial_lock_for_upload,
    )

    job_result, job_config, trial_results, trial_dirs = _load_job_from_disk(job_dir)
    archive_bytes = 0
    for trial_result in trial_results:
        trial_dir = trial_dirs[trial_result.trial_name]
        _read_trial_lock_for_upload(trial_dir)
        archive_bytes += len(_create_trial_archive(trial_dir))
    return {
        "job_id": str(job_result.id),
        "job_name": job_config.job_name,
        "trial_count": len(trial_results),
        "trial_archive_bytes": archive_bytes,
    }


def scan_archive(archive: Path, secrets: Iterable[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                findings.append({"path": member.name, "kind": "archive_link"})
                continue
            if not member.isfile() or member.name.endswith("SANITIZATION_REPORT.json"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            try:
                text = extracted.read().decode("utf-8")
            except UnicodeDecodeError:
                findings.append({"path": member.name, "kind": "archive_binary_file"})
                continue
            _scan_content(
                text,
                suffix=Path(member.name).suffix.lower(),
                path=member.name,
                secrets=secrets,
                findings=findings,
            )
    return findings


def _trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(
        child
        for child in job_dir.iterdir()
        if child.is_dir() and not child.is_symlink() and (child / "result.json").exists()
    )


def sanitize_job(
    source_job: Path,
    output_root: Path,
    *,
    env_files: Iterable[Path] = (),
    force: bool = False,
    validate_harbor: bool = True,
    run_kingfisher: bool = True,
) -> Path:
    source_job = source_job.resolve()
    output_root = output_root.resolve()
    destination = output_root / source_job.name
    failed_report = output_root / f"{source_job.name}.SANITIZATION_FAILED.json"
    if source_job == destination or source_job in destination.parents:
        raise ValueError("destination must not be the source job or inside it")
    if destination.exists() and not force:
        raise FileExistsError(f"destination already exists: {destination}")
    for required in ("config.json", "lock.json", "result.json"):
        if not (source_job / required).is_file():
            raise FileNotFoundError(f"missing Harbor job file: {source_job / required}")
    trials = _trial_dirs(source_job)
    if not trials:
        raise ValueError(f"no trial directories found in {source_job}")

    secret_files = [*env_files, *_inferred_secret_files(source_job)]
    secrets = collect_known_secrets(secret_files)
    audit = Audit()
    sanitizer = Sanitizer(secrets, audit)
    output_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{source_job.name}.sanitize-", dir=output_root))
    stage_job = temp / source_job.name
    stage_job.mkdir()
    try:
        for name in JOB_FILES:
            path = source_job / name
            if not path.exists():
                continue
            _sanitize_file(path, stage_job / name, sanitizer)
            _record_include(audit, stage_job, stage_job / name)
        trial_names = {trial.name for trial in trials}
        for path in sorted(source_job.iterdir()):
            if path.name not in JOB_FILES and path.name not in trial_names:
                audit.exclude(source_job, path, "job entry not allowlisted")

        for source_trial in trials:
            stage_trial = stage_job / source_trial.name
            stage_trial.mkdir()
            for name in TRIAL_FILES:
                path = source_trial / name
                if not path.exists():
                    continue
                _sanitize_file(path, stage_trial / name, sanitizer)
                _record_include(audit, stage_job, stage_trial / name)
            _copy_agent(
                source_trial / "agent",
                stage_trial / "agent",
                source_job=source_job,
                stage_job=stage_job,
                sanitizer=sanitizer,
            )
            inferred_state = source_job.parent.parent
            trajectory_path = stage_trial / "agent" / "trajectory.json"
            if not trajectory_path.exists():
                combined = _combined_trajectory(
                    inferred_state, source_trial / "agent", sanitizer
                )
                if combined is not None:
                    _write_json(trajectory_path, combined)
                    _record_include(audit, stage_job, trajectory_path)
            for tree in RUNTIME_TREES:
                _copy_runtime_tree(
                    source_trial / tree,
                    stage_trial / tree,
                    source_job=source_job,
                    stage_job=stage_job,
                    sanitizer=sanitizer,
                )
            allowed_trial_entries = {
                *TRIAL_FILES,
                "agent",
                *RUNTIME_TREES,
            }
            for path in sorted(source_trial.iterdir()):
                if path.name not in allowed_trial_entries:
                    audit.exclude(source_job, path, "trial entry not allowlisted")

        known_values = tuple(secrets.values())
        audit.findings.extend(scan_tree(stage_job, known_values))
        kingfisher: dict[str, Any] | None = None
        with tempfile.TemporaryDirectory(prefix="harbor-sanitize-archive-") as tmp:
            archive = Path(tmp) / f"{source_job.name}.tar.gz"
            _create_audit_archive(stage_job, archive)
            audit.findings.extend(scan_archive(archive, known_values))
            if run_kingfisher:
                try:
                    tree_scan = _scan_with_kingfisher(
                        stage_job, Path(tmp) / "kingfisher-tree.json"
                    )
                    archive_scan = _scan_with_kingfisher(
                        archive, Path(tmp) / "kingfisher-archive.json"
                    )
                    kingfisher = {
                        "version": tree_scan["version"],
                        "tree_findings": tree_scan["findings"],
                        "archive_findings": archive_scan["findings"],
                        "live_validation": False,
                        "update_check": False,
                    }
                    total = tree_scan["findings"] + archive_scan["findings"]
                    if total:
                        audit.findings.append(
                            {"path": ".", "kind": "kingfisher", "count": total}
                        )
                except Exception as exc:
                    audit.findings.append(
                        {
                            "path": ".",
                            "kind": "kingfisher_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            archive_sha256 = _sha256(archive)
            archive_bytes = archive.stat().st_size

        compatibility: dict[str, Any] | None = None
        if validate_harbor:
            try:
                compatibility = _validate_harbor_compatibility(stage_job)
            except Exception as exc:
                audit.findings.append(
                    {
                        "path": ".",
                        "kind": "harbor_validation",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        report = {
            "schema_version": SCHEMA_VERSION,
            "source_job": source_job.name,
            "destination_job": destination.name,
            "status": "clean" if not audit.findings else "blocked",
            "trial_count": len(trials),
            "included": sorted(audit.included, key=lambda item: item["path"]),
            "excluded": sorted(audit.excluded, key=lambda item: item["path"]),
            "redactions": dict(sorted(audit.redactions.items())),
            "findings": audit.findings,
            "audit_archive": {"sha256": archive_sha256, "bytes": archive_bytes},
            "kingfisher": kingfisher,
            "harbor_validation": compatibility,
        }
        _write_json(stage_job / "SANITIZATION_REPORT.json", report)
        if audit.findings:
            _write_json(failed_report, report)
            raise ValueError(
                "sanitization blocked with "
                f"{len(audit.findings)} finding(s); see {failed_report}"
            )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(stage_job, destination)
        failed_report.unlink(missing_ok=True)
        temp.rmdir()
        return destination
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--secret-env-file", type=Path, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    results = []
    for job_dir in args.job_dir:
        destination = sanitize_job(
            job_dir,
            args.output_root,
            env_files=args.secret_env_file,
            force=args.force,
        )
        report = json.loads((destination / "SANITIZATION_REPORT.json").read_text())
        results.append(
            {
                "job": destination.name,
                "status": report["status"],
                "files": len(report["included"]),
                "redactions": sum(report["redactions"].values()),
            }
        )
    print(json.dumps({"jobs": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
