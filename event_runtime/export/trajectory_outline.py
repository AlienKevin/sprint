#!/usr/bin/env python3
"""Author semantic rollout chapters for one sanitized public trajectory.

This is deliberately an offline publication step. It lets a read-only Codex
agent browse the exact already-sanitized public trace, validates the
schema-constrained response against immutable step IDs, and atomically writes
a sidecar consumed by the website. It never runs in the live timeline refresh
path and it never controls an experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "trajectory-outline/v1"
PROMPT_VERSION = "rollout-outline/v1"
SOURCE_ACCESS_VERSION = "direct-public-trajectory/v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "high"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def capture_stamp() -> str:
    return utc_now().replace("-", "").replace(":", "")


def capture_path(repository: Path, run_id: str) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("-") or "unknown-run"
    return (
        repository.resolve()
        / ".artifacts"
        / "trajectory-outlines"
        / safe_run_id
        / f"{capture_stamp()}-{os.getpid()}"
    )


def capture_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "manifest.json", payload)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_deepseek(payload: dict[str, Any]) -> bool:
    model = str((payload.get("run") or {}).get("model") or "").lower()
    return "deepseek" in model


def trace_groups(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Mirror the viewer's grouping of Codex tool-only records."""
    steps = [step for step in payload.get("steps") or [] if isinstance(step, dict)]
    if _is_deepseek(payload):
        return [[step] for step in steps]
    groups: list[list[dict[str, Any]]] = []
    for step in steps:
        has_narrative = bool(step.get("message") or step.get("reasoning_content"))
        if has_narrative or not groups:
            groups.append([step])
        else:
            groups[-1].append(step)
    return groups


def group_id(group: list[dict[str, Any]]) -> str:
    return str(group[0].get("step_id") or "")


def public_number(step: dict[str, Any]) -> int:
    value = step.get("public_step_id", step.get("attempt_step_id", 0))
    return int(value) if isinstance(value, (int, float)) else 0


def prompt(trajectory_path: Path, *, repository: Path) -> str:
    resolved_path = trajectory_path.resolve()
    try:
        display_path = resolved_path.relative_to(repository.resolve())
    except ValueError:
        display_path = resolved_path
    return f"""Use your read-only shell and file-inspection tools to browse the exact
sanitized public trajectory at `{display_path}` and author a semantic table of
contents for a technical reader. This is the same JSON file served to the
website. Analyze it directly in whatever way is most useful; do not ask for or
rely on a preprocessed summary. Return only the JSON shape required by the
output schema.

The trace's `public_step_id` is the step number visible to readers and its
`step_id` is the stable boundary anchor. In a DeepSeek trace every record is a
viewer step. In a Codex/Luna trace, a record containing `message` or
`reasoning_content` begins a viewer step and any following tool-only records
belong to that step until the next narrative record. Boundary IDs must be the
primary record's exact `step_id`, so they match what the viewer shows.

Choose chapter boundaries when the agent's objective, method, experiment, or
conclusion meaningfully changes—not at fixed intervals and not from keyword
frequency. Cover the entire ordered rollout with 2–20 contiguous chapters
and never collapse the entire trace into one chapter. Every boundary must use
an exact eligible `step_id` from the public trace.

Titles must be concise, concrete action/outcome phrases. Summaries must be one
or two information-dense sentences explaining what the agent actually tried,
why it changed direction, and what happened. Name concrete methods and results
when the trace supports them—for example PPO, behavior cloning, reward shaping,
phase-conditioned hopping, a verifier failure, measured distance, or a lane
violation. Expand or explain specialist shorthand instead of emitting vague
labels such as "phase", "wait", "monitoring", "testing candidates", or
"working on the policy". Do not invent facts, infer hidden success, or copy
credentials. Use start_step_id/end_step_id to partition every viewer step
exactly once, in order, with no gaps or overlaps. Use stable descriptive slugs
for chapter IDs. The synopsis should state the overall strategy, major pivots,
and final outcome in two or three sentences."""


@contextlib.contextmanager
def isolated_codex_environment(
    trajectory_path: Path,
    schema_path: Path,
    *,
    codex_bin: str,
) -> Any:
    """Expose an exact trace copy through an OS-enforced read-only workspace."""
    resolved_codex = shutil.which(codex_bin) or codex_bin
    auth_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth_path = auth_root / "auth.json"
    if not auth_path.is_file():
        raise RuntimeError(f"Codex authentication is unavailable at {auth_path}")

    with tempfile.TemporaryDirectory(prefix="trajectory-outline-") as raw_temp:
        temp = Path(raw_temp)
        temp.chmod(0o711)
        source_root = temp / "source"
        source_root.mkdir(mode=0o755)
        source = source_root / "trajectory.json"
        source.write_bytes(trajectory_path.read_bytes())
        schema = source_root / "outline.schema.json"
        schema.write_bytes(schema_path.read_bytes())
        source.chmod(0o444)
        schema.chmod(0o444)
        source_root.chmod(0o555)

        codex_home = temp / "codex-home"
        codex_home.mkdir(mode=0o700)
        shutil.copyfile(auth_path, codex_home / "auth.json")
        (codex_home / "auth.json").chmod(0o600)
        subprocess.run(
            ["sudo", "-n", "chown", "-R", "nobody:nogroup", str(codex_home)],
            check=True,
            capture_output=True,
            text=True,
        )

        # The npm-installed executable is below the user's non-listable home.
        # Add traverse-only access temporarily, then restore the exact mode.
        user_home = Path.home()
        original_mode = stat.S_IMODE(user_home.stat().st_mode)
        transient_output = codex_home / "final-response.json"
        try:
            if not original_mode & stat.S_IXOTH:
                subprocess.run(
                    [
                        "sudo",
                        "-n",
                        "chmod",
                        f"{original_mode | stat.S_IXOTH:o}",
                        str(user_home),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            yield {
                "prefix": [
                    "sudo",
                    "-n",
                    "-u",
                    "nobody",
                    "env",
                    f"HOME={codex_home}",
                    f"CODEX_HOME={codex_home}",
                    f"PATH={os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
                    str(resolved_codex),
                ],
                "repository": source_root,
                "trajectory": source,
                "schema": schema,
                "output": transient_output,
            }
        finally:
            if stat.S_IMODE(user_home.stat().st_mode) != original_mode:
                subprocess.run(
                    ["sudo", "-n", "chmod", f"{original_mode:o}", str(user_home)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    str(codex_home),
                ],
                check=False,
                capture_output=True,
                text=True,
            )


def _validate_text(value: Any, *, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} must contain {minimum}–{maximum} characters")
    if _CONTROL_RE.search(value) or "<script" in value.lower():
        raise ValueError(f"{name} contains unsafe markup or control characters")
    return value


def validate_authored(
    authored: dict[str, Any], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    groups = trace_groups(payload)
    if not groups:
        raise ValueError("cannot outline an empty trajectory")
    by_id = {group_id(group): index for index, group in enumerate(groups)}
    if not all(by_id) or len(by_id) != len(groups):
        raise ValueError("trajectory group IDs must be unique and non-empty")
    chapters = authored.get("chapters")
    if not isinstance(chapters, list):
        raise ValueError("chapters must be an array")
    minimum = 1 if len(groups) < 4 else 2
    if not minimum <= len(chapters) <= 20:
        raise ValueError(
            f"expected {minimum}–20 chapters, received {len(chapters)}"
        )

    validated: list[dict[str, Any]] = []
    expected_start = 0
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for position, raw in enumerate(chapters, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"chapter {position} must be an object")
        chapter_id = str(raw.get("id") or "")
        if not _SLUG_RE.fullmatch(chapter_id) or len(chapter_id) > 80:
            raise ValueError(f"chapter {position} has an invalid id")
        if chapter_id in seen_ids:
            raise ValueError(f"chapter id {chapter_id!r} is duplicated")
        start_id = str(raw.get("start_step_id") or "")
        end_id = str(raw.get("end_step_id") or "")
        if start_id not in by_id or end_id not in by_id:
            raise ValueError(f"chapter {chapter_id!r} references an unknown group")
        start_index, end_index = by_id[start_id], by_id[end_id]
        if start_index != expected_start:
            raise ValueError(f"chapter {chapter_id!r} leaves a gap or overlaps")
        if end_index < start_index:
            raise ValueError(f"chapter {chapter_id!r} ends before it starts")
        title = _validate_text(
            raw.get("title"), name=f"chapter {chapter_id} title", minimum=8, maximum=80
        )
        summary = _validate_text(
            raw.get("summary"),
            name=f"chapter {chapter_id} summary",
            minimum=30,
            maximum=500,
        )
        normalized_title = title.casefold()
        if normalized_title in seen_titles:
            raise ValueError(f"chapter title {title!r} is duplicated")
        start_group, end_group = groups[start_index], groups[end_index]
        validated.append(
            {
                "id": chapter_id,
                "start": {
                    "step_id": start_id,
                    "public_step_id": public_number(start_group[0]),
                },
                "end": {
                    "step_id": end_id,
                    "public_step_id": max(public_number(step) for step in end_group),
                },
                "title": title,
                "summary": summary,
            }
        )
        seen_ids.add(chapter_id)
        seen_titles.add(normalized_title)
        expected_start = end_index + 1
    if expected_start != len(groups):
        raise ValueError("chapters do not cover the end of the trajectory")
    _validate_text(authored.get("synopsis"), name="synopsis", minimum=40, maximum=700)
    return validated


def build_artifact(
    authored: dict[str, Any],
    payload: dict[str, Any],
    *,
    trajectory_sha256: str,
    model: str,
    reasoning_effort: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    chapters = validate_authored(authored, payload)
    run = payload.get("run") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.get("run_id"),
        "trajectory": {
            "schema_version": payload.get("schema_version"),
            "source_fingerprint": payload.get("source_fingerprint"),
            "sha256": trajectory_sha256,
            "step_count": len(payload.get("steps") or []),
        },
        "generator": {
            "agent": "codex",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "source_access": SOURCE_ACCESS_VERSION,
            "generated_at": generated_at or utc_now(),
        },
        "synopsis": authored["synopsis"].strip(),
        "chapters": chapters,
    }


def artifact_matches(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    *,
    trajectory_sha256: str,
    model: str,
    reasoning_effort: str,
) -> bool:
    trajectory = artifact.get("trajectory") or {}
    generator = artifact.get("generator") or {}
    return bool(
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("run_id") == (payload.get("run") or {}).get("run_id")
        and trajectory.get("source_fingerprint") == payload.get("source_fingerprint")
        and trajectory.get("sha256") == trajectory_sha256
        and trajectory.get("step_count") == len(payload.get("steps") or [])
        and generator.get("agent") == "codex"
        and generator.get("model") == model
        and generator.get("reasoning_effort") == reasoning_effort
        and generator.get("prompt_version") == PROMPT_VERSION
        and generator.get("source_access") == SOURCE_ACCESS_VERSION
    )


def invoke_codex(
    trajectory_path: Path,
    *,
    repository: Path,
    model: str,
    reasoning_effort: str,
    codex_bin: str,
    timeout_seconds: int,
    capture_dir: Path,
) -> dict[str, Any]:
    schema = Path(__file__).with_name("trajectory_outline.schema.json")
    capture_dir.mkdir(parents=True, exist_ok=False)
    output = capture_dir / "final-response.json"
    events = capture_dir / "events.jsonl"
    stderr = capture_dir / "stderr.log"
    with isolated_codex_environment(
        trajectory_path, schema, codex_bin=codex_bin
    ) as isolated:
        isolated_repository = isolated["repository"]
        isolated_trajectory = isolated["trajectory"]
        isolated_output = isolated["output"]
        if sha256_bytes(isolated_trajectory.read_bytes()) != sha256_bytes(
            trajectory_path.read_bytes()
        ):
            raise RuntimeError("isolated trajectory copy does not match its source")
        synthesis_prompt = prompt(
            isolated_trajectory, repository=isolated_repository
        )
        command = [
            *isolated["prefix"],
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--cd",
            str(isolated_repository),
            "--json",
            "--output-schema",
            str(isolated["schema"]),
            "--output-last-message",
            str(isolated_output),
            synthesis_prompt,
        ]
        (capture_dir / "prompt.txt").write_text(synthesis_prompt)
        atomic_json(
            capture_dir / "command.json",
            {
                "argv": [*command[:-1], "<prompt in prompt.txt>"],
                "isolation": "unprivileged-user-read-only-source/v1",
                "source_sha256": sha256_bytes(isolated_trajectory.read_bytes()),
            },
        )
        with events.open("w", encoding="utf-8") as event_stream, stderr.open(
            "w", encoding="utf-8"
        ) as error_stream:
            completed = subprocess.run(
                command,
                text=True,
                stdout=event_stream,
                stderr=error_stream,
                timeout=timeout_seconds,
                check=False,
            )
        if isolated_output.is_file():
            output.write_bytes(isolated_output.read_bytes())
    if completed.returncode != 0:
        tail = stderr.read_text(errors="replace")[-4_000:].strip()
        raise RuntimeError(f"Codex outline generation failed: {tail}")
    try:
        authored = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Codex did not produce valid outline JSON") from exc
    if not isinstance(authored, dict):
        raise RuntimeError("Codex outline response must be a JSON object")
    return authored


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o644)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def generate(
    trajectory_path: Path,
    *,
    output_path: Path,
    repository: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    codex_bin: str = "codex",
    timeout_seconds: int = 1_800,
    force: bool = False,
    capture_dir: Path | None = None,
) -> dict[str, Any]:
    raw = trajectory_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("public trajectory must be a JSON object")
    trajectory_sha256 = sha256_bytes(raw)
    if not force and output_path.is_file():
        try:
            cached = json.loads(output_path.read_text())
            if artifact_matches(
                cached,
                payload,
                trajectory_sha256=trajectory_sha256,
                model=model,
                reasoning_effort=reasoning_effort,
            ):
                validate_authored(
                    {
                        "synopsis": cached.get("synopsis"),
                        "chapters": [
                            {
                                "id": chapter.get("id"),
                                "start_step_id": (chapter.get("start") or {}).get("step_id"),
                                "end_step_id": (chapter.get("end") or {}).get("step_id"),
                                "title": chapter.get("title"),
                                "summary": chapter.get("summary"),
                            }
                            for chapter in cached.get("chapters") or []
                        ],
                    },
                    payload,
                )
                return cached
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    run_id = str((payload.get("run") or {}).get("run_id") or trajectory_path.stem)
    capture_dir = capture_dir or capture_path(repository, run_id)
    started_at = utc_now()
    try:
        authored = invoke_codex(
            trajectory_path,
            repository=repository,
            model=model,
            reasoning_effort=reasoning_effort,
            codex_bin=codex_bin,
            timeout_seconds=timeout_seconds,
            capture_dir=capture_dir,
        )
        artifact = build_artifact(
            authored,
            payload,
            trajectory_sha256=trajectory_sha256,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    except BaseException as exc:
        capture_manifest(
            capture_dir,
            {
                "schema_version": "trajectory-outline-capture/v1",
                "status": (
                    "interrupted"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "failed"
                ),
                "run_id": run_id,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "trajectory": str(trajectory_path.resolve()),
                "trajectory_sha256": trajectory_sha256,
                "started_at": started_at,
                "finished_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    atomic_json(output_path, artifact)
    capture_manifest(
        capture_dir,
        {
            "schema_version": "trajectory-outline-capture/v1",
            "status": "complete",
            "run_id": run_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "trajectory": str(trajectory_path.resolve()),
            "trajectory_sha256": trajectory_sha256,
            "outline": str(output_path.resolve()),
            "chapter_count": len(artifact["chapters"]),
            "started_at": started_at,
            "finished_at": utc_now(),
        },
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="durable directory for Codex JSONL events, final response, and status",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.trajectory.with_suffix(".outline.json")
    run_id = str((json.loads(args.trajectory.read_text()).get("run") or {}).get("run_id") or args.trajectory.stem)
    capture_dir = args.capture_dir or capture_path(args.repository, run_id)
    artifact = generate(
        args.trajectory,
        output_path=output,
        repository=args.repository,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        codex_bin=args.codex_bin,
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        capture_dir=capture_dir,
    )
    print(
        json.dumps(
            {
                "run_id": artifact["run_id"],
                "chapters": len(artifact["chapters"]),
                "output": str(output),
                "generator": artifact["generator"],
                "capture": str(capture_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
