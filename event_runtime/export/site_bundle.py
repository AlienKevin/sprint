#!/usr/bin/env python3
"""Build the exact current-batch website payload used for deployment."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


DYNAMIC_DIRECTORIES = frozenset({"data", "captures", "replay"})
IGNORED_DIRECTORIES = frozenset({".git", ".vercel", "__pycache__", "node_modules"})
INDEX_FAMILIES = ("policies", "timelines", "timeline-overviews", "trajectories")


def is_atomic_staging_file(path: Path) -> bool:
    return path.name.startswith(".") and path.name.endswith(".tmp")


def _safe_relative(path: str) -> Path:
    relative = Path(unquote(urlsplit(path).path).lstrip("/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe public path in website data: {path!r}")
    return relative


def _link_or_copy(source: Path, destination: Path, *, root: Path) -> None:
    if source.is_symlink():
        resolved = source.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"website symlink escapes its root: {source}") from exc
        source = resolved
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    root: Path,
    skipped_top_level: frozenset[str] = frozenset(),
) -> None:
    if not source.is_dir():
        return
    for current, directories, files in os.walk(source):
        current_path = Path(current)
        relative_dir = current_path.relative_to(source)
        if relative_dir == Path("."):
            directories[:] = sorted(
                name
                for name in directories
                if name not in IGNORED_DIRECTORIES and name not in skipped_top_level
            )
        else:
            directories[:] = sorted(
                name for name in directories if name not in IGNORED_DIRECTORIES
            )
        for name in sorted(files):
            item = current_path / name
            if is_atomic_staging_file(item) or item.suffix == ".pyc":
                continue
            _link_or_copy(
                item,
                destination / relative_dir / name,
                root=root,
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid website JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"website JSON must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o644)


def _run_ids(batch: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(
        str(arm.get("run_id") or "")
        for arm in batch.get("arms", [])
        if isinstance(arm, dict)
    )
    if (
        not values
        or any(not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise RuntimeError("current batch must contain unique, non-empty run IDs")
    return values


def _public_asset_references(payloads: Iterable[dict[str, Any]]) -> set[Path]:
    references: set[Path] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            relative = _safe_relative(value)
            if relative.parts and relative.parts[0] in {"captures", "replay"}:
                references.add(relative)

    for payload in payloads:
        visit(payload)
    return references


def _resolve_asset(source: Path, relative: Path) -> tuple[Path, Path]:
    candidate = source / relative
    if candidate.is_file():
        return candidate, relative
    # Vercel clean URLs turn /replay/name into replay/name.html on disk.
    if relative.parts and relative.parts[0] == "replay" and not relative.suffix:
        html_relative = relative.with_suffix(".html")
        html = source / html_relative
        if html.is_file():
            return html, html_relative
    raise RuntimeError(f"current website data references a missing asset: /{relative}")


def _copy_current_dynamic_tree(
    source: Path, destination: Path, batch_path: Path
) -> tuple[str, ...]:
    batch = _read_json(batch_path)
    run_ids = _run_ids(batch)
    selected = set(run_ids)
    copied_payloads: list[dict[str, Any]] = [batch]

    _link_or_copy(
        batch_path,
        destination / "data/batches/current.json",
        root=source,
    )
    batch_id = str(batch.get("batch_id") or "")
    if batch_id:
        historical = source / "data" / "batches" / f"{batch_id}.json"
        if historical.is_file():
            _link_or_copy(
                historical,
                destination / "data" / "batches" / historical.name,
                root=source,
            )

    performance = source / "data" / "performance" / "current.json"
    if performance.is_file():
        performance_payload = _read_json(performance)
        observed = {
            str(row.get("run_id"))
            for row in performance_payload.get("runs", [])
            if isinstance(row, dict) and row.get("run_id")
        }
        if not observed.issubset(selected):
            raise RuntimeError(
                "performance/current.json contains runs outside the current batch"
            )
        _link_or_copy(
            performance,
            destination / "data/performance/current.json",
            root=source,
        )
        copied_payloads.append(performance_payload)

    for family in INDEX_FAMILIES:
        index_source = source / "data" / family / "index.json"
        entries: list[dict[str, Any]] = []
        index_payload: dict[str, Any] | None = None
        if index_source.is_file():
            index_payload = _read_json(index_source)
            entries = [
                row
                for row in index_payload.get("runs", [])
                if isinstance(row, dict) and str(row.get("run_id") or "") in selected
            ]
            filtered = dict(index_payload)
            filtered["runs"] = entries
            _write_json(destination / "data" / family / "index.json", filtered)
            copied_payloads.append(filtered)

        paths: set[Path] = {
            Path("data") / family / f"{run_id}.json" for run_id in run_ids
        }
        for entry in entries:
            path = entry.get("path")
            if isinstance(path, str):
                relative = _safe_relative(path)
                if relative.parts[:2] != ("data", family):
                    raise RuntimeError(
                        f"{family} index points outside its public directory: {path}"
                    )
                paths.add(relative)
        for relative in sorted(paths):
            item = source / relative
            if not item.is_file():
                continue
            payload = _read_json(item)
            declared = payload.get("run_id")
            if declared is not None and str(declared) not in selected:
                raise RuntimeError(f"selected artifact declares another run: {item}")
            _link_or_copy(item, destination / relative, root=source)
            # Policy and performance payloads are the public sources of replay
            # links. Avoid walking large raw trajectory strings that may merely
            # mention local /data paths in terminal output.
            if family == "policies":
                copied_payloads.append(payload)

    for relative in sorted(_public_asset_references(copied_payloads)):
        item, resolved_relative = _resolve_asset(source, relative)
        _link_or_copy(item, destination / resolved_relative, root=source)
    return run_ids


def build_site_bundle(
    source: Path,
    destination: Path,
    *,
    require_current: bool = False,
    include_project_link: bool = True,
) -> dict[str, Any]:
    """Materialize one immutable current-results-only deployment directory."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise RuntimeError("site bundle destination must be outside the source tree")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"site bundle destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    _copy_tree(
        source,
        destination,
        root=source,
        skipped_top_level=DYNAMIC_DIRECTORIES,
    )
    current = source / "data" / "batches" / "current.json"
    mode = "current_batch"
    run_ids: tuple[str, ...] = ()
    if current.is_file():
        run_ids = _copy_current_dynamic_tree(source, destination, current)
    elif require_current:
        raise RuntimeError(f"current batch index is missing: {current}")
    else:
        # Static-only fixtures and first deployments do not yet have a current
        # experiment. Preserve their complete tree without weakening the
        # current-batch filter used by production once current.json exists.
        mode = "full_tree_without_current_batch"
        for name in sorted(DYNAMIC_DIRECTORIES):
            _copy_tree(source / name, destination / name, root=source)

    if include_project_link:
        project = source / ".vercel" / "project.json"
        if project.is_file():
            _link_or_copy(
                project,
                destination / ".vercel/project.json",
                root=source,
            )

    files = [path for path in destination.rglob("*") if path.is_file()]
    return {
        "schema_version": 1,
        "mode": mode,
        "run_ids": list(run_ids),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-current",
        action="store_true",
        help="permit a static-only bundle before a current batch exists",
    )
    args = parser.parse_args()
    report = build_site_bundle(
        args.web,
        args.output,
        require_current=not args.allow_missing_current,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
