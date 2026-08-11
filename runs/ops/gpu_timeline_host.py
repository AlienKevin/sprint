#!/usr/bin/env python3
"""Host-side GPU timeline emit/summarize via Modal Volume (preemption-safe)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_DIR = Path(__file__).resolve().parents[2] / "events/g1-100-metres/environment"
sys.path.insert(0, str(SCRIPT_DIR))

import sprintctl  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sprint_gpu_timeline", ENV_DIR / "sprint-gpu-timeline.py"
)
timeline = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(timeline)


def _telem_prefix(run_id: str) -> str:
    return f"runs/{run_id}/telemetry"


def host_append_event(
    run: dict[str, Any],
    *,
    phase: str,
    action: str,
    job_id: str = "",
    attempt: int = 0,
    lease_id: str = "",
    epoch_s: int | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(run["run_id"])
    event = {
        "schema_version": 1,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch_s": int(time.time() if epoch_s is None else epoch_s),
        "run_id": run_id,
        "job_id": job_id or None,
        "attempt": int(attempt),
        "lease_id": lease_id or None,
        "phase": phase,
        "action": action,
        "detail": detail or {},
        "event_id": uuid.uuid4().hex[:12],
        "source": "host",
    }
    prefix = _telem_prefix(run_id)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(event, indent=2, sort_keys=True) + "\n")
        tmp = Path(handle.name)
    try:
        os.chmod(tmp, 0o600)
        sprintctl.volume_upload(
            run,
            tmp,
            f"{prefix}/gpu_timeline/events/{event['epoch_s']}_{event['event_id']}.json",
        )
    finally:
        tmp.unlink(missing_ok=True)

    remote_jsonl = f"{prefix}/gpu_timeline.jsonl"
    existing = sprintctl.volume_get_text(run, remote_jsonl) or ""
    if "✓ Finished" in existing:
        existing = existing.split("✓ Finished")[0]
    line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        handle.write(existing)
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(line)
        tmp2 = Path(handle.name)
    try:
        os.chmod(tmp2, 0o600)
        sprintctl.volume_upload(run, tmp2, remote_jsonl)
    finally:
        tmp2.unlink(missing_ok=True)

    state_dir = Path(str(run["state_dir"]))
    local = state_dir / "telemetry"
    local.mkdir(parents=True, exist_ok=True)
    with (local / "gpu_timeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return event


def host_write_summary(run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run["run_id"])
    prefix = _telem_prefix(run_id)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "telemetry"
        root.mkdir(parents=True)
        text = sprintctl.volume_get_text(run, f"{prefix}/gpu_timeline.jsonl")
        if text:
            (root / "gpu_timeline.jsonl").write_text(text)
        events_dir = root / "gpu_timeline" / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        sprintctl.run_command(
            sprintctl.modal_command(
                "volume",
                "get",
                str(run["volume_name"]),
                f"{prefix}/gpu_timeline/events",
                str(events_dir),
            ),
            run=run,
            check=False,
            timeout=120,
        )
        summary = timeline.summarize_events(timeline._load_events(root))
        summary["run_id"] = run_id
        summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = root / "gpu_time_summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        sprintctl.volume_upload(run, out, f"{prefix}/gpu_time_summary.json")
        state_dir = Path(str(run["state_dir"]))
        local = state_dir / "telemetry" / "gpu_time_summary.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(out.read_text())
        return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] not in {"emit", "summarize"}:
        print(
            "Usage: gpu_timeline_host.py emit --run-id ID --phase P "
            "--action enter|exit [--job-id J]\n"
            "       gpu_timeline_host.py summarize --run-id ID",
            file=sys.stderr,
        )
        return 2
    cmd = argv[0]
    run_id = None
    phase = None
    action = None
    job_id = ""
    i = 1
    while i < len(argv):
        if argv[i] == "--run-id":
            run_id = argv[i + 1]
            i += 2
        elif argv[i] == "--phase":
            phase = argv[i + 1]
            i += 2
        elif argv[i] == "--action":
            action = argv[i + 1]
            i += 2
        elif argv[i] == "--job-id":
            job_id = argv[i + 1]
            i += 2
        else:
            print(f"unknown arg {argv[i]}", file=sys.stderr)
            return 2
    if not run_id:
        return 2
    os.environ.setdefault("MODAL_PROFILE", "kevinli020508")
    _, run = sprintctl.load_run(run_id)
    if cmd == "emit":
        if not phase or not action:
            return 2
        print(
            json.dumps(
                host_append_event(
                    run, phase=phase, action=action, job_id=job_id
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(host_write_summary(run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
