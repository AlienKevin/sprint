#!/usr/bin/env python3
"""Supervise a durable CPU-agent trial: relaunch after loss unless operator stop.

Designed for long bakeoffs where Modal may preempt a GPU worker or Harbor/Codex
may exit. Remounts the same ``sprint-$RUN_ID`` volume via the original launcher
config. Prevents restart storms with a host lock + exponential backoff.

Does **not** launch paid bakeoffs by itself; callers pass ``--launch-cmd``
(or use ``--dry-run`` / unit tests with mocks).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs" / "ops"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

DEFAULT_MIN_BACKOFF_S = 30
DEFAULT_MAX_BACKOFF_S = 600
DEFAULT_MAX_RESTARTS = 50
UNRECOVERABLE_EXIT_CODES = {78}
SUPERVISOR_EXHAUSTED_EXIT = 75


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def state_dir_for(run_id: str) -> Path:
    return OPS / run_id


def stop_requested(state_dir: Path) -> tuple[bool, str]:
    """Operator stop must remain stopped."""
    if (state_dir / "FINALIZED.json").is_file():
        return True, "FINALIZED"
    for name in ("STOP_REQUESTED.json", "STOP_ACK.json"):
        path = state_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return True, f"{name}:unreadable"
        reason = str(payload.get("reason") or "")
        if name == "STOP_ACK.json" and reason and reason != "operator_stop":
            # Non-operator ack (e.g. agent crash marker) does not block relaunch.
            continue
        if name == "STOP_REQUESTED.json" or reason == "operator_stop":
            return True, f"{name}:{reason or 'operator_stop'}"
    # Also honor empty sentinel file used by some launchers.
    if (state_dir / "STOP").is_file():
        return True, "STOP"
    return False, ""


def next_backoff_s(
    attempt: int,
    *,
    min_s: float = DEFAULT_MIN_BACKOFF_S,
    max_s: float = DEFAULT_MAX_BACKOFF_S,
) -> float:
    """Exponential backoff capped at max_s. attempt is 1-based failure count."""
    if attempt < 1:
        attempt = 1
    delay = float(min_s) * (2 ** (attempt - 1))
    return float(min(max_s, delay))


def load_supervise_state(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "supervise-state.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "restarts": 0,
            "consecutive_failures": 0,
            "last_launch_at": None,
            "last_exit_code": None,
            "stopped": False,
        }
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "restarts": 0, "consecutive_failures": 0}


def save_supervise_state(state_dir: Path, payload: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "supervise-state.json"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def append_cpu_lifecycle(state_dir: Path, payload: dict[str, Any]) -> None:
    """Persist CPU sandbox boundaries independently of Harbor artifacts."""
    path = state_dir / "telemetry" / "cpu_lifecycle.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def next_cpu_attempt(state_dir: Path) -> int:
    try:
        run = json.loads((state_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    return int(run.get("cpu_launch_attempt") or 0) + 1


def write_failure_artifact(
    state_dir: Path, *, run_id: str, reason: str, state: dict[str, Any]
) -> None:
    """Persist a loud, machine-readable terminal supervisor failure."""
    payload = {
        "schema_version": 1,
        "at": utc_now(),
        "run_id": run_id,
        "reason": reason,
        "restarts": int(state.get("restarts") or 0),
        "consecutive_failures": int(state.get("consecutive_failures") or 0),
        "last_exit_code": state.get("last_exit_code"),
        "recoverable": False,
    }
    path = state_dir / "SUPERVISOR_FAILED.json"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def should_relaunch(
    *,
    stop: bool,
    harbor_alive: bool,
    max_restarts: int,
    restarts: int,
    consecutive_failures: int,
) -> tuple[bool, str]:
    if stop:
        return False, "operator_stop"
    if harbor_alive:
        return False, "already_alive"
    if restarts >= max_restarts:
        return False, "max_restarts"
    if consecutive_failures >= max_restarts:
        return False, "max_consecutive_failures"
    return True, "relaunch"


def run_loop(
    run_id: str,
    *,
    launch_fn: Callable[[], int],
    alive_fn: Callable[[], bool],
    stop_fn: Callable[[], tuple[bool, str]],
    sleep_fn: Callable[[float], None] = time.sleep,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    min_backoff_s: float = DEFAULT_MIN_BACKOFF_S,
    max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
    max_iterations: int | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Core supervise loop (injectable for unit tests)."""
    state_dir = state_dir or state_dir_for(run_id)
    state = load_supervise_state(state_dir)
    history: list[dict[str, Any]] = []
    iterations = 0

    lock_path = state_dir / "supervise.lock"
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "run_id": run_id,
                "stopped": True,
                "reason": "supervise_lock_busy",
                "history": history,
            }

        while True:
            iterations += 1
            if max_iterations is not None and iterations > max_iterations:
                break
            Path("/data/.keepalive").touch()
            stop, stop_reason = stop_fn()
            alive = bool(alive_fn())
            ok, reason = should_relaunch(
                stop=stop,
                harbor_alive=alive,
                max_restarts=max_restarts,
                restarts=int(state.get("restarts") or 0),
                consecutive_failures=int(state.get("consecutive_failures") or 0),
            )
            entry = {
                "ts": utc_now(),
                "alive": alive,
                "stop": stop,
                "stop_reason": stop_reason,
                "decision": reason,
            }
            if not ok:
                entry["action"] = (
                    "halt" if stop or reason.startswith("max_") else "wait"
                )
                history.append(entry)
                state["stopped"] = bool(stop)
                state["last_decision"] = reason
                save_supervise_state(state_dir, state)
                if stop or reason.startswith("max_"):
                    if reason.startswith("max_"):
                        write_failure_artifact(
                            state_dir, run_id=run_id, reason=reason, state=state
                        )
                    break
                sleep_fn(min_backoff_s)
                continue

            # Backoff before relaunch (not before first ever launch if restarts==0
            # and never launched; still backoff after failures).
            failures = int(state.get("consecutive_failures") or 0)
            if failures > 0 or int(state.get("restarts") or 0) > 0:
                delay = next_backoff_s(
                    max(1, failures), min_s=min_backoff_s, max_s=max_backoff_s
                )
                entry["backoff_s"] = delay
                sleep_fn(delay)
                # Re-check stop after sleep.
                stop, stop_reason = stop_fn()
                if stop:
                    entry["action"] = "halt_after_backoff"
                    entry["stop_reason"] = stop_reason
                    history.append(entry)
                    state["stopped"] = True
                    save_supervise_state(state_dir, state)
                    break

            entry["action"] = "launch"
            history.append(entry)
            cpu_attempt = next_cpu_attempt(state_dir)
            launch_started_at = utc_now()
            append_cpu_lifecycle(
                state_dir,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "event": "cpu_launch_started",
                    "attempt": cpu_attempt,
                    "at": launch_started_at,
                },
            )
            code = int(launch_fn())
            append_cpu_lifecycle(
                state_dir,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "event": "cpu_launch_exited",
                    "attempt": cpu_attempt,
                    "at": utc_now(),
                    "exit_code": code,
                    "started_at": launch_started_at,
                },
            )
            state["restarts"] = int(state.get("restarts") or 0) + 1
            state["last_launch_at"] = utc_now()
            state["last_exit_code"] = code
            if code in UNRECOVERABLE_EXIT_CODES:
                state["consecutive_failures"] = failures + 1
                state["last_decision"] = f"unrecoverable_exit:{code}"
                save_supervise_state(state_dir, state)
                write_failure_artifact(
                    state_dir,
                    run_id=run_id,
                    reason=f"unrecoverable_exit:{code}",
                    state=state,
                )
                break
            if code == 0:
                state["consecutive_failures"] = 0
            else:
                state["consecutive_failures"] = failures + 1
            save_supervise_state(state_dir, state)
            # After launch returns (process ended), loop to decide again.
    finally:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)

    return {
        "run_id": run_id,
        "stopped": bool(state.get("stopped")),
        "restarts": int(state.get("restarts") or 0),
        "consecutive_failures": int(state.get("consecutive_failures") or 0),
        "history": history,
        "state": state,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--launch-argv-json",
        default="",
        help="JSON argv array used to relaunch the same run and Volume.",
    )
    parser.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    parser.add_argument("--min-backoff-s", type=float, default=DEFAULT_MIN_BACKOFF_S)
    parser.add_argument("--max-backoff-s", type=float, default=DEFAULT_MAX_BACKOFF_S)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decisions only; never exec the launch argv.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single supervise iteration (for smoke / tests).",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MODAL_PROFILE", "kevinli020508")

    state_dir = state_dir_for(args.run_id)
    launch_argv: list[str] = []
    if args.launch_argv_json:
        try:
            parsed = json.loads(args.launch_argv_json)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --launch-argv-json: {exc}")
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(isinstance(item, str) and item for item in parsed)
        ):
            parser.error("--launch-argv-json must be a non-empty string array")
        launch_argv = parsed

    def stop_fn() -> tuple[bool, str]:
        return stop_requested(state_dir)

    def alive_fn() -> bool:
        if args.dry_run:
            return False
        try:
            from event_runtime.control import run as sprintctl

            _, run = sprintctl.load_run(args.run_id)
            return bool(sprintctl.harbor_alive(run))
        except Exception:  # noqa: BLE001
            return False

    def launch_fn() -> int:
        if args.dry_run or not launch_argv:
            print(json.dumps({"dry_run_launch": True, "run_id": args.run_id}))
            return 0
        print(f"supervise launch: {json.dumps(launch_argv)}", flush=True)
        return int(subprocess.call(launch_argv))

    if not args.dry_run and not launch_argv:
        print("--launch-argv-json required unless --dry-run", file=sys.stderr)
        return 2

    result = run_loop(
        args.run_id,
        launch_fn=launch_fn,
        alive_fn=alive_fn,
        stop_fn=stop_fn,
        max_restarts=args.max_restarts,
        min_backoff_s=args.min_backoff_s,
        max_backoff_s=args.max_backoff_s,
        max_iterations=1 if args.once else None,
        state_dir=state_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    decision = str(result.get("state", {}).get("last_decision") or "")
    if decision.startswith("unrecoverable_exit:"):
        return int(decision.rsplit(":", 1)[-1])
    if decision.startswith("max_"):
        return SUPERVISOR_EXHAUSTED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
