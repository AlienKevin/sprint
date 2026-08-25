#!/usr/bin/env python3
"""Start one authoritative CPU-agent process as a non-restarting user service."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs" / "ops"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def append_lifecycle(state_dir: Path, payload: dict[str, object]) -> None:
    path = state_dir / "telemetry" / "cpu_lifecycle.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_run_id(parser: argparse.ArgumentParser, run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,48}", run_id):
        parser.error("unsafe run id")


def parse_argv(parser: argparse.ArgumentParser, raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid launch argv: {exc}")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        parser.error("launch argv must be a non-empty string array")
    expected = (ROOT / "event_runtime/control/launch.sh").resolve()
    if Path(value[0]).resolve() != expected:
        parser.error(f"launch argv must begin with {expected}")
    return value


def run_once(run_id: str, launch_argv: list[str]) -> int:
    """Run the trial exactly once and durably record its process boundary."""
    state_dir = OPS / run_id
    started_at = utc_now()
    started_epoch = time.time()
    append_lifecycle(
        state_dir,
        {
            "schema_version": 1,
            "event": "cpu_launch_started",
            "run_id": run_id,
            "attempt": 1,
            "at": started_at,
            "pid": os.getpid(),
            "execution_policy": "single_process_no_resume",
        },
    )

    child: subprocess.Popen[bytes] | None = None
    requested_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        child = subprocess.Popen(launch_argv, cwd=ROOT)
        raw_code = int(child.wait())
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    stop_requested = any(
        (state_dir / name).exists()
        for name in (
            "STOP",
            "STOP_REQUESTED.json",
            "BUDGET_STOP_REQUESTED.json",
            "STOP_ACK.json",
        )
    )
    if stop_requested or requested_signal is not None:
        reason = "requested_stop"
    elif raw_code == 0:
        reason = "agent_process_completed"
    elif raw_code < 0:
        reason = "agent_process_signal"
    else:
        reason = "agent_process_failed"
    finished_at = utc_now()
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt": 1,
        "execution_policy": "single_process_no_resume",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": max(0.0, time.time() - started_epoch),
        "raw_exit_code": raw_code,
        "reason": reason,
        "stop_requested": stop_requested,
        "forwarded_signal": requested_signal,
        "recoverable_in_place": False,
    }
    append_lifecycle(
        state_dir,
        {
            **payload,
            "event": "cpu_launch_exited",
            "at": finished_at,
        },
    )
    atomic_write(state_dir / "CPU_TRIAL_EXIT.json", payload)
    return raw_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch-argv-json", required=True)
    parser.add_argument("--secret-env", action="append", default=[])
    parser.add_argument("--launch-env", action="append", default=[])
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--run-once", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    validate_run_id(parser, args.run_id)
    launch_argv = parse_argv(parser, args.launch_argv_json)

    if args.run_once:
        return run_once(args.run_id, launch_argv)

    if args.batch_id and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{2,48}", args.batch_id
    ):
        parser.error("unsafe batch id")
    secret_envs = list(dict.fromkeys(args.secret_env))
    allowed_secret_envs = {
        "OPENROUTER_API_KEY",
    }
    unsupported_secrets = sorted(set(secret_envs) - allowed_secret_envs)
    if unsupported_secrets:
        parser.error(
            "unsupported secret environment variable: " + ", ".join(unsupported_secrets)
        )
    launch_envs = list(dict.fromkeys(args.launch_env))
    allowed_launch_envs = {
        "OPENROUTER_MODEL",
        "SPRINT_OPENROUTER_PROVIDER_ENDPOINT",
        "SPRINT_OPENROUTER_QUANTIZATION",
    }
    unsupported_launch = sorted(set(launch_envs) - allowed_launch_envs)
    if unsupported_launch:
        parser.error(
            "unsupported launch environment variable: " + ", ".join(unsupported_launch)
        )
    missing = [name for name in [*secret_envs, *launch_envs] if not os.environ.get(name)]
    if missing:
        parser.error(f"{', '.join(missing)} is not set")

    controller_python = ROOT / "harbor/.venv/bin/python3"
    if not controller_python.is_file() or not os.access(controller_python, os.X_OK):
        parser.error(f"missing executable controller runtime: {controller_python}")
    uv = os.environ.get("UV") or shutil.which("uv") or "/home/ubuntu/.local/bin/uv"
    if not Path(uv).is_file() or not os.access(uv, os.X_OK):
        parser.error("UV must name an executable absolute path")
    tool_dirs = [str(controller_python.parent), str(Path(uv).resolve().parent)]
    vercel = shutil.which("vercel")
    if vercel:
        tool_dirs.append(str(Path(vercel).resolve().parent))
    service_path = os.pathsep.join(
        dict.fromkeys([*tool_dirs, *os.environ.get("PATH", "").split(os.pathsep)])
    )

    unit = f"sprint-trial-{args.run_id}.service"
    state_dir = OPS / args.run_id
    log_path = Path(f"/data/sprint-launch-{args.run_id}.log")
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "batch_id": args.batch_id or None,
        "unit": unit,
        "log_path": str(log_path),
        "launch_argv": launch_argv,
        "process_manager_restart": "no",
        "cpu_execution_policy": "single_process_no_resume",
        "controller_python": str(controller_python),
        "path": service_path,
        "launch_env_names": launch_envs,
    }
    atomic_write(state_dir / "trial-launch.json", metadata)

    runner = [
        str(controller_python),
        "-u",
        str(Path(__file__).resolve()),
        "--run-once",
        "--run-id",
        args.run_id,
        "--launch-argv-json",
        json.dumps(launch_argv, separators=(",", ":")),
    ]
    command = [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={unit}",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        f"--property=WorkingDirectory={ROOT}",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        "--setenv=MODAL_PROFILE",
        f"--setenv=UV={Path(uv).resolve()}",
        f"--setenv=PATH={service_path}",
    ]
    command.extend(f"--setenv={name}" for name in secret_envs)
    command.extend(f"--setenv={name}" for name in launch_envs)
    if args.batch_id:
        command.append(f"--setenv=SPRINT_BATCH_ID={args.batch_id}")
    command.extend(runner)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"failed to start {unit}; no fallback was used", file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
