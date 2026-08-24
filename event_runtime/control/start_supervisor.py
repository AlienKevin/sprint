#!/usr/bin/env python3
"""Start an event trial supervisor as a self-restarting systemd user service."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
OPS = ROOT / "runs" / "ops"


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch-argv-json", required=True)
    parser.add_argument("--secret-env", action="append", required=True)
    parser.add_argument("--launch-env", action="append", default=[])
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--max-restarts", type=int, default=50)
    parser.add_argument("--min-backoff-s", type=float, default=30)
    parser.add_argument("--max-backoff-s", type=float, default=600)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,48}", args.run_id):
        parser.error("unsafe run id")
    try:
        launch_argv = json.loads(args.launch_argv_json)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid launch argv: {exc}")
    if (
        not isinstance(launch_argv, list)
        or not launch_argv
        or not all(isinstance(item, str) and item for item in launch_argv)
    ):
        parser.error("launch argv must be a non-empty string array")
    secret_envs = list(dict.fromkeys(args.secret_env))
    allowed_secret_envs = {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "SPRINT_DEEPSEEK_PRICING_SNAPSHOT",
    }
    unsupported_secret_envs = sorted(set(secret_envs) - allowed_secret_envs)
    if unsupported_secret_envs:
        parser.error(
            "unsupported secret environment variable: "
            + ", ".join(unsupported_secret_envs)
        )
    launch_envs = list(dict.fromkeys(args.launch_env))
    allowed_launch_envs = {
        "OPENROUTER_MODEL",
        "SPRINT_CODEX_DEEPSEEK_CONTEXT_WINDOW",
        "SPRINT_CODEX_DEEPSEEK_MODEL",
        "SPRINT_OPENROUTER_PROVIDER_ENDPOINT",
        "SPRINT_OPENROUTER_QUANTIZATION",
    }
    unsupported_launch_envs = sorted(set(launch_envs) - allowed_launch_envs)
    if unsupported_launch_envs:
        parser.error(
            "unsupported launch environment variable: "
            + ", ".join(unsupported_launch_envs)
        )
    if args.batch_id and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{2,48}", args.batch_id
    ):
        parser.error("unsafe batch id")
    missing_secret_envs = [name for name in secret_envs if not os.environ.get(name)]
    if missing_secret_envs:
        parser.error(f"{', '.join(missing_secret_envs)} is not set")
    missing_launch_envs = [name for name in launch_envs if not os.environ.get(name)]
    if missing_launch_envs:
        parser.error(f"{', '.join(missing_launch_envs)} is not set")
    uv = os.environ.get("UV") or shutil.which("uv") or "/home/ubuntu/.local/bin/uv"
    if not Path(uv).is_file() or not os.access(uv, os.X_OK):
        parser.error("UV must name an executable absolute path")
    uv = str(Path(uv).resolve())
    harbor_python = ROOT / "harbor/.venv/bin/python3"
    if not harbor_python.is_file() or not os.access(harbor_python, os.X_OK):
        parser.error(f"missing executable controller runtime: {harbor_python}")
    # Keep the venv entrypoint path. Resolving its symlink to uv's base Python
    # discards pyvenv.cfg discovery and therefore the installed Modal package.
    harbor_python = harbor_python.absolute()
    tool_dirs = [str(harbor_python.parent), str(Path(uv).parent)]
    vercel = shutil.which("vercel")
    if vercel:
        tool_dirs.append(str(Path(vercel).resolve().parent))
    service_path = os.pathsep.join(
        dict.fromkeys([*tool_dirs, *os.environ.get("PATH", "").split(os.pathsep)])
    )

    unit = f"sprint-lane-{args.run_id}.service"
    state_dir = OPS / args.run_id
    log_path = Path(f"/data/sprint-launch-{args.run_id}.log")
    supervisor_argv = [
        str(harbor_python),
        "-u",
        str(MODULE_DIR / "supervisor.py"),
        "--run-id",
        args.run_id,
        "--launch-argv-json",
        json.dumps(launch_argv, separators=(",", ":")),
        "--max-restarts",
        str(args.max_restarts),
        "--min-backoff-s",
        str(args.min_backoff_s),
        "--max-backoff-s",
        str(args.max_backoff_s),
    ]
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "batch_id": args.batch_id or None,
        "unit": unit,
        "log_path": str(log_path),
        "launch_argv": launch_argv,
        "max_restarts": args.max_restarts,
        "process_manager_restart": "on-failure",
        "restart_prevent_exit_status": [75, 78],
        "uv": uv,
        "controller_python": str(harbor_python),
        "path": service_path,
        "launch_env_names": launch_envs,
    }
    atomic_write(state_dir / "supervisor.json", metadata)

    command = [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={unit}",
        "--property=Restart=on-failure",
        "--property=RestartSec=15s",
        "--property=RestartPreventExitStatus=75 78",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        "--setenv=MODAL_PROFILE",
        f"--setenv=UV={uv}",
        f"--setenv=PATH={service_path}",
    ]
    command.extend(f"--setenv={name}" for name in secret_envs)
    command.extend(f"--setenv={name}" for name in launch_envs)
    if args.batch_id:
        command.append(f"--setenv=SPRINT_BATCH_ID={args.batch_id}")
    command.extend(supervisor_argv)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            f"failed to start {unit}; no unmonitored fallback was used",
            file=os.sys.stderr,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
