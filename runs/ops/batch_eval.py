#!/usr/bin/env python3
"""Reproducible six-trial Sprint launch, monitoring, and website publishing."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import frontier_update  # noqa: E402
import sprintctl  # noqa: E402


UV = Path(os.environ.get("UV", "/home/ubuntu/.local/bin/uv"))
WEB = ROOT / "sprint-web"
BATCH_ROOT = SCRIPT_DIR / "batches"
HARBOR_REVISION = "2f50d4c78bac5420b50d5cd15bc549a9bb19fa9d"
CODEX_VERSION = "0.147.0"
TRIALS_PER_MODEL = 3
REASONING_EFFORT = "max"
RUN_HOURS = 24.0
POLL_SECONDS = 30
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,48}$")
ALERT_PATTERNS = {
    "provider_rate_limit": re.compile(
        r"\b(?:429|rate.?limit|too many requests)\b", re.I
    ),
    "provider_quota": re.compile(
        r"\b(?:insufficient_quota|quota exceeded|billing limit)\b", re.I
    ),
    "provider_auth": re.compile(
        r"\b(?:401|403|invalid api key|authentication failed)\b", re.I
    ),
    "modal_infrastructure": re.compile(
        r"\b(?:modal.*(?:internal|connection|timeout)|sandbox.*failed|container.*lost)\b",
        re.I,
    ),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic_json(path: Path, payload: Any, mode: int = 0o600) -> None:
    frontier_update.atomic_write_json(path, payload, mode=mode)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if name.strip() in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"} and value:
            values[name.strip()] = value
    return values


def matrix(
    batch_id: str, trials_per_model: int = TRIALS_PER_MODEL
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    specs = (
        ("deepseek", "deepseek/deepseek-v4-flash", "run-deepseek.sh"),
        ("luna", "openai/gpt-5.6-luna", "run-luna.sh"),
    )
    for family, model, wrapper in specs:
        for trial in range(1, trials_per_model + 1):
            run_id = f"{batch_id}-{family}-{trial}"
            if not RUN_ID_RE.fullmatch(run_id):
                raise ValueError(f"generated run ID is invalid: {run_id}")
            arms.append(
                {
                    "run_id": run_id,
                    "family": family,
                    "model": model,
                    "resolved_model_version": (
                        "DeepSeek-V4-Flash-0731"
                        if family == "deepseek"
                        else "gpt-5.6-luna"
                    ),
                    "reasoning_effort": REASONING_EFFORT,
                    "codex_version": CODEX_VERSION,
                    "wrapper": str(ROOT / "runs" / wrapper),
                    "trial": trial,
                    "status": "planned",
                }
            )
    return arms


def batch_dir(batch_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(batch_id):
        raise ValueError("batch ID must be 3-49 safe filename characters")
    return BATCH_ROOT / batch_id


def batch_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "batch.json"


def read_batch(batch_id: str) -> dict[str, Any]:
    path = batch_path(batch_id)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"unknown batch: {batch_id}") from exc
    if payload.get("batch_id") != batch_id:
        raise ValueError("batch state has mismatched ID")
    return payload


def provider_models(url: str, key: str) -> set[str]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"provider model-list request failed: HTTP {exc.code}"
        ) from exc
    return {
        str(row.get("id"))
        for row in payload.get("data", [])
        if isinstance(row, dict) and row.get("id")
    }


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout


def vercel_project_link_ready() -> bool:
    """Accept Vercel CLI metadata while requiring the exact Sprint project."""
    try:
        frontier_update.verify_project_link(WEB)
    except RuntimeError:
        return False
    return True


def preflight(
    *,
    batch_id: str,
    env_file: Path,
    modal_profile: str,
    require_fresh: bool = True,
    check_providers: bool = True,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    keys = load_env(env_file)
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        checks[f"secret_{name.lower()}"] = len(keys.get(name, "")) >= 16
    checks["harbor_revision"] = (
        ROOT / "harbor/.sprint-upstream-commit"
    ).read_text().strip() == HARBOR_REVISION
    checks["goal_template"] = (ROOT / "runs/codex-goal-slash.j2").read_bytes() == (
        ROOT / "runs/codex-goal.j2"
    ).read_bytes()
    checks["warm_images"] = (
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "check_modal_image_warmup.py")],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    checks["vercel_project_link"] = vercel_project_link_ready()
    command_env = dict(os.environ)
    command_env["MODAL_PROFILE"] = modal_profile
    checks["modal_auth"] = False
    try:
        json.loads(
            run_checked(
                [
                    str(UV),
                    "run",
                    "--project",
                    str(ROOT / "harbor"),
                    "modal",
                    "app",
                    "list",
                    "--json",
                ],
                env=command_env,
            )
        )
        checks["modal_auth"] = True
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        pass
    checks["vercel_auth"] = (
        subprocess.run(
            ["vercel", "whoami"],
            cwd=WEB,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if check_providers and checks["secret_openai_api_key"]:
        models = provider_models(
            "https://api.openai.com/v1/models", keys["OPENAI_API_KEY"]
        )
        checks["openai_luna_visible"] = "gpt-5.6-luna" in models
    else:
        checks["openai_luna_visible"] = not check_providers
    if check_providers and checks["secret_deepseek_api_key"]:
        models = provider_models(
            "https://api.deepseek.com/models", keys["DEEPSEEK_API_KEY"]
        )
        checks["deepseek_v4_flash_visible"] = "deepseek-v4-flash" in models
    else:
        checks["deepseek_v4_flash_visible"] = not check_providers
    planned = matrix(batch_id)
    checks["six_unique_runs"] = len({arm["run_id"] for arm in planned}) == 6
    checks["fresh_run_ids"] = not any(
        (SCRIPT_DIR / arm["run_id"]).exists() for arm in planned
    )
    if require_fresh:
        checks["fresh_batch_id"] = not batch_path(batch_id).exists()
    ready = all(bool(value) for value in checks.values())
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "batch_id": batch_id,
        "modal_profile": modal_profile,
        "env_file": str(env_file),
        "checks": checks,
        "ready": ready,
    }


def start_monitor_service(batch_id: str, env_file: Path, modal_profile: str) -> None:
    vercel = shutil.which("vercel")
    if not vercel:
        raise RuntimeError("vercel CLI is not available for the batch monitor")
    service_path = os.pathsep.join(
        dict.fromkeys(
            [
                str(Path(sys.executable).resolve().parent),
                str(UV.resolve().parent),
                str(Path(vercel).resolve().parent),
                *os.environ.get("PATH", "").split(os.pathsep),
            ]
        )
    )
    unit = f"sprint-batch-{batch_id}-monitor"
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit}.service"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    run_checked(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--collect",
            "--property=Restart=on-failure",
            "--property=RestartSec=30",
            f"--setenv=PATH={service_path}",
            f"--setenv=UV={UV.resolve()}",
            f"--setenv=MODAL_PROFILE={modal_profile}",
            sys.executable,
            str(Path(__file__).resolve()),
            "monitor",
            "--batch-id",
            batch_id,
            "--env-file",
            str(env_file),
            "--modal-profile",
            modal_profile,
            "--loop",
        ]
    )


def launch(batch_id: str, env_file: Path, modal_profile: str) -> dict[str, Any]:
    report = preflight(
        batch_id=batch_id,
        env_file=env_file,
        modal_profile=modal_profile,
    )
    if not report["ready"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError("preflight failed: " + ", ".join(failed))
    state_dir = batch_dir(batch_id)
    state_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    payload = {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": started,
        "reasoning_effort": REASONING_EFFORT,
        "codex_version": CODEX_VERSION,
        "trials_per_model": TRIALS_PER_MODEL,
        "run_hours": RUN_HOURS,
        "modal_profile": modal_profile,
        "preflight": report,
        "arms": matrix(batch_id),
        "alerts": [],
        "deploy": {},
        "status": "launching",
    }
    atomic_json(batch_path(batch_id), payload)
    keys = load_env(env_file)
    base_env = dict(os.environ)
    base_env.update(keys)
    base_env.update(
        {
            "MODAL_PROFILE": modal_profile,
            "CONFIRM_LAUNCH": "1",
            "REASONING_EFFORT": REASONING_EFFORT,
            "CODEX_VERSION": CODEX_VERSION,
            "SPRINT_BATCH_ID": batch_id,
            "UV": str(UV),
        }
    )
    launched: list[dict[str, Any]] = []
    try:
        for arm in payload["arms"]:
            arm["launch_started_at"] = utc_now()
            atomic_json(batch_path(batch_id), payload)
            env = dict(base_env)
            env["RUN_ID"] = arm["run_id"]
            env["MODEL"] = arm["model"]
            output = run_checked([arm["wrapper"]], env=env)
            arm["status"] = "launched"
            arm["launch_output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
            arm["launched_at"] = utc_now()
            arm["deadline_at"] = (
                parse_time(arm["launched_at"]) + dt.timedelta(hours=RUN_HOURS)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            launched.append(arm)
            atomic_json(batch_path(batch_id), payload)
            time.sleep(2)
    except Exception as exc:
        arm["status"] = "launch_error"
        arm["launch_error"] = f"{type(exc).__name__}: {exc}"
        payload["status"] = "rolling_back_partial_launch"
        for started_arm in launched:
            try:
                sprintctl.request_stop(
                    started_arm["run_id"], reason="partial_batch_launch_rollback"
                )
                started_arm["status"] = "stopping_after_launch_rollback"
                started_arm["stop_requested_at"] = utc_now()
            except Exception as stop_exc:  # noqa: BLE001
                started_arm["rollback_error"] = f"{type(stop_exc).__name__}: {stop_exc}"
        payload["status"] = "launch_error"
        atomic_json(batch_path(batch_id), payload)
        raise
    payload["status"] = "running"
    payload["launched_at"] = utc_now()
    atomic_json(batch_path(batch_id), payload)
    start_monitor_service(batch_id, env_file, modal_profile)
    return payload


def log_alerts(run_id: str) -> list[dict[str, str]]:
    paths = [
        Path(f"/data/sprint-launch-{run_id}.log"),
        SCRIPT_DIR / run_id / "monitor.log",
        SCRIPT_DIR / run_id / "frontier-worker.log",
        SCRIPT_DIR / run_id / "controller-errors.jsonl",
    ]
    alerts: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 256_000))
                text = handle.read().decode(errors="replace")
        except OSError:
            continue
        for kind, pattern in ALERT_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                alerts.append(
                    {
                        "run_id": run_id,
                        "kind": kind,
                        "source": path.name,
                        "count_in_tail": str(len(matches)),
                    }
                )
    return alerts


def public_batch(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batch_id": payload["batch_id"],
        "updated_at": payload.get("updated_at"),
        "status": payload.get("status"),
        "reasoning_effort": payload["reasoning_effort"],
        "codex_version": payload["codex_version"],
        "run_hours": payload["run_hours"],
        "arms": [
            {
                key: arm.get(key)
                for key in (
                    "run_id",
                    "family",
                    "model",
                    "resolved_model_version",
                    "trial",
                    "status",
                    "launched_at",
                    "deadline_at",
                    "finalized_at",
                    "ledger",
                    "snapshot_heartbeat_ok",
                    "frontier_worker_alive",
                    "harbor_alive",
                )
            }
            for arm in payload["arms"]
        ],
        "alerts": payload.get("alerts", [])[-100:],
    }


def _frontier_ready_for_publish(state: dict[str, Any]) -> bool:
    if any(
        item.get("status") in {"queued", "running"}
        for item in state.get("capture_queue", [])
    ):
        return False
    captures = state.get("captures", {})
    return all(
        not policy.get("replay_path")
        or captures.get(policy_hash, {}).get("valid") is True
        for policy_hash, policy in state.get("policies", {}).items()
    )


def mark_deployed_runs(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Persist durable per-run proof that its current policy index was deployed."""
    alerts: list[dict[str, str]] = []
    deploy_state = payload.get("deploy") or {}
    current_hash = frontier_update.site_tree_hash(WEB)
    if (
        deploy_state.get("site_status") not in {"deployed", "noop"}
        or deploy_state.get("last_deployed_site_hash") != current_hash
    ):
        return alerts
    for arm in payload["arms"]:
        state_dir = SCRIPT_DIR / arm["run_id"]
        frontier_path = state_dir / "frontier-state.json"
        index_path = WEB / "data" / "policies" / f"{arm['run_id']}.json"
        try:
            frontier = json.loads(frontier_path.read_text())
            if not index_path.is_file() or not _frontier_ready_for_publish(frontier):
                continue
            marker = {
                "schema_version": 1,
                "run_id": arm["run_id"],
                "batch_id": payload["batch_id"],
                "deployed_at": deploy_state.get("last_deployed_at") or utc_now(),
                "deployment_site_sha256": current_hash,
                "policy_index_sha256": frontier_update.sha256_file(index_path),
                "production_alias": "https://g1-sprint.vercel.app",
            }
            local_marker = state_dir / "BATCH_SITE_DEPLOYED.json"
            try:
                existing = json.loads(local_marker.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
            if (
                existing.get("batch_id") == marker["batch_id"]
                and existing.get("policy_index_sha256") == marker["policy_index_sha256"]
            ):
                continue
            temp = batch_dir(payload["batch_id"]) / f".{arm['run_id']}-site.json"
            try:
                atomic_json(temp, marker)
                _, run = sprintctl.load_run(arm["run_id"])
                sprintctl.volume_upload(
                    run,
                    temp,
                    f"runs/{arm['run_id']}/state/BATCH_SITE_DEPLOYED.json",
                )
                atomic_json(local_marker, marker)
            finally:
                temp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            alerts.append(
                {
                    "run_id": arm["run_id"],
                    "kind": "site_marker",
                    "source": type(exc).__name__,
                    "count_in_tail": "1",
                }
            )
    return alerts


def monitor_cycle(batch_id: str, *, deploy: bool = True) -> dict[str, Any]:
    path = batch_path(batch_id)
    with frontier_update.file_lock(path.with_suffix(".lock")):
        payload = read_batch(batch_id)
        now = dt.datetime.now(dt.timezone.utc)
        finalized = 0
        cycle_alerts: list[dict[str, str]] = []
        for arm in payload["arms"]:
            run_id = arm["run_id"]
            run_path = SCRIPT_DIR / run_id / "run.json"
            if not run_path.is_file():
                if arm["status"] not in {"planned", "launch_error"}:
                    arm["status"] = "missing_run_state"
                continue
            try:
                status = sprintctl.monitor_once(run_id, upload=True)
                for key in (
                    "ledger",
                    "snapshot_heartbeat_ok",
                    "frontier_worker_alive",
                    "remote_error",
                    "agent_container_id",
                    "harbor_alive",
                    "stop_ack",
                ):
                    arm[key] = status.get(key)
                arm["last_monitor_at"] = utc_now()
            except Exception as exc:
                arm["monitor_error"] = f"{type(exc).__name__}: {exc}"
                cycle_alerts.append(
                    {
                        "run_id": run_id,
                        "kind": "monitor_error",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
            deadline = (
                parse_time(arm["deadline_at"]) if arm.get("deadline_at") else None
            )
            if deadline and now >= deadline and arm.get("stop_requested_at") is None:
                sprintctl.request_stop(run_id, reason="fixed_24h_batch_deadline")
                arm["stop_requested_at"] = utc_now()
                arm["status"] = "stopping"
            cycle_alerts.extend(log_alerts(run_id))

        known = {
            (item.get("run_id"), item.get("kind"), item.get("source"))
            for item in payload.get("alerts", [])
        }
        for alert in cycle_alerts:
            key = (alert.get("run_id"), alert.get("kind"), alert.get("source"))
            if key not in known:
                alert["first_seen_at"] = utc_now()
                payload.setdefault("alerts", []).append(alert)
                known.add(key)
        payload["updated_at"] = utc_now()
        public_path = WEB / "data" / "batches" / f"{batch_id}.json"
        atomic_json(public_path, public_batch(payload), mode=0o644)

        if deploy:
            deploy_state = payload.setdefault("deploy", {})
            try:
                frontier_update.deploy_if_needed(
                    deploy_state,
                    web=WEB,
                    debounce_seconds=60,
                )
            except Exception as exc:
                deploy_state["site_status"] = "error"
                cycle_alerts.append(
                    {
                        "run_id": "batch",
                        "kind": "website_deploy",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
        cycle_alerts.extend(mark_deployed_runs(payload))

        for arm in payload["arms"]:
            run_id = arm["run_id"]
            finalized_path = SCRIPT_DIR / run_id / "FINALIZED.json"
            if not finalized_path.is_file() and (
                arm.get("harbor_alive") is False or arm.get("stop_ack")
            ):
                try:
                    complete, result = sprintctl.finalize(run_id)
                    arm["finalization_conditions"] = result.get("conditions", {})
                    if complete:
                        arm["status"] = "finalized"
                except Exception as exc:  # noqa: BLE001
                    cycle_alerts.append(
                        {
                            "run_id": run_id,
                            "kind": "finalization",
                            "source": type(exc).__name__,
                            "count_in_tail": "1",
                        }
                    )
            if finalized_path.is_file():
                arm["status"] = "finalized"
                arm["finalized_at"] = arm.get("finalized_at") or utc_now()
                finalized += 1

        known = {
            (item.get("run_id"), item.get("kind"), item.get("source"))
            for item in payload.get("alerts", [])
        }
        for alert in cycle_alerts:
            key = (alert.get("run_id"), alert.get("kind"), alert.get("source"))
            if key not in known:
                alert["first_seen_at"] = utc_now()
                payload.setdefault("alerts", []).append(alert)
                known.add(key)
        all_finalized = finalized == len(payload["arms"])
        payload["status"] = "complete" if all_finalized else "running"
        payload["updated_at"] = utc_now()
        atomic_json(public_path, public_batch(payload), mode=0o644)
        if deploy and all_finalized:
            try:
                frontier_update.deploy_if_needed(
                    payload.setdefault("deploy", {}),
                    web=WEB,
                    debounce_seconds=0,
                )
            except Exception as exc:  # noqa: BLE001
                cycle_alerts.append(
                    {
                        "run_id": "batch",
                        "kind": "final_site_deploy",
                        "source": type(exc).__name__,
                        "count_in_tail": "1",
                    }
                )
        deployed_current = bool(
            not deploy
            or (
                payload.get("deploy", {}).get("site_status") in {"deployed", "noop"}
                and payload.get("deploy", {}).get("last_deployed_site_hash")
                == frontier_update.site_tree_hash(WEB)
            )
        )
        if all_finalized and not deployed_current:
            payload["status"] = "finalizing_site"
            payload["updated_at"] = utc_now()
            atomic_json(public_path, public_batch(payload), mode=0o644)
        known = {
            (item.get("run_id"), item.get("kind"), item.get("source"))
            for item in payload.get("alerts", [])
        }
        for alert in cycle_alerts:
            key = (alert.get("run_id"), alert.get("kind"), alert.get("source"))
            if key not in known:
                alert["first_seen_at"] = utc_now()
                payload.setdefault("alerts", []).append(alert)
                known.add(key)
        atomic_json(path, payload)
        return payload


def stop_batch(batch_id: str) -> dict[str, Any]:
    payload = read_batch(batch_id)
    for arm in payload["arms"]:
        if (
            arm.get("status") != "finalized"
            and (SCRIPT_DIR / arm["run_id"] / "run.json").is_file()
        ):
            sprintctl.request_stop(arm["run_id"], reason="operator_batch_stop")
            arm["stop_requested_at"] = arm.get("stop_requested_at") or utc_now()
            arm["status"] = "stopping"
    payload["updated_at"] = utc_now()
    atomic_json(batch_path(batch_id), payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "launch", "monitor", "status", "stop"):
        command = sub.add_parser(name)
        command.add_argument("--batch-id", required=True)
        command.add_argument("--env-file", type=Path, default=ROOT / ".env")
        command.add_argument(
            "--modal-profile", default=os.environ.get("MODAL_PROFILE", "kevinli020508")
        )
        if name == "launch":
            command.add_argument("--confirm", action="store_true")
        if name == "monitor":
            command.add_argument("--loop", action="store_true")
            command.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
            command.add_argument("--no-deploy", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        output = preflight(
            batch_id=args.batch_id,
            env_file=args.env_file.resolve(),
            modal_profile=args.modal_profile,
        )
    elif args.command == "launch":
        if not args.confirm:
            raise SystemExit("launch requires --confirm")
        output = launch(args.batch_id, args.env_file.resolve(), args.modal_profile)
    elif args.command == "monitor":
        while True:
            output = monitor_cycle(args.batch_id, deploy=not args.no_deploy)
            print(json.dumps(public_batch(output), indent=2), flush=True)
            if not args.loop or output.get("status") == "complete":
                break
            time.sleep(max(10, args.poll_seconds))
    elif args.command == "stop":
        output = stop_batch(args.batch_id)
    else:
        output = read_batch(args.batch_id)
    print(
        json.dumps(
            output if args.command == "preflight" else public_batch(output), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
