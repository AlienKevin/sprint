#!/usr/bin/env python3
"""Scrub g1-sprint Harbor job dirs and upload privately to Harbor Hub.

Auth: uses HARBOR_API_KEY (sk-harbor-...) from the environment or
/data/harbor-adapters-experiments/.env. Dead UUID / revoked keys are skipped.
When auth fails, scrubbed bundles are still prepared under hub-scrubbed/.

Upload path is always: copy → scrub → leak-gate → (auth) → harbor upload.
Does not modify live job directories. Does not kill training runs.
Never prints secret values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SCRUB_ROOT = RUNS / "hub-scrubbed"
LEDGER = RUNS / "HUB_JOBS.md"
STATE = RUNS / ".hub_track_state.json"
ENV_FILE = ROOT / ".env"
KEEPALIVE = Path("/data/.keepalive")
HARBOR_CANDIDATES = (ROOT / "harbor",)
UV = Path(os.environ.get("UV", "/home/ubuntu/.local/bin/uv"))

JOB_ROOT_NAMES = (
    "jobs",
    "jobs-lane",
    "jobs-kimi-k3",
    "jobs-terra",
    "jobs-kimi",
    "jobs-sprint",
)

# Pattern redaction (values never logged). Order: specific before generic.
SECRET_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-oat01-[A-Za-z0-9_-]{20,}"), "[REDACTED_ANT_OAT]"),
    (re.compile(r"sk-ant-api0[0-9]-[A-Za-z0-9_-]{20,}"), "[REDACTED_ANT_API]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "[REDACTED_ANT]"),
    (re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENAI_PROJ]"),
    (re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENROUTER]"),
    (re.compile(r"sk-harbor-[A-Za-z0-9_-]{20,}"), "[REDACTED_HARBOR]"),
    # Modal Shared Endpoint bearer: wk-<id>.ws-<secret>
    (re.compile(r"\bwk-[A-Za-z0-9_-]+\.ws-[A-Za-z0-9_-]+\b"), "[REDACTED_MODAL_WK]"),
    # Generic OpenAI-style keys (after specific prefixes above).
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_SK]"),
    (
        re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-+/=]{16,}"),
        r"\1[REDACTED_BEARER]",
    ),
    (
        re.compile(r"(?i)(?<![A-Za-z0-9_-])(bearer\s+)[A-Za-z0-9._\-+/=]{16,}"),
        r"\1[REDACTED_BEARER]",
    ),
    (
        re.compile(
            r"(?i)(bearer\s+)(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
        ),
        r"\1[REDACTED_JWT]",
    ),
    # URLs with embedded credentials.
    (
        re.compile(r"(https?://)([^/\s:@]+):([^/\s:@]+)@"),
        r"\1[REDACTED_USER]:[REDACTED_PASS]@",
    ),
    (
        re.compile(
            r"(CLAUDE_CODE_OAUTH_TOKEN=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(ANTHROPIC_API_KEY=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(ANTHROPIC_AUTH_TOKEN=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(OPENROUTER_API_KEY=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(OPENAI_API_KEY=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(DEEPSEEK_API_KEY=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(KIMI_MODEL_API_KEY=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(HARBOR_API_KEY=)(?!\$\{)(?!\[REDACTED)([^\s\"']+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(MODAL_(?:TOKEN_(?:ID|SECRET)|PROXY_TOKEN_(?:ID|SECRET))=)"
            r"(?!\*+)(?!\[REDACTED)([^\s\"']+)"
        ),
        r"\1[REDACTED]",
    ),
    # Harbor partial redactions like sk-a****gAA still leak prefix/suffix.
    (re.compile(r"sk-[A-Za-z0-9_-]{1,12}\*{2,}[A-Za-z0-9_-]{0,12}"), "[REDACTED]"),
]

# Post-scrub self-check: any hit blocks upload. Patterns must not match our
# [REDACTED_*] placeholders.
LEAK_GATE: list[tuple[str, re.Pattern[str]]] = [
    ("sk-ant", re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")),
    ("sk-proj", re.compile(r"sk-proj-[A-Za-z0-9_-]{8,}")),
    ("sk-or", re.compile(r"sk-or-v1-[A-Za-z0-9_-]{8,}")),
    ("sk-harbor", re.compile(r"sk-harbor-[A-Za-z0-9_-]{8,}")),
    ("sk-generic", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("wk-modal", re.compile(r"\bwk-[A-Za-z0-9_-]+\.ws-[A-Za-z0-9_-]+\b")),
    (
        "bearer",
        re.compile(
            r"(?i)(?:authorization:\s*)?bearer\s+(?!\[REDACTED)[A-Za-z0-9._\-+/=]{16,}"
        ),
    ),
    ("url-userinfo", re.compile(r"https?://[^/\s:@\[]+:[^/\s:@\[]+@")),
    (
        "env-assign",
        re.compile(
            r"(?i)\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN|"
            r"CLAUDE_CODE_OAUTH_TOKEN|DEEPSEEK_API_KEY|KIMI_MODEL_API_KEY|"
            r"OPENROUTER_API_KEY|HARBOR_API_KEY|"
            r"MODAL_TOKEN_(?:ID|SECRET)|MODAL_PROXY_TOKEN_(?:ID|SECRET))"
            r"=(?!\$\{)(?!\[REDACTED)[^\s\"']{8,}"
        ),
    ),
    ("partial-sk", re.compile(r"sk-[A-Za-z0-9_-]{1,12}\*{2,}[A-Za-z0-9_-]{0,12}")),
]

SENSITIVE_ENV_KEYS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_MODEL_API_KEY",
    "HARBOR_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_PROXY_TOKEN_ID",
    "MODAL_PROXY_TOKEN_SECRET",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def touch_keepalive() -> None:
    KEEPALIVE.parent.mkdir(parents=True, exist_ok=True)
    KEEPALIVE.write_text(utc_now() + "\n")


def harbor_cwd() -> str:
    for path in HARBOR_CANDIDATES:
        if path.is_dir():
            return str(path)
    return str(Path.cwd())


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def collect_literal_secrets() -> set[str]:
    secrets: set[str] = set()
    for key in SENSITIVE_ENV_KEYS:
        val = os.environ.get(key)
        if val and len(val) >= 8:
            secrets.add(val)
    oauth_path = RUNS / ".oauth_token"
    if oauth_path.exists():
        try:
            val = oauth_path.read_text().strip()
            if val and len(val) >= 8:
                secrets.add(val)
        except OSError:
            pass
    secrets_dir = RUNS / ".secrets"
    if secrets_dir.is_dir():
        for path in secrets_dir.glob("*.env"):
            try:
                for line in path.read_text().splitlines():
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    _, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v and len(v) >= 8:
                        secrets.add(v)
            except OSError:
                pass
    return secrets


def discover_job_dirs() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def add(job_dir: Path) -> None:
        resolved = job_dir.resolve()
        if resolved in seen:
            return
        if job_dir.is_dir() and (job_dir / "config.json").exists():
            seen.add(resolved)
            found.append(job_dir)

    for name in JOB_ROOT_NAMES:
        root = RUNS / name
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            add(child)
    for root in sorted(RUNS.glob("jobs*")):
        if not root.is_dir() or root.name in JOB_ROOT_NAMES:
            continue
        for child in sorted(root.iterdir()):
            add(child)
    # Durable launcher jobs: runs/ops/<run-id>/harbor-jobs/<job-name>/
    ops = RUNS / "ops"
    if ops.is_dir():
        for harbor_jobs in sorted(ops.glob("*/harbor-jobs")):
            if not harbor_jobs.is_dir():
                continue
            for child in sorted(harbor_jobs.iterdir()):
                add(child)
    return found


def job_label(job_dir: Path) -> str:
    # Prefer durable ops/<run>/harbor-jobs/<job> → ops/<run>/<job>
    parts = job_dir.parts
    if "ops" in parts and "harbor-jobs" in parts:
        try:
            ops_i = parts.index("ops")
            hj_i = parts.index("harbor-jobs")
            run_id = parts[ops_i + 1]
            job_name = parts[hj_i + 1]
            return f"ops/{run_id}/{job_name}"
        except (ValueError, IndexError):
            pass
    return f"{job_dir.parent.name}/{job_dir.name}"


def scrub_text(text: str, literals: set[str]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text
    for secret in sorted(literals, key=len, reverse=True):
        n = out.count(secret)
        if n:
            out = out.replace(secret, "[REDACTED]")
            counts["literal_secret"] = counts.get("literal_secret", 0) + n
    for pat, repl in SECRET_RES:
        out, n = pat.subn(repl, out)
        if n:
            counts[pat.pattern[:40]] = counts.get(pat.pattern[:40], 0) + n
    return out, counts


def is_text_file(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return False
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def copy_and_scrub(job_dir: Path, literals: set[str]) -> tuple[Path, dict[str, int]]:
    label = job_label(job_dir)
    dest = SCRUB_ROOT / Path(label)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(job_dir, dest, symlinks=False, ignore_dangling_symlinks=True)

    totals: dict[str, int] = {}
    for path in dest.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if not is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scrubbed, counts = scrub_text(text, literals)
        if scrubbed != text:
            path.write_text(scrubbed, encoding="utf-8")
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
    return dest, totals


def leak_scan(root: Path) -> list[dict[str, str]]:
    """Return residual secret-prefix hits (path + kind only; no matched text)."""
    hits: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if not is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for kind, pat in LEAK_GATE:
            if pat.search(text):
                rel = str(path.relative_to(root))
                hits.append({"kind": kind, "path": rel})
                break
    return hits


def assert_scrubbed_clean(root: Path) -> tuple[bool, str]:
    hits = leak_scan(root)
    if not hits:
        return True, "leak-gate clean"
    # Aggregate kinds only — never echo matched secret text.
    kinds: dict[str, int] = {}
    for hit in hits:
        kinds[hit["kind"]] = kinds.get(hit["kind"], 0) + 1
    summary = ", ".join(f"{k}={n}" for k, n in sorted(kinds.items()))
    return False, f"leak-gate FAIL ({len(hits)} files): {summary}"


def auth_ok() -> tuple[bool, str]:
    key = os.environ.get("HARBOR_API_KEY", "")
    if not key:
        return False, "HARBOR_API_KEY unset"
    if not key.startswith("sk-harbor-"):
        return False, f"key format rejected (prefix={key[:12]!r}, not sk-harbor-)"
    env = os.environ.copy()
    env["HARBOR_API_KEY"] = key
    cwd = harbor_cwd()
    uv_bin = str(UV if UV.is_file() else "uv")
    try:
        proc = subprocess.run(
            [
                uv_bin,
                "run",
                "python",
                "-c",
                "import asyncio; from harbor.upload.db_client import UploadDB; "
                "print(asyncio.run(UploadDB().get_user_id()))",
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"auth probe error: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-1] if err else "unknown auth failure"
        safe, _ = scrub_text(tail, set())
        return False, safe[:200]
    uid = (proc.stdout or "").strip().splitlines()[-1]
    return True, f"user_id={uid}"


def try_upload(scrubbed_dir: Path) -> tuple[bool, str]:
    env = os.environ.copy()
    cwd = harbor_cwd()
    uv_bin = str(UV if UV.is_file() else "uv")
    cmd = [uv_bin, "run", "harbor", "upload", str(scrubbed_dir), "--private"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"upload error: {exc}"
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    safe_lines = []
    for line in out.splitlines():
        scrubbed, _ = scrub_text(line, set())
        if re.search(r"sk-(ant|or|harbor|proj)-", scrubbed, re.I):
            safe_lines.append("[line redacted: contained key-like token]")
        else:
            safe_lines.append(scrubbed)
    safe = "\n".join(safe_lines)[-2000:]
    if proc.returncode != 0:
        return False, safe or f"upload exit {proc.returncode}"
    url_m = re.search(r"https?://\S+", safe)
    id_m = re.search(
        r"(?:job[_ ]?id|uploaded job|hub)[:\s]+([0-9a-f-]{36})", safe, re.I
    )
    bits = []
    if url_m:
        bits.append(url_m.group(0).rstrip(").,"))
    if id_m:
        bits.append(f"id={id_m.group(1)}")
    if bits:
        return True, " ".join(bits)
    return True, safe.splitlines()[-1] if safe else "uploaded (no URL parsed)"


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {"jobs": {}, "auth_last": None, "auth_ok": False}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def write_ledger(state: dict, auth_msg: str) -> None:
    lines = [
        "# Harbor Hub job ledger (g1-sprint)",
        "",
        f"Updated: {utc_now()}",
        "",
        "## Auth",
        "",
        f"- Status: {'OK' if state.get('auth_ok') else 'BLOCKED'}",
        f"- Detail: {auth_msg}",
        "- Expected key format: `sk-harbor-...` (UUID keys return 401).",
        "- Checked: `/data/harbor-adapters-experiments/.env`, `~/.harbor/api_key.env`, process env, history candidates.",
        "- Unblock: `harbor auth login --no-browser` then `export HARBOR_API_KEY=...`, or mint a fresh Hub key and put it in the adapters `.env`.",
        "",
        "## Scrub policy",
        "",
        "- Live job dirs under `runs/jobs*` / `runs/ops/*/harbor-jobs` are never modified.",
        "- Upload-ready copies land in `runs/hub-scrubbed/<label>/`.",
        "- Redacts: `sk-` / `sk-proj-` / `sk-ant-` / `sk-or-v1-` / `sk-harbor-`, Modal `wk-*.ws-*`, Bearer/JWT, URL userinfo, sensitive env assignments, Harbor partial `sk-a****…`, plus host literal secrets.",
        "- Leak-gate: after scrub, scan for known prefixes; **upload refused** if any remain.",
        "- Entrypoint: `python3 runs/hub_track_upload.py --once` (also `qwopctl hub-upload`).",
        "- See `runs/SECRETS_AND_HUB.md`.",
        "",
        "## Jobs",
        "",
        "| Local path | Task / model | Scrubbed path | Hub | Scrub hits | Status |",
        "|---|---|---|---|---|---|",
    ]
    jobs = state.get("jobs", {})
    for key in sorted(jobs):
        j = jobs[key]
        hub = j.get("hub") or "—"
        scrub = j.get("scrubbed") or "—"
        hits = j.get("scrub_hits", 0)
        status = j.get("status") or ""
        notes = j.get("notes") or ""
        cell = f"{status}" + (f" — {notes}" if notes else "")
        identity = f"{j.get('task', '?')} / {j.get('model', '?')}"
        lines.append(
            f"| `{key}` | `{identity}` | `{scrub}` | {hub} | {hits} | {cell} |"
        )
    if not jobs:
        lines.append("| — | — | — | — | — | no jobs discovered yet |")
    missing = [
        name
        for name in ("jobs-kimi-k3", "jobs-terra")
        if not (RUNS / name).exists()
    ]
    lines.append("")
    lines.append("## Pending roots")
    lines.append("")
    if missing:
        lines.append(
            "- Not present yet (poller watching): "
            + ", ".join(f"`runs/{n}/`" for n in missing)
        )
    else:
        lines.append("- `jobs-kimi-k3` and `jobs-terra` are present.")
    lines.append("")
    lines.append("## Poller")
    lines.append("")
    lines.append(
        f"- Last poll: {state.get('last_poll', '—')}; interval_s={state.get('interval_s', '—')}"
    )
    lines.append(f"- Auth last checked: {state.get('auth_last', '—')}")
    lines.append(
        "- Log: `runs/logs/hub_track_upload.log`; script: `runs/hub_track_upload.py`"
    )
    lines.append("")
    LEDGER.write_text("\n".join(lines) + "\n")


def process_once(
    state: dict,
    *,
    force_rescrub: bool = False,
    scrub_only: bool = False,
) -> dict:
    touch_keepalive()
    load_env_file()
    literals = collect_literal_secrets()
    ok, auth_msg = auth_ok()
    state["auth_ok"] = ok
    state["auth_last"] = utc_now()
    state["last_poll"] = utc_now()

    for job_dir in discover_job_dirs():
        key = job_label(job_dir)
        entry = state.setdefault("jobs", {}).setdefault(
            key,
            {
                "local": str(job_dir),
                "status": "discovered",
                "hub": None,
                "scrubbed": None,
                "scrub_hits": 0,
                "notes": "",
                "mtime": None,
                "task": "?",
                "model": "?",
            },
        )
        try:
            cfg = json.loads((job_dir / "config.json").read_text())
            agents = cfg.get("agents") or []
            if agents:
                entry["model"] = (
                    agents[0].get("model_name")
                    or agents[0].get("model")
                    or entry.get("model")
                    or "?"
                )
            trials = [p.name for p in job_dir.iterdir() if p.is_dir()]
            if trials:
                entry["task"] = ",".join(sorted(trials))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        try:
            mtime = max(p.stat().st_mtime for p in job_dir.rglob("*") if p.is_file())
        except ValueError:
            mtime = job_dir.stat().st_mtime
        prev_mtime = entry.get("mtime")
        need_scrub = (
            force_rescrub
            or not entry.get("scrubbed")
            or not Path(entry["scrubbed"]).exists()
            or prev_mtime is None
            or mtime > float(prev_mtime)
        )
        if need_scrub:
            scrubbed, totals = copy_and_scrub(job_dir, literals)
            entry["scrubbed"] = str(scrubbed)
            entry["scrub_hits"] = int(sum(totals.values()))
            entry["mtime"] = mtime
            entry["scrubbed_at"] = utc_now()
            entry["scrub_kinds"] = sorted(totals)[:8]
            clean, gate_msg = assert_scrubbed_clean(scrubbed)
            entry["leak_gate_ok"] = clean
            entry["leak_gate"] = gate_msg
            if not clean:
                entry["status"] = "scrub_leak_blocked"
                entry["notes"] = gate_msg
            elif entry.get("status") == "uploaded":
                entry["status"] = "rescrubbed_pending_reupload"
            elif entry.get("status") not in {"uploaded"}:
                entry["status"] = "scrubbed_ready"

        # Always re-check leak gate before upload (even if scrub skipped).
        scrub_path = entry.get("scrubbed")
        if scrub_path and Path(scrub_path).exists():
            clean, gate_msg = assert_scrubbed_clean(Path(scrub_path))
            entry["leak_gate_ok"] = clean
            entry["leak_gate"] = gate_msg
            if not clean:
                entry["status"] = "scrub_leak_blocked"
                entry["notes"] = gate_msg
                continue

        if entry.get("status") == "scrub_leak_blocked":
            continue

        if not ok:
            if entry.get("status") != "uploaded":
                entry["status"] = "scrubbed_ready_auth_blocked"
                hit_note = (
                    f"redacted forms ({entry.get('scrub_hits', 0)} hits)"
                    if entry.get("scrub_hits")
                    else "no raw secret patterns found"
                )
                entry["notes"] = (
                    f"{hit_note}; leak-gate ok; upload blocked until fresh sk-harbor key"
                )
            continue

        if scrub_only:
            if entry.get("status") not in {"uploaded"}:
                entry["status"] = "scrubbed_ready"
                entry["notes"] = "scrubbed; leak-gate ok; upload skipped (--scrub-only)"
            continue

        if entry.get("status") in {
            "scrubbed_ready",
            "rescrubbed_pending_reupload",
            "upload_failed",
            "discovered",
            "scrubbed_ready_auth_blocked",
        }:
            success, msg = try_upload(Path(entry["scrubbed"]))
            if success:
                entry["status"] = "uploaded"
                entry["hub"] = msg
                entry["notes"] = f"uploaded {utc_now()}"
            else:
                entry["status"] = "upload_failed"
                entry["notes"] = f"upload failed {utc_now()}: {msg[:300]}"
    write_ledger(state, auth_msg)
    save_state(state)
    print(
        f"[{utc_now()}] auth={'OK' if ok else 'BLOCKED'} ({auth_msg[:120]}) "
        f"jobs={len(state.get('jobs', {}))} scrub_only={scrub_only}",
        flush=True,
    )
    return state


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scrub Harbor job dirs then privately upload to Hub."
    )
    ap.add_argument("--once", action="store_true", help="Single pass then exit")
    ap.add_argument("--interval", type=int, default=180, help="Poll seconds")
    ap.add_argument("--hours", type=float, default=6.0, help="How long to poll")
    ap.add_argument("--force-rescrub", action="store_true")
    ap.add_argument(
        "--scrub-only",
        action="store_true",
        help="Scrub + leak-gate only; never call harbor upload",
    )
    ap.add_argument(
        "--check",
        type=Path,
        metavar="DIR",
        help="Scan an existing scrubbed bundle for residual secret prefixes; exit 2 on hit",
    )
    args = ap.parse_args()

    touch_keepalive()

    if args.check is not None:
        root = args.check
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
        clean, msg = assert_scrubbed_clean(root)
        print(msg)
        return 0 if clean else 2

    state = load_state()
    state["interval_s"] = args.interval
    deadline = time.time() + args.hours * 3600
    while True:
        process_once(
            state,
            force_rescrub=args.force_rescrub,
            scrub_only=args.scrub_only,
        )
        args.force_rescrub = False
        if args.once or time.time() >= deadline:
            break
        time.sleep(max(30, args.interval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
