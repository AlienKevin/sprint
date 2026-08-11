#!/usr/bin/env python3
"""Detect runs that were cut off upstream but recorded as clean completions.

`FINALIZED.json` reports `complete: true` when Harbor's conditions are all met --
harbor_exited, stop_ack, ledger_terminal and so on. Those conditions are all
satisfied when a provider kills the agent mid-work, because the agent process
does exit cleanly and the ledger does go terminal. The run is then
indistinguishable from a genuine finish.

That happened on 2026-08-03: `lane-luna-20260803T104255Z` ended on

    {"type":"error","message":"Selected model is at capacity. ..."}
    {"type":"turn.failed", ...}

with three of four todo items still open, and was recorded `complete: true`.
Its "43 submissions, 0 valid" reads as a model result and is not one.

This is read-only and deliberately standalone rather than patched into
sprintctl.finalize, because sprintctl is imported by the live monitor of every
running arm. Fold it in when no paid arm is in flight.

    python3 event_runtime/preflight/check_upstream_truncation.py            # all runs
    python3 event_runtime/preflight/check_upstream_truncation.py RUN_ID ... # specific runs
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parent

# Terminal upstream failures. Deliberately broad: a false positive costs one
# manual look, a false negative silently publishes a truncated arm as a result.
PATTERNS = [
    (re.compile(r"at capacity", re.I), "provider_at_capacity"),
    (re.compile(r"\brate[ _-]?limit", re.I), "rate_limited"),
    (re.compile(r"\bquota\b", re.I), "quota_exhausted"),
    (re.compile(r"insufficient[_ ]quota|billing|payment required", re.I), "billing"),
    (re.compile(r"overloaded", re.I), "overloaded"),
    (re.compile(r"\b(401|403)\b.*(unauthor|forbidden)|invalid[_ ]api[_ ]key", re.I), "auth"),
    (re.compile(r'"type"\s*:\s*"turn\.failed"'), "turn_failed"),
    (re.compile(r"context[_ ]length[_ ]exceeded|too many tokens", re.I), "context_overflow"),
]

TRACE_NAMES = ("codex.txt", "claude-code.txt", "agent.txt", "kimi-code.txt")


def traces(run_dir: Path) -> list[Path]:
    out: list[Path] = []
    for name in TRACE_NAMES:
        out.extend(p for p in run_dir.rglob(name) if "agent" in p.as_posix())
    return out


def tail_bytes(path: Path, n: int = 200_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > n:
                fh.seek(size - n)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def inspect(run_dir: Path) -> dict:
    fin = run_dir / "FINALIZED.json"
    claimed = False
    if fin.is_file():
        try:
            claimed = bool(json.loads(fin.read_text()).get("complete"))
        except (OSError, json.JSONDecodeError):
            pass
    hits: dict[str, int] = {}
    tail_hits: list[str] = []
    for tr in traces(run_dir):
        text = tail_bytes(tr)
        if not text:
            continue
        for pat, label in PATTERNS:
            n = len(pat.findall(text))
            if n:
                hits[label] = hits.get(label, 0) + n
        # what the very end of the trace says is the strongest signal
        for line in text.strip().splitlines()[-6:]:
            for pat, label in PATTERNS:
                if pat.search(line):
                    tail_hits.append(label)
                    break
    return {
        "run_id": run_dir.name,
        "claimed_complete": claimed,
        "signals": hits,
        "terminal_signals": sorted(set(tail_hits)),
        # The verdict that matters: Harbor said finished, the trace says cut off.
        "truncated_but_marked_complete": bool(claimed and tail_hits),
    }


def main(argv: list[str]) -> int:
    if argv:
        dirs = [OPS / a for a in argv]
    else:
        dirs = sorted(p for p in OPS.glob("lane-*") if p.is_dir() and not p.name.startswith("."))
    bad = 0
    for d in dirs:
        if not d.is_dir():
            print(f"  {d.name}: no such run")
            continue
        r = inspect(d)
        if r["truncated_but_marked_complete"]:
            bad += 1
            print(f"  !! {r['run_id']}")
            print(f"       marked complete BUT trace ends on: {', '.join(r['terminal_signals'])}")
            print(f"       signals: {r['signals']}")
            print("       -> treat as incomplete_upstream; its scores are not a model result")
        elif r["signals"]:
            print(f"  ~  {r['run_id']}: upstream errors seen mid-run "
                  f"{r['signals']} (recovered; complete={r['claimed_complete']})")
        else:
            print(f"  ok {r['run_id']} (complete={r['claimed_complete']})")
    if bad:
        print(f"\n{bad} run(s) recorded as complete were actually truncated upstream.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
