#!/usr/bin/env python3
"""Write a mode-0600 Harbor --env-file under /data/qwop-run-secrets/.

Secrets go in the env file (loaded into the Harbor process via dotenv) so they
never appear in process argv / ``ps``. Non-secret agent knobs may still use
``--ae KEY=value``.

Usage:
  python3 runs/ops/write_harbor_env_file.py --label terra-... \\
      OPENAI_API_KEY \\
      OPENAI_BASE_URL=https://openrouter.ai/api/v1

Keys without ``=`` are read from the current environment. Prints the env-file
path on stdout. Never prints secret values.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SECRETS_ROOT = Path("/data/qwop-run-secrets")
LABEL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{1,80}$")
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--label",
        required=True,
        help="Subdir under /data/qwop-run-secrets/ (safe filename chars).",
    )
    ap.add_argument(
        "--filename",
        default="harbor.env",
        help="Env file name inside the label dir (default harbor.env).",
    )
    ap.add_argument(
        "assignments",
        nargs="+",
        help="KEY or KEY=VALUE. Bare KEY reads os.environ[KEY].",
    )
    args = ap.parse_args()

    if not LABEL_RE.match(args.label):
        print("label must be 2-81 safe filename characters", file=sys.stderr)
        return 2
    if "/" in args.filename or args.filename in {".", ".."} or not args.filename:
        print("invalid filename", file=sys.stderr)
        return 2

    pairs: list[tuple[str, str]] = []
    for item in args.assignments:
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item, os.environ.get(item, "")
            if not value:
                print(f"{key} is unset or empty", file=sys.stderr)
                return 1
        if not KEY_RE.match(key):
            print(f"invalid env key: {key!r}", file=sys.stderr)
            return 2
        if "\n" in value or "\r" in value or "\0" in value:
            print(f"{key} contains a newline or NUL", file=sys.stderr)
            return 1
        pairs.append((key, value))

    dest_dir = SECRETS_ROOT / args.label
    dest_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    # Tighten dir perms even if it already existed.
    os.chmod(dest_dir, 0o700)
    dest = dest_dir / args.filename
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            os.chmod(tmp, 0o600)
            for key, value in pairs:
                fh.write(f"{key}={value}\n")
        os.replace(tmp, dest)
        os.chmod(dest, 0o600)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
