#!/usr/bin/env python3
"""Fail closed if an agent environment contains undeclared credentials."""

from __future__ import annotations

import re
import sys
from pathlib import Path

CLOUD_CONTROL_PLANE_NAMES = frozenset(
    {
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROFILE",
        "MODAL_CONFIG_PATH",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
    }
)
OPTIONAL_AGENT_NAMES = frozenset({"OPENAI_BASE_URL"})
NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def validate(path: Path, secret_name: str) -> set[str]:
    allowed = {secret_name, *OPTIONAL_AGENT_NAMES}
    names: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed agent env line {line_number}")
        name = line.split("=", 1)[0]
        if not NAME_RE.fullmatch(name):
            raise ValueError(f"invalid agent env name on line {line_number}")
        names.append(name)

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate agent env variables: {duplicates}")
    present = set(names)
    forbidden = sorted(present & CLOUD_CONTROL_PLANE_NAMES)
    if forbidden:
        raise ValueError(f"refusing cloud control-plane credentials: {forbidden}")
    unexpected = sorted(present - allowed)
    if unexpected:
        raise ValueError(f"refusing unexpected agent variables: {unexpected}")
    if secret_name not in present:
        raise ValueError(f"missing required agent credential: {secret_name}")
    return present


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: python -m event_runtime.control.credentials ENV_FILE SECRET_NAME",
            file=sys.stderr,
        )
        return 2
    try:
        validate(Path(args[0]), args[1])
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
