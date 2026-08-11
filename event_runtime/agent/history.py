#!/usr/bin/env python3
"""List this trial's archived policy receipts; official results remain blind."""

import json
import os
import sys

RECEIPTS = "/app/submissions/receipts"


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__)
        return 0
    try:
        names = sorted(name for name in os.listdir(RECEIPTS) if name.endswith(".json"))
    except (FileNotFoundError, PermissionError):
        names = []
    if not names:
        print("no policies submitted yet")
        return 0

    print(f"{len(names)} local submission request(s)")
    for name in names:
        try:
            with open(os.path.join(RECEIPTS, name), encoding="utf-8") as handle:
                receipt = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        note = str(receipt.get("note") or "")
        submission_id = receipt.get("submission_id", name[:-5])
        prefix = f"{submission_id}" + (f"  {note}" if note else "")
        print(f"{prefix}  submitted; official result hidden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
