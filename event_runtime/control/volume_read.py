#!/usr/bin/env python3
"""Read one exact Modal Volume file without a directory-list RPC.

``modal volume get`` resolves even an exact filename through
``VolumeListFiles``.  The benchmark controller reads several well-known
heartbeat, budget, and index files frequently, so using that CLI path turns
six concurrent trials into an account-wide directory-list storm.  Modal's
``Volume.read_file`` uses the exact-file RPC instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume_name")
    parser.add_argument("remote_path")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="fail before emitting more than this many bytes (0 means unlimited)",
    )
    parser.add_argument("--output", help="write bytes to this local file")
    args = parser.parse_args()

    import modal

    try:
        volume = modal.Volume.from_name(
            args.volume_name,
            create_if_missing=False,
        )
        emitted = 0
        output = Path(args.output) if args.output else None
        sink = output.open("wb") if output else sys.stdout.buffer
        try:
            for chunk in volume.read_file(args.remote_path):
                emitted += len(chunk)
                if args.max_bytes and emitted > args.max_bytes:
                    if output:
                        sink.close()
                        output.unlink(missing_ok=True)
                    return 4
                sink.write(chunk)
        finally:
            if output and not sink.closed:
                sink.close()
    except FileNotFoundError:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
