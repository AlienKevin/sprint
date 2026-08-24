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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume_name")
    parser.add_argument("remote_path")
    args = parser.parse_args()

    import modal

    try:
        volume = modal.Volume.from_name(
            args.volume_name,
            create_if_missing=False,
        )
        for chunk in volume.read_file(args.remote_path):
            sys.stdout.buffer.write(chunk)
    except FileNotFoundError:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
