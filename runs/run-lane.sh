#!/usr/bin/env bash
# Compatibility entrypoint. The durable launcher owns run identity, snapshots,
# safe stop, and monitoring.
set -euo pipefail
exec /data/qwop-bench/runs/run-lane-durable.sh "$@"
