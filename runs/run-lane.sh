#!/usr/bin/env bash
# Compatibility entrypoint. The durable launcher owns run identity, snapshots,
# safe stop, and monitoring.
set -euo pipefail
ROOT="${SPRINT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$ROOT/runs/run-lane-durable.sh" "$@"
