#!/usr/bin/env bash
# Stage the agent's working directory: somewhere to submit from, somewhere for
# local receipts to land, and the final answer's home.
set -euo pipefail

mkdir -p /app/submissions/queue /app/submissions/notes \
         /app/submissions/receipts /logs/artifacts/telemetry

# The reviewed nominal verifier source is published read-only at /app/verifier.
# The agent can run it only through its own training-GPU queue. Official scoring
# remains a separate trusted process and returns no result to the agent.
