#!/usr/bin/env bash
# Harness check: does the whole loop work on a real GPU?
#
# Runs the task with a stand-still TorchScript policy through the oracle agent.
# It will not finish 100 m and is not meant to. What is under test is the loop:
# submit -> continuous verification in an isolated container -> Isaac rollout ->
# result handed back -> archived -> final verify. No model credentials needed.
set -euo pipefail

SP=/tmp/claude-1000/-data-qwop-bench/b30c9aaa-c092-4bc7-b4da-4d4719508d31/scratchpad
cd /tmp/harbor-fork

# The Isaac images were built under this Modal workspace; image caches are
# per-workspace, so using another profile would rebuild 20 GB from scratch.
export MODAL_PROFILE=kevinli020508

exec uv run harbor run \
    --path "$SP/harness-task" \
    --agent oracle \
    --env modal \
    -n 1 \
    --jobs-dir /data/qwop-bench/runs/jobs \
    "$@"
