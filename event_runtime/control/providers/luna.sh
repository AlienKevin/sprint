#!/usr/bin/env bash
# Compatibility entrypoint; all controlled OpenAI models share openai.sh.
set -euo pipefail

ROOT="${EVENT_REPOSITORY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export MODEL="${MODEL:-openai/gpt-5.6-luna}"
export OPENROUTER_PRESET="${OPENROUTER_PRESET:-@preset/sprint-gpt-5-6-luna-openai-standard}"
exec "$ROOT/event_runtime/control/providers/openai.sh"
