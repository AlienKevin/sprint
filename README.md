# sprint

Private backup of the key QWOP-bench lane-eval launchers, challenge tasks,
`sprintbench` package, ops tooling, and sprint-web site sources.

Source host tree: `/data/qwop-bench`. This repo is not a full mirror. It omits
secrets, job artifacts, Modal caches, capture JSON, and trained `*.pt` policies.

## Layout

| Path | Role |
|------|------|
| `runs/run-lane-durable.sh` | Preferred durable lane launcher (Claude Code / Codex) |
| `runs/run-lane.sh` | Non-durable lane launcher |
| `runs/run-*.sh` | Model-specific wrappers (opus, luna, terra, deepseek, kimi, harness) |
| `runs/ops/` | `qwopctl`, frontier update, telemetry helpers, RUNBOOK, tests |
| `runs/*.j2` | Claude Code / Codex goal prompt templates |
| `challenge/g1-sprint-100m-lane/` | Lane task (instruction, Dockerfile, watcher scripts, tests) |
| `challenge/g1-sprint-100m/` | Open (non-lane) task |
| `sprintbench/` | Package, Modal app, scripts, embedded challenge copy |
| `sprint-web/` | Site HTML / timelines (large capture JSON gitignored) |

## Launch a durable lane run

Requires a Harbor checkout at the pinned continuous-verification commit, Modal
access, and agent credentials in the environment (never committed here).

```bash
# From a host where paths match the launchers (or edit ROOT/HARBOR in the scripts)
cd /path/to/sprint   # or /data/qwop-bench

RUN_ID="lane-$(date -u +%Y%m%dT%H%M%SZ)"

# Claude Code
export CLAUDE_CODE_OAUTH_TOKEN='…'
runs/run-lane-durable.sh --dry-run --run-id "$RUN_ID" --agent-kind claude-code
runs/run-lane-durable.sh --run-id "$RUN_ID" --agent-kind claude-code

# Codex (pin CLI version for fair comparisons)
export OPENAI_API_KEY='…'
runs/run-lane-durable.sh --dry-run --run-id "$RUN_ID" \
  --agent-kind codex --model 'openai/…' --reasoning-effort high \
  --codex-version 0.146.0
runs/run-lane-durable.sh --run-id "$RUN_ID" \
  --agent-kind codex --model 'openai/…' --reasoning-effort high \
  --codex-version 0.146.0
```

Stop the agent without killing Harbor:

```bash
python3 runs/ops/qwopctl.py stop --run-id "$RUN_ID"
```

Full procedure, Harbor pin, telemetry paths, and snapshot rules:
`runs/ops/RUNBOOK.md`.

## Secrets

Do not commit `.env`, `.oauth_token`, `runs/.secrets/`, restic passwords, or
API keys. Launchers read credentials from the environment or local secret files
outside this repo.
