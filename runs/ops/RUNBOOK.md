# Durable lane run

This runbook supports `claude-code` and `codex`. It stops the chosen agent
without stopping Harbor. Harbor stays alive while continuous graders drain,
artifacts copy back, and the final verifier runs.

Runs have no agent time cap. The supported stop path is the manual `qwopctl
stop` command below. The Modal sandbox timeout is 86,400 seconds and serves
only as an infrastructure or orphan backstop. If that backstop fires, do not
expect a normal `STOP_ACK`.

Never send `SIGTERM` to Harbor for a normal stop. Never use `modal container
stop`.

## Harbor Codex process and state

At pinned Harbor commit
`c6d90cb50ff4a19e725e4724e004c21ecbd96f49`, the Codex adapter runs:

```text
bash -c 'set -o pipefail; ...; codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check --model MODEL --json \
  --enable unified_exec ... -- PROMPT 2>&1 </dev/null |
  tee /logs/agent/codex.txt'
```

Without the durable hook, the npm install used by Harbor would run
`node .../@openai/codex/bin/codex.js`, which spawns the package's native
`vendor/.../bin/codex` and forwards `SIGINT`, `SIGTERM`, and `SIGHUP`.
The durable hook resolves that same native executable and makes it the isolated
group leader:

```text
Modal Sandbox.exec bash
├── durable Codex wrapper
│   └── setsid .../vendor/.../bin/codex exec ...  [new PGID]
│       └── Codex tool children
└── tee /logs/agent/codex.txt
```

The wrapper records the exact group leader, PGID, and `/proc` start time in
`/run/qwop-agent/codex-process`. The watcher does not select a process by a
broad `pgrep`. Harbor and `tee` stay outside the Codex process group.

The adapter sets `CODEX_HOME=/tmp/codex-home`. Live event JSONL files go under
`/tmp/codex-home/sessions/YYYY/MM/DD/`; other trace state can include
`history.jsonl`, `session_index.jsonl`, `config.toml`, logs, and state SQLite
files. Stdout event JSONL goes to `/logs/agent/codex.txt`. On normal adapter
cleanup, Harbor copies sessions to `/logs/agent/sessions` and writes
`/logs/agent/trajectory.json`.

The durable wrapper also copies an allowlist of Codex trace state to
`/logs/agent/codex-state` before it exits. Snapshots include that directory and
the live allowlisted paths. They exclude `auth.json`, `/tmp/codex-secrets`,
model/plugin caches, `.cache`, and `models_cache.json`. `/app` and Harbor agent
and artifact logs remain in every snapshot, so source, checkpoints,
submissions, notes, and Codex event JSONL survive. Codex snapshots do not add
the broad `/root` source used by the Claude Code recovery set.

## Telemetry

GPU + CPU/system sampling is on by default for durable keepalives. Data lands
in `/logs/artifacts/telemetry/` (Hub: `artifacts/logs/artifacts/telemetry/`)
and host mirrors under `runs/ops/<run_id>/telemetry/` plus
`runs/monitor-<run_id>-telemetry.csv`. See `runs/TELEMETRY.md`.

## Before launch

```bash
cd /data/qwop-bench
test "$(git -C /data/harbor-continuous rev-parse HEAD)" = \
  c6d90cb50ff4a19e725e4724e004c21ecbd96f49
test "$(git -C /data/harbor-continuous branch --show-current)" = \
  continuous-verification
test -z "$(git -C /data/harbor-continuous status --porcelain)"
RUN_ID="lane-$(date -u +%Y%m%dT%H%M%SZ)"
```

### Claude Code dry run and launch

```bash
export CLAUDE_CODE_OAUTH_TOKEN='replace-with-token'
runs/run-lane-durable.sh --dry-run --run-id "$RUN_ID" \
  --agent-kind claude-code
runs/run-lane-durable.sh --run-id "$RUN_ID" \
  --agent-kind claude-code
```

The launcher keeps the current `claude-opus-5` and `max` effort defaults. It
does not set `CLAUDE_FORCE_OAUTH`.

### Codex dry run and launch

Codex CLI is pinned (default `0.146.0`) via `--codex-version` → Harbor
`--ak version=…`. See `runs/CODEX_PIN.md`. Do not omit the pin for fair
Luna/DeepSeek comparisons.

```bash
export OPENAI_API_KEY='replace-with-api-key'
CODEX_MODEL='openai/replace-with-model'
CODEX_ENDPOINT='https://api.example.com/v1'
runs/run-lane-durable.sh --dry-run --run-id "$RUN_ID" \
  --agent-kind codex --model "$CODEX_MODEL" \
  --endpoint "$CODEX_ENDPOINT" --reasoning-effort high \
  --codex-version 0.146.0
runs/run-lane-durable.sh --run-id "$RUN_ID" \
  --agent-kind codex --model "$CODEX_MODEL" \
  --endpoint "$CODEX_ENDPOINT" --reasoning-effort high \
  --codex-version 0.146.0
```

Omit `--endpoint` for the default OpenAI endpoint. The launcher accepts only a
credential-free HTTPS URL with no query or fragment. Pass the key through
`OPENAI_API_KEY`; never put a key in the model, endpoint, or Harbor arguments.
The launcher rejects short key values because Harbor's output scrubber could
replace common short text.

Dry runs print no credential and create no app, Volume, state directory, or
job. Real launches write the selected credential and restic password under
`/data/qwop-run-secrets/$RUN_ID/` with mode `0600`.

To detach either launch, use the same arguments with `nohup`:

```bash
nohup runs/run-lane-durable.sh --run-id "$RUN_ID" \
  --agent-kind codex --model "$CODEX_MODEL" \
  --endpoint "$CODEX_ENDPOINT" --reasoning-effort high \
  --codex-version 0.146.0 \
  >"/data/qwop-launch-$RUN_ID.log" 2>&1 </dev/null &
```

## Status and manual safe stop

```bash
python3 runs/ops/qwopctl.py status --run-id "$RUN_ID"
python3 runs/ops/qwopctl.py stop --run-id "$RUN_ID"
```

Both outputs report `agent_kind`. The controller selects the app recorded for
this run, then creates `/run/qwop-stop` in its exact agent container. It does
not select the newest job or container.

For Claude Code, the watcher sends `SIGINT` to the exact Claude PID, waits up to
90 seconds, then sends `SIGTERM` to that PID if needed. For Codex, it sends
`SIGINT` to the recorded native group leader, which is the same signal the npm
launcher documents and forwards. After 90 seconds it sends `SIGTERM` only to
the isolated Codex process group. The wrapper treats exit 130 or 143 as clean
only when the watcher first wrote `/run/qwop-agent/expected-interrupt`.

The watcher takes a final snapshot and writes `STOP_ACK` only after that
snapshot succeeds. Repeating `stop` is safe. Harbor, the container keepalive,
and grader containers receive no signal.

## Wait and finalize

```bash
python3 runs/ops/qwopctl.py wait --run-id "$RUN_ID" \
  --timeout-seconds 10800
python3 runs/ops/qwopctl.py finalize --run-id "$RUN_ID"
```

Success requires `STOP_ACK`, a clean terminal ledger, checksummed archives for
all attempt directories, the final artifact manifest and verifier result,
`finished_at` in trial and job results, Harbor exit, and a current site state.
`finalize` is safe to repeat.

## Recovery

Recovery reads the named Volume and does not need a live Harbor app:

```bash
python3 runs/ops/qwopctl.py check --run-id "$RUN_ID"
python3 runs/ops/qwopctl.py recover --run-id "$RUN_ID" \
  --destination "/data/qwop-recovered/$RUN_ID"
```

Both outputs report `agent_kind`. `check` runs `restic check` and verifies
attempt archive SHA-256 files. `recover` restores the latest snapshot under the
destination and retains the downloaded host-attempt archives. Codex state
appears under `logs/agent/codex-state` and, when captured live, under
`tmp/codex-home`. Pass `--snapshot SNAPSHOT_ID` to restore an older generation.

## Frontier and deployment

The monitor reparses the full ledger whenever it changes. It captures only
policies that enter the current frontier and checks dominance again before
capture and deploy. Capture, render, and deploy share a file lock. Site changes
wait five minutes before deployment, and an unchanged site causes no Vercel
deployment.

For a read-only frontier status with explicit paths:

```bash
python3 runs/ops/frontier_update.py status \
  --job "$JOB" --trial "$TRIAL" \
  --state "/data/qwop-bench/runs/ops/$RUN_ID/frontier-state.json"
```

The updater checks `.vercel/project.json` before any production deployment. It
will deploy only project `prj_dgvTovRNwdSDcefYmo6oXfju9M3p` in scope
`alienkevins-projects`.
