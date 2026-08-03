# Codex harness pin (Luna ↔ DeepSeek)

Pinned: **`0.146.0`** (`@openai/codex@0.146.0`)  
Checked: 2026-08-02  

## Sources

| Source | Result |
|---|---|
| npm dist-tag `latest` | `0.146.0` |
| GitHub `openai/codex` latest release | `rust-v0.146.0` / `0.146.0` (2026-07-29) |
| Host `codex --version` | `codex-cli 0.146.0` |
| npm `alpha` | `0.147.0-alpha.4` (pre-release; **not** used) |
| `/tmp/harbor-fork` | absent on this host |
| Harbor packaging | installs CLI in sandbox via `npm install -g @openai/codex@VERSION` |

Harbor agent kwarg: `--ak version=0.146.0` → `BaseInstalledAgent._version` → install pin (not image-floating `@latest`).

## Where it is set

| File | Mechanism |
|---|---|
| `runs/run-lane-durable.sh` | Default `CODEX_VERSION=0.146.0`; `--codex-version`; for `--agent-kind codex` passes `--ak version=$CODEX_VERSION` |
| `runs/run-luna.sh` | `CODEX_VERSION=0.146.0` → `--codex-version` on durable launcher |
| `runs/run-deepseek.sh` | **same** `CODEX_VERSION=0.146.0` → `--codex-version` |

Dry-run / `run.json` record `codex_version` for audit.

Override only if both arms change together:

```bash
CODEX_VERSION=0.146.0 CONFIRM_LAUNCH=1 runs/run-luna.sh
CODEX_VERSION=0.146.0 CONFIRM_LAUNCH=1 runs/run-deepseek.sh
```

## Smoke coordination (2026-08-02)

- In-flight Luna smoke `smoke-luna-20260802T211110Z` (agent task `3cc44022…`) launched **without** `--ak version=…` (would have installed `@latest` at install time).
- At interrupt: `agent_container_id=null`, empty `agent/` dir, trial.log only `Selected strategy: _ModalDirect` — **Codex had not started**.
- Harbor/monitor stopped so a re-smoke can use the pin. **Re-smoke required** before trusting smoke results against this pin.

## `reasoning_effort` (do not block pin)

Per `BLINDSPOTS_LUNA_DEEPSEEK.md` probes: Luna rejects `max` (accepts `xhigh`); DeepSeek accepts `max` (`xhigh`→high). Native-max: Luna `xhigh`, DeepSeek `max`. Prefer `max` for both only if later probes change that.
