# Secrets injection and Harbor Hub scrub

## Launch pattern (no secrets in `ps`)

Harbor agents read credentials via `_get_env` (agent `--ae` env first, then
process environ). Prefer **env-file injection**:

1. Write secrets to `/data/qwop-run-secrets/<label>/harbor.env` mode `0600`
   (`python3 runs/ops/write_harbor_env_file.py --label … KEY=…`).
2. Pass `harbor run --env-file <that path>` (Harbor `load_dotenv` into the
   Harbor process only — values never appear in argv).
3. `unset` host copies of the secret before `exec`.
4. Non-secret knobs (`CLAUDE_FORCE_OAUTH=1`, `KIMI_MODEL_BASE_URL=…`,
   `BASH_ENV=…`) may stay on `--ae KEY=value`.

| Launcher | Auth injection |
|---|---|
| `run-lane-durable.sh` / `run-luna.sh` / `run-deepseek.sh` | env-file (durable) |
| `run-terra.sh` | env-file (`OPENAI_API_KEY` from OpenRouter) |
| `run-opus.sh` | env-file (OAuth or Anthropic API key) |

Do **not** pass `--ae OPENAI_API_KEY=…`, `--ae CLAUDE_CODE_OAUTH_TOKEN=…`, or
`--ae KIMI_MODEL_API_KEY=…`.

## Hub upload path (scrub first)

Default entrypoint:

```bash
python3 runs/hub_track_upload.py --once
# or
python3 runs/ops/qwopctl.py hub-upload
```

Pipeline: discover jobs → copy under `runs/hub-scrubbed/` → redact →
**leak-gate** → (if Hub auth OK) `harbor upload --private`.

Scrub/gate only (no upload):

```bash
python3 runs/hub_track_upload.py --once --scrub-only
python3 runs/ops/qwopctl.py hub-upload --scrub-only
```

Self-check an existing bundle:

```bash
python3 runs/hub_track_upload.py --check runs/hub-scrubbed/jobs-terra/2026-07-27__20-37-40
```

Exit code `2` if known secret prefixes remain. Upload is refused on leak-gate
failure even when Hub auth works.

### Redacted forms

- Key prefixes: `sk-`, `sk-proj-`, `sk-ant-`, `sk-or-v1-`, `sk-harbor-`, Modal `wk-*.ws-*`
- Bearer / JWT tokens; URL userinfo (`https://user:pass@host`)
- Env assignments: OpenAI / DeepSeek / Anthropic / OR / Kimi / Harbor / Modal
- Literal values from process env and `runs/.secrets/*.env`

Live job dirs are never modified. Ledger: `runs/HUB_JOBS.md`.

## Luna smoke (2026-08-02)

`smoke-luna-20260802T211110Z` already launched via durable `--env-file`
(`/data/qwop-run-secrets/smoke-luna-…/harbor.env`). No `--ae` secret remediation
needed for that process. Future Luna/DeepSeek launches keep using
`run-luna.sh` / `run-deepseek.sh` → durable env-file only.

## Residual risks

- Secrets still exist in Harbor process `/proc/<pid>/environ` after dotenv load
  (better than argv; avoid `cat /proc/…/environ` dumps in shared logs).
- Agent session files / trial logs can still capture keys if the model prints
  them; scrub + leak-gate catch common prefixes before Hub upload.
- Hub upload stays blocked while `HARBOR_API_KEY` returns 401 — scrub/gate still
  runs and refreshes `hub-scrubbed/`.
