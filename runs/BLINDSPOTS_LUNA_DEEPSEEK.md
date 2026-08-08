# Blindspots: Luna vs DeepSeek V4 Flash (Codex / Modal)

Status: **investigation only — no Harbor/Modal jobs launched.**
Date: 2026-08-02
Profile: `MODAL_PROFILE=kevinli020508`
Harbor pin: vendored `harbor/` @ `b69b181bceae132ca0018790dfed3654556a9ec3` (`continuous-verification`)
Task: `challenge/g1-sprint-100m-lane` (concise instruction, self-collision gate, `/app/train` PhysX self-collisions)

Secrets (mode `600`, never print):
- `runs/.secrets/openai-luna.env` → `OPENAI_API_KEY`
- `runs/.secrets/deepseek.env` → `DEEPSEEK_API_KEY` + `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://api.deepseek.com`

`/data/.keepalive` touched.

Terra / Kimi / Opus: not started. Modal `app list` shows no live lane Terra/Opus/Kimi apps.

---

## 1. Harbor Codex kwargs (verified)

| Knob | Harbor wire format | Notes |
|---|---|---|
| Agent | `--agent codex` | Durable launcher: `--agent-kind codex` |
| Model | `--model openai/gpt-5.6-luna` or `deepseek/deepseek-v4-flash` | Codex strips to last path segment → `gpt-5.6-luna` / `deepseek-v4-flash` |
| Reasoning | `--ak reasoning_effort=VALUE` → `-c model_reasoning_effort=VALUE` | Harbor default for Codex is **`high`** if unset (`run-lane-durable.sh` same). No enum validation in Harbor — any string is forwarded. |
| Prompt | `--ak prompt_template_path=...` | Durable launcher hardcodes `runs/codex-goal.j2` |
| Endpoint | env `OPENAI_BASE_URL` (+ Harbor writes `openai_base_url` into Codex `config.toml`) | Durable: `--endpoint HTTPS_URL` → env-file only |
| Auth | `OPENAI_API_KEY` | Durable: `--env-file` under `/data/qwop-run-secrets/$RUN_ID/` |

Durable dry-run (Luna shape) succeeded with redacted auth; `launch: false`.

### Prompt / `/goal`

- Local Codex CLI **0.146.0** binary includes real slash command:
  `Usage: /goal [<objective>|clear|edit|pause|resume]`
  plus goal continuation runtime (`ext/goal/...`, `goals/continuation.md`).
- Harbor Codex agent has **no** `_apply_goal` helper (unlike Claude Code). The prompt template must start with `/goal ...` if we want the slash command.
- Current `runs/codex-goal.j2` is **keep-going prose only** (no `/goal`). Terra comment “Codex has no /goal” is **stale**.
- Fair recommendation for **both** arms: update template to:

```text
/goal Keep working on the following task until the operator stops the run. Do not stop
early to summarize, ask whether to continue, or declare the job done. After each
scored submission, immediately keep training and try to beat the best time.
Prefer submitting improved policies often so continuous verification can score
them.

Task:
{{ instruction }}
```

Draft file: `runs/codex-goal-slash.j2`. Durable launcher still points at `codex-goal.j2` — either copy over before launch or patch the launcher.

---

## 2. DeepSeek + Codex (official harness)

Official docs: [Integrate with Codex](https://api-docs.deepseek.com/quick_start/agent_integrations/codex/).

| Item | Setting |
|---|---|
| Base URL | Provider `base_url = "https://api.deepseek.com/"` (`wire_api = "responses"`) |
| Model ids | `deepseek-v4-flash` (Codex-supported now); `deepseek-v4-pro` early August 2026 |
| Catalog | Vendored `codex-deepseek-models.json` → 1M context, `auto_compact_token_limit: null` |
| Harbor env | `OPENAI_API_KEY` + `OPENAI_BASE_URL` still set for auth/endpoint bookkeeping |
| Sandbox rewrite | `QWOP_CODEX_PROVIDER=deepseek` → wrapper removes `openai_base_url`, writes `[model_providers.deepseek]` + `models.json` |
| Durable wiring | `--endpoint https://api.deepseek.com` + env-file key; see `runs/DEEPSEEK_EARLY_EXIT.md` |

Do **not** rely on Harbor’s `openai_base_url` alone — that keeps the OpenAI provider (~258k + remote compaction). Luna still uses the default OpenAI Codex path (no rewrite).

---

## 3. Max reasoning: OpenAI Luna vs DeepSeek

> **CORRECTION 2026-08-03.** The table below probed `/v1/chat/completions`. Codex
> does **not** use that endpoint — it uses the **Responses API** (`wire_api =
> "responses"`). Re-probed on the Responses wire, `gpt-5.6-luna` **accepts `max`**
> and echoes `reasoning.effort=max` back (HTTP 200). The prior `xhigh`
> recommendation for Luna was an artifact of probing the wrong endpoint, and is
> superseded. Both arms run **`max`**. Verified against the current key at
> the repository-local `.env` (which remains gitignored).
>
> | Endpoint | `gpt-5.6-luna` + `max` |
> |---|---|
> | `/v1/chat/completions` | 400 unsupported (`none\|low\|medium\|high\|xhigh`) |
> | `/v1/responses` (Codex path) | **200, effort=max** |
>
> Consistent with `runs/ops/lane-luna-*/run.json` recording `reasoning_effort:
> "max"` on a run that scored 18 submissions before preemption.

Live tiny chat probes (keys not logged) — **chat/completions only, see correction above**:

| Value | `gpt-5.6-luna` (api.openai.com) | `deepseek-v4-flash` |
|---|---|---|
| `max` | **400** unsupported (`none\|low\|medium\|high\|xhigh` only) | **200** OK, reasoning present |
| `ultra` | **400** unsupported | **400** unknown variant |
| `xhigh` | **200** OK | **200** OK (docs: Flash maps `xhigh` → **high**, not max) |
| `high` | **200** OK | **200** OK |

| Arm | Pass to Harbor `--reasoning-effort` | Why |
|---|---|---|
| Luna | **`max`** | Accepted on the Responses wire Codex uses |
| DeepSeek | **`max`** | Highest value DeepSeek accepts |
| Opus 5 | **`high`** | Operator-specified 2026-08-03 |

Do **not** reuse Terra’s `ultra` for official OpenAI Luna (rejected on both wires).

Harbor durable default `high` is **not** max — must set explicitly for the Codex arms.

---

## 4. Cost / rate limits

| Layer | Finding |
|---|---|
| Modal GPU | Prior lane ops ~\$18–28/h per agent lane job + up to 2 concurrent verifiers (A10G). Two parallel arms ≈ **2×** that envelope. |
| OpenAI Luna | `/models` lists `gpt-5.6-luna` (accepted). Billing/usage endpoints need session key / scopes — **spend unknown** from this key. |
| DeepSeek | `/user/balance` → available (**CNY ~111.63** topped-up). Token burn at `reasoning_effort=max` on a long agent loop can drain this quickly; watch balance. |
| Rate limits | Not probed beyond tiny completions (no 429). Unknown RPM/TPM for both under Codex tool storms. |

---

## 5. Fairness checklist

Hold constant:
- Task tree digest SHA256 `c33f2f467c9682825b2e18e5cf8ec054133ff6b8efbbc0a9f84927520551d7bd` (34 files)
- Same Modal profile / Harbor commit / image build from same `g1-sprint-100m-lane`
- Same agent kind: **Codex** (not claude-code / kimi-code)
- Same `/goal` template text
- Same sandbox timeout backstop (86400s); no agent wall-clock cap (manual `qwopctl stop`)
- Continuous verifier settings from `task.toml` (`max_concurrent=2`, `max_submissions=60`)

Intentional differences (required for native max reasoning):
- Model id + API endpoint + API key
- `reasoning_effort`: `xhigh` (Luna) vs `max` (DeepSeek)

Codex feature risks that can differ by model/provider:
- `/goal` continuation semantics if one backend errors on goal tooling
- Tool-call + `reasoning_content` round-trip (DeepSeek requires returning reasoning on tool turns)
- Context / output limits (DeepSeek docs: max thinking wants large output window)
- Codex `@latest` install at job start can race if arms launch far apart — see §9

---

## 6. `/goal` character budget

| Template | Chars | Under 4000? |
|---|---|---|
| Instruction only | 2118 | YES |
| `/goal` + instruction | 2124 | YES |
| Keep-going `codex-goal.j2` rendered | 2449 | YES |
| `/goal` + keep-going + instruction (recommended) | 2454 | YES |

Codex binary formats an “objective must be at most N characters” error; N not recoverable from stripped strings. Current instruction+wrap is safely under the historical 4000 Claude budget.

---

## 7. Secrets on cmdline (`--ae`) / `ps` risk

- **Unsafe (Terra-style):** `--ae OPENAI_API_KEY=...` appears in `ps` / stop docs.
- **Safe (durable launcher):** writes key to mode-`0600` env-file, passes `--env-file`, unsets host env before `harbor run`.
- Draft scripts below use durable path only. Never put keys in model/endpoint/`--ae`.

---

## 8. `gpt-5.6-luna` model id

- Listed on official `GET https://api.openai.com/v1/models` → **YES**
- Chat Completions with `reasoning_effort=xhigh` → **200**
- Also present nearby: `gpt-5.6-terra`, `gpt-5.6-sol` (not used)

---

## 9. Missing pieces / draft launch plan (DO NOT LAUNCH YET)

Gaps to close before go:
1. **Replace or sync** `runs/codex-goal.j2` with `/goal` wrap (`codex-goal-slash.j2` drafted).
2. **Codex version pin** — done: both arms use **`0.146.0`** via `--ak version=0.146.0`. See `runs/CODEX_PIN.md`.
3. Launch via `runs/run-lane-durable.sh` (not legacy `run-terra.sh`).
4. Distinct `RUN_ID`s, jobs under `runs/ops/$RUN_ID/`, secrets under `/data/qwop-run-secrets/$RUN_ID/`.
5. Start host monitors (default on). Continuous monitoring after approval.
6. Do not touch Terra/Kimi/Opus.

Draft wrappers (default dry-run / require `CONFIRM_LAUNCH=1`):
- `runs/run-luna.sh`
- `runs/run-deepseek.sh`

Intended post-approval commands (keys via `source` of secret files, never inline):

```bash
# Luna
set -a; source runs/.secrets/openai-luna.env; set +a
CONFIRM_LAUNCH=1 RUN_ID=lane-luna-<UTC> \
  runs/run-luna.sh

# DeepSeek (parallel)
set -a; source runs/.secrets/deepseek.env; set +a
CONFIRM_LAUNCH=1 RUN_ID=lane-deepseek-<UTC> \
  runs/run-deepseek.sh
```

Effective Harbor shape after scripts:

| | Luna | DeepSeek |
|---|---|---|
| `--agent-kind` | `codex` | `codex` |
| `--model` | `openai/gpt-5.6-luna` | `deepseek/deepseek-v4-flash` |
| `--endpoint` | *(omit → OpenAI default)* | `https://api.deepseek.com` |
| `--reasoning-effort` | `xhigh` | `max` |
| `--codex-version` / `--ak version=` | `0.146.0` | `0.146.0` |
| prompt | `/goal` + keep-going + instruction | same |
| env | modal / kevinli020508 | same |

---

## Decision needed from operator

1. Approve `/goal` template swap on `codex-goal.j2`.
2. Approve asymmetric efforts (`xhigh` vs `max`) as “native max,” or force identical string (would under-max DeepSeek if both use `xhigh`, or break Luna if both use `max`).
3. Codex pin `0.146.0` is already wired (see `CODEX_PIN.md`); confirm before full launch.
4. **Go / no-go to launch.**
5. Re-smoke Luna after pin (prior smoke interrupted pre-Codex; see `CODEX_PIN.md`).
