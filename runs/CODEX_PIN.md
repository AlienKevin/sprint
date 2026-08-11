# Codex harness pin (Luna and DeepSeek)

Pinned: **`0.147.0`** (`@openai/codex@0.147.0`)
Checked: 2026-08-08

## Source

`npm view @openai/codex version` returned `0.147.0`. The exact package is
baked into the offline Sprint image and verified with `codex --version` during
the image build. Agent setup has no network access and cannot float to another
release.

## Where it is enforced

| File | Mechanism |
|---|---|
| `events/g1-100-metres/environment/Dockerfile` | Bakes `@openai/codex@0.147.0` and checks its version |
| `runs/run-lane-durable.sh` | Requires `CODEX_VERSION=0.147.0`, records it in `run.json`, and passes `--ak version=...` to Harbor |
| `runs/run-luna.sh` | Uses the same pin with the official OpenAI endpoint |
| `runs/run-deepseek.sh` | Uses the same pin with DeepSeek's official Codex provider configuration |

Both model arms must change together if this pin is advanced. A requested
version that differs from the baked version is rejected before a sandbox is
created.

## Cost-reconstruction settings

Luna additionally pins Codex `service_tier=default` and disables hosted web
search. Harbor records every request's cache reads, cache writes, ordinary
input, output, reasoning output, total tokens, context size, model, effort,
tier, timestamp, long-context classification, pricing snapshot, and calculated
list-price cost. Luna tariff selection is request-time aware because the mutable
Luna alias had a 5x price reduction on 2026-08-10. Sprint finalization rejects
an incomplete or non-reconciling token ledger. The calculated value is a
reproducible estimate at the timestamp-selected published standard tariff, not
provider-billed or invoice-exact spend. The OpenAI Costs export remains
authoritative for contracted rates, credits, adjustments, and requests billed
without a locally completed response.

DeepSeek uses its own provider pricing and does not receive the OpenAI service
tier setting.
