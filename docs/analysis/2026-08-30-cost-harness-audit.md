# Cost, strategy, and harness audit: the selected 100m trials

Audit date: 2026-08-30. All event times below are UTC. This is a retrospective analysis of local, recorded evidence; no new model requests, cloud jobs, or controlled experiments were run for this report.

## 1. Scope and conclusions

The three selected trials achieved different results through substantially different programs of work. Their common approximately $10 budget does not mean they bought equivalent training, equivalent model reasoning, or equivalent useful computation.

| Model | Selected run | Displayed trial in the active batch | Harness | Official best effective speed |
| --- | --- | ---: | --- | ---: |
| DeepSeek | `s10-vexp-r120-20260828-deepseek-1` | 1 | DeepSeek harness `0.1.1-rc.2` | 1.004530 m/s; 77.635m at 60s, unfinished |
| GLM | `claude-goalfix2-20260828-1750-glm-2` | 3 | Claude Code | 10.101329 m/s; 100m in 9.9s |
| Luna | `s10-vexp-r123-20260828-luna-4` | 3 | Codex `0.149.1` | 3.640803 m/s; 100m in 27.466s |

The displayed trial numbers come from the active batch manifest, not the numeric suffixes of run IDs. Model identities are respectively `deepseek/deepseek-v4-flash-vision-exp`, `z-ai/glm-5.3-flash`, and `openai/gpt-5.6-luna`.

The detailed strategy sections quote the archival verifier metrics. The website's capture-audited performance snapshot can differ slightly because it recomputes the legal prefix from captured frames: the displayed winners are 1.004526, 10.101341, and 3.640795 m/s respectively. Use that snapshot for website comparisons, not an agent's local training score. Some unsuccessful trials have larger differences between local claims and capture-audited scores.

The principal findings are:

- DeepSeek abandoned custom neural PPO training and spent the remaining run optimizing a parameterized scripted crawler. Its large GPU category includes simulation/search and local verification, not just neural-network training.
- GLM successfully trained a large-batch PPO runner. It also spent appreciable compute on a hung smoke test and a completed training run whose requested output filenames were wrong, necessitating another run.
- Luna's winner was **the full-geometry neural actor wrapped in `ppo_full_ensemble2.pt`**, with the recorded winning environment using **1.5 times the cross-track/heading input**. It was not the unmodified actor, a muted-feedback branch, or the final fall-tolerant training run.
- Luna's higher API cost is quantitatively explained by more requests, much more cumulative input, its cache-read/write tariff, and more output. **No request in this selected Luna trace incurred the recorded >272k-token long-context surcharge.**
- Different harnesses, context management, model tariffs, strategies, and outcomes are entangled. This evidence cannot establish that one harness caused a model to be less efficient or that swapping harnesses would preserve its behavior.

Selection sources: [active batch manifest](/data/sprint/web/data/batches/current.json), [performance snapshot](/data/sprint/web/data/performance/current.json). Official result sources: [DeepSeek policies](/data/sprint/web/data/policies/s10-vexp-r120-20260828-deepseek-1.json), [GLM policies](/data/sprint/web/data/policies/claude-goalfix2-20260828-1750-glm-2.json), [Luna policies](/data/sprint/web/data/policies/s10-vexp-r123-20260828-luna-4.json).

**Score precision/source distinction:** the policy-index scores quoted in the strategy sections preserve the historical policy JSON values. The currently audited website performance snapshot reports **DeepSeek1.004526, GLM10.101341, Luna3.640795m/s** for the same winning submissions. These small differences arise between the policy-index representation and the website's audited capture-based readout. Use `performance/current.json` for website-facing comparisons; do not silently substitute raw policy-index values for that source. This distinction can be larger for other trials. The monetary tables below use the reconciled cost sources independently of score precision.

## 2. What the cost categories actually measure

### 2.1 End-of-trial totals

The API column below is the benchmark's calculated/normalized API usage, not necessarily the provider-reported amount actually charged. Compute columns are reconciled provider **pre-credit resource spend**. The table excludes the sealed official verifier from the agent budget.

| USD | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| Benchmark API | 1.34760688 | 0.75164969 | 3.28008303 |
| Training/GPU **whole worker** | 6.29877259 | 6.58695469 | 5.08680281 |
| CPU-agent **whole sandbox** | 2.40358472 | 2.55848240 | 1.52919451 |
| **Agent total** | **10.04996419** | **9.89708678** | **9.89608035** |
| Official verifier, excluded | 0.37198226 | 0.13565031 | 0.61152076 |
| Provider-reported API, for comparison | 0.76692294 | 0.375824845 | 3.28008303 |

These are final totals, not the exact cost at which each winning policy was first queued or first scored. A small budget overshoot is not evidence that all work after exactly $10 was deliberately authorized; cost observation, enforcement, and draining occur asynchronously.

The site's first-enqueue convention gives different costs for the winning policy:

| Winning policy | Cost at result processing | Cost at first agent queue | Original queue turn |
| --- | ---: | ---: | ---: |
| DeepSeek #7 | $10.049964 | $9.990312 | 313 |
| GLM #9 | $9.844495 | $8.187165 | 179 |
| Luna #24 | $9.896080 | $8.177805 | 648 |

Queue costs include API events and allocated compute only through the proven queue cutoff. Compute is a retrospective estimate: the allocation/tariff curve is calibrated to the terminal provider settlement, not an exact per-policy invoice. The full-trial totals above are unchanged. Source: each point's `queued_at`, `cost_at_queue_usd`, component costs and provenance in [the performance snapshot](/data/sprint/web/data/performance/current.json).

This convention attributes the **eventual verified score** to the agent's first submission request. A GPU job can be queued with automatic submission of its future output, so the convention does not prove that the finished artifact or score was already available at that instant. Later processing times remain separate evidence. In particular, policies #13–15 in Luna's best trial were queued at turn 590 and policies #16, #17, #30–32 at turn 596; their later bridge-observation times are not new agent actions at the interruption turn.

The updated chart changes the timing of the breakthroughs, not their verified quality. Luna's 3.64 m/s result moves away from the apparent final-budget cliff to about $8.18. GLM already reaches 8.47 m/s at about $5.77 and later 10.10 m/s at about $8.19; DeepSeek's best is 1.00 m/s at about $9.99. GLM therefore offers the strongest observed high-performance cost trade-off in the displayed best trials. The roughly one-cent difference between GLM's and Luna's final breakthroughs is not meaningful precision given the retrospective compute allocation.

These staircase curves show best-so-far quality within selected independent trials, not mean learning curves across five repetitions. Individual plotted submissions can be dominated; the chart is not a single global Pareto frontier or evidence of a guaranteed model-level advantage. The five-trial finish counts and failure cases in the observations should accompany any interpretation of the best-run curves.

Provider compute files explicitly exclude invoice-level credits, reservations, subscription fees and taxes. Workspace-level volume storage is not attributed into these run totals: endpoint size is not a time-weighted invoice measurement. The API audits also report `invoice_exact: false`; provider-reported per-request usage is not a complete account invoice.

### 2.2 CPU and RAM are present in both compute labels

| Reconciled component, USD | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| Training worker: A10/A10G | 3.09156017 | 3.23130141 | 2.49865280 |
| Training worker: CPU | 2.39778576 | 2.50963983 | 1.93395727 |
| Training worker: RAM | 0.80942666 | 0.84601345 | 0.65419274 |
| Agent sandbox: CPU | 1.43366403 | 1.52605571 | 0.91212783 |
| Agent sandbox: RAM | 0.96992069 | 1.03242669 | 0.61706668 |

Thus, interpreting the entire $6.30/$6.59/$5.09 training category as the price of GPU silicon alone would substantially overstate GPU-device spending. Conversely, the CPU-agent category is not a measurement of CPU instructions executed: it includes reserved memory and allocated lifetime.

All three runs used the same recorded resource contract:

| Role | CPU reservation | RAM reservation | GPU | Concurrent workers per trial |
| --- | ---: | ---: | --- | ---: |
| Agent sandbox | 2 physical cores / 4 vCPU equivalent | 8 GiB | None | 1 |
| Training worker | 6 physical cores / 12 vCPU equivalent | 12 GiB | 1 A10G | 1 |
| Official verifier | 4 physical cores / 8 vCPU equivalent | 10 GiB | 1 A10G | 1 |

The pinned Sandbox tariff is $0.00003942 per physical-core-second, $0.00000667 per GiB-second, and $0.000306 per A10-second. This implies a reservation floor of **$0.47592/hour for the agent** and **$2.241/hour for the training worker**. The pinned billing basis is the maximum of requested and actual CPU/memory, not utilization-proportional billing. These are Sandbox rates; do not substitute a different Functions tariff.

Sources: [pinned tariff and reconciliation code](/data/sprint/event_runtime/cost/modal.py:29); [DeepSeek resource contract](/data/sprint/runs/ops/s10-vexp-r120-20260828-deepseek-1/run.json), [GLM contract](/data/sprint/runs/ops/claude-goalfix2-20260828-1750-glm-2/run.json), [Luna contract](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/run.json); reconciled [DeepSeek compute](/data/sprint/runs/ops/s10-vexp-r120-20260828-deepseek-1/telemetry/modal-cost.json), [GLM compute](/data/sprint/runs/ops/claude-goalfix2-20260828-1750-glm-2/telemetry/modal-cost.json), [Luna compute](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/telemetry/modal-cost.json). Official references recorded by the audit: [Modal pricing](https://modal.com/pricing), [Sandbox resources](https://modal.com/docs/guide/sandbox-resources). This local-only report does not revalidate live prices.

### 2.3 Allocated time is not useful-compute time

| Measurement | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| Agent allocated duration | 18,206s / 5.057h | 19,378s / 5.383h | 11,599s / 3.222h |
| Agent CPU counter, core-seconds | 1,746.46 | 1,745.44 | 1,659.59 |
| Approximate use of two-core reserved capacity | 4.8% | 4.5% | 7.2% |
| A10 provider-equivalent seconds: device cost / tariff | 10,103.1378 | 10,559.8085 | 8,165.5320 |
| Worker telemetry execution-window seconds | 8,896.411 | 10,184.221 | 7,725.853 |
| Started allocations identified by compute audit | 52 | 20 | 18 |
| Interval-weighted sampled GPU utilization inside worker jobs | 33.47% | 52.25% | 53.15% |
| Sampled CUPTI SM-active window mean | 0.2968% | 19.4191% | 8.7139% |

GPU-utilization percentage and SM-active percentage are distinct counters. The latter is the raw percentage-unit `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` metric, not a fraction requiring another multiplication by 100. Neither is a count of useful training progress, FLOPs purchased, or achieved task quality. Sampling windows and coverage differ from complete billed intervals.

The difference between A10 provider-equivalent time and worker execution windows is about20m07 for DeepSeek,6m16 for GLM and7m20 for Luna. This can include allocation startup, shutdown and lifecycle coverage differences; it must not all be labeled idle time or waste. The provider report is hourly/by application and does not establish exact dollars for individual jobs.

Low agent CPU usage is consistent with waiting for model responses, GPU jobs, polling and other I/O. It does not make the reservation free. Exact per-poll dollars are not recoverable from these aggregate counters. It also does not prove the whole reservation could safely have been removed: peaks, memory and runtime requirements were not tested under smaller contracts.

Counter sources: each run's `telemetry/unified-timeline.json`, `telemetry/durable-gpu-samples.jsonl`, `telemetry/durable-gpu-attempts/`, and reconciled `telemetry/modal-cost.json`, under the linked run directories above.

## 3. API pricing, cache behavior, and compaction

### 3.1 Count requests and tokens, not visible trajectory steps

Input totals count the prompt on **every request**, including repeatedly read cached prefixes. They are not unique tokens written by the agent. Output already includes any reasoning tokens counted in that output field; do not add the reasoning subtotal again.

| Recorded usage | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| API requests | 314 | 220 | 756 |
| Cumulative input tokens | 74,565,989 | 18,791,437 | 114,273,205 |
| Cache-read input tokens | 74,298,240 | 17,838,528 | 111,315,254 |
| Cache-write input tokens | 0 separately recorded | 0 separately recorded | 2,955,683 |
| Ordinary uncached input | 267,749 | 952,909 | 2,268 |
| Cache-read fraction of cumulative input | 99.6409% | 94.9290% | 97.4115% |
| Output tokens | 143,653 | 147,115 | 262,003 |
| Separately reported reasoning subtotal | 60,887 | 0 reported; not proof of no reasoning | 126,560 |
| Largest recorded input request | 392,265 | 168,214 | 252,517 |

“No separately recorded cache writes” for DeepSeek/GLM is an accounting representation, not proof that no provider cache was populated. Luna's ordinary uncached bucket is tiny because nearly all new input is in its separately charged cache-write bucket.

Sources: full per-request [DeepSeek API audit](/data/sprint/runs/ops/s10-vexp-r120-20260828-deepseek-1/usage/run-usage-audit.json), [GLM API audit](/data/sprint/runs/ops/claude-goalfix2-20260828-1750-glm-2/usage/run-usage-audit.json), [Luna API audit](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/usage/run-usage-audit.json). Request counts are `.requests.length`/`.request_count`; token totals sum those requests.

### 3.2 Pricing reconstruction

Rates below are USD per million tokens on the **benchmark accounting basis**, not claims about today's public model price list.

| Input/output category | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| Ordinary uncached input | 0.44 | 0.15, snapshot-supported normalized rate | 0.20 |
| Cache read | 0.014 | 0.03, snapshot-supported normalized rate | 0.02 |
| Cache write | Not separate | Not separate | 0.25 |
| Output | 1.32 | 0.50, snapshot-supported normalized rate | 1.20 |

DeepSeek requests preserve a peak tariff and a time-of-day effective tariff. The benchmark normalizes to the peak $0.44/$0.014/$1.32 rates; the selected run's provider-reported total is $0.76692294 versus normalized $1.34760688, a **$0.58068394 accounting adjustment**. Do not describe that adjustment as more tokens consumed or extra GPU work.

GLM's endpoint snapshots directly record effective rates $0.075 uncached/$0.015 cache-read/$0.25 output per million, a0.5 discount fraction, and gross-up multiplier2. The $0.15/$0.03/$0.50 normalized, pre-discount rates are **directly supported by the historical effective-price snapshot and its explicit multiplier2**. Every one of the220 requests independently reconstructs at those rates (maximum floating-point residual about1.7e-18), not merely the aggregate. They are not a claim about today's independently published list price. For example, [request e271a6c8b2304d933e5f1275cc858308](/data/sprint/runs/ops/claude-goalfix2-20260828-1750-glm-2/provider-api-usage/api-usage/requests/e271a6c8b2304d933e5f1275cc858308.json) records112,117 input tokens including111,872 cached,966 output tokens, benchmark cost$0.00387591 and provider cost$0.001937955. Normalized GLM API cost is exactly twice its recorded provider amount: $0.75164969 versus $0.375824845.

Luna's captured pricing snapshot directly records $0.20 ordinary input, $0.02 cache-read, $0.25 cache-write and $1.20 output per million, without a promotional gross-up. It also records a full-request long-context override above272,000 input tokens:2× input-category prices and1.5× output price. Every selected-run request is below that threshold; all `.long_context_pricing_applied` flags are false. Therefore **the long-context surcharge explains none of this selected Luna trial's observed API bill**.

The exact decomposition is:

| Benchmark API component, USD | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| Ordinary uncached input | 0.11780956 | 0.14293635 | 0.00045360 |
| Cache-read input | 1.04017536 | 0.53515584 | 2.22630508 |
| Cache-write input | 0 | 0 | 0.73892075 |
| Output | 0.18962196 | 0.07355750 | 0.31440360 |
| **Total** | **1.34760688** | **0.75164969** | **3.28008303** |

Each component is `token_count × corresponding_rate / 1,000,000`; these sum exactly to the audit totals at shown precision. Cached input is cheap but not free. Luna's repeated cache reads alone cost $2.2263, more than DeepSeek's or GLM's entire normalized API bill. Luna's cache writes add another $0.7389. Its higher bill does not require poor cache hit rate: its hit rate is over97%.

### Why DeepSeek costs more than GLM despite cheaper cached input

The remembered comparison is supported by the recorded prices and usage: DeepSeek's normalized cached-input rate is $0.014/M, less than half GLM's $0.03/M, but it reads **4.17 times as many cached tokens** (74.30M versus 17.84M). Consequently its cache-read bill is **1.94 times larger** ($1.0402 versus $0.5352), and its complete API bill is **1.79 times larger** ($1.3476 versus $0.7516).

There are two volume effects: DeepSeek makes **1.43 times as many requests** (314 versus 220), and its average total input per request is **2.78 times larger** (about 237k versus 85k). Its output-token total is actually slightly smaller than GLM's. This is primarily an input-history-volume explanation, not an explanation based on generating more output. The observed GLM compaction and DeepSeek's uninterrupted history growth are consistent with that explanation, but the traces do not isolate how much of the difference compaction caused rather than different tool outputs, strategies, or turn counts.

Compaction is not a free, otherwise-identical intervention: it changes the retained context and can replace a reusable prefix. Repeated prefix reads are discounted, not zero-cost; cache writes and output still count. [OpenAI's prompt-caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching) explains those cache mechanics, while the model-specific dollar calculations above use the saved historical billing snapshots rather than current prices.

Historical snapshot provenance: [DeepSeek endpoint](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-vision-exp/endpoints), [GLM endpoint](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints), [Luna endpoint](https://openrouter.ai/api/v1/models/openai/gpt-5.6-luna/endpoints), and [OpenAI pricing](https://developers.openai.com/api/docs/pricing). These are the source URLs preserved in local snapshots. The evidence for the experimental models and exact historical rates here is the saved request data; current pages were not fetched for this local-only report.

### 3.3 What context resets do—and what the traces establish

Compaction and prompt caching solve different problems. Compaction replaces conversation history with a shorter representation; prompt caching discounts reuse of an unchanged prefix. A compaction can reduce future prompt size while requiring new summary tokens and a newly written prefix. Long un-compacted histories can remain highly cacheable but accumulate substantial cache-read cost over many turns. These mechanisms do not alone predict which harness has the lower total bill.

DeepSeek has **no observed compaction** in this selected trace: no large prompt reset was found, and inputs grow as high as392,265. This is not proof that the harness has no compaction implementation or that another trial would never compact.

GLM has an observed context reset around **167–168k tokens**, followed by a request with26,683 input tokens at20:41:48.935153Z. The provider-usage sequence immediately before that reset reaches168,214. This is an observed boundary region, **not an exact configured automatic-compaction threshold**. Different local/provider counters and a summary request can give slightly different numbers around the same event.

Luna has six large prompt resets, concentrated around roughly245k input tokens (range of preceding recorded requests241,928–252,517). Its per-request record reports a model context window of258,400. That field is not itself proof of an exact compaction trigger, and the272k pricing threshold is a different quantity.

| Luna next request | Previous input | New input | New request usage timestamp |
| --- | ---: | ---: | --- |
| `api_call_78` | 252,517 | 8,917 | 18:32:05.965Z |
| `api_call_130` | 242,330 | 9,916 | 18:39:52.164Z |
| `api_call_332` | 241,928 | 9,030 | 19:33:10.823Z |
| `api_call_477` | 242,100 | 9,223 | 20:01:27.716Z |
| `api_call_565` | 243,694 | 8,656 | 20:32:09.607Z |
| `api_call_669` | 242,101 | 8,893 | 20:53:52.893Z |

These rows are reproducible adjacent-request drops, not an estimate of tokens saved or extra cost caused by each compaction. Summaries can omit information or alter subsequent behavior; that quality effect is not measured by this audit. Direct harness event/configuration corroboration and its limitations are described in the source notes below.

## 4. DeepSeek: from unsuccessful PPO to scripted crawling

### 4.1 Neural phase, 09:33–10:50

DeepSeek wrote custom PPO, but its early work repeatedly repaired infrastructure and policy semantics: observation dimensions, CPU/GPU export compatibility, double-squashed actions and self-collision handling. The completed early main PPO job `a7123ae7a295` used64 environments ×128 rollout steps ×150 iterations, or1.23M configured transitions. GLM's later500-iteration runs used49.15M transitions—40 times as many. Small training scale is a plausible contributor to weak results, not a controlled explanation of the performance gap.

A benchmark job `80558987b6d8` hung while creating/closing successive environments and hit a five-minute GPU-activity watchdog. The agent's diagnosis attributes this to simulator environment lifecycle behavior; the exact underlying simulator fault was not proved. After an export wrapper correction, a learned policy still covered only about0.58m before self-collision.

Public anchors: [#62 benchmark watchdog](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s28255), [#67 action-distribution bug](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s33126), [#104 weak corrected policy](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s49556).

### 4.2 Search phase, 10:50–14:33

At10:50 DeepSeek switches to zero/crouching/sinusoidal heuristics, then fixed crawling postures and lane feedback. By11:42 it designs an evolutionary search over start delay, frequency, joint amplitudes, posture offsets and steering gains. The archived implementation uses population32, elite selection and Gaussian/random mutation, evaluating simulator effective speed directly. This is **parameter search over a scripted controller**, not continuing neural learning.

Jobs `fb0313077212`, `1b1b974a05b1` and `8d0acc38fbc2` increase search rollout horizons from12 to15 to20 seconds. Full60-second testing then reveals that the last crawler can remain legal for roughly65m. The agent calls the pattern “100m-capable,” but the official results do not support a100m finish.

Subsequent full60-second evolution (`57666c1ce2a3`, about19m) and an extended search that fails then restarts (`7f365831d18a`, `49eee6c9f4ba`) do not beat that candidate. Around13:40 DeepSeek returns to smaller frequency/amplitude grids; later start-delay/frequency refinement yields the official77.635m result. The final20-variant grid appears at14:24.

Public anchors: [#135 switches to heuristics](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s64402), [#190 full parameter search](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s98148), [#242 sustainable crawler and longer horizon](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s111833), [#272 plateau](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s124113), [#309 final targeted grid](http://localhost:64853/trajectory?run=s10-vexp-r120-20260828-deepseek-1#a1-s135927).

### 4.3 Cost implication

The registry contains52 jobs:10 PPO-command jobs,1 diagnostic,1 benchmark,4 early learned-policy local tests,29 later scripted searches/local tests and7 submission forwarders. Final statuses are40 succeeded,9 failed,3 terminated, all recorded attempt1. Manual fixes/restarts used new job IDs, not recorded provider attempt2 retries.

Many short allocations, repeated local verification and a small32-candidate parallel population explain the observed cost shape better than “DeepSeek trained a large neural model for $6.30.” The low sampled SM activity is consistent with limited GPU-kernel parallel work, but is not by itself proof of which Python, simulator or synchronization bottleneck dominated.

Sources: [raw DeepSeek trajectory](/data/sprint/web/data/trajectories/s10-vexp-r120-20260828-deepseek-1.json), [job registry](/data/sprint/runs/ops/s10-vexp-r120-20260828-deepseek-1/gpu-job-registry), [source archive for full parameter search](/data/sprint/runs/ops/s10-vexp-r120-20260828-deepseek-1/gpu-job-work/fb0313077212/app.tar.gz), member `app/optimize_full_parallel.py`.

## 5. GLM: large-batch PPO with costly lifecycle and export failures

### 5.1 Smoke-test and normalization failures, 18:12–18:38

The smoke job `d06eb458cf5b` completes three training iterations in6.9 seconds, prints `[train] done`, and then stops making progress. It remains allocated for about18.5 minutes before cancellation. GLM suspects the second evaluation environment in the same process, removes in-process evaluation and separates training from testing. This is a particularly clear example of billed lifetime being much longer than the useful training work.

The first larger job `90e896b138c8` then exposes a PPO bug: rollout uses normalized observations while the update uses raw observations. Logs show KL around0.18 and learning rate pinned at the floor. GLM cancels, fixes the update and relaunches `46eab2975a23`.

Public anchors: [#44 smoke log](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s44), [#51 cancel hung evaluation](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s51), [#58 normalization fix](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s58).

### 5.2 Learning legal locomotion, 18:40–21:03

The custom actor and critic are512/256/128 MLPs. Major jobs use4096 environments with24 rollout steps; a500-iteration run collects49.15M transitions. The first learns movement but fails lane constraints. Run2 (`66893d0bcf16`, about29m) trains faster movement yet still fails legally. Scratch run3 (`0459f5aa45cd`) adds stronger whole-body extent/lane constraints and produces the first clean100m result, official8.470678m/s.

This distinction is important: high training velocity was not equivalent to a legal race finish. The reward/constraint changes and subsequent tests are part of the cost of aligning the training task with scoring. [#135 discusses whole-body extent and safety margins](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s135).

### 5.3 Completed training whose output was lost, 21:07–21:59

Run4 (`a0faa74175b2`) completes500 more iterations, logs49.15M transitions and saves iteration1000. But its declared return paths are `_125`, `_250`, `_375`, `_500`; a resumed run actually writes `_625`, `_750`, `_875`, `_1000`. The job is marked failed because the declared files are missing. The useful weights are unavailable to the agent, which reruns from the older checkpoint as run5 (`0ac05eada5ba`) after fixing output numbering.

The first job lasts about22m43, and the repeat about23m42. Calling the first “failed learning” would be inaccurate: it is a completed computation with an artifact-return failure. Exact per-job dollar attribution is unavailable, but the repeated allocation is directly observed.

Public anchors: [#152 completed-run/missing-output log](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s152), [#166 rerun plan and naming correction](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s166).

### 5.4 Best runner followed by unsuccessful continuation, 22:09 onward

The four returned run5 checkpoints are batch-evaluated. The strongest finishes locally around9.92s; the official best is9.9s, or10.101329m/s. A missing full training checkpoint then leads GLM to restore the actor from TorchScript while initializing a new critic/optimizer. Early updates damage the gait and the job is canceled. Subsequent critic-warmup and reward variants also fail to improve the best. Late repeated submissions and polling reflect uncertainty about submission acknowledgment and best-versus-last selection, not an additional breakthrough.

Public anchors: [#182 best local checkpoint](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s182), [#183 missing resume state](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s183), [#190 damaged actor-only continuation](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s190), [#203 another regression](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s203), [#209 submission/polling uncertainty](http://localhost:64853/trajectory?run=claude-goalfix2-20260828-1750-glm-2#a1-s209).

The registry has20 jobs:10 training commands including smoke,9 other jobs with worker starts, and1 final job without a worker start timestamp. Compute-side allocation counting can include startup before a worker execution window. Statuses:12 succeeded,4 failed,4 terminated. All registry attempts are1.

Sources: [raw GLM trajectory](/data/sprint/web/data/trajectories/claude-goalfix2-20260828-1750-glm-2.json), [lost-output job](/data/sprint/runs/ops/claude-goalfix2-20260828-1750-glm-2/gpu-job-registry/a0faa74175b2.json), [successful rerun source archive](/data/sprint/runs/ops/claude-goalfix2-20260828-1750-glm-2/gpu-job-work/0ac05eada5ba/app.tar.gz), member `app/sprint_train.py`.

## 6. Luna: curriculum learning, input-gain variants, and an after-stop winner

### 6.1 Broad exploration and PPO restarts, 18:12–19:20

Luna first tries scripted phase sweeps and retargeted installed motion clips without useful legal progress. Library-based RL jobs `8b7c436894fa` and `077286494658` fail; it writes PPO directly. The custom smoke job then hangs during exit after producing output. A direct-speed run is stopped after weak/stalled learning. An unbounded Gaussian curriculum regresses at its1→2m/s transition, prompting bounded/tanh actions, less entropy and training-only torso-contact resets.

Public anchors: [#168 custom PPO decision](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s171), [#188 exit problem](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s191), [#234 direct-speed cancellation](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s237), [#275 training reset changes](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s278), [#282 bounded-policy restart](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s285).

### 6.2 Three substantial completed learning jobs

| Job | Strategy | Configured transitions | Recorded runtime | Relationship to best |
| --- | --- | ---: | --- | --- |
| `1edc543bc01d` | Minimal-geometry PPO;1024 envs,1200 iterations,32 steps; speed curriculum | 39.32M | 19:20–19:51; about30m21 | First working long-trained actor; legally weak |
| `b6764c1a08b3` | Full-geometry PPO;512 envs,1000 iterations,32 steps;1/2/3/4m/s curriculum | 16.38M | 20:09–20:45; about35m46 | Base neural actor used by winner |
| `7294eafc964d` | Full-geometry fall-tolerant PPO;512 envs,800 iterations,32 steps; stronger lane penalties | 13.11M | 20:48–21:17; about28m41 | Final four candidates do not improve best |

Actor and critic networks are256/128/128 MLPs. Minimal geometry removes much collision detail; the later full asset more closely matches evaluation. The filename `ppo_nofall.pt` is misleading if read as “never falls”: the final job uses `--allow-fall`, meaning no early fall termination, plus weaker upright penalties, stronger lane penalties and perturbed resets. Its four queued candidates have maximum official effective speed0.147113m/s.

Sources: [minimal actor returned, #437](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s440), [full-training registry](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/gpu-job-registry/b6764c1a08b3.json), [final-training registry](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/gpu-job-registry/7294eafc964d.json), [full-training source archive](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/gpu-job-work/b6764c1a08b3/app.tar.gz), member `app/train_g1_ppo.py`.

### 6.3 The exact winning artifact and branch

SHA256 of archived `app/ppo_full_ensemble2.pt` is `7a933a7feef9801618134c819ca5134658ff1de2b7e571daa3d40699afedaa2e`, matching official policy24. The archived `make_policy_ensembles.py` defines family2 as row-wise variants of the same neural actor:

| Row modulo3 | Observation columns120:122 before the actor | Meaning |
| --- | --- | --- |
| 0 | Original values | Raw actor |
| 1 | Multiplied by0.5 | Weaker cross-track/heading input |
| 2 | Multiplied by1.5 | **Stronger cross-track/heading input** |

The recorded winning capture is `representative_lane: 2`, with `env_id: 2`, identifying the stronger-input branch. It finishes in27.466s, with max whole-body lateral extent0.5981608m against0.61m and no measured disqualifying self-collision. This is a hand-coded input transform around a learned policy, not a separately trained third model. Increasing input gain is not proof of a simple linear increase in steering torque: the neural actor is nonlinear.

The unmodified `ppo_full.pt` has SHA543e4173… and official effective speed0.030091m/s; the full ensemble2 artifact achieves3.640803m/s. This large difference makes the wrapper a central part of the observed success. It does not establish repeated-run robustness or a controlled average effect.

Public anchors: [#646 creates wrappers](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s649), [#648 queues the seven full-actor variants](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s651). Local sources: [winning source/artifact archive](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/gpu-job-work/9897cf8473d5/app.tar.gz), members `app/make_policy_ensembles.py` and `app/ppo_full_ensemble2.pt`; [winning capture metrics](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/captures/frontier-7a933a7feef9.json).

### 6.4 Queue management and post-stop scoring

Luna queues alternatives and repeatedly reorganizes them. At20:33 it cancels lower-value queued work to preserve budget for the full/fall-tolerant family. `Promise.all` often runs status/log/cost checks or packaging commands concurrently; the training contract remains **one A10G worker per trial**, so those calls are not evidence of parallel GPU training jobs.

The winning seven-policy submission was queued at20:48:14. Its GPU job `9897cf8473d5` never executes: attempt0, terminated. After the agent trace is interrupted at21:25:20, the submission bridge recovers the queued snapshot (`source: enqueue_snapshot_at_budget_stop`) and forwards the winning artifact at21:28:30. It is officially submitted at21:31:34 and finishes scoring at21:33:34. This recovered submission is real, but **Luna did not observe that official success during the recorded agent trace**, and the forwarding job did not itself consume a completed GPU execution window.

Public anchors: [#579 trims queued work](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s582), [#655 final four slots planned](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s658), [#758 interruption](http://localhost:64853/trajectory?run=s10-vexp-r123-20260828-luna-4#a1-s761). Local sources: [queued job record](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/gpu-job-registry/9897cf8473d5.json), [winning artifact recovery](/data/sprint/runs/ops/s10-vexp-r123-20260828-luna-4/submission-bridge/204814-e3b4.json).

The registry contains21 jobs,18 with actual execution:8 RL attempts including2 library failures;4 probe/script/motion jobs;2 local verification jobs;4 started submission forwarders;3 attempt0 queued jobs. Statuses are12 succeeded,3 failed,6 terminated. The three attempt0 jobs must not be counted as three completed GPU-training runs.

## 7. Polling, harness effects, and interpretation limits

| Public trajectory measure | DeepSeek | GLM | Luna |
| --- | ---: | ---: | ---: |
| Visible steps | 315 | 223 | 758 |
| Steps containing `event gpu status` | 144 | 7 | 257 |
| Steps containing `event gpu logs` | 134 | 38 | 259 |
| Steps containing `event cost` | 40 | 20 | 156 |
| Steps containing a numeric sleep command | 124 | 48 | 168 |
| Steps containing `Promise.all` | 0 | 0 | 177 |

These are counts of public steps whose tool-argument text contains the command, not parsed command executions, API calls, or mutually exclusive categories. One step can contain all three polling commands. Tool schemas differ across harnesses; direct comparison requires this qualification.

Nevertheless, repeated log/status/cost polling is visible in Luna and DeepSeek, while GLM commonly uses longer individual waits and fewer visible steps. Repeated observations enlarge histories and can trigger more model requests. This is a plausible mechanism connecting orchestration style with API and CPU-lifetime cost, but it is not a separately measured dollar allocation. Some polling discovers real failures, protects artifacts, or permits early cancellation and can save later compute.

All three run contracts say `official_results_hidden_agent_local_verification`. Discussion of an agent's local evaluation or belief must not be confused with access to retrospective official results. Official best scores in this report are known to the auditor after the run. Failed/unfinished effective speed also differs from ordinary mean velocity; do not divide distance by time and substitute that for the official metric.

Other limitations:

- The selected trials are best outcomes, not a randomized matched comparison. They have different harnesses and models simultaneously, and were not run with controlled identical policies, requests, context management, seeds or training trajectories.
- No causal, controlled harness experiment was performed. We cannot assign a percentage of Luna's cost to “Codex overhead,” prove that compaction made one model worse, or forecast savings from substituting another harness without changing behavior.
- A registry failure may be an artifact/lifecycle failure rather than failed learning; a successful job may return a poor policy. Neither status alone measures useful work.
- Shortened screening rollouts and full official scoring are not interchangeable. The DeepSeek horizon change illustrates this directly.
- Provider job logs are not all materialized as standalone local `worker.log` files. Log claims here use the preserved tool observations in raw public trajectories, cross-checked with job registries and archived source.
- Per-job exact invoice dollars, exact cost of a poll/compaction, and exact latent bottlenecks remain unresolved. Duration×reservation floor is an estimate, not a provider invoice allocation.
- The run IDs identify the historical sample; the active manifest is authoritative for which15 trials belong to the current comparison. Prefix matching would include/exclude the wrong runs.

## Appendix A. All15 trials in the active batch

Manifest: `agents-100m-sampled-15-20260829`, updated2026-08-29T07:33:42Z. Rows below enumerate `web/data/batches/current.json.arms`, then join each exact `run_id` to its usage audit and reconciled compute file. There is no run-prefix inference. All totals below retain the benchmark API basis and exclude the official verifier.

| Family / displayed trial | Exact run ID | API USD | Whole training worker USD | Whole agent sandbox USD | Agent total USD | API requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek1 | `s10-vexp-r120-20260828-deepseek-1` | 1.347606880 | 6.29877259 | 2.40358472 | 10.049964190 | 314 |
| DeepSeek2 | `s10-vexp-r120-20260828-deepseek-2` | 1.301101984 | 6.53796731 | 2.14110008 | 9.980169374 | 284 |
| DeepSeek3 | `s10-vexp-r121-20260828-deepseek-3` | 2.531795352 | 5.38081520 | 2.15299503 | 10.065605582 | 398 |
| DeepSeek4 | `s10-vexp-r122-20260828-deepseek-4` | 1.886606608 | 5.51482736 | 2.51936088 | 9.920794848 | 373 |
| DeepSeek5 | `s10-vexp-r120-20260828-deepseek-5` | 1.728813296 | 6.12438226 | 2.21273436 | 10.065929916 | 316 |
| GLM1 | `claude-goal-20260828-0237-main-glm-2` | 0.660582910 | 6.79463503 | 2.50271535 | 9.957933290 | 182 |
| GLM2 | `claude-goalfix2-20260828-1750-glm-1` | 1.000240720 | 6.34154864 | 2.50588007 | 9.847669430 | 257 |
| GLM3 | `claude-goalfix2-20260828-1750-glm-2` | 0.751649690 | 6.58695469 | 2.55848240 | 9.897086780 | 220 |
| GLM4 | `claude-goalfix2-20260828-1750-glm-3` | 0.631492550 | 6.89706153 | 2.35812620 | 9.886680280 | 159 |
| GLM5 | `claude-goalfix2-20260828-1750-r4-glm-4` | 0.604842510 | 6.94444912 | 2.38854711 | 9.937838740 | 157 |
| Luna1 | `s10-vexp-r123-20260828-luna-2` | 4.645198140 | 3.61276337 | 1.62512509 | 9.883086600 | 1,041 |
| Luna2 | `s10-vexp-r123-20260828-luna-3` | 4.166165280 | 3.98374616 | 1.67217265 | 9.822084090 | 964 |
| Luna3 | `s10-vexp-r123-20260828-luna-4` | 3.280083030 | 5.08680281 | 1.52919451 | 9.896080350 | 756 |
| Luna4 | `s10-vexp-r123-20260828-luna-5` | 5.171857500 | 2.85653048 | 1.65843057 | 9.686818550 | 1,150 |
| Luna5 | `s10-vexp-r124-20260828-luna-1` | 3.250955370 | 5.30267298 | 1.46679902 | 10.020427370 | 777 |

| Five-trial family aggregate | Benchmark API USD | Provider-reported API USD | Training worker USD | Agent sandbox USD | Agent total USD | Excluded verifier USD | API requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek | 8.79592412 | 4.766442872 | 29.85676472 | 11.42977507 | 50.08246391 | 1.22089701 | 1,685 |
| GLM | 3.64880838 | 1.824404190 | 33.56464901 | 12.31375113 | 49.52720852 | 0.80440033 | 975 |
| Luna | 20.51425932 | 20.514259320 | 20.84251580 | 7.95172184 | 49.30849696 | 2.17893851 | 4,688 |

The family aggregates reinforce the descriptive pattern—Luna spends more on API and less on allocated compute—but do not remove model/harness confounding or establish that the five selected trials are representative of every possible run.

## Appendix B. Reproduction and source conventions

For every manifest arm with run ID `R`, the monetary join is:

```text
API             = /data/sprint/runs/ops/R/usage/run-usage-audit.json
                   .calculated_api_usage_usd
Provider API    = same file .provider_billed_api_usage_usd
Agent sandbox   = /data/sprint/runs/ops/R/telemetry/modal-cost.json
                   .by_role_usd.cpu_agent
Training worker = same file .by_role_usd.training_gpu
Verifier        = same file .by_role_usd.verifier_gpu
Agent total     = API + Agent sandbox + Training worker
```

Pricing snapshots, token fields and discount transformations are preserved per request in the API files, not reconstructed from current marketing pages. GPU command/attempt/status information comes from each run's `gpu-job-registry/*.json`; training sources come from the corresponding `gpu-job-work/JOB/app.tar.gz`. Only archive members needed for this audit were read; no policy execution was required for artifact SHA verification.

Public trajectory links use the local preview origin `http://localhost:64853` and the site's `/trajectory?run=...#a1-s...` anchors; their fragment is the raw `step_id`, not necessarily the displayed public step number. Replace the preview origin with the deployed site origin when publishing. Local evidence links use absolute paths so they remain unambiguous in the shared workspace.
