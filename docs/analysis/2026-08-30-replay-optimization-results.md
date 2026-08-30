# Replay optimization validation

Date: 2026-08-30. This report records staged measurements for shared replay assets (priority 1) and persistent comparison rendering (priority 3).

Status: baseline, priority 1, and priority 3 implementation/measurement complete. Correctness checks passed; fixed-seek priority 3 visual comparison is recorded below.

## Method

- One Chromium test session, real WebGL through ANGLE/SwiftShader software rendering, viewport 1280 × 900, device pixel ratio 1. This is a shared development VM, not the user's hardware GPU or a representative phone.
- A separate read-only HTTP fixture serves the current website with Brotli quality 5 and browser `Cache-Control` rules from `web/vercel.json`. It does not modify or replace the user's preview server. The fixture's compressed responses are warmed before measured rounds so on-demand compression is not part of the repeated comparison.
- The baseline fixture freezes the trajectory page, its JavaScript/styles, Vercel header configuration, the complete inline comparison shell, hero, and original collision example before implementation. Existing capture/trace data are read-only and shared between stages.
- Each stage runs three independent rounds for two scenarios: fallen Luna trial 2 policy 9, adding/removing/re-adding policy 8; and moving winning GLM trial 2 policy 9, adding/removing/re-adding policy 6. Each round clears Chromium's HTTP cache before navigation. Shader/driver state is not reset between rounds.
- Each stage is measured both on unthrottled loopback and with CDP network emulation set to 500,000 bytes/second (4 Mbps) and 40 ms latency. These are controlled test conditions, not claimed real-world network performance.
- Readiness is the parent page's `g1:policies-state` message after scene initialization/first draw. Cold measurements include page and trace dependencies; selection measurements start at the policy-button action. Diagnostic collection happens after readiness and is excluded from reported ready time. Readiness is the first usable frame, not a GPU-completion fence or completion of the requested half-second focus-camera transition. Functional framing checks wait separately for camera transitions to settle.
- Wire bytes come from CDP `Network.loadingFinished.encodedDataLength`, including response headers; cached resources contribute zero. Request counts include browser-cache-served requests. Whole-page cold bytes may vary slightly with the live exported index and favicon; policy-change bytes are the cleaner repeat-transfer comparison.
- Functional observations record selected order, iframe/API/canvas identity, paused time, visual/recorded playback ends, camera state, decoded pose inspection, and page errors. Separate fixed-time screenshots and pose signatures compare the hero, collision example, and comparison renderer.
- All CPU-heavy benchmark/server invocations use `run-heavy`. Other unrelated VM jobs continue; do not infer precise hardware-GPU speedups from these timings. No training or model API calls are involved.

## Baseline (before priority 1)

Three-round median readiness, unthrottled compressed fixture:

| Scenario | Cold page | Add second | Remove second | Re-add second |
| --- | ---: | ---: | ---: | ---: |
| Fallen Luna policies 9 / 8 | 4,539 ms | 1,600 ms | 1,746 ms | 2,051 ms |
| Moving GLM policies 9 / 6 | 2,336 ms | 1,589 ms | 1,412 ms | 1,698 ms |

Every policy-set change replaced the iframe, replay API, and canvas. Warm removal and re-addition each transferred 1,764,634 bytes for the comparison document even though already fetched captures were HTTP-cache hits. Adding the second fallen capture transferred 1,948,817 bytes; adding the second moving capture transferred 2,118,472 bytes.

Observed correctness: all selected labels/orders and colors matched, every comparison remained initially paused at zero, and no page errors were recorded. Fallen policy 9 ended visually at 3.06 s; adding policy 8 extended the visual end to 3.12 s while preserving the 60 s recorded end. Moving policy 9 finished at 9.898345 s; its pair with policy 6 ended at 10.466825 s.

Raw baseline diagnostic data are retained in `/tmp/replay-baseline-metrics.json`; the checked-in [compact metrics](2026-08-30-replay-optimization-metrics.json) retain all 144 measured actions across the three stages and two network conditions. The former audit's measurements used an uncompressed `no-store` preview and must not be compared directly with this compressed fixture.

## Priority 1

Shared HQ geometry and Three.js are now separate content-hashed scripts under `/assets/replay/`, served with `public, max-age=31536000, immutable`. Existing captures and scene behavior are unchanged; portable self-contained exports remain supported.

**Measured result:** warm removal/re-addition transfers fell from **1,764,634 to 31,996 bytes (98.19% less)**. New second-capture loads fell from 1,948,817 to 216,179 bytes for the fallen pair and 2,118,472 to 385,834 bytes for the moving pair. Shared scripts were memory-cache hits with zero wire bytes; captures were browser-cache hits after their first fetch. First cold-page transfer remained approximately 2.56 MB / 2.46 MB because each shared resource is still needed once.

Three-round median readiness under **4 Mbps / 40 ms**:

| Scenario / action | Before | Priority 1 | Change |
| --- | ---: | ---: | ---: |
| Fallen, cold page | 10,425 ms | 7,571 ms | −27.4% |
| Fallen, add second | 6,775 ms | 1,821 ms | −73.1% |
| Fallen, remove second | 5,581 ms | 1,658 ms | −70.3% |
| Fallen, re-add second | 4,942 ms | 1,755 ms | −64.5% |
| Moving, cold page | 7,157 ms | 6,783 ms | −5.2% |
| Moving, add second | 5,729 ms | 2,327 ms | −59.4% |
| Moving, remove second | 4,982 ms | 1,622 ms | −67.4% |
| Moving, re-add second | 4,966 ms | 1,741 ms | −64.9% |

Unthrottled compressed-fixture priority 1 medians:

| Scenario | Cold page | Add second | Remove second | Re-add second |
| --- | ---: | ---: | ---: | ---: |
| Fallen Luna policies 9 / 8 | 5,384 ms | 2,123 ms | 1,620 ms | 1,649 ms |
| Moving GLM policies 9 / 6 | 2,005 ms | 1,511 ms | 1,424 ms | 1,498 ms |

Cold and unthrottled results vary substantially with software-GPU scheduling; do **not** interpret the cold timing differences as a robust cold-start improvement. The strongest established improvement is repeat-transfer elimination and lower constrained-network policy-change latency. Scene/API/canvas identity still changes on every selection, so repeated rendering initialization remains for priority 3.

Correctness:

- All 48 priority 1 measured actions had finite ready timestamps, correct selected order/labels, unchanged paused starts/playback ends, and no page errors. Together with baseline, 96 measured actions were retained.
- Nine fixed-seek snapshots (hero, original collision, and standalone comparison; 0 / 0.8 / 2 s) had **exactly equal** pose, camera, framing, results, metadata, and playback signatures. Hero's 77.36 s presentation continuation is preserved.
- Visual comparisons show the same robot shapes/poses/framing. Screenshots are not byte-identical: procedural floor grain differs slightly despite seeded randomness (typical full-image mean absolute RGB error ≈0.02/255, virtually no pixels differing by more than 8 levels). Two initial collision screenshots also differ in the Play button's compositor-paint timing; the difference disappears in the later snapshot. This is not claimed as pixel-zero verification.
- Blocking the engine script and mesh script separately produced a specific visible load error, disabled Play, and no half-initialized replay API. Unblocking and reloading restored a working player.
- Publisher verified lossless round trips for all 479 existing generated replay documents; aggregate raw HTML shrank from 1,948,883,222 to 418,570,181 bytes, with exactly two shared assets. This is on-disk/raw HTML reduction, not compressed network savings for a single visit.
- Current-15-trial deployment-bundle validation passed: 410 files / 1,147,334,592 bytes, including the comparison shell and its shared assets. No deployment command was run as part of this validation.

The normal local preview still uses uncompressed `no-store` responses. These HTTP-cache gains were confirmed on the production-equivalent fixture, not misrepresented as an optimization of the preview server. Persistent renderer reuse should help independently of those headers.

## Priority 3

Comparison policy changes now keep the iframe, replay API, canvas, renderer, shared geometry, and unchanged actors. Parsed/packed captures are reused through a bounded in-document cache. A newly selected capture is fetched once; warm removal and re-addition require **zero HTTP requests and zero wire bytes**, not just browser-cache hits.

Three-round median readiness under **4 Mbps / 40 ms**, showing each implementation stage separately:

| Scenario / action | Baseline | Priority 1 | Priority 1 + 3 |
| --- | ---: | ---: | ---: |
| Fallen, cold page | 10,425 ms | 7,571 ms | 7,801 ms |
| Fallen, add second | 6,775 ms | 1,821 ms | 518 ms |
| Fallen, remove second | 5,581 ms | 1,658 ms | 501 ms |
| Fallen, re-add second | 4,942 ms | 1,755 ms | 50 ms |
| Moving, cold page | 7,157 ms | 6,783 ms | 7,258 ms |
| Moving, add second | 5,729 ms | 2,327 ms | 815 ms |
| Moving, remove second | 4,982 ms | 1,622 ms | 587 ms |
| Moving, re-add second | 4,966 ms | 1,741 ms | 899 ms |

Priority 3 reduced median new-capture selection readiness by **71.6% / 65.0%** relative to priority 1 for the fallen / moving scenarios. Warm re-addition improved by 97.2% / 48.4% in this run. The software-rendering scheduler produced substantial variation: for example, fallen re-addition was 802 / 46 / 50 ms across three rounds. These are renderer-readiness signals, not compositor/GPU-finish measurements or guarantees of sub-100-ms visible transitions. The repeat-request elimination is deterministic; the exact latency ratios are not. No additional cold-start improvement is claimed.

Unthrottled compressed-fixture medians:

| Scenario / action | Baseline | Priority 1 | Priority 1 + 3 |
| --- | ---: | ---: | ---: |
| Fallen, cold page | 4,539 ms | 5,384 ms | 5,217 ms |
| Fallen, add second | 1,600 ms | 2,123 ms | 179 ms |
| Fallen, remove second | 1,746 ms | 1,620 ms | 588 ms |
| Fallen, re-add second | 2,051 ms | 1,649 ms | 947 ms |
| Moving, cold page | 2,336 ms | 2,005 ms | 1,706 ms |
| Moving, add second | 1,589 ms | 1,511 ms | 865 ms |
| Moving, remove second | 1,412 ms | 1,424 ms | 22 ms |
| Moving, re-add second | 1,698 ms | 1,498 ms | 706 ms |

Repeat-transfer comparison (same bytes in both network conditions):

| Action | Baseline | Priority 1 | Priority 1 + 3 |
| --- | ---: | ---: | ---: |
| Add fallen capture | 1,948,817 B / 3 requests | 216,179 B / 5 requests | 184,183 B / 1 request |
| Add moving capture | 2,118,472 B / 3 requests | 385,834 B / 5 requests | 353,838 B / 1 request |
| Warm remove | 1,764,634 B / 2 requests | 31,996 B / 4 requests | 0 B / 0 requests |
| Warm re-add | 1,764,634 B / 3 requests | 31,996 B / 5 requests | 0 B / 0 requests |

### Correctness and memory

- All 48 priority 3 measured actions had finite readiness, correct selected order/playback ends, and no captured page or iframe errors. All 36 changed-set actions retained the same iframe/API/canvas; each page built exactly one scene. All 24 warm remove/re-add actions issued zero requests. All 12 first additions fetched only the newly selected recording.
- All nine priority 3 fixed-seek pose/camera/framing/results/playback signatures exactly matched both baseline and priority 1. Screenshot comparisons retained the same robot appearance and framing; full-image mean absolute RGB differences were 0.021–0.032/255, with maximum channel differences of 8–10, consistent with procedural floor grain. No pixel-zero claim is made.
- Re-running shared-engine and mesh failure checks after priority 3 again showed specific visible errors, disabled Play, and no half-initialized API; clearing the blocked requests and reloading recovered successfully.
- Sixteen separate real-browser functional checks passed: paused seek, renderer/shared-geometry/actor reuse, Back/Forward, focus-only preserved time and no requests, changing a playing set, failed capture with visible error, retry, superseded delayed requests, eight-policy identity/order, all-policy stopping, Replay restart, repeated churn, cache/GPU resource stability, and no uncaught exceptions.
- Policy-set changes preserve baseline behavior: reset to time zero and pause. Focus-only changes preserve time/play state and the half-second camera transition. The eight-policy Luna comparison retains the 60 s recorded duration but stops visually at the latest selected policy end, 3.12 s; the 3.06 s policy alone still stops at 3.06 s.
- After loading eight unique captures, 20 remove/re-add cycles (40 policy-set changes) made zero HTTP requests. The original retained actor and all 28 shared geometry identities stayed fixed. Removed actors were disposed; the renderer stayed at **37 GPU geometries / 6 textures** with two active actors throughout sampled cycles 1 / 5 / 10 / 20.
- Accounted capture-cache storage stayed at **15,485,648 bytes / 8 entries**, within its 64 MiB / 12-entry limits; active captures accounted for 9,659,408 bytes. Fetch and pack counters stayed constant while cache hits increased. This bound is for the cache's accounted storage, **not total browser/GPU memory**.
- After explicit garbage collection, JavaScript heap changed from **39,852,088 to 40,014,864 bytes (+0.41%)** across those 40 changes; embedder heap changed from 230,261,208 to 230,806,968 bytes (+0.24%). These short-run samples show no accumulating GPU-resource/cache growth, not proof against every possible long-running memory leak.
- Publisher rechecked all 479 embedded recording `DATA` hashes after scene propagation: identical to their pre-optimization values. Hero's 77.36 s presentation continuation remains intact. Focused implementation and bundle tests passed before timing.
- Final broad regression: **260 tests passed in 62.02 s**, including export performance, trajectory outline, policy-queue cost/provenance, and website-bundle tests. JavaScript syntax and `git diff --check` passed. The environment lacks PyTorch, so `tests/test_g1_100_metres_scoring.py` and separate control-batch collection could not run; these optimizations do not modify scoring or simulation code.
- Final current-15-trial bundle built and validated: **411 files / 1,148,303,644 bytes** (`/tmp/sprint-p3-bundle.Zchl63`), including the persistent comparison shell and shared assets. No deployment command was run for this check.

Persistent renderer/capture reuse also avoids repeat work on the normal `no-store` preview. Shared-asset HTTP caching still depends on the production headers and is only claimed for the compressed/cacheable fixture used here.
