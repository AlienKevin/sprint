# Replay and asset-loading performance audit

Date: 2026-08-30. Source and asset measurements were taken around 18:42–18:48 UTC from the current `/data/sprint` working tree.

**Historical baseline audit. Priority 1 (shared assets) and priority 3 (persistent comparison rendering) were subsequently implemented; priority 2 (lower rendering quality) was not. See the [before/after verification report](2026-08-30-replay-optimization-results.md) for measurements and validation status. Descriptions of the current loading path below refer to this audit's original snapshot, not the optimized implementation.**

Separate approved behavior change: after the baseline measurements, the three Observations replays were changed to start paused (`autoplay=0`). Browser checks confirmed all three remain at zero until Play is clicked. This prevents automatic playback, but does not eliminate initial downloads, first-frame initialization, or offscreen rendering after the user starts playback. Hero and Scoring autoplay remain unchanged. The loading measurements below are from before this opt-out; its functional verification is subsequent.

## Executive summary

The clearest measured source-level inefficiency is duplication of the robot meshes and Three.js in every replay document. The robot meshes account for about 1.59 MB of a 1.76 MB Brotli-compressed comparison shell. Changing the selected policy set creates a new iframe, so that shell and its initialization are repeated.

Large raw capture sizes are not equivalent to large production transfers: the 6.22 MB capture of Luna trial 2 policy 9 compresses to approximately 169 KB because most of its minute-long tail is repetitive. Its full decoded frame array still has to be parsed and packed, however. The recently implemented trailing-stall playback cutoff changes the playback duration, not the amount downloaded or initialized.

The recommended implementation order balances measured opportunity and estimated effort:

1. Extract shared mesh and Three.js assets into content-versioned, cacheable resources.
2. Introduce a lighter rendering budget for small/mobile replay windows, primarily shadows and pixel density.
3. Retain the comparison renderer across policy-set changes, adding/removing actors and reusing bounded decoded-capture data.

The first two are smaller changes. The third is not a trivial fix, but directly targets the repeated-selection bottleneck: in a controlled two-policy initialization, the first WebGL render consumed 1.18 s of 1.40 s total initialization. Compact capture exports and tighter visibility control remain useful follow-ups, not the highest-priority fixes for that measured delay. Absolute timings come from SwiftShader software rendering, not the user's hardware GPU; no end-user speedup is promised.

## Scope and measurement limitations

- Read-only source inspection, local/public HTTP requests, compression, isolated Node.js CPU measurements, and browser resource/CPU profiling. Browser-only diagnostic instrumentation and rendering-quality changes were discarded by navigation. No new simulation, training, deployment, or performance implementation.
- Bulk compression and CPU work used `run-heavy`.
- File sizes below are exact byte counts at the measurement snapshot. Working-tree changes after that snapshot can change generated HTML sizes slightly.
- Compression measurements use gzip level 6 and Brotli quality 5 through Node's zlib. They are reproducible estimates, not claims about Vercel's exact compression settings.
- Local HTTP transfer durations are loopback measurements. They do not represent the user's forwarded preview, mobile network, browser parsing, first rendered frame, or production end-to-end latency.
- Isolated Node.js CPU measurements exclude browser DOM work, GPU upload, shader compilation, rasterization, and frame scheduling. They must not be presented as mobile browser timings.
- Engineering effort estimates below are rough implementation-and-test estimates, not commitments or measured performance improvements.

## Current loading path

### Homepage

The homepage has six replay iframes: one eagerly loaded hero race, two scoring examples, and three observation examples. The five examples use native `loading="lazy"`; this does not unload or pause a replay after it has loaded. Browser lazy-loading thresholds also need not coincide with actual visibility.

At the measured baseline, the shared scene creates an antialiased WebGL renderer, caps device pixel ratio at 2, enables soft shadow maps, and normally starts playback after 500 ms. There is no replay visibility observer that pauses an offscreen scene. Only the hero loaded in the initial browser viewport; native lazy loading was working. Multiple examples can still remain active after they have been visited, but the effect of that contention was not quantified. The separately approved Observations start-paused change reduces automatic contention, but is not viewport-aware pausing.

Sources: [homepage iframe markup](/data/sprint/web/index.html), [renderer setup and playback lifecycle](/data/sprint/web/renderers/g1-100-metres/scene.js:143).

The homepage also fetches policy-index, timeline-index, performance, batch, and pricing data together, then refreshes every 30 seconds and on focus/page-show events. Timeline-index and performance alone total 1,817,462 raw bytes. The site's data URLs use `no-store`, so repeated stable data is another avoidable workload, although compressed meshes dominate the replay shell's transfer cost.

Source: [loadSnapshot and refresh](/data/sprint/web/app.js:261).

### Trajectory and policy selection

1. The trajectory page fetches its index, then the full selected trajectory and outline. It parses the full trace and computes its SHA-256 before publishing the loaded event.
2. The overview prefetches timeline-overview and policy metadata when a run is present in the URL. Policy availability checks may issue HEAD requests for captures whose readiness metadata is stale.
3. Selecting a different set of policies replaces the replay iframe. The URL includes the selected captures and a new replay generation token.
4. The comparison shell embeds the registry, HQ meshes, scene source, and Three.js. Capture requests run concurrently with `cache: 'force-cache'`.
5. The shell parses captures, packs the visible links into another frame array, constructs a new scene, and reports readiness after a requestAnimationFrame callback.

Sources: [trajectory loading](/data/sprint/web/trajectory.js:72), [overview prefetch](/data/sprint/web/trajectory-overview.js:73), [iframe replacement](/data/sprint/web/trajectory-overview.js:847), [comparison-shell generator](/data/sprint/web/renderers/g1-100-metres/render_trial_comparison.py:78), [capture loading and scene creation](/data/sprint/web/renderers/g1-100-metres/render_trial_comparison.py:188).

Already present: concurrent capture fetches, HTTP cache preference for captures, generation checks against stale responses, an explicit loading state, and reuse of the current iframe for some focus-only interactions. Not present: a persistent renderer across policy-set changes, an application-level decoded-capture cache across iframe lifetimes, or independently cached shared geometry assets.

## Payload measurements

All sizes in this table are bytes. Compression columns are local measurements at the specified settings.

| Resource | Raw | gzip 6 | Brotli 5 |
| --- | ---: | ---: | ---: |
| Hero `model-race.html` | 11,037,672 | 4,125,146 | 4,071,623 |
| `replay/trial-comparison.html` | 3,362,364 | 1,880,992 | 1,763,664 |
| Shared embedded HQ meshes | 2,592,378 | 1,696,887 | 1,593,107 |
| Embedded Three.js | 603,445 | 149,992 | 136,810 |
| Comparison registry | 79,273 | 5,111 | 4,235 |
| Comparison scene source | 58,352 | 19,308 | 18,618 |
| Luna trial 2 policy 9 capture | 6,218,257 | 229,791 | 169,254 |
| Winning GLM capture | 1,059,605 | 324,022 | 332,152 |
| Winning Luna capture | 2,903,743 | 888,927 | 922,543 |
| Timeline index | 1,241,538 | 127,822 | 76,202 |
| Performance snapshot | 575,924 | 65,757 | 33,699 |
| `trajectory-overview.js` | 58,115 | 14,495 | 14,000 |
| `trajectory.js` | 35,371 | 10,722 | 10,358 |

The hero's embedded policy data is 7,755,095 raw bytes, or 2,301,876 bytes at Brotli 5. Geometry and the trajectory samples are its important components; micro-optimizing the small registry is unlikely to be the first priority.

The sample full Luna trajectory is 7,176,644 raw bytes, its outline 5,661 bytes, its timeline overview 517,956 bytes, and its policy metadata 9,936 bytes. Deep-link loading therefore has a separate trace/DOM dependency before the replay can be restored, not just the capture request.

### Six homepage replay documents

These totals describe loading all six examples, not the initial viewport. The scoring collision example was `3c89dd3bbca5` when measured.

| Replay | Raw bytes | Brotli 5 bytes |
| --- | ---: | ---: |
| Hero race | 11,037,672 | 4,071,623 |
| Lane example `8e2feee19eac` | 3,490,009 | 1,823,974 |
| Collision example `3c89dd3bbca5` | 3,432,026 | 1,802,509 |
| DeepSeek observation `76d7d31f8c51` | 7,324,826 | 2,927,450 |
| Luna observation `dd28af63bf84` | 7,244,443 | 1,870,820 |
| GLM observation `5eb6fee88ef0` | 3,369,801 | 1,785,874 |
| **Total** | **35,898,777** | **14,282,250** |

A separately compressed copy of HQ meshes plus Three.js is 1,729,917 bytes. Five additional copies are 8,649,585 bytes by arithmetic. This gives the approximate scale of duplicate transfer that shared cacheable assets could avoid across a full-page visit; it is not an exact prediction of post-change transfer, because compression boundaries and cache behavior change.

## Local preview versus deployed site

At the measurement time:

| Environment / request | Observed behavior |
| --- | --- |
| Local `http://localhost:59453/model-race` | Python SimpleHTTP, `Cache-Control: no-store`, no content encoding, 11,037,672-byte body |
| Local comparison shell and capture | Same uncompressed, `no-store` behavior |
| Public `https://g1-sprint.vercel.app/model-race` | HTTP/2, Brotli, `Cache-Control: public, max-age=60`, `x-vercel-cache: HIT` |
| Public hero GET with Brotli accepted | 3,844,715 encoded body bytes; approximately 0.2185 s total and 0.0424 s TTFB from this VM |
| Public comparison shell / Luna policy 9 capture | Both returned 404 |

The deployed hero reported `Last-Modified: Sun, 30 Aug 2026 07:15:18 GMT`. Thus the public deployment was an older/different bundle than the current preview. Its timings must not be attributed to the current comparison viewer. The preview port shown to the user can also differ from the server port used for these VM-local checks.

The repository's [Vercel header configuration](/data/sprint/web/vercel.json) sets a global 60-second browser cache lifetime and explicitly disables caching for the root page, version file, and `/data/*`. Local SimpleHTTP does not implement those production headers or compression. A `force-cache` request does not make a `no-store` response persist in cache.

Do not simply mark every capture URL immutable: its current filename incorporates the policy hash, not necessarily the complete capture-content hash. Corrected metadata or regenerated captures may reuse that policy URL. Long-lived immutable caching should use content-versioned asset URLs or an equivalent explicit invalidation scheme.

## CPU and geometry observations

The HQ mesh set contains 28 meshes, 110,997 vertices, and 212,355 triangles. Each new scene decodes quantized positions and indices, computes vertex normals, and computes bounds. This work is repeated even when the same robot geometry was used in a preceding iframe.

Five-sample isolated Node.js measurements, with garbage collection before each sample:

| Operation | Median | Range |
| --- | ---: | ---: |
| Parse Luna policy 9's complete capture JSON | 31.02 ms | 30.80–42.52 ms |
| Parse its 7.18 MB complete trajectory JSON | 16.96 ms | 15.61–20.64 ms |
| Decode HQ geometry, compute normals and bounds | 46.97 ms | 34.73–96.17 ms |

These numbers establish real repeated work, but do not establish the dominant browser bottleneck. In particular, they do not explain seconds of loading without browser evidence about scripting, DOM/layout, GPU upload, shader compilation, contention, and first-frame rendering.

Source: [geometry reconstruction](/data/sprint/web/renderers/g1-100-metres/scene.js:276).

## Recommended top three changes

### 1. Share and cache the mesh and Three.js assets

**Estimated effort:** half a day to one day, including generator changes and regression checks.

Extract the identical HQ mesh payload and Three.js into shared content-versioned assets. Give those immutable URLs a long cache lifetime; keep changing policy-selection state and capture metadata separate. Make each generated replay await the shared resources instead of embedding them.

**Measured basis:** about 1.73 MB compressed shared payload is duplicated per document; the HQ meshes account for roughly 90% of the compressed comparison shell. This is a larger repeat-transfer target than the registry or scene source.

**Expected benefit, not yet timed:** less duplicate transfer on additional examples and changed selections, especially on constrained mobile connections. The first cold load still needs the geometry. Separate assets do not by themselves remove per-iframe geometry/GPU initialization.

**Risks and validation:** maintain correct cache invalidation, CORS/CSP compatibility if relevant, local preview behavior, deployment asset inclusion, and any intended self-contained offline export. Validate first-load failure handling and warm cache use. Shared assets should not alter meshes, poses, camera identities, or scoring metadata.

### 2. Use a lighter rendering budget for compact/mobile replays

**Estimated effort:** two to four hours for a scoped quality preset and verification on representative hardware; allow additional time if visual tuning is needed.

Cap pixel density more conservatively for small embedded players and simplify or disable expensive shadows there. Preserve the current high-quality desktop presentation until a visual comparison justifies changing it. Do not alter robot poses, collision geometry used for scoring, or replay timing.

**Measured basis:** first WebGL rendering took 1,179.6 ms of 1,397.5 ms scene initialization in a controlled two-policy replay. Profiling separately found about 1,185 ms in shader-program inspection, compared with 19 ms decoding meshes and 17 ms computing normals. This makes rendering a more promising target than small JavaScript or registry reductions in this environment.

A warmed browser-only diagnostic measured approximately 8.4 ms CPU draw submission with baseline settings, 4.6 ms without shadows, and 2.2 ms without shadows at half pixel density. The test browser's baseline DPR was 1, so the half-DPR experiment was a deliberately low-quality diagnostic, **not** a recommendation to ship DPR 0.5. WebGL is asynchronous: these are not GPU-completion times, FPS measurements, or measured cold-start improvements.

**Expected benefit, not yet measured on user hardware:** cheaper first rendering and playback, particularly in small previews or constrained devices. This recommendation has lower implementation effort than persistent-scene refactoring, but trades visual quality for performance.

**Risks and validation:** compare readable body/contact detail, shadows, aliasing, and labels at mobile and desktop sizes. Measure cold first-frame and steady-state frame times on a physical GPU, not only SwiftShader. Keep a clear quality fallback; do not promise the diagnostic draw-submission ratio as an end-user speedup.

### 3. Keep the comparison renderer alive across policy-set changes

**Estimated effort:** one to three days, including lifecycle/history and stale-selection regression tests. This is the largest of the three changes, not a quick cache-header edit.

Keep one scene, camera, renderer, shared geometry, and track alive. On selection changes, add/remove the affected robot actors and load only new captures. Keep decoded captures in a bounded cache, with explicit disposal and eviction, rather than preloading every policy in a trial. Continue using generation identifiers to reject stale asynchronous responses.

**Measured basis:** adding a second policy took 5.98 s in the initial test; removing and re-adding it took 1.60 s and 2.89 s. Each policy-set change recreated the iframe and downloaded the entire shell plus all selected captures. In contrast, focus-only switching reused the existing iframe, made no new requests, and retained the paused clock. Scene initialization, particularly the first render, dominated the instrumented two-policy startup.

**Expected benefit, not yet timed:** avoids rebuilding the unchanged scene, re-uploading shared geometry, and reconstructing every existing actor on each selection. This targets policy-switch latency more directly than capture compaction or shared HTTP assets alone.

**Risks and validation:** preserve Back/Forward behavior, policy order/identity, camera focus, paused state, removed-policy cleanup, and generation-race handling. Bound cache memory and free removed GPU resources appropriately. Test rapid add/remove/reselect, changing trials, one/eight selected policies, late capture errors, and renderer recovery. Measure before/after on identical hardware. No new model API calls or training are required, but retaining decoded data raises browser memory use.

## Lower-priority follow-ups

### Gate initialization and pause offscreen playback

**Estimated effort:** two to four hours, plus browser verification.

Use actual visibility to set non-hero iframe sources near the viewport and pause loaded scenes when fully offscreen or the document is hidden. Resume only according to the user's prior playback intent; preserve explicit pauses, the separately approved Observations start-paused behavior, and reduced-motion preferences. Merely adding the Observations autoplay opt-out is not the remaining recommendation here.

**Measured/source basis:** six homepage replay iframes, baseline automatic playback after loading, DPR up to 2, soft shadows, and no replay offscreen pause. Browser profiling must determine how many scenes are simultaneously active in realistic navigation, and distinguish the baseline from the subsequent Observations autoplay opt-out.

**Expected benefit, not yet timed:** lower competing CPU/GPU work, memory pressure, and battery use while browsing. This can improve perceived responsiveness even when download time is not the problem. It does not directly remove the selected comparison viewer's cold-start cost.

**Risks and validation:** do not reset an active replay's progress on ordinary scrolling, cause layout shifts, or silently resume a manually paused replay. Verify keyboard navigation, reduced motion, visibility transitions, and the existing replay-height contract.

### Export a separate render-only capture payload

**Estimated effort:** half a day to one day, including provenance and regression checks.

Keep the authoritative capture untouched. Export a rendering derivative containing the 28 links actually used by the renderer and the frames needed through the already validated visual playback end. Compute the end from the full authoritative recording, so later recoveries are not lost.

**Measured Luna policy 9 example:**

- Original: 44 bodies × 7 pose values plus time = 309 numeric fields per frame; 3,001 frames; 6,218,257 raw bytes / 169,254 Brotli-5 bytes.
- Rendered links only: 28 bodies × 7 plus time = 197 fields; 3,972,002 raw bytes / 108,335 Brotli-5 bytes.
- Rendered links and a seven-second visual tail: 351 frames; 462,781 raw bytes / 55,840 Brotli-5 bytes.

The reduced example omits 16 non-rendered links and the validated trailing stall. It is a byte-count experiment, not an implemented endpoint or a measured loading-time improvement. Metadata needed for displayed outcomes and provenance must still accompany the derivative.

That byte-count experiment used the older approximately seven-second playback cutoff at the time of measurement. The subsequently requested one-second-after-final-advance rule shortens this example further. A future rendering export must use the current validated cutoff, not hard-code the seven-second illustration above.

**Expected benefit, not yet timed:** smaller decoded arrays, less parsing/packing and memory use, plus some network reduction. Savings vary sharply by policy: a successful moving runner does not have the same repetitive tail as a fallen robot.

**Risks and validation:** retain every link used by camera/pose code, preserve interpolation and scoring labels, avoid shortening genuine slow locomotion or delayed recoveries, and preserve the hero DeepSeek continuation through 77.36 s. This is strictly a presentation export, not a new score or a modification to the official archive.

## Other findings and priorities

- **Small JavaScript/registry micro-optimizations:** low priority relative to duplicated assets and first rendering. The registry compresses to 4.2 KB, and instrumented scene source compilation took only 1.6 ms.
- **Stable snapshot caching and refresh deduplication:** avoid refetching the entire unchanged homepage snapshot on every timer/focus trigger, while retaining freshness for active trials. Important but secondary to the measured replay payload duplication.
- **Trace rendering and deep-link readiness:** inspect whether full trajectory rendering or hashing unnecessarily delays policy replay restoration. The full Luna trace is materially larger than its outline and policy metadata.
- **Preview compression:** useful for realistic preview testing and remote-forwarded preview bandwidth. Production already returned Brotli for the deployed hero, so this is not a new production optimization there.

## Browser profiling results

The current preview was measured sequentially in one Chromium browser session using SwiftShader software WebGL, baseline DPR 1. Other test browsers were closed to avoid competing browser workloads. This remained a shared development VM, not an isolated benchmarking machine. Local HTTP used uncompressed `no-store` responses. Timings below are diagnostic observations, not statistical benchmarks or estimates for the user's GPU/network. Initial/warm timing differences also include shader/driver warm-up and scheduling, not only HTTP caching.

| Scenario | Observed timing | Interpretation |
| --- | ---: | --- |
| Homepage cold DOMContentLoaded | 0.122 s | Basic document became ready well before the replay |
| Homepage warm DOMContentLoaded | 0.206 s | Not a rendering-ready measurement |
| Hero cold resource transfer | 0.247 s for 11.04 MB | Loopback transport was not the multi-second bottleneck |
| Hero warm resource transfer | 0.199 s, same bytes | `no-store` prevented a warm HTTP cache benefit |
| Homepage cold / warm load | 6.94 / 7.36 s | Hero iframe load was approximately 6.93 / 7.23 s |
| Luna trial 2, policy 9: page navigation to replay ready | 2.65 s | Includes page/trace dependencies |
| Add policy 8: click to replay ready | 5.98 s | Fresh iframe, two captures, new scene |
| Remove policy 8: click to replay ready | 1.60 s | Fresh iframe, one capture, new scene |
| Re-add policy 8: click to replay ready | 2.89 s | Still fresh iframe and downloads; some execution/driver work warmed |
| Focus an already selected policy | 0 new requests | Existing iframe retained; paused playback time unchanged |

Each policy-set change transferred a roughly 3.36 MB comparison shell plus 6.22 MB of captures for the single-policy case or 12.36 MB for the pair. `Response.json()` completion took approximately 42 ms for the single capture and 102–311 ms for the pair. Those browser intervals include body reading and scheduling; they are not isolated JSON-parse CPU times.

Browser-only instrumentation of a fresh two-policy scene measured source compilation at 1.6 ms, scene initialization at 1,397.5 ms, and the first `renderer.render()` at 1,179.6 ms (84.4% of initialization). A separate CPU profile found approximately 1,185 ms in WebGL `getProgramInfoLog`, 120 ms in `texImage2D`, 19 ms in mesh base64 decoding, and 17 ms in normals computation. Much remaining time was attributed to native program/GPU waits, not cleanly attributable JavaScript functions.

The homepage's initial viewport loaded only the hero replay; lazy example frames had not loaded. Its timeline-index and performance snapshot were nevertheless fetched twice within the initial 12 seconds through initial/focus/page-show refresh paths. Deduplicating in-flight initial snapshot requests is a small follow-up, but not an explanation for the large first-render delay.

The new half-second camera focus transition was checked for preserved paused time, correct target, and no new requests. Software rendering blocked a nominal 250 ms timer for several seconds, so this environment could not certify visually smooth animation timing. Unit tests cover the wall-clock easing independently.

After the source baseline, all three Observations URLs with `autoplay=0` stayed at time zero with Play shown through checks at 5.8, 10.5, and 27.5 seconds. Clicking Play advanced the replay normally. These are functional checks, not an offscreen-performance A/B.

Raw local artifacts: `/tmp/perf-home-cold.json`, `/tmp/perf-home-warm.json`, `/tmp/perf-home-warm.cpuprofile`, and `/tmp/perf-trial-{first,add,remove,repeat}.json`. Playback end values in these older raw profiles precede the revised one-second stall rule; use their loading measurements, not their former cutoff values.

### What still needs measurement before claiming a speedup

- Repeat cold/warm tests on a physical GPU and a representative phone, including network throttling and production-equivalent compression/cache headers.
- Measure each proposed change separately with the same policy selections. Compare transferred bytes, first usable frame, long tasks, frame pacing, and retained memory.
- Include a moving winning policy, not only repetitive fallen captures. Profile the effect of scrolling past multiple loaded examples and returning to the hero.
- Retain functional regression coverage for identity, selection/focus order, history, stale generations, pause/replay, legal/DQ labels, and the extended DeepSeek recording.

No optimization should be declared successful solely because a transfer-size estimate decreased. Compare the same cold/warm interaction scenarios before and after each isolated change.
