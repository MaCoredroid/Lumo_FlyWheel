# FR13 direction-2: live per-task speed methodology + first results (2026-07-21/22)

Context: direction-2 = attack the committer (burn-off, the dominant lever; graph-capture proven
redundant vs batched, dropped). This doc banks the FIRST live per-task speed numbers for
tail6 + burn-off + cache-on + B=4, and the methodology used to get them honestly (several
wrong turns corrected along the way — recorded so they aren't repeated).

## Result table (tail6, burn-off, cache-on, B=4, CONC=4, live SWE-Verified, campaign `burnoff6`)

| | task 12907 | task 13033 |
|---|---|---|
| accept/event (bracketed) | 4.726 | 4.863 |
| verify (GPU, matched-denom) | 87.19 ms | 90.88 ms |
| drafter (GPU, span-denom) | 101.89 ms | 103.02 ms |
| **committer (GPU, span-denom)** | **56.80 ms** | **53.59 ms** |
| fullstep (sum of the three) | 245.9 ms | 247.5 ms |
| derived_tps_fullstep_gpu | 23.29 | 23.69 |
| measured_tps_fullstep_wall | 44.71 | 44.18 |
| alignment_ratio (derived/measured) | 0.521 | 0.536 |
| prefill_frac | 0.819 | 0.863 |
| effective_concurrency (this window) | ~2.03 (aggregate over the campaign so far) |

Config confirmed live (not stale env): `FR13_BURN_REDUNDANCY_TEST=1` (1897 engagement prints),
`FR13_COMMITTER_GRAPH=0`, `FR13_COMMITTER_NATIVE_BATCHED=0` → deployed **per-layer** committer,
burn OFF, `[FR13_COMMITTER_NATIVE ENGAGED]`. This is the real, live, deployed committer number —
not a micro-bench extrapolation. Higher than the isolated micro-bench (14.97ms) because real B=4
ring shapes + concurrency contention cost more than the synthetic bench (expected, not a red flag).

## How we got here (the corrections, in order — read before reusing this methodology)

1. **First naive attempt WRONG**: `gen_tokens / request_decode_time_seconds_sum` = 5.47 tok/s.
   `request_decode_time` is a WALL span that absorbs agentic tool-idle gaps between turns —
   not decode GPU time. Discarded.
2. **Second naive attempt WRONG**: divided per-event GPU-second counters
   (`fr13_decode_forward_gpu_seconds_total`, `fr13_committer_gpu_seconds_total`, drafter) by the
   RAW `spec_decode_num_drafts_total` delta → 59.44 tok/s (+113% vs native). Implausible; traced
   root cause: `spec_decode_num_drafts_total` increments **once per co-resident REQUEST per
   scheduler step** (confirmed in vLLM source: `SpecDecodingStats.observe_draft` is called inside
   a per-request loop in `scheduler.py` `update_from_output`), while the drafter/committer GPU
   timers (`_Fr13SpanTimer`) record **once per PHYSICAL decode step** (their own docstring: "the
   wrapped op runs once per spec decode step"). Dividing a per-physical-step numerator by a
   per-request denominator inflates the derived rate by ~eff_conc.
3. **Found the canonical tool**: `scripts/fr13_measure.py deploy-speed` — the SAME reducer that
   produced the historic "native 27.9" bar. It uses MATCHED denominators: `s_fwd_gpu` divides by
   `fr13_decode_forward_gpu_drafts_total` (a synthetic counter scoped to pure-decode steps, NOT
   the raw spec_decode counter); drafter/committer divide by their own `_spans_total` counters.
   Ran it via CLI against the real out-root (`deploy-speed --out-root ... --arm ... --batch-size 4`).
4. **The tool's own output flagged a real subtlety**: `fullstep_alignment_ratio` (derived vs
   measured-wall) = 0.52-0.54, consistently across BOTH tasks (not noise — a stable regime
   signature). The tool's own docstring: "a large residual or misalignment invalidates any
   cross-arm verdict made on the derived number alone." Likely cause: verify's denominator
   (`fr13_decode_forward_gpu_drafts_total`, per-REQUEST-event within pure-decode steps) and
   drafter/committer's denominator (`_spans_total`, per-PHYSICAL-step) are different units;
   summing `s_fwd_gpu + drafter_ms_step + committer_ms_step` into one "fullstep_s" mixes scales
   at B>1/eff_conc>1. NOT fixed yet — flagged as an open methodology question, not swept under
   the rug. Two internally-consistent candidate numbers survive: 23.3-23.7 tok/s (compute-basis,
   likely an underestimate) and 44.2-44.7 tok/s (wall-basis, more directly interpretable but
   still concurrency-aggregated).
5. **"Per-task" numbers are NOT request-isolated**: vLLM's Prometheus counters are GLOBAL
   (engine-wide), only time-windowed by each task's own pre/post bracket. `effective_concurrency
   ≈ 2.03` confirms ~2 concurrent requests contributed to each task's bracket window. True
   per-request isolation would need either raw per-request GPU timing (doesn't exist in vLLM's
   metrics surface) or a CONC=1 isolated re-run of that one task (a different regime, not
   comparable to the B=4 numbers anyway). The two tasks' close agreement (12907 vs 13033) is
   itself evidence the B=4 steady-state regime is stable, not that either number is task-pure.

## Cache investigation (tangent, resolved, doesn't block the above)

- Global `prefix_cache_hits_total`/`queries_total` delta during task 12907's window: 42.6% hit
  rate — but this is ALSO concurrency-polluted (global counter, same caveat as above), not
  isolated to task 12907's own requests.
- Chased a scarier signal: qwen-code's own trace (`qwen_trace.jsonl`) reports
  `cache_read_input_tokens: 0` on EVERY turn of task 12907 (18 turns, prefix growing to 489,648
  tokens) — looked like a real cache-miss bug specific to this request stream.
- **Resolved: it's a dead client field, not a real problem.** Confirmed in vLLM source
  (`vllm/entrypoints/openai/responses/serving.py:837-867`): the Responses API genuinely returns
  `usage.input_tokens_details.cached_tokens` AND a per-turn breakdown
  (`cached_tokens_per_turn`) — real, wired, populated data. qwen-code's client reads a
  DIFFERENT (Anthropic-style) field name that vLLM never sends, so it always logs 0 regardless
  of the true value. The real per-turn cache data was never persisted anywhere (proxy dumps
  empty, no raw response capture) — **not retroactively recoverable** for already-run tasks.
- Fix for future runs (not yet built): have the offload proxy log
  `usage.input_tokens_details` per response (cheap, no qwen-code changes needed) to get genuine
  per-request cache visibility going forward.

## Native comparison: reused existing data, no fresh arm2 needed

`output/fr13_stateless_b4_16/nativemtp5apc_slb4` = native MTP-5 + APC (cache-on) + B=4, covering
the **exact same 16-task subset** (`subset_b4_sixteen.json`, all 16 instance_ids present, all with
`vllm_metrics_pre/post.txt` brackets). Ran the same canonical reducer:

```
accept_per_event = 3.511      (bracketed, matched subset+cache+B4)
prefill_frac = 0.197          (much LOWER than tail6 burn-off's 0.82-0.86 — see caveat below)
per_request_decode_tps = 9.795
derived_tps (wall, concurrency-summed) = 9.71
s_per_fwd_gpu / derived_tps_gpu / effective_concurrency = null
```

**Gap: this historic native capture has ZERO GPU-timer instrumentation** (`fr13_committer_gpu`,
`fr13_drafter_gpu`, `fr13_decode_forward_gpu` all absent from its metrics files — the capture
predates that instrumentation). So there is **no native fullstep_gpu / component breakdown** to
compare against tail6's 23.3-23.7 / 44.2-44.7. Only wall-basis numbers exist for native. A
matched fullstep_gpu comparison would need a FRESH native run with
`FR13_SFWD/DFWD/CFWD_GPU_TIMER=1` — a real "native-with-timers" capture, not literally the
burn-on/burn-off A/B arm2 (which is not needed — burn redundancy is already settled by the
red-team + isolated micro-bench).

**prefill_frac mismatch (0.197 native vs 0.82-0.86 tail6 burn-off) is itself a finding**: either
genuine workload difference (agentic nondeterminism — the live agent's own tool-call choices
differ run to run at temp 0.6) or a real regime difference between the two captures. Until this
is understood/matched, even the wall-basis numbers (9.71 native vs 44.2-44.7 tail6— NOT
comparable at face value, wildly different prefill mix) should not be read as a speed verdict.

## Accept regression check: CORRECTED (2026-07-22, user caught the error)

**The "4.317" comparison above was WRONG — 4.317 is PB-CONFOUNDED, not clean tail6.** Ladder text
confirms: "tail6-pb HYBRID (K=8 chain + fallback replay on accept>6 overflow... keeps accept
4.317)" — that's the piggyback investigation number, not the plain non-pb baseline. Comparing
burn-off (non-pb) against it was an apples-to-oranges error.

**The real canonical reference** is the "MEASURED DECOMPOSITION (2026-07-21)" entry in the ladder:
cache-OFF, `eff_conc=2.04` (nearly identical to this run's 2.03), matched denominators, same 16
tasks: native accept_per_event=3.415 (step 158ms, derived_tps_fullstep_gpu=27.9,
**measured_tps_fullstep_wall=27.82**, alignment_ratio≈1.0 — clean, unlike this run's 0.52); tail6
accept_per_event=4.953 (step 282ms, committer=88ms burn-ON, derived_tps=21.1).

Corrected comparison: this run's accept (4.726, 4.863; cache-ON, burn-OFF) vs that clean
cache-OFF baseline (4.953) is **roughly flat, -1.7% to -4.6%** — not a clear improvement or
regression. Other "clean" non-pb tail6 accept citations in the ladder span 4.9-5.7 (rg1: 4.9-5.7,
rg2c: 5.36) — real run-to-run variance of ~15%+ even under matched config, live agentic temp-0.6.
A single new run sitting a few percent below one historic point is not strong evidence either way
without more samples.

**Native's measured TPS (27.82) is real but cache-OFF.** Searched all existing native captures
with GPU-timer instrumentation (fr13_native_vs_tail6, fr13_commnative_gate,
fr13_native_tail6_decomp, fr13_commnative, fr13_commnative_ab) — every one has
`FR13_ENABLE_APC=0`. **No cache-ON native + GPU-timer capture exists anywhere in the archive.**
The project's own standing rule (ladder line 53-54): "Cross-pair comparisons stay
within-cache-config; the final bar compare = tree-cache-ON vs native-cache-ON." That pairing does
not exist yet — a genuine, confirmed gap, not an oversight. Closing it needs a fresh native run
with `FR13_ENABLE_APC=1` + `FR13_SFWD/DFWD/CFWD_GPU_TIMER=1` on the same 16-task subset (NOT a
burn-on/burn-off A/B — burn redundancy is separately settled; this is purely about getting
native's own timed numbers under the cache-ON regime this project ships).

## Open items
- alignment_ratio ~0.53 mismatch: unresolved, needs either fixing the fullstep formula's
  denominator-unit mismatch or picking one basis (wall vs compute) as canonical for cross-arm
  verdicts going forward.
- Native fullstep_gpu component data: does not exist historically; needs a fresh
  timers-enabled native capture if a component-level (drafter/verify/committer) comparison to
  native is ever required.
- prefill_frac mismatch between the reused native capture and the live tail6 burn-off run: not
  yet explained; blocks any face-value wall-basis TPS comparison between them.
- Cache proxy-logging fix: queued, not built (would give real per-request cache visibility for
  future runs).
