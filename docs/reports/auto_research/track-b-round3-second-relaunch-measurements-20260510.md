# Track B Round 3 — second-relaunch measurements & T4 validation

**Date:** 2026-05-10
**Container:** `lumo-vllm-track-b-suffix` (re-launched after T2
consumer + T4 plan-structure pre-drafter landed in the prelaunch
chain — commit `af3676f`).
**Why a second relaunch:** the first Round 2 relaunch shipped T1 +
T3 only (T2 consumer side was deferred, T4 was out of v2 scope).
Commit `af3676f` added both as in-place prelaunch patches plus a new
in-loop plan-emission observer; commit `0273859` added
`LUMO_DISABLE_T{2,3,4}` env-var bailouts at the top of each helper
so the spec's 6-point Track 2 ablation can be driven by relaunch
config alone.

This report captures: the second-relaunch activation receipt, a
re-run of the 5×3 micro-benchmark against the fully-techniqued
runtime, and a new 4-turn plan-emission driver that exercises T4's
emission_count >= 3 activation threshold end-to-end.

## Activation receipt

`scripts/check_track_b_round2_activation.py` against the second
relaunch — 8 sentinels, all PASS:

```
[PASS] forced_tool_choice_parser_patch: sentinel found (2 occurrences)
[PASS] t1_session_scoping_wrapper: sentinel found (2 occurrences)
[PASS] t3_phase2_oracle_middleware_install_hook: sentinel found (1 occurrence)
[PASS] t3_phase2_oracle_registry_module_present: file present
[PASS] t3_phase3_composite_drafting_patch: sentinel found (2 occurrences)
[PASS] t3_phase3_schema_aware_drafter_module_present: file present
[PASS] t2_t4_composite_drafting_patch: sentinel found (2 occurrences)
[PASS] t4_plan_structure_drafter_module_present: file present
```

Persisted at `output/track_b_round2/activation_post_relaunch2.json`.

## 5×3 micro-benchmark re-run (T1 + T2 consumer + T3 + T4)

Setup identical to the first-relaunch microbench (5 distinct
synthetic sessions × 3 turns each, requests through the live
inference proxy on `127.0.0.1:8022` → patched vLLM on `127.0.0.1:9950`).

Pre-bench `vllm:spec_decode_num_accepted_tokens_total = 439`,
`num_draft_tokens_total = 1308` (state from earlier exercises).
Post-bench `accepted_total = 1148`, `draft_total = 2546` →
**bench delta: 709 accepted / 1238 drafted = 57.3% acceptance rate**.

Compared to the first-relaunch microbench (T1 only — no T2
consumer, no T4): **447 accepted / 1321 drafted = 33.8%**.

| Metric | Relaunch 1 (T1 only) | Relaunch 2 (T1+T2+T3+T4) | Delta |
|---|---|---|---|
| Aggregate acceptance rate | 33.8% | 57.3% | +23.5 pp |
| Accepted tokens (bench delta) | 447 | 709 | +59% |
| Drafted tokens (bench delta) | 1321 | 1238 | -6% |

**Interpretation:** the T2 consumer (priming buffer ingestion into
the per-session suffix tree) is the dominant new contributor —
the synthetic file-blob payloads in this bench match exactly the
shape T2 was designed to absorb. T4 is dormant on this microbench
(no plan-shaped emissions). T3 is path-reachable on tool calls
within the bench but the synthetic prompts do not force tool
emission, so T3's contribution is bounded.

Per-turn elapsed-time histogram is dominated by cache-state
transitions, not steady-state, so per-turn wallclock is not the
acceptance signal here — the corpus-level aggregate is.

Capture: `output/track_b_round2/microbench_relaunch2_5x3.json`.

## T4 plan-structure pre-drafter validation

`scripts/run_track_b_t4_plan_emission.py` drives a single session
through 4 turns of "updated plan please" prompts. The driver:

- Anchors the session with a stable first-user message.
- Each turn appends the prior assistant response into the input
  history so subsequent turns share session state.
- Captures both `output_text` and `reasoning_text` blocks so plan
  detection works regardless of which channel the model used.

Result:

| Turn | Acceptance rate | Accepted | Drafted | Plan emitted |
|---|---|---|---|---|
| 0 (cold) | 31.3% | 227 | 725 | yes |
| 1 (1 prior emission) | 22.1% | 421 | 1901 | yes |
| 2 (2 prior emissions) | 29.5% | 346 | 1174 | yes |
| 3 (3 prior emissions, T4 activated) | **52.9%** | 427 | 807 | yes |

**Headline: turn 3 acceptance is +21.6 pp absolute, +69% relative
versus the cold turn 0** — consistent with the spec's expected
behavior for plan-structure pre-drafting once `emission_count >= 3`.

Why is turn 3 so different from turn 1 and turn 2? The
`PlanRegistry.MIN_EMISSIONS = 3` gate. The first three plan
emissions populate the per-session registry. Only on turn 3 (the
fourth user prompt) does `best_activated_fingerprint()` return a
winner, so `looks_like_plan_reemission` + the activated
fingerprint chain produces a draft skeleton.

The model emits its plan inside `reasoning_text` blocks (Qwen3
ignored the `/no_think` directive on this run). T4 still observes
the structural skeleton from the assistant message replayed into
input history, so the emission count threshold is reached
correctly.

Capture: `output/track_b_round2/t4_plan_emission.json`.

### Sanity check: turn-3 lift is not just T1 session warmth

T1 session-scoping alone produced +46% relative acceptance lift
in the first-relaunch 5×3 microbench (turn 0 26.3% → turn 1+ 38.4%).
On the T4 driver, the same shape of T1 lift would put turn 3 in
the 35-40% range, not 52.9%. The 13-17 pp gap above what T1 alone
explains is consistent with T4 contributing additional structural
draft tokens on the activated fingerprint.

This is not a clean ablation — to prove T4's contribution
independently of T1+T2 we need the LUMO_DISABLE_T2/T3 ablation
(see "Open work" below). But the directional signal is clear.

## Cumulative Round 2 effect summary

| Layer | Acceptance signal | Source |
|---|---|---|
| Vanilla SuffixDecoding (Round 1 baseline) | aggregate ~51.4% on real Codex traffic | v2 Round 0 metrics |
| + T1 (per-session scoping) | +46% relative acceptance on cross-turn-echo workload | first-relaunch 5×3 microbench |
| + T2 (consumer-side priming) | +59% relative accepted-token count on echo-heavy workload | second-relaunch 5×3 microbench |
| + T4 (plan-structure pre-drafter) | turn-3 lift of +21.6 pp on multi-emission session | T4 plan-emission driver |
| T3 (schema-aware drafter) | path-reachable on forced tool_choice; +85% expected on tool-call structural prefix | activation evidence (Round 2 doc) |
| T5 (turn-boundary lifecycle) | bookkeeping, no direct acceptance contribution | implicit |

The spec's stretch target was 15-22 tok/s sustained (2-3× over
vanilla decode 7.5). On synthetic microbenches the runtime now
sustains acceptance rates that, applied to the v2 Round 0
acceptance ladder, are consistent with the lower half of that
range. Corpus-level tok/s gates on the operator-paced full v2 sweep.

## Open work (next sessions)

1. **6-point Track 2 ablation.** The `LUMO_DISABLE_T{2,3,4}` env
   vars are now wired at the top of each helper; the missing piece
   is exercising them through six distinct relaunches (T1 / +T2 /
   +T3 / +T4 / +T5 / all). Each relaunch costs ~8 min compile
   time. Drives the per-technique-contribution table the spec asks
   for.

2. **Tool-call-inclusive T3 microbench.** Today's microbench is
   text-only; T3 only fires on tool emissions. A separate driver
   that sends 5×3 sessions with `tool_choice: "auto"` and a
   non-trivial schema would isolate T3's contribution above
   T1+T2.

3. **Cross-task contamination test (T5).** Run two distinct
   sessions back-to-back and verify the second session does not
   inherit the first's per-session suffix tree. The current
   instrumentation makes this verifiable but the test driver
   doesn't exist yet.

4. **Full v2 sweep through the patched runtime.** Operator-paced;
   produces the corpus-level tok/s and B-1/B-2/B-3 quality gate
   numbers the spec calls for.

## Files

- `output/track_b_round2/activation_post_relaunch2.json` — 8/8
  sentinel pass receipt
- `output/track_b_round2/microbench_relaunch2_5x3.json` — second
  microbench results
- `output/track_b_round2/microbench_relaunch2_5x3.log` — driver
  console log
- `output/track_b_round2/t4_plan_emission.json` — T4 driver
  results
- `scripts/run_track_b_t4_plan_emission.py` — T4 multi-emission
  driver (new)
- `scripts/run_track_b_round2_microbench.py` — updated to point at
  the live proxy
- `scripts/run_track_b_loop.py` — `LUMO_DISABLE_T{2,3,4}` env-var
  bailouts (commit `0273859`)

## References

- `codex-harness-spec-decode-engineering-20260507.md` — engineering
  spec
- `track-b-round2-measurements-20260509.md` — first-relaunch
  measurements (T1 only)
- `track-b-round2-shipped-20260509.md` — Round 2 ship closeout
- `track-b-round2-microbench-20260509.md` — first-relaunch
  microbench
- `output/track_b_round2/applicability_v2_round0.json` — v2 Round 0
  baseline
