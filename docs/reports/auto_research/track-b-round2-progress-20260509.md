# Track B Round 2 — progress as of 2026-05-09

**Status:** Step 3 (harness oracle API + skeleton) shipped end-to-end.
Steps 4-9 (per-technique drafter implementations) are gated on
vLLM source rebuilds (30-60 min each) and properly out of single-
session scope.

## Shipped this session

| commit | what |
| --- | --- |
| 53ab3ec | Step 3 design doc: HarnessOracleSnapshot API skeleton |
| 2f92dda | Step 3 phase 1: proxy-side X-Lumo-Oracle synthesis (session_id, turn_index, dialect) |
| eb27444 | Step 3 phase 2: extended synthesis (is_session_open, tool_schemas, expected_tool_call) |
| 059addc | Step 3 phase 3: vLLM-side harness_oracle.py skeleton module |
| f0f82ab | Round 2 applicability analyzer + v2 Round 0 report |

End-to-end verified against the live Round 1 baseline
(`lumo-vllm-track-b-suffix`): a sidecar proxy at port 8033 emits
`X-Lumo-Oracle` and records oracle fields in JSONL captures. The
vLLM-side skeleton round-trips against the proxy's encoder.

## v2 Round 0 applicability — corpus-wide ceilings

84 completed turns across 13 tasks × 4 runs:

| technique | fires on | covers decode | speedup target | reduction ceiling |
| --- | ---: | ---: | ---: | ---: |
| T3 schema-aware tool drafter | 94% | 95% | 3.0× | 187 s = 63% of corpus decode |
| T1 cross-turn ngram | 100% | 100% | 1.5× | 98 s = 33% of corpus decode |
| T2 read_file priming | 6% | 5% | 2.0× | 7 s = 2.5% of corpus decode |
| T4 plan-structure | n/a | — | — | not detectable from current capture |

**Headline:** T3 is the highest-leverage Round 2 technique by a 2×
margin over T1. T2 is correctly de-prioritised — only ~5% of decode
time is reachable. T1 + T3 compound (different time slots) so the
combined ceiling is ~285 s decode reduction = ~14.6% of corpus
wallclock.

## Blocking constraints for Steps 4-9

Each technique requires:
1. A new vLLM proposer module under `vllm/v1/spec_decode/`
2. A drafter-coordinator extension to dispatch to it
3. A vLLM rebuild (30-60 min on GB10 ARM + CUDA 13)
4. Per-technique microbenchmark + acceptance-rate measurement
5. Integration test against the running baseline

Steps 1-2 are pure addition (no behavioural delta until wired);
they could be staged in our repo and dropped into the container via
the prelaunch hook *if* the hook can land arbitrary `.py` files
into `vllm/v1/spec_decode/`. The current prelaunch hook only
patches existing files — extending it to drop new modules is a
minor change but introduces a new failure mode (stale module
copies between runs). Defer until a consumer is actually ready.

Step 3-5 (rebuild + measurement) are operator-paced — each needs
a dedicated baseline relaunch + a Track B sweep, ~3-4 hours of
runtime per technique iteration.

## What an operator can do next

1. **Land Technique 1 (cross-turn ngram session scoping)** as
   commit #1 of Round 2: it's the smallest delta (extends
   existing SuffixDecoding's suffix tree to be keyed by
   session_id) and the proxy already provides session_id. Expected
   measurement: T1 ceiling is 33% of corpus decode = ~10% of
   wallclock; the actual figure depends on cross-turn n-gram
   acceptance rates we don't have data for yet.
2. **Schedule Technique 3 (schema-aware tool drafter)** as the
   biggest-payoff next step. Needs XGrammar-2's `traverse_draft_tree`
   primitive (already in our XGrammar 0.2.0 build) plus a new
   proposer that consumes `tool_schemas` + `expected_tool_call`
   from the oracle.
3. **Skip T2/T4** in the first Round 2 cut. T2 has too little
   coverage to justify the integration cost; T4 lacks an emitter.

## What did NOT ship

- **LMCache wiring** (Round 2 Step 1 in the original spec) —
  upstream incompatibility with hybrid-attention models documented
  in `track-b-lmcache-integration-staged-20260509.md`. BLOCKED
  multi-week per spec.
- **Codex-side oracle emitter** for `primed_texts` and
  `plan_fingerprint` — gated on the Codex Rust source patch
  (the binary on this host is the prebuilt `@openai/codex-linux-arm64`
  wheel). Defer to when the proxy-side ceiling for T2/T4 looks
  like it's worth chasing.
- **vLLM-side consumer** of the X-Lumo-Oracle header — gated on
  a vLLM rebuild. The skeleton module is ready; the drafter
  coordinator extension is not.

## Test posture

- 56 tests pass across:
  - `test_inference_proxy.py` (38: 29 prior + 9 new oracle synthesis)
  - `test_vllm_harness_oracle.py` (10: round-trip, isolation, defaults)
  - `test_build_track_b_round2_applicability.py` (8: technique gating, math)
- Round 1 baseline (`lumo-vllm-track-b-suffix`) is up and serving
  traffic on 127.0.0.1:9950. No regressions from this session's
  changes (proxy work is sidecar; vLLM-side module unloaded).

## Files added/changed

```
docs/reports/auto_research/
  track-b-harness-oracle-api-skeleton-20260509.md      (new, 321 lines)
  track-b-round2-progress-20260509.md                  (this doc)
src/lumo_flywheel_serving/
  inference_proxy.py                                    (+267 lines)
  vllm_harness_oracle.py                                (new, 152 lines)
scripts/
  build_track_b_round2_applicability.py                 (new, 308 lines)
tests/
  test_inference_proxy.py                               (+255 lines)
  test_vllm_harness_oracle.py                           (new, 158 lines)
  test_build_track_b_round2_applicability.py            (new, 121 lines)
```
