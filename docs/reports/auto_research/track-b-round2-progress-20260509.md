# Track B Round 2 — progress as of 2026-05-09

**Status:** Step 3 (harness oracle API + skeleton) shipped end-to-end.
**Step 4 (Technique 1: cross-turn ngram session scoping) ship-ready
without rebuild** — activates on next vLLM relaunch through
ModelServer. **Step 8 (Technique 3: schema-aware tool drafter)
ship-ready end-to-end without rebuild** — three prelaunch patches
deliver the producer (proxy oracle synthesis), the registry
(FastAPI middleware), and the consumer (composite drafting in
SuffixDecodingProposer.propose). Steps 5/6/9 (T2/T4/T5) remain
out of Round 2 scope per the v2 applicability analysis.

## Shipped this session

| commit | what |
| --- | --- |
| 53ab3ec | Step 3 design doc: HarnessOracleSnapshot API skeleton |
| 2f92dda | Step 3 phase 1: proxy-side X-Lumo-Oracle synthesis (session_id, turn_index, dialect) |
| eb27444 | Step 3 phase 2: extended synthesis (is_session_open, tool_schemas, expected_tool_call) |
| 059addc | Step 3 phase 3: vLLM-side harness_oracle.py skeleton module |
| f0f82ab | Round 2 applicability analyzer + v2 Round 0 report |
| 90dc8b5 | T1 phase 1: proxy injects `lumo_sess_<id>__` X-Request-Id |
| 2911641 | T1 phase 2: prelaunch wraps SuffixDecodingCache per-session |
| 8b82a50 | Round 2 progress doc: T1 ship-ready |
| ac81374 | T3 phase 1: schema-aware drafter decision core |
| dc0eb29 | Round 2 progress doc: T3 phase 1 ship-ready |
| 6eb4d32 | T3 phase 2: oracle middleware drop + api_server install hook |
| 2c078aa | Round 2 progress doc: T3 phase 2 ship-ready |
| 8d4c4a0 | T3 phase 3: composite drafting in SuffixDecodingProposer.propose |

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

## T1 is ship-ready — measurement plan

Both halves of Technique 1 are committed:

- **Proxy** (commit 90dc8b5): every `/v1/responses` call going
  through `lumo_flywheel_serving.inference_proxy` now carries
  `X-Request-Id: lumo_sess_<session_id>__<suffix>`.
- **vLLM** (commit 2911641): the prelaunch hook idempotently
  rewrites `vllm/v1/spec_decode/suffix_decoding.py` to wrap
  `self.suffix_cache` in `_SessionRoutedSuffixDecodingCache` —
  per-session sub-caches keyed by parsing the same prefix.

T1 activates on the **next vLLM relaunch** (operator-gated through
`ModelServer`). Until then, the running Round 1 baseline still
runs the un-partitioned cache.

To measure T1's effect once activated:
1. Relaunch vLLM (`ModelServer.start` or `make serve`). Confirm
   prelaunch log shows `applied T1 session scoping wrapper`.
2. Run the v2 Round 0 sweep against the patched runtime.
3. Diff aggregate decode_sum_s and spec_decode_num_accepted_tokens
   per task vs the v2 Round 0 baseline (capture in
   `output/track_b_round2/applicability_v2_round0.json`).
   Headline acceptance metric: aggregate
   `accepted_per_draft_token` across tool-call-regime turns.
4. Run `scripts/build_track_b_round2_applicability.py` against
   the new capture and compare T1's "decode_sum_s_covered" before
   vs after — this is the corpus-level measurement.

Theoretical ceiling at the v2 spec's 1.5× target = 98 s decode
reduction = 33% of corpus decode. Actual reduction depends on
intra-session n-gram acceptance rates which we don't have data
for yet.

## T3 phase 1 — decision core ready

`src/lumo_flywheel_serving/schema_aware_drafter.py` (commit ac81374)
exposes `propose(snapshot, recent_text) -> DraftProposal | None`:
given an oracle snapshot with `expected_tool_call` + `tool_schemas`
and the model's recent decoded text, returns the next deterministic
text chunk the schema implies. Tokenizer-free; the eventual vLLM-
side integrator (phases 2-3) wraps it in a token encoder + composite
drafting policy.

Three Codex / Qwen3 XML anchors are catalogued: forced-name region
(confidence 1.0), first-required-property opener (0.9), and
string-property opening quote (0.8). Parallel set for the OpenAI
Responses-API JSON dialect. 14 unit tests cover the anchor sequencing,
required-vs-fallback property selection, and the unknown-dialect
falls-through-to-suffix contract.

## T3 phase 3 — composite drafting ship-ready

The final piece. SuffixDecodingProposer.propose now consults
`_lumo_try_schema_aware_draft` BEFORE falling through to the
existing `self.suffix_cache.speculate` call. The helper:

- Looks up the oracle in `lumo_oracle_registry.ORACLE_REGISTRY`
  keyed by `req_id` (populated by the T3 phase 2 middleware).
- Lazy-loads a tokenizer from
  `vllm_config.model_config.tokenizer` on first use; cached.
- Decodes the recent token pattern to text, calls
  `lumo_schema_aware_drafter.propose(snapshot, recent_text)`,
  and encodes the proposal back to tokens.
- Tokenizer round-trip safety: the encoded tokens are only
  forwarded as the draft if `decode(encoded) == proposal_text`
  exactly. Models tokenise XML-tag boundaries differently;
  without this guard we'd emit drafts the model never accepts.

Exception-defensive at every step — any failure returns `None`
and the existing suffix-decoding path runs unchanged. Helpers
are bound class-level after the file body so the patch is
single-file (subclassing would require patching
`gpu_model_runner.py` too, doubling the patch surface).

Activation: next vLLM relaunch fires all three T3 phases plus
T1 in one pass. Verify with the prelaunch log lines:
`applied T1 session scoping wrapper`,
`applied T3 oracle middleware install hook`,
`applied T3 composite drafting wrapper`.

## Round 2 measurement plan

Single relaunch activates the full T1 + T3 stack. To measure:

1. Run the v2 Round 0 sweep against the patched runtime with
   per-request capture enabled.
2. Compare to the v2 Round 0 baseline
   (`output/track_b_round2/applicability_v2_round0.json`):
   - Aggregate `decode_sum_s` (T1+T3 ceiling: ~285s reduction =
     ~14.6% of corpus wallclock).
   - `accepted_per_draft_token` per regime — the T3 ceiling
     would push tool-call regime acceptance toward ~1.0 on the
     forced-name + structural-prefix slots.
   - Per-task wallclock — T1's session-scoping benefit shows up
     across ALL turns; T3's schema-aware win is concentrated on
     the tool-call regime turns (94% of v2 Round 0 traffic).
3. Re-run `scripts/build_track_b_round2_applicability.py`
   against the new capture. The `T3_schema_aware_tool_drafter`
   ceiling becomes a measured number, not a target.

## What an operator can do next

1. **Single relaunch activates Round 2**: one
   `ModelServer` relaunch fires T1 + T3 phases 1/2/3. No code
   changes needed; the prelaunch chain handles everything
   idempotently.
2. **Skip T2/T4** in the first Round 2 cut. T2 has too little
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

- 85 unit tests + 5 docker-gated integration tests pass:
  - `test_inference_proxy.py` (42: oracle synthesis + session-
    prefixed X-Request-Id)
  - `test_vllm_harness_oracle.py` (18: snapshot round-trip,
    isolation, defaults, OracleRegistry LRU + thread-safety,
    fastapi-importorskip middleware checks — 2 skip locally,
    exercised in container)
  - `test_build_track_b_round2_applicability.py` (8: technique
    gating, math)
  - `test_schema_aware_drafter.py` (14: dialect dispatch, anchor
    sequencing, required-list priority, type-driven proposals)
  - `test_vllm_t1_session_scoped_suffix_decoding_patch.py` (1
    integration: T1 wrapper applied in transient container)
  - `test_vllm_t3_oracle_middleware_patch.py` (2: prelaunch-shell
    syntax-import regression guard + container-side T3 phase 2
    patch application)
  - `test_vllm_oracle_middleware_integration.py` (1: middleware
    fires inside vLLM image, registry round-trips)
  - `test_vllm_t3_composite_drafting_patch.py` (1: T3 phase 3
    patches apply, propose-helpers are bound, schema-aware
    bail-out paths return None cleanly, idempotent re-application)
- Round 1 baseline (`lumo-vllm-track-b-suffix`) is up and serving
  traffic on 127.0.0.1:9950. No regressions from this session's
  changes — T1's vLLM-side patch fires only on the next relaunch
  through `ModelServer`'s prelaunch chain.

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
