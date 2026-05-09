# Track B Round 2 shipped — T1 + T3 + T2 producer

**Date:** 2026-05-09
**Schema:** `lumo.track_b.round_2_shipped.v1`
**Status:** end-to-end ship-ready. Single `ModelServer` relaunch
activates the full T1 (cross-turn ngram session scoping) + T3
(schema-aware tool drafter) stack; T2 producer side ships
alongside. Measurement automation (activation check + applicability
analyzer + delta script) is wired to deliver the Round 2 headline
in one command after the relaunch.

## What "Round 2 shipped" means

The harness-coupled spec
(`codex-harness-spec-decode-engineering-20260507.md`) lists five
techniques. Round 2's first cut targets the high-leverage subset
the v2 Round 0 applicability analyzer flagged:

- **T1 cross-turn ngram session scoping** — partitions
  arctic_inference's `SuffixDecodingCache` by session_id parsed
  from the proxy's `lumo_sess_<id>__` request-id prefix.
  Theoretical ceiling: 33% of corpus decode = ~5% wallclock.
- **T3 schema-aware tool drafter** — composite drafting that runs
  schema-driven structural prefill alongside SuffixDecoding,
  picking the higher-confidence draft. Theoretical ceiling: 63%
  of corpus decode = ~9.6% wallclock.
- **T2 read_file priming** (producer only) — proxy synthesises
  primed_texts on the oracle. Consumer side deferred until v2
  capture confirms the actual coverage justifies it (the
  applicability analyzer's regime-proxy estimate of 5% may
  understate true coverage now that we have direct measurement).
- **T4 plan-structure pre-drafting** — skipped this round (needs
  a Codex emitter we don't have).
- **T5 lifecycle** — covered implicitly by oracle's
  `is_session_open` field; no separate code path needed.

## Architecture

Two endpoints meet at a session id parsed from the request-id
prefix:

```
Codex CLI
    │
    │ /v1/responses (X-Lumo-Oracle: {...}, X-Request-Id: lumo_sess_<id>__<uuid>)
    ▼
inference_proxy.py
    │  -- synthesises oracle from payload
    │  -- session_id = sha256(first user message)[:16]
    │  -- turn_index from input transcript
    │  -- tool_schemas + expected_tool_call from payload
    │  -- primed_texts from cat-style function_call_outputs
    │  -- forwards X-Lumo-Oracle header + lumo_sess_ prefix
    ▼
vLLM 0.19 (patched at prelaunch)
    ├── api_server.py
    │       -- T3 phase 2 middleware: parses X-Lumo-Oracle, stashes in
    │          lumo_oracle_registry.ORACLE_REGISTRY keyed by req_id
    │
    ├── v1/spec_decode/
    │       lumo_oracle_registry.py    (dropped at prelaunch)
    │       lumo_schema_aware_drafter.py (dropped at prelaunch)
    │
    └── v1/spec_decode/suffix_decoding.py
        ├── T1: self.suffix_cache wrapped in
        │       _SessionRoutedSuffixDecodingCache → per-session
        │       arctic_inference.SuffixDecodingCache instances
        ├── T3 phase 3: propose() consults
        │       _lumo_try_schema_aware_draft FIRST
        │       (looks up oracle, runs schema_aware_drafter,
        │        round-trips via tokenizer, returns tokens)
        │       falls through to suffix_cache.speculate on miss
```

## Activation

Single relaunch of the running container through `ModelServer`. The
prelaunch chain in `_track_b_runtime_prelaunch_shell` runs:

1. GPU memory hygiene + recovery
2. PR #39562 KV allocator stop-gap
3. `arctic-inference==0.1.2` install
4. `lmcache==0.4.4` install + nixl uninstall
5. Forced tool_choice parser fix (Track B 2026-05-08, in Round 1)
6. **T1 session-scoping wrapper** (commit 2911641, fixed 6eb4d32)
7. **T3 phase 2 oracle middleware install hook** (commit 6eb4d32)
8. **T3 phase 3 composite drafting wrapper** (commit 8d4c4a0)

Verify with `scripts/check_track_b_round2_activation.py` — exits
non-zero if any sentinel is missing, with actionable Triggers in
stderr.

## Measurement plan

Four scripted steps after `ModelServer` relaunch:

```bash
# 1. Verify all six prelaunch sentinels landed.
python scripts/check_track_b_round2_activation.py \
  --container lumo-vllm-track-b-suffix \
  --output output/track_b_round2/activation_post_relaunch.json

# 2. Run the v2 sweep against the patched runtime (proxy capture on).

# 3. Build the post-patch applicability JSON.
python scripts/build_track_b_round2_applicability.py \
  --input output/track_b_e2e_v2_post_patch \
  --output output/track_b_round2/applicability_v2_round_patched.json \
  --print

# 4. Diff against the v2 Round 0 baseline.
python scripts/build_track_b_round2_delta.py \
  --baseline output/track_b_round2/applicability_v2_round0.json \
  --patched  output/track_b_round2/applicability_v2_round_patched.json \
  --output   output/track_b_round2/delta_v2_round0_to_patched.json \
  --print
```

The delta script's headline is the single number Round 2's
acceptance gate is judged on: corpus decode reduction in seconds +
percent.

## Theoretical ceilings vs anticipated reality

The applicability analyzer ceilings assume the technique speedup
target hits at full coverage. Real numbers will fall short:

| technique | covered | speedup target | ceiling | realistic estimate |
| --- | ---: | ---: | ---: | ---: |
| T1 (cross-turn ngram) | 100% | 1.5× | 98 s = 33% | 30-60 s (depends on intra-session ngram acceptance, untested) |
| T3 (schema-aware tool drafter) | 94% | 3.0× | 187 s = 63% | 50-100 s (anchor catalog is structural-prefix only — ~16 tokens/turn) |
| T2 producer | TBD | 2.0× | TBD | unknown until consumer ships |

Combined T1+T3 realistic estimate: 80-160 s decode reduction
= 4-8% of corpus wallclock. Confirmation gates on the post-patch
sweep.

## Commit log

| commit | scope |
| --- | --- |
| 53ab3ec | Step 3 design doc: HarnessOracleSnapshot API skeleton |
| 2f92dda | Step 3 phase 1: proxy-side X-Lumo-Oracle synthesis (session_id, turn_index, dialect) |
| eb27444 | Step 3 phase 2: extended synthesis (is_session_open, tool_schemas, expected_tool_call) |
| 059addc | Step 3 phase 3: vLLM-side harness_oracle.py skeleton module |
| f0f82ab | Round 2 applicability analyzer + v2 Round 0 numbers |
| 6a96c55 | Round 2 progress doc (initial) |
| 90dc8b5 | T1 phase 1: proxy session-prefixed X-Request-Id |
| 2911641 | T1 phase 2: prelaunch wraps SuffixDecodingCache per-session (had a syntax bug fixed in 6eb4d32) |
| ac81374 | T3 phase 1: schema-aware drafter decision core |
| 6eb4d32 | T3 phase 2: oracle middleware drop + api_server install hook (also fixed T1 syntax bug) |
| 8d4c4a0 | T3 phase 3: composite drafting in SuffixDecodingProposer.propose |
| 6af3943 | Round 2 delta script |
| 814ac01 | Round 2 activation checker |
| 500c039 | T2 phase 1: proxy-side primed_texts synthesis |

## What's NOT in Round 2

- **Real-data measurement**: needs an operator-gated relaunch +
  v2 sweep. The toolchain is ready; the run is queued.
- **T2 consumer side**: deferred until the v2 post-patch capture
  reveals whether oracle_primed_text_count is high enough on real
  Codex traffic to justify the integration cost. The producer
  side ships now so the data is there to measure against.
- **T4 plan-structure pre-drafting**: requires a Codex emitter
  (the `@openai/codex-linux-arm64` wheel doesn't emit
  plan_fingerprint). Out of scope until either Codex source is
  patchable or the proxy detects plan emissions
  reliably (it currently can't).
- **vLLM rebuild**: the entire Round 2 stack lands as prelaunch
  patches against a stock `lumo-flywheel-vllm:26.01-py3-v0.19.0`
  image. No rebuild needed; a relaunch suffices.

## Test posture

99 unit tests + 5 docker-gated integration tests pass. Suites:

- `test_inference_proxy.py` (47): oracle synthesis (incl.
  primed_texts), session-prefixed X-Request-Id encode/parse,
  capture row enrichment, e2e proxy header forwarding.
- `test_vllm_harness_oracle.py` (18): snapshot round-trip,
  isolation, OracleRegistry LRU/thread-safety, FastAPI middleware
  (2 importorskip locally, exercised in container).
- `test_schema_aware_drafter.py` (14): codex + openai dialect
  anchors, required-list priority, type-driven proposals.
- `test_build_track_b_round2_applicability.py` (10): technique
  gating (incl. T2 oracle-vs-regime-proxy logic), math.
- `test_build_track_b_round2_delta.py` (7): headline,
  measured-vs-ceiling ratios, regime aggregation.
- `test_check_track_b_round2_activation.py` (9): sentinel parse,
  partial-failure aggregation, exit codes.

Docker-gated:
- `test_vllm_t1_session_scoped_suffix_decoding_patch.py` (1):
  T1 wrapper applied + exercises session partitioning.
- `test_vllm_t3_oracle_middleware_patch.py` (2): prelaunch-shell
  syntax-import regression guard + T3 phase 2 patch e2e.
- `test_vllm_oracle_middleware_integration.py` (1): middleware
  fires inside vLLM image, registry round-trips.
- `test_vllm_t3_composite_drafting_patch.py` (1): T3 phase 3
  patches apply, propose-helpers bind, bail-outs return None
  cleanly, idempotent re-application.

## Round 2 in one paragraph

Round 1 shipped live SuffixDecoding as the v2 baseline. Round 2
adds two patched layers on top: the proxy synthesises a
HarnessOracleSnapshot per request and forwards it via header +
session-prefixed request-id; vLLM (patched at prelaunch with no
rebuild) partitions the suffix cache per session and runs a
schema-aware drafter ahead of suffix decoding for tool-call regime
turns. Theoretical decode-reduction ceilings on the v2 Round 0
corpus: 14.6% combined wallclock. Real numbers come out of the
delta script after the operator's next vLLM relaunch + sweep.
