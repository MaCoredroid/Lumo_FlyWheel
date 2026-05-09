# Track B Round 2 — session summary 2026-05-09

**Status:** Round 2 fully shipped and activated live in the
`lumo-vllm-track-b-suffix` container. T1 (cross-turn ngram session
scoping) demonstrably working in production, T2 producer side
firing, T3 (schema-aware tool drafter) wired through propose() and
ready to be exercised by forced-tool-choice traffic. Next step is
the corpus-level v2 sweep against the patched runtime.

## What this session shipped

19 commits, ~3000 lines of code + tests + docs, no vLLM rebuilds.
Round 2 lands as prelaunch-applied patches against the stock
`lumo-flywheel-vllm:26.01-py3-v0.19.0` image.

### Step 3 — harness oracle API skeleton (Round 2 prerequisite)

| commit | scope |
| --- | --- |
| 53ab3ec | Step 3 design doc: `HarnessOracleSnapshot` API skeleton |
| 2f92dda | Step 3 phase 1: proxy-side `X-Lumo-Oracle` synthesis (session_id, turn_index, dialect) |
| eb27444 | Step 3 phase 2: extended synthesis (is_session_open, tool_schemas, expected_tool_call) |
| 059addc | Step 3 phase 3: vLLM-side `vllm_harness_oracle.py` skeleton module |
| f0f82ab | Round 2 applicability analyzer + v2 Round 0 numbers |
| 6a96c55 | Round 2 session-state report (initial) |

The proxy synthesises the oracle from the inbound `/v1/responses`
payload (no Codex source change), forwards via header. v2 Round 0
applicability numbers reveal **T3 covers 95% of decode time and is
the highest-leverage technique by 2× over T1**, T2 is only 5%, T4
needs an emitter we don't have.

### Step 4 — Technique 1 (cross-turn ngram session scoping)

| commit | scope |
| --- | --- |
| 90dc8b5 | T1 phase 1: proxy injects `lumo_sess_<id>__<uuid>` X-Request-Id |
| 2911641 | T1 phase 2: prelaunch wraps `SuffixDecodingCache` per-session |
| 8b82a50 | Round 2 progress doc: T1 ship-ready |

Proxy injects a session-prefixed `X-Request-Id`. vLLM's
`_base_request_id` already promotes that to the engine req_id,
so `SuffixDecodingProposer` sees the prefix and a prelaunch-
patched `_SessionRoutedSuffixDecodingCache` wrapper routes to
per-session sub-caches. Non-prefixed traffic falls through to
the default bucket (bit-for-bit backward compat).

### Step 8 — Technique 3 (schema-aware tool drafter)

| commit | scope |
| --- | --- |
| ac81374 | T3 phase 1: schema-aware drafter decision core |
| dc0eb29 | Round 2 progress doc: T3 phase 1 ship-ready |
| 6eb4d32 | T3 phase 2: oracle middleware drop + api_server install hook (also fixed a T1 syntax bug) |
| 2c078aa | Round 2 progress doc: T3 phase 2 ship-ready |
| 8d4c4a0 | T3 phase 3: composite drafting in `SuffixDecodingProposer.propose` |
| 3f4f354 | Round 2 progress doc: T3 fully ship-ready |

Three coordinated prelaunch patches turn the schema-aware drafter
from a standalone module into the actual draft-source for tool-
call regime turns:

1. **Decision core** (text → `DraftProposal`). Pure Python,
   tokenizer-free, dialect-aware (codex Qwen3 XML +
   Responses-API JSON). Anchors at confidence 1.0/0.9/0.8.
2. **Oracle middleware**. FastAPI middleware patched into
   `build_app` parses `X-Lumo-Oracle` and stashes per-request
   snapshots in a thread-safe LRU registry keyed by
   `X-Request-Id`.
3. **Composite drafting**. `SuffixDecodingProposer.propose`
   consults `_lumo_try_schema_aware_draft` first, falls through
   to suffix-decoding speculation on miss. Tokenizer round-trip
   safety guards against drafts the model never accepts.

### Step 6 — Technique 2 (read_file priming, producer only)

| commit | scope |
| --- | --- |
| 500c039 | T2 phase 1: proxy-side primed_texts synthesis |

Proxy detects `cat`-style shell calls in prior turns, pairs them
with their function_call_outputs, emits primed_texts on the
oracle. Consumer side deferred until v2 capture confirms coverage.

### Round 2 measurement automation

| commit | scope |
| --- | --- |
| 6af3943 | Round 2 delta script — baseline vs patched applicability |
| 814ac01 | Round 2 activation checker — verify prelaunch patches landed |
| fb79a86 | Round 2 progress doc: operator runbook + measurement targets |
| 1d3b2ea | Track B Round 2 closeout report |
| 9176a3b | Spec doc: Round 2 ship-ready status update |

Three operator-facing scripts:

- `check_track_b_round2_activation.py` — runs against the live
  container, asserts every sentinel landed.
- `build_track_b_round2_applicability.py` — already shipped in
  Step 3; re-run after the v2 sweep produces the patched-runtime
  applicability JSON.
- `build_track_b_round2_delta.py` — pairs baseline + patched
  applicability JSONs, emits the corpus-decode-reduction headline.

### Live activation

| commit | scope |
| --- | --- |
| 5747dca | Round 2 live activation evidence |

End-of-session relaunch executed:

1. Stopped the prior baseline (`docker stop`).
2. Ran host-memory recovery via `LUMO_SUDO_PASSWORD`
   (sync + drop_caches + swap cycle), freed 117 GiB.
3. Launched fresh container via `/tmp/launch_suffix_vllm.sh`
   which pulls the patched prelaunch shell from
   `_track_b_runtime_prelaunch_shell()`.
4. All 8 prelaunch sentinels fired in order — no errors.
5. Engine ready at 19:40:42Z (~8 min cold start).
6. Activation checker — all 6 PASS.
7. Smoke test through sidecar proxy: 2-turn same-session
   exchange showed **turn 1 acceptance jumped from 13% to 83%**
   (4/30 → 25/30), decode time **32.7s → 0.83s = 39× faster**.
   This is the cross-turn-ngram benefit T1 was designed to
   deliver.

## Test posture

- **99 unit tests + 5 docker-gated integration tests pass** across
  the new + touched suites:
  - `test_inference_proxy.py` (47): oracle synthesis + session-
    prefixed X-Request-Id + primed_texts heuristic
  - `test_vllm_harness_oracle.py` (18): registry, middleware
  - `test_schema_aware_drafter.py` (14): anchor catalogue
  - `test_build_track_b_round2_applicability.py` (10): T2 oracle-
    vs-regime gating
  - `test_build_track_b_round2_delta.py` (7): headline math
  - `test_check_track_b_round2_activation.py` (9): exit codes
  - `test_vllm_t1_session_scoped_suffix_decoding_patch.py` (1):
    in-container T1 wrapper exercise
  - `test_vllm_t3_oracle_middleware_patch.py` (2): syntax
    regression guard + T3 phase 2 e2e
  - `test_vllm_oracle_middleware_integration.py` (1): middleware
    fires inside vLLM image
  - `test_vllm_t3_composite_drafting_patch.py` (1): T3 phase 3
    propose() helpers + bail-outs

- The `test_prelaunch_shell_imports_cleanly` test catches the
  class of failure that bit the original T1 commit — embedded
  triple-quotes prematurely closing the outer Python r-string.

## Ground truth from the live smoke

The 2-turn smoke test produced these capture rows (sidecar proxy
→ patched vLLM, both turns share same first-user-message anchor):

| field | turn 0 | turn 1 |
| --- | ---: | ---: |
| oracle_session_id | `sess_a6abacd5af12741d` | same |
| oracle_turn_index | 0 | 1 |
| oracle_is_session_open | True | False |
| oracle_dialect | codex | codex |
| oracle_tool_schema_count | 2 | 2 |
| oracle_primed_text_count | 0 | 1 |
| spec_decode_num_accepted_tokens | 4 | 25 |
| spec_decode_num_draft_tokens | 30 | 30 |
| accepted_per_draft_token | 0.133 | 0.833 |
| decode_sum_s | 32.74 | 0.83 |

39× decode speedup is contrived (same prompt, same file content).
Real Codex traffic won't see 39×; the realistic combined T1+T3
estimate is 4-8% wallclock reduction on the v2 Round 0 corpus.
What the smoke proves: the partitioning works as designed and
the per-session tree carries response tokens across turns.

## What's left

1. **Run the full v2 sweep** against the patched runtime —
   produces the corpus-level decode-reduction number via
   `build_track_b_round2_applicability.py` →
   `build_track_b_round2_delta.py`. Operator-paced (~3-4 hr
   sweep + analysis).
2. **Forced-tool-choice exercise** to validate T3's schema-aware
   path on real traffic — the auto-tool-choice smoke didn't
   exercise it (no `expected_tool_call` was set).
3. **T2 consumer side** — gated on whether the v2 capture's
   `oracle_primed_text_count` is high enough to justify the
   integration cost. Producer fired correctly on the 2-turn
   smoke, so the data will be there.
4. **T4 plan-structure** — needs a Codex emitter; out of Round 2
   scope per the v2 applicability analysis.

## Files added/changed (delta vs start of session)

```
docs/reports/auto_research/
  track-b-harness-oracle-api-skeleton-20260509.md          (new)
  track-b-round2-progress-20260509.md                       (new)
  track-b-round2-shipped-20260509.md                        (new)
  track-b-round2-activation-evidence-20260509.md            (new)
  track-b-round2-session-summary-20260509.md                (this doc)
  codex-harness-spec-decode-engineering-20260507.md         (status update)

src/lumo_flywheel_serving/
  inference_proxy.py                                        (+~400 lines: oracle, primed_texts, session-id)
  vllm_harness_oracle.py                                    (new, ~270 lines)
  schema_aware_drafter.py                                   (new, ~210 lines)

scripts/
  run_track_b_loop.py                                       (+~280 lines: T1+T3 prelaunch patches)
  build_track_b_round2_applicability.py                     (new, ~310 lines)
  build_track_b_round2_delta.py                             (new, ~210 lines)
  check_track_b_round2_activation.py                        (new, ~210 lines)

tests/
  test_inference_proxy.py                                   (+~390 lines)
  test_vllm_harness_oracle.py                               (new, ~290 lines)
  test_schema_aware_drafter.py                              (new, ~170 lines)
  test_build_track_b_round2_applicability.py                (new, ~150 lines)
  test_build_track_b_round2_delta.py                        (new, ~140 lines)
  test_check_track_b_round2_activation.py                   (new, ~140 lines)
  test_vllm_oracle_middleware_integration.py                (new, ~140 lines)
  test_vllm_t3_oracle_middleware_patch.py                   (new, ~170 lines)
  test_vllm_t3_composite_drafting_patch.py                  (new, ~230 lines)
```

## Single-paragraph framing

Round 1 shipped live SuffixDecoding as the v2 baseline. Round 2
adds two patched layers on top: the proxy synthesises a
HarnessOracleSnapshot per request and forwards it via header +
session-prefixed request-id; vLLM (patched at prelaunch with no
rebuild) partitions the suffix cache per session and runs a
schema-aware drafter ahead of suffix decoding for tool-call regime
turns. The full stack is **live in the running container**, the
session-scoped suffix tree is demonstrably partitioning correctly
(end-of-session smoke showed 39× decode speedup on a same-session
turn-1 continuation), and the four-script measurement toolchain
produces the corpus-level number once an operator runs the v2
sweep.
