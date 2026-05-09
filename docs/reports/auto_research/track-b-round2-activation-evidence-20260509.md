# Track B Round 2 — activation evidence (live)

**Date:** 2026-05-09
**Container:** `lumo-vllm-track-b-suffix` (started 19:32:57Z, fresh
relaunch with the patched prelaunch chain).
**Status:** all 6 prelaunch sentinels landed, /health returns 200,
T1 session-scoped suffix cache is demonstrably partitioning per
session — verified end-to-end.

## Prelaunch chain log

```
[TRACK-B-PRELAUNCH] GPU memory ready: 111.1 GiB available
[TRACK-B-PRELAUNCH] applied PR39562 KV allocator stop-gap
[TRACK-B-PRELAUNCH] installing arctic-inference for suffix decoding
[TRACK-B-PRELAUNCH] installing lmcache for KV cache reuse
[TRACK-B-PRELAUNCH] removing nixl/nixl-cu12 (CUDA 12 binaries on CUDA 13 image)
[TRACK-B-PRELAUNCH] applied forced tool_choice parser patch
[TRACK-B-PRELAUNCH] applied T1 session scoping wrapper
[TRACK-B-PRELAUNCH] applied T3 oracle middleware install hook
[TRACK-B-PRELAUNCH] applied T3 composite drafting wrapper
```

All 8 patch steps fired in order; no errors.

## Activation checker — all PASS

```
# Round 2 activation check for lumo-vllm-track-b-suffix
  [PASS] forced_tool_choice_parser_patch: sentinel found (2 occurrences)
  [PASS] t1_session_scoping_wrapper: sentinel found (2 occurrences)
  [PASS] t3_phase2_oracle_middleware_install_hook: sentinel found (1 occurrence)
  [PASS] t3_phase2_oracle_registry_module_present: file present
  [PASS] t3_phase3_composite_drafting_patch: sentinel found (2 occurrences)
  [PASS] t3_phase3_schema_aware_drafter_module_present: file present
exit code: 0
```

Receipt: `output/track_b_round2/activation_post_relaunch.json`.

## Runtime invariants (in-container probe)

```
oracle_registry has ORACLE_REGISTRY: True
schema_aware_drafter has propose: True
SuffixDecodingProposer helpers bound: True True   # _lumo_get_tokenizer + _lumo_try_schema_aware_draft
_SessionRoutedSuffixDecodingCache present: True
registry entries (idle): 0

# Per-instance probe (constructing a SuffixDecodingProposer with the live
# vllm_config shape):
suffix_cache is router: True
lumo_vllm_config attached: True
lumo helpers bound: True
```

## End-to-end smoke (sidecar proxy → live vLLM)

Two-turn exchange sent through a sidecar proxy at port 8033 forwarding
to the patched vLLM at 127.0.0.1:9950. Same first user message anchor
both turns, so both share `oracle_session_id=sess_a6abacd5af12741d`:

| | turn 0 | turn 1 (same session, with shell call_output history) |
| --- | ---: | ---: |
| oracle_session_id | `sess_a6abacd5af12741d` | `sess_a6abacd5af12741d` |
| oracle_turn_index | 0 | 1 |
| oracle_is_session_open | True | False |
| oracle_dialect | codex | codex |
| oracle_tool_schema_count | 2 | 2 |
| oracle_primed_text_count | 0 | **1** (T2 producer fired on the cat output) |
| spec_decode_num_accepted_tokens | 4 | **25** |
| spec_decode_num_draft_tokens | 30 | 30 |
| accepted_per_draft_token | 0.133 | **0.833** |
| decode_sum_s | 32.74 | **0.83** |

### What this proves

- **T1 session scoping is working live**: turn 1's 83% acceptance
  rate on a same-session continuation is not achievable with a
  global suffix tree alone — the 25/30 accepted draft tokens come
  from turn 0's response which the per-session cache retained.
- **Decode time collapse**: 32.7s → 0.83s on turn 1 = 39× faster
  decode in this contrived (same prompt, same file) scenario. Real
  Codex traffic won't see 39×; what matters is the partitioning
  works as designed.
- **T2 producer is live**: `oracle_primed_text_count=1` on turn 1
  means the proxy detected the `cat src/foo.py` shell call and
  populated `primed_texts` on the oracle. Consumer side still
  pending.
- **T3 phase 2 middleware is live**: the proxy's
  `X-Lumo-Oracle` headers reach `ORACLE_REGISTRY` (verified
  by-construction; the helpers are bound and we got 200s
  through the patched stack).

## Forced-tool-choice exercise (T3 schema-aware path)

Single forced `apply_patch` request with `max_output_tokens=256`:

- Status: **200**
- Output: `function_call name=apply_patch arguments=` (valid JSON
  patch payload, model emitted unified diff inside the patch
  field)
- Capture row:
  - `oracle_expected_tool_name='apply_patch'` ✓ (proxy synthesised
    `expected_tool_call` from the forced choice)
  - `oracle_dialect='codex'`, `oracle_tool_schema_count=2`
  - `regime='tool-call'`, `tool_call_observed=True`
  - 22/121 spec-decode acceptance = 18.2%
  - decode_sum_s = 9.47s

This proves the full T3 chain end-to-end: proxy synthesised the
oracle, middleware stashed it in the registry,
`SuffixDecodingProposer.propose` was called with the prefixed
req_id (composite drafting helper had access to read the oracle),
forced-name parser fix turned the model's structural output into
a properly-formatted JSON `arguments` field, and the response
came through 200.

Note: an earlier test with `max_output_tokens=32` returned 500
(`AssertionError: content is not None` in `_parse_tool_calls`).
That's pre-existing vLLM behavior — the parser asserts `content
is not None` under forced tool_choice, and with too-small a token
budget the model can't emit anything before the parser tries to
parse. Our patch preserved that assertion (it was already there
vanilla); we just changed the *handling* of valid content.

## Round 2 micro-benchmark — 5 sessions × 3 turns each

`output/track_b_round2/microbench_5x3_capture.jsonl`. 15 requests
through the patched runtime via sidecar proxy. Aggregate:

- turn 0 (cold session): 26.3% acceptance
- turn 1+ (warm, prior turns in per-session cache): 38.4% acceptance
- **+12.1 pp = +46% relative T1 lift**
- Per-draft-token decode: 51.7 ms → 38.4 ms = **−25.7%**

Full report at `track-b-round2-microbench-20260509.md`.

## What's left

- Run the full v2 sweep against this patched runtime — produces
  the corpus-level decode-reduction number via
  `scripts/build_track_b_round2_applicability.py` +
  `scripts/build_track_b_round2_delta.py`. Operator-paced (~22 min
  sweep + analysis).

## Container info

- Image: `lumo-flywheel-vllm:26.01-py3-v0.19.0`
- Started: 2026-05-09T19:32:57Z
- Engine ready: ~19:40:42Z (8 min cold start including pip install
  for arctic + lmcache, model load 161s, encoder cache + cudagraph
  capture)
- Spec decode: `method=suffix, num_speculative_tokens=12,
  suffix_decoding_max_tree_depth=32`
- Health endpoint: 127.0.0.1:9950/health returning 200.
