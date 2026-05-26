# Round 5 FP8-KV Remeasure Plan — D-fp8 vs E3-fp8

**Generated:** 2026-05-25
**Status:** Reminder / run plan. Not executed yet.
**Goal:** remeasure Config D and Config E3 under a truly realized FP8 KV-cache
runtime, so we know whether the current `auto`/BF16 KV regime is leaving
decode bandwidth on the table.

---

## Why this exists

The Round-5 B=4 sweep compared:

| Run | Config | Speculation | Realized KV |
|---|---|---|---|
| `q36a_D_b4` | D | suffix stack | `auto` / likely BF16 |
| `q36a_E3_b4` | E | `qwen3_5_mtp`, linear depth 3 | `auto` / likely BF16 |

The bundles requested `kv_cache_dtype: fp8_e5m2`, but
`ModelServer._initial_kv_cache_dtype()` rewrites FP8-checkpoint +
`fp8_e5m2` KV to `auto`, and the run metrics report
`vllm:cache_config_info{cache_dtype="auto", ...}`. Therefore the existing
D/E3 comparison is fair between D and E3, but it does **not** answer whether
FP8 KV improves the Qwen3.6 FP8 serving stack.

This remeasure pair answers that one question:

```text
same task slice, same temperature, same B=4, same D/E3 speculation
only change: realized KV cache dtype = FP8
```

---

## Runs to create

Use explicit tags so they cannot be confused with the current `auto`/BF16-KV
baseline:

| New tag | Control tag | Config | Speculation | Required realized KV |
|---|---|---|---|---|
| `q36a_D_fp8kv_b4` | `q36a_D_b4` | D | suffix stack | FP8 |
| `q36a_E3_fp8kv_b4` | `q36a_E3_b4` | E | `qwen3_5_mtp`, linear depth 3 | FP8 |

Fixed conditions:

- same 16 SWE-Bench Verified astropy instances:
  `docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json`
- `B=4`
- `temp=0.6`, `top_p=0.95`
- `agent_wall_s=1800`
- `eval_timeout_s=1800`
- same x86 runner / DGX serving topology as `round5-b4-sweep-runbook-20260525.md`
- same steptrace and `per_req_spec_trace.jsonl` instrumentation

---

## Required config-surface fix

Do **not** rerun by only setting `kv_cache_dtype: fp8_e5m2` in the bundle. That
is the stale path that currently becomes `auto`.

Before launching, add a narrow config/runtime path that can request a vLLM KV
dtype that actually resolves to FP8 for this checkpoint. Candidate names to test
against the installed vLLM surface:

```text
fp8
fp8_e4m3
fp8_e5m2
```

The first smoke goal is not SWE-Bench. It is engine startup plus this proof:

```text
vllm:cache_config_info{cache_dtype="fp8" ...}
```

or the exact concrete FP8 label vLLM emits.

If vLLM rejects all FP8 KV dtypes for the Qwen3.6 FP8 checkpoint, stop and
record the blocker. Do not silently fall back to `auto`.

---

## Preflight gates

### Gate 0 — realized dtype proof

For each candidate runtime dtype:

1. Launch a tiny smoke server.
2. Capture startup log line:
   `dtype=... kv_cache_dtype=...`
3. Capture `/metrics`:
   `vllm:cache_config_info{cache_dtype=...}`
4. Fail closed if realized dtype is `auto`.

### Gate 1 — distribution / correctness sanity

Because FP8 KV changes attention numerics, this is not just a drafter-side
change. Run B-1/B-2/B-3 before the 16-task SWE run:

- greedy byte-exact where applicable
- fixed-prompt KL / logprob tolerance
- short Codex smoke with no malformed output/tool-call regression

If FP8 KV violates the gate, do not run the expensive pair.

### Gate 2 — instrumentation sanity

For D-fp8 and E3-fp8 smoke calls, verify:

- steptrace counters advance
- `per_req_spec_trace.jsonl` rows are valid JSON
- D has suffix drafting active
- E3 has `draft=3` rows
- metrics still expose realized FP8 KV after requests

---

## Measurement math

Use the same math as:

`docs/reports/auto_research/swe-bench-config-decode-comparison-spec-20260525.md`

Primary speed is run-level steptrace:

```text
steptrace_decode_tps =
  delta(vllm:generation_tokens_total)
  / delta(vllm:request_decode_time_seconds_sum)
```

Do not promote task-local `decode_sum_s` under B=4; it is overlap-contaminated.

Report, for all four rows (`D`, `D-fp8`, `E3`, `E3-fp8`):

1. resolved / 16
2. steptrace decode TPS
3. accept ratio
4. draft/event
5. accepted/event
6. timeout count
7. dormant seconds and active generation percentage
8. configured KV dtype and realized KV dtype

---

## Decision rules

FP8 KV is a win if:

- B-1/B-2/B-3 pass
- realized KV is actually FP8
- resolved count is no worse than the paired `auto`/BF16-KV control, or any
  drop is understood and acceptable
- steptrace TPS improves meaningfully:
  - `D-fp8` vs `D`: at least +5%
  - `E3-fp8` vs `E3`: at least +5%

If FP8 KV helps E3 but hurts D, keep the result separated by drafter. Suffix
decoding and MTP may stress attention/KV differently.

If FP8 KV is neutral or worse, keep current `auto`/BF16 KV and do not spend
kernel effort on FP8 tree-attn until there is a separate reason.

---

## Expected outcomes to watch for

Possible outcomes:

1. **FP8 KV improves both D and E3.** Promote FP8 KV as a new baseline regime,
   then rerun any future F/tree experiment in that regime.
2. **FP8 KV improves E3 only.** Native MTP is more bandwidth-bound on KV than
   suffix D, or D's CPU/proposer overhead masks the gain.
3. **FP8 KV hurts correctness.** Keep BF16/auto KV for SWE-Bench/Codex despite
   the theoretical bandwidth savings.
4. **FP8 KV cannot launch on this vLLM/Qwen3.6 checkpoint combo.** Record as a
   serving-surface blocker; current D/E/F results remain `auto`/BF16-KV only.

---

## Next steps — D-Pro suffix-trie recovery

Separate from the KV-dtype remeasure, there is now enough evidence to justify a
small D-Pro canary: current Config D uses SuffixDecoding, but the serving path
appears to verify a flat linear suffix draft rather than the full suffix trie.

Concrete evidence:

1. `q36a_D_b4` is the suffix-stack control:
   `method=suffix`, `num_speculative_tokens=12`.
2. vLLM's native `SuffixDecodingProposer.propose()` returns
   `list[list[int]]` and appends only `draft.token_ids`.
3. ArcticInference's `SuffixDecodingDraft` contains richer tree metadata:
   `token_ids`, `parents`, `probs`, `score`, and `match_len`.
4. ArcticInference's simulator walks `result.parents` to verify accepted
   branches, which proves the parent links are intended tree structure.
5. ArcticInference's vLLM plugin path also emits `result.token_ids` into the
   vLLM draft-token surface, so the serving integration still consumes a linear
   sequence.
6. `SuffixDecodingCache.speculate(..., use_tree_spec=False)` defaults to
   non-tree speculation unless requested; the vLLM suffix wrapper does not pass
   `use_tree_spec=True`.

Useful public references:

- vLLM suffix proposer API/source:
  <https://docs.vllm.ai/en/v0.18.1/api/vllm/v1/spec_decode/suffix_decoding/>
- ArcticInference suffix docs:
  <https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html>
- SuffixDecoding paper:
  <https://openreview.net/pdf?id=uwL0vbeEVn>

The research question is not "can suffix draft long continuations?" D already
does that cheaply. The question is:

```text
Is D losing because the suffix trie lacks a correct continuation,
or because vLLM only submits one linear path while the right continuation
exists elsewhere in the trie?
```

Recommended D-Pro sequence:

### D-Pro-0 — observe, do not optimize yet

Patch/instrument the suffix path to expose, per speculative event:

- `len(token_ids)`
- `len(parents)`
- root width and depth histogram
- `score`, `match_len`, and top node probabilities
- current linear path chosen by vLLM
- whether the eventual target-accepted token path existed elsewhere in the
  raw suffix trie

Run this on a tiny slice first. Do not launch a full 16-task round until this
answers whether missed acceptance is a proposal-quality issue or a
linearization issue.

### D-Pro-1 — MTP-guided suffix-trie trim

If D-Pro-0 shows the trie often contains better alternatives than the emitted
linear path, test:

```text
build suffix trie up to depth 10-12
use Qwen3.6 MTP n=3 as a neural prior over the first 1-3 trie levels
boost / keep branches whose prefixes agree with MTP top-k probabilities
trim low-score suffix branches
verify the selected proposal losslessly
```

This uses MTP where it is strongest: short-horizon local probability. It keeps
suffix where it is strongest: cheap long-tail continuation from repeated agent
patterns.

Two verifier flavors are possible:

| Variant | Verification | Why test |
|---|---|---|
| `D-pro-tree-attn` | packed suffix tree + tree mask | lowest duplicate compute, but depends on TREE_ATTN and source patches |
| `D-pro-batched-paths` | top suffix paths as separate linear drafts | FlashAttention-compatible, simpler correctness story, more duplicated compute |

The D-Pro promotion gate should be stricter than normal tuning:

```text
D-Pro must beat q36a_E3_b4 on steptrace TPS,
or prove a clear mechanism with accepted/event lift worth a larger build.
```

Suggested first target:

```text
accepted/event > 2.7
steptrace TPS > 15.1
no resolved-count regression on the paired slice
```

If D-Pro-0 shows the correct continuation is rarely present in the suffix trie,
do not build D-Pro-1. In that case D's weakness is suffix proposal quality, not
vLLM linearization.

---

## Reminder

The current D/E/F docs should always say **configured KV** and **realized KV**
separately. The current historical D/E/F rows are not FP8-KV measurements even
when their bundles contain `kv_cache_dtype: fp8_e5m2`.
