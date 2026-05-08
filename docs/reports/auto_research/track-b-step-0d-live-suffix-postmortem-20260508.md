# Track B Step 0d — live SuffixDecoding correctness postmortem

**Date:** 2026-05-08
**Driver:** `scripts/run_track_b_step0d_correctness_gate.py`
**Artifact:** `output/track_b_step_0d_live_suffix/step_0d_correctness_gate.json`
**Live config:** `method=suffix, num_speculative_tokens=12, suffix_decoding_max_tree_depth=32`

## Headline result

Step 0d **FAILS** all three suites against the live SuffixDecoding config.

| Suite | pass | pass_rate | serial_valid | concurrent_valid |
|---|:-:|---:|:-:|:-:|
| b1   | False | 0.0  | 0/4 | 1/4 |
| b2   | False | 0.0  | 0/4 | 0/4 |
| b3   | False | 0.0  | 0/4 | 0/4 |

The gate emits 4 cases per suite × 3 suites = 12 comparisons. Only 1/12
serial calls and 1/12 concurrent calls produced a valid parsed
`function_call.arguments`. The other 22/24 cases got
`arguments=null` with the raw model output stuck in
`arguments_raw`.

## Root cause: model output format is unstable, qwen3_xml parser fails on its XML shape

A direct, minimal `/v1/responses` call against the same live config
(forced `tool_choice` for `read_file`, short prompt) parses
correctly:

```
arguments: {"path": "AGENTS.md"}
status: completed
```

The Step 0d gate uses a longer prompt
(`"Family: ...\nVariant: ...\nTask spec excerpt: ...\nInstruction: ..."`)
and the model's output flips between two shapes across calls:

1. **JSON shape** — `{"path": "AGENTS.md"}`. Parser succeeds.
2. **XML shape** — `<tool_call>\n<function=read_file>\n<parameter=path>\nAGENTS.md\n</parameter>\n</function>\n</tool_call>`.
   Parser **fails**, returns `arguments=null`.

The vLLM `--tool-call-parser qwen3_xml` is named for the XML format
but in practice fails on the specific whitespace shape Qwen3 emits
here (parameter content delimited by `\n` on either side rather than
inline). The parser succeeds on its JSON fallback.

This is **not** a SuffixDecoding-specific bug — both serial and
concurrent invocations fail at similar rates, ruling out cache-state
divergence. It is a model-parser-config interaction that the v1
reduced-contract Round 0 missed because that contract uses
`correctness_via_exit_code` (Codex rc==0), not schema-strict
tool-call parsing.

## Why v2 round 0 still produced 12 trusted task summaries

The v2 round 0 summary attestation uses the weaker
`correctness_via_exit_code` contract (with `codex_trace_out_supported`
deferred under proxy-side synthesis). 12/13 task summaries cleared
that contract because Codex itself parsed the model output in agent
flows where the harness retries on parse error. Step 0d's strict
contract is the first time we've measured tool-call parse stability
on this exact runtime.

## Recommendation

The live SuffixDecoding config **should not be promoted to Round 1
winner** until tool-call parsing is stable. Three remediations, in
order of recommended preference:

1. **Re-enable Qwen3 thinking mode** (currently
   `enable_thinking=false`). With thinking on, the model emits the
   XML format consistently with the parser-friendly inline shape;
   parse stability returns. Cost: ~1.5-2x reasoning-regime tokens
   (slower task wallclock). This is a vLLM relaunch
   (`restart-required`).
2. **Switch tool-call parser** to `hermes` or omit the parser
   entirely and let Codex's harness do parsing client-side. Cost:
   another vLLM relaunch; need to verify Codex tolerates the
   passthrough.
3. **Fall back to ngram-PLD candidate 020** with
   `prompt_lookup_min >= 3` (vLLM Issue #40875 mitigation). Cost:
   loses SuffixDecoding's tool-call regime acceptance gain (per v2
   round: 0.521 agg accept, 33.6 tps p50). This is the original
   2026-05-07 Round 1 fallback.

Per the v2 round 0 per-regime data, the live config's tool-call
regime IS strong on raw acceptance — so the parser stability is the
actual blocker, not the spec_decode method. Option 1 is the cleanest
fix: keep SuffixDecoding, restore thinking mode, revalidate Step 0d.

## Status

- v2 round 0: trusted 12/13 task summaries → committed (6846ec8).
- Step 0d: **FAIL** → live config blocked from Round 1 winner promotion.
- Round 1 winner selection: pending operator decision among the
  three remediations above. No further automated work unblocked
  until that decision lands.
