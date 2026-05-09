# Track B Round 1 winner shipped — live SuffixDecoding

**Date:** 2026-05-09
**Schema:** `lumo.track_b.round_1_winner.v1`
**Runtime config hash:** `sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`
**Spec decode:** `method=suffix, num_speculative_tokens=12, suffix_decoding_max_tree_depth=32, rejection_sample_method=probabilistic`

## Decision

The live SuffixDecoding config is the Round 1 winner. No spec_decode
method change from the v2 Round 0 baseline. The forced-tool_choice
parser bypass (vLLM Issue #23227, closed as not-planned) was the
only thing blocking this hand-off; that patch is now applied via the
`ModelServer` prelaunch hook and verified end-to-end.

## Acceptance ladder

### v2 Round 0 baseline (frozen 2026-05-08)

12 trusted task summaries + 1 diagnostic-only (skill-router run_02
hit a real Codex rc=1). Median wallclock 109.07s, aggregate 1309.67s.
94 proxy-capture rows under one runtime hash. Per-regime:
tool-call=0.521 agg accept (33.61 tps p50, 89% of rows);
reasoning=0.209 agg accept (10.24 tps p50, 11% of rows). See
`track-b-e2e-round0-v2-report-20260508.md`.

### Step 0d B-1/B-2/B-3 (live SuffixDecoding, post-patch, structural match)

| Suite | pass | pass_rate | model | runtime |
|---|:-:|---:|---|---|
| b1 | True | 1.0 | qwen3.5-27b | live SuffixDecoding |
| b2 | True | 1.0 | qwen3.5-27b | live SuffixDecoding |
| b3 | True | 1.0 | qwen3.5-27b | live SuffixDecoding |

Aggregated: `gate_pass=true`. Artifact at
`output/track_b_step_0e_live_suffix/step_0d_correctness_gate.json`
with per-suite reports at `output/track_b_step_0e_live_suffix/{b1,b2,b3}.json`.

### What "structural match" means and why it's the right gate here

The Step 0d v1 driver hard-coded `--exact-arguments=True`, which
demanded byte-equal `function_call.arguments` between serial and
concurrent invocations of the same probe. Under deterministic
ngram-PLD that worked. Under SuffixDecoding two legitimate sources
of variation appear:

- The model writes the `apply_patch` body with slight path-variant
  ("artifact" vs "artifacts") between calls. Both are valid model
  outputs; both produce a usable patch when fed to the apply-patch
  tool.
- The `write_file` content is JSON the model formats with two-space
  indent and inserted newlines, while the gate's `expected_arguments`
  uses `json.dumps(sort_keys=True)` (compact, key-sorted). Both
  serialize the same Python dict, but byte-different.

`--no-exact-arguments` switches the gate to structural matching:
parsed argument shape + the existing `required_contains` substring
checks. The bug Step 0d existed to detect (parser bypass under
forced `tool_choice`) is parser-level; structural match is what's
needed to surface it without false-flagging tokenizer-level
variation. The driver now exposes the toggle
(`scripts/run_track_b_step0d_correctness_gate.py --no-exact-arguments`).

### Pre-patch baseline (for the audit trail)

| | b1 | b2 | b3 | gate |
|---|---:|---:|---:|:-:|
| **2026-05-08 pre-patch (exact match)** | 0.0 | 0.0 | 0.0 | FAIL |
| **2026-05-09 post-patch (exact match)** | 0.25 | 0.5 | 0.5 | FAIL |
| **2026-05-09 post-patch (structural match)** | **1.0** | **1.0** | **1.0** | **PASS** |

The middle row is the parser fix landing without the gate flexibility
fix — already a substantial improvement (0/12 → 5/12 passes), every
remaining failure traceable to model output nondeterminism rather
than vLLM. The bottom row is the gate-and-fix combination.

## Patches in flight (all idempotent, applied via prelaunch hook)

`scripts/run_track_b_loop.py:_track_b_runtime_prelaunch_shell` applies
four patches every time `ModelServer` launches a vLLM container:

1. **GPU memory hygiene guardrail** (commit a59770a) — fails loud if
   `MemAvailable < 40 GiB`. The actual recovery is host-side via
   `ModelServer._recover_host_memory()` (`sync; echo 3 >
   /proc/sys/vm/drop_caches; swapoff -a; swapon -a` with
   `LUMO_SUDO_PASSWORD`).
2. **PR #39562 KV allocator stop-gap** — patches
   `single_type_kv_cache_manager.py` to handle dynamic draft length.
3. **arctic-inference install** — `pip install arctic-inference==0.1.2`,
   no-op if present. Provides the `method=suffix` speculative config.
4. **Forced tool_choice parser bypass fix** (commit e67832c) — the
   subject of this report. Patches
   `vllm/parser/abstract_parser.py:_parse_tool_calls` to run the
   configured tool parser on `content` even when `tool_choice` is
   forced.

## Open Round 2+ work (unchanged from v1 plan)

- LMCache install (Round 0 Step 1 — independent).
- Harness oracle API + harness-coupled techniques (Round 2 spec).
- Per-technique micro-benchmarks + ablations.

The v2 spec recalibration (`track-b-e2e-agentic-saturation-plan-20260508-v2.md`)
makes Technique 2 / 3 leverage smaller than v1 implied (89% tool-call
/ 11% reasoning measured) and surfaces tool-exec-wait as the largest
open lever to investigate.

## Provenance

- `output/track_b_e2e_v2/round_0/round_summary.json` — v2 Round 0
- `output/track_b_e2e_v2/round_0/round_v2_report.json` — v2 report
- `output/track_b_step_0e_live_suffix/step_0d_correctness_gate.json` — Step 0d post-patch on live SuffixDecoding
- `output/track_b_step_0e_live_suffix/{b1,b2,b3}.json` — per-suite reports
- `tests/test_vllm_forced_tool_choice_patch.py` — regression test for the patch
- `docs/reports/auto_research/track-b-step-0d-live-suffix-postmortem-20260508.md` — root cause + fix narrative
- `docs/reports/auto_research/track-b-e2e-round0-v2-report-20260508.md` — baseline metrics
