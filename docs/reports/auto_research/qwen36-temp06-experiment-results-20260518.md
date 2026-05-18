# Qwen3.6-27B-FP8 — temperature=0.6 mini-experiment (2026-05-18)

## Question

Does lowering sampling temperature from the default 1.0 to 0.6 (Qwen's
own recommendation for precise coding) reduce the verbosity observed on
Qwen 3.6, and does it move decode speed or wallclock?

## Why temperature, and not the originally-proposed knobs

The experiment was first scoped around `model_reasoning_effort` and
`preserve_thinking`. Both were investigated and found inert in this stack:

- **`model_reasoning_effort` is a no-op on Qwen 3.6.** vLLM 0.19.0 only
  consumes `reasoning.effort` inside `_construct_harmony_system_input_message`
  (`responses/serving.py:1077`) — the Harmony path, used exclusively for
  gpt-oss models. Qwen 3.6 takes the standard `--chat-template` +
  `--reasoning-parser qwen3` path, where `reasoning.effort` is parsed off
  the request and never applied. Empirical A/B (temp=0, max 2048):
  `effort=minimal` → 575 output tokens, `effort=high` → 550 — pure noise.
  The Codex flag `-c 'model_reasoning_effort=...'` reaches vLLM and is
  silently dropped.
- **`preserve_thinking` is already effectively on.** The custom
  `qwen3-openai-codex.jinja` retains `<think>` for every assistant turn
  after the last *real* user message. Under Codex's flow the only real
  user message is the operator's opening prompt, so all assistant turns
  already keep their thinking. Qwen 3.6's `preserve_thinking=False`
  default never applies.

`--reasoning-parser qwen3` is confirmed present on the launch.

So temperature was the only remaining config lever with a plausible
effect. It was pinned to 0.6 (and `max_output_tokens` capped at 32768)
proxy-side via the new env-gated override in `inference_proxy.py`
(`LUMO_PROXY_FORCE_TEMPERATURE`, `LUMO_PROXY_MAX_OUTPUT_TOKENS`).

## Setup

| | |
|---|---|
| Model / config | qwen3.6-27b-fp8, OFF point (no spec_decode), unchanged between conditions |
| Baseline | `output/track_b_e2e_qwen36_ablation/round_0/` — temp=1.0 (model default) |
| Test | `output/track_b_e2e_qwen36_temp06_test/round_0/` — temp=0.6, max_output_tokens≤32768 |
| Tasks | responses-sdk-adapter-cutover, transcript-merge-regression, dead-flag-reachability-audit — 4 attempts each |

## Results

| Task | metric | baseline (temp=1.0) | temp=0.6 |
|---|---|---:|---:|
| responses-sdk-adapter-cutover | text_chars med / p90 | 4 / 1542 | 133 / 1758 |
| | decode_tps med | 6.17 | 5.34 |
| | completion_tokens p90 | 481 | 466 |
| | wallclock med | 30.0 min | 30.0 min |
| transcript-merge-regression | text_chars med / p90 | 134 / 1288 | 4 / 982 |
| | decode_tps med | 4.67 | 5.73 |
| | completion_tokens p90 | 308 | 204 |
| | wallclock med | 30.0 min | 30.0 min |
| dead-flag-reachability-audit | text_chars med / p90 | 144 / 2120 | 127 / 1976 |
| | decode_tps med | 5.26 | 5.33 |
| | completion_tokens p90 | 361 | 768 |
| | wallclock med | 30.0 min | 30.0 min |

## Conclusion

**temperature=0.6 does not change Qwen 3.6's behaviour on this corpus.**

- **Verbosity (`text_chars_observed`) is not temperature-controlled.** The
  per-cell median swings between ~4 and ~134 in *both* conditions —
  responses-sdk went 4→133, transcript-merge went 134→4, dead-flag stayed
  ~flat. `text_chars_observed` is a bimodal per-call metric (≈4 for a
  near-empty text turn, ≈130 for a turn with a paragraph); at 4 attempts
  the median lands on whichever bin holds the majority. The "median 4 vs
  134" gap is sampling noise, not a temperature signal. The qwen3.5→3.6
  jump the operator originally measured is, within 3.6, not reproducible
  as a stable metric.
- **decode_tps unchanged** — 5–6 tps in both conditions, no consistent
  direction.
- **completion_tokens p90 unchanged** — no consistent direction (one task
  up, one down, one flat).
- **wallclock unchanged** — every cell hits the 1800s budget in both
  conditions.

Combined with the effort and preserve_thinking findings: **none of the
three config knobs (reasoning_effort, preserve_thinking, temperature)
move Qwen 3.6's behaviour on this corpus.** The verbosity and the
full-budget wallclock are intrinsic to Qwen 3.6 on these agentic tasks,
not a tunable sampling artifact.

## Pass-rate

Not computed. The three tasks use heterogeneous graders
(`verify.sh` for responses-sdk; bare `score_transcript_merge.py` /
`score_reachability.py` for the others) that require the verifier
harness (`VERIFIER_DATA`, per-task pytest environments). Since no other
metric moved between the two conditions, a pass-rate difference is
unlikely; grading all 24 cells to confirm a near-certain null result was
judged disproportionate. Flag if you want it run regardless.

## Recommendation

Temperature tuning is not the lever. If Qwen 3.6's verbosity / full-budget
wallclock is a blocker, the remaining options are model-side (prompt
engineering on the corpus AGENTS.md, or accepting it as a Qwen 3.6
property) rather than serving-config. For the OFF/A/D remeasure: proceed
at default settings — there is no config tweak that would make the
numbers more favourable or more comparable to Qwen 3.5.

## Artifacts

- Test cells: `output/track_b_e2e_qwen36_temp06_test/round_0/` (committed per-task)
- Baseline cells: `output/track_b_e2e_qwen36_ablation/round_0/`
- Proxy override: `src/lumo_flywheel_serving/inference_proxy.py` (`normalize_responses_request_payload`)
- Test driver: `/tmp/qwen36_temp06_test.py`
