# Track B Round 2 SuffixDecoding Findings

Measured: 2026-05-07

## Objective

Push the post-PR #39562 real-content decode baseline from roughly 10-11 tok/s toward 30 tok/s on an authored Codex benchmark workload.

## New Runtime Surface

The Track B controller now accepts vLLM `spec_decode.method: suffix` in addition to `ngram`.

Runtime prelaunch now also:

- Applies the PR #39562 KV allocator stop-gap in `single_type_kv_cache_manager.py`.
- Installs `arctic-inference==0.1.2`, which vLLM's suffix proposer imports lazily.

This makes vLLM's built-in SuffixDecoding proposer testable as Technique 1 from `docs/reports/auto_research/codex-harness-spec-decode-engineering-20260507.md`.

## Measurement Shape

- Script: `scripts/measure_track_b_real_content_task.py`
- Benchmark family/variant: `release-note-to-plan-translation/v1-clean-baseline`
- Prompt content: `AGENTS.md`, `.scenario_variant`, `release_notes/*`, `repo_inventory/*`
- Prompt size: 4449 chars, 619 words, 1068 input tokens in measured requests
- Policy: one cold completion discarded, Prometheus metrics sampled only around the warm window
- Output cap: 2048
- Acceptance target: 30 decode tok/s

## Results

| Candidate | Config | Warm concurrency | Decode tok/s | Wall output tok/s | Accepted / draft tokens | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `053` | suffix, k=12, tree=32, factor=2.0, min_prob=0.05, strict | 1 | **16.123641** | 16.088674 | 809 / 2981 = 0.271385 | Best suffix point, fail 30 |
| `053` | same | 4 | 12.978076 | 49.269557 | 4269 / 18021 = 0.236890 | Decode worse; wall throughput is concurrency only |
| `054` | suffix, k=24, tree=64, factor=4.0, min_prob=0.0, strict | 1 | 13.592849 | 13.575734 | 958 / 7295 = 0.131323 | Longer drafts hurt |
| `055` | suffix, k=12, tree=32, factor=2.0, min_prob=0.05, probabilistic | 1 | 14.314585 | 14.294272 | 980 / 4152 = 0.236031 | Probabilistic did not help |

Artifacts:

- `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/053/real_content_suffix_release_plan_c1_run_01.json`
- `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/053/real_content_suffix_release_plan_c4_run_01.json`
- `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/054/real_content_suffix_release_plan_c1_run_01.json`
- `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/055/real_content_suffix_release_plan_c1_run_01.json`

## Decision

Built-in SuffixDecoding is a real uplift over the prior real-content ngram baseline:

- Prior best ngram real-content c1: about 10-11 decode tok/s.
- Best suffix real-content c1: 16.123641 decode tok/s.

It does not reach the 30 decode tok/s objective. The limiting signal is accepted-token ratio, not draft length. Candidate 054 drafted longer spans, but accepted-token ratio fell to 0.131323 and decode speed regressed. Candidate 055's probabilistic sampler also regressed.

## Next Productive Path

Do not continue blind suffix-only restart tuning. The next path toward 30 decode tok/s needs extra harness information, not just larger suffix parameters:

- A harness oracle that primes the suffix cache with exact tool observations and read-file content before the model emits edit/plan text.
- A schema-aware tool-call drafter for tool-call frames, measured on the existing tool-call gate workload.
- A deterministic or low-temperature acceptance-specific measurement only if the target Codex serving posture actually uses that sampling behavior; otherwise it would be a proxy artifact.

## Follow-Up Tool-Call Workload Closeout

After the user allowed any authored benchmark family/variant and specifically requested tool-call emission coverage, candidate `056` closed the 30 tok/s target on the tool-call-inclusive workload:

- Workload: `policy-aware-request-resolution/v1-clean-baseline`
- Gate: `scripts/run_track_b_tool_call_gate.py --tool-choice-mode auto --no-exact-arguments --measure-throughput --target-decode-tps 30`
- Artifact: `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/056/tool_call_b2_policy_v1_c4_auto_structural_512_pr39562_suffix056_throughput_script.json`
- Result: PASS, `4/4` tool-call cases, `66.295983` decode tok/s, `83.034414` wall output tok/s.

This directly exercises the PR #39562-patched runtime and Qwen3 XML tool-call parser path implicated by vLLM Issue #40875. The long release-plan text workload remains below target at `15.148751` decode tok/s for candidate `056`, so the completion is scoped to the authored tool-call Codex workload.
