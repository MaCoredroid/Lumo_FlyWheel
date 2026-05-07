# Track B Round 1 Correctness Gate

Measured: 2026-05-07

## Objective

Test the PR #39562-patched runtime against an authored benchmark workload that emits tool calls, so the Qwen3 tool-call XML path implicated by vLLM Issue #40875 is exercised directly before choosing a `prompt_lookup_min=2` Track B candidate.

## Runtime Under Test

- Model: `qwen3.5-27b`
- Endpoint: direct vLLM `/v1/responses` on `http://127.0.0.1:9950/v1`
- Active candidate: `028`
- Candidate config: ngram speculative decode, `num_speculative_tokens=2`, `prompt_lookup_min=2`, `prompt_lookup_max=8`
- PR #39562 stop-gap: applied before vLLM launch in `single_type_kv_cache_manager.py`, changing the required-block equality guard to allow `num_required_blocks <= len(req_blocks)`.
- Authored workload: `policy-aware-request-resolution/v1-clean-baseline`

## Gate Artifacts

| Gate | Artifact | Result |
| --- | --- | --- |
| B1 batch serial/concurrent equivalence | `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/028/b1_result_pr39562_c4.json` | PASS, 4/4, match rate 1.0 |
| B2 authored-workload equivalence | `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/028/b2_policy_v1_pr39562_c4.json` | PASS, 4/4, match rate 1.0 |
| B3 longer-prefix authored-workload equivalence | `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/028/b3_policy_v1_pr39562_c4.json` | PASS, 4/4, match rate 1.0 |
| Tool-call gate, auto tool selection | `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/028/tool_call_b2_policy_v1_c4_auto_pr39562_only.json` | PASS, 4/4, pass rate 1.0 |
| Tool-call gate, forced Responses tool choice | `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/028/tool_call_b2_policy_v1_c4_pr39562_only.json` | FAIL, 2/4, pass rate 0.5 |

## Tool-Call Coverage

The passing auto-tool gate used the authored benchmark context and asked the model to emit these Codex-like tool calls:

- `read_file` for `AGENTS.md`
- `exec_command` for `pytest -q` in `.`
- `apply_patch` adding a tool-gate marker
- `write_file` writing `artifacts/tool_gate.json`

Auto tool selection was used intentionally for the Issue #40875 check. In this mode the model emits Qwen-style tool-call XML and vLLM parses it into Responses `function_call` items. All four serial calls and all four concurrent calls parsed into valid function-call objects, and normalized serial/concurrent calls matched.

## Forced-Tool Caveat

Forced `tool_choice` is not clean on this runtime. Candidate 028 produced a 2/4 pass rate in the forced Responses path:

- `apply_patch`: serial request hit HTTP 500; concurrent request parsed.
- `write_file`: serial response produced invalid/truncated JSON arguments; concurrent request hit HTTP 500.

This is a separate Responses forced-tool path risk. It should not be treated as proof of Issue #40875 corruption in the auto XML parser path, because the auto-path gate passed. It does mean forced tool choice should not be used as a shipping assumption without an additional parser/server fix.

## Decision

Candidate `028` is the current Round 1 lead for the PR #39562-patched runtime on a tool-call-inclusive authored workload:

- It passed B1/B2/B3 equivalence at `c4`.
- It passed direct auto tool-call XML emission and parsing at `c4`.
- Prior speed matrix evidence for the same candidate on `release-note-to-plan-translation/v1-clean-baseline` was `10.547060` decode tok/s at `c1`, `9.860336` decode tok/s at `c4`, and `31.874462` wall output tok/s at `c4`.

This does not complete the 30 decode tok/s goal. The current evidence clears correctness for the best tested `prompt_lookup_min=2` candidate, but the measured decode throughput remains about 10-11 tok/s. Further harness-coupled decode work is still required for the 30 decode tok/s objective.
