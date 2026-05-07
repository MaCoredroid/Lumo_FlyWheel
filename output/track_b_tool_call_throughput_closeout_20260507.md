# Track B Tool-Call Throughput Closeout

Measured: 2026-05-07

## Objective

Close the Track B decode-speed goal using an authored Codex workload that directly emits tools, so the PR #39562-patched runtime and the Qwen3 XML tool-call parser path from vLLM Issue #40875 are exercised together.

## Runtime

- Model: `qwen3.5-27b`
- Endpoint: direct vLLM `/v1/responses` on `http://127.0.0.1:9950/v1`
- Active candidate: `056`
- Active tuned-config bundle: `712fd011-4b16-4051-9e8c-875405b70f5b`
- Spec decode: built-in vLLM suffix decoding, `num_speculative_tokens=12`, tree depth `32`, max cached requests `1000`, spec factor `2.0`, min token probability `0.05`, probabilistic rejection sampling
- PR #39562 stop-gap: confirmed in container prelaunch logs, replacing the KV allocator equality guard with `num_required_blocks <= len(req_blocks)`
- Qwen3 tool parser: confirmed in launch args as `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
- Launch shape also included `--default-chat-template-kwargs '{"enable_thinking": false}'`; this did not improve the long text workload, but it remained active for the tool-call throughput gate.

## Workload

- Authored family/variant: `policy-aware-request-resolution/v1-clean-baseline`
- Gate script: `scripts/run_track_b_tool_call_gate.py`
- Mode: Responses `tool_choice: auto`, so the model emits Qwen-style XML and vLLM parses it into Responses `function_call` items.
- Tool cases:
  - `read_file` for `AGENTS.md`
  - `exec_command` for `pytest -q`
  - `apply_patch` containing `tool_gate_marker` and `tool gate`
  - `write_file` containing family, variant, and checked status

## Result

| Artifact | Pass | Decode tok/s | Wall output tok/s | Notes |
| --- | ---: | ---: | ---: | --- |
| `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/056/tool_call_b2_policy_v1_c4_auto_structural_512_pr39562_suffix056_throughput_script.json` | 4/4 | **66.295983** | 83.034414 | PASS target 30 |

Metric details:

- Generation tokens: `916`
- Decode sum: `13.816825034096837` s
- Prompt tokens: `5138`
- Request count: `8` (`4` serial plus `4` concurrent)
- Pass target gate: `true`

Arithmetic self-check:

- `916 / 13.816825034096837 = 66.295983`
- A previous wrapper run of the same active runtime and workload measured `71.745152` decode tok/s.

## Why This Is Much Higher Than Earlier Numbers

The earlier 10-16 tok/s measurements were from the long `release-note-to-plan-translation` text-generation prompt, where outputs ran to the 2048-token cap and suffix acceptance stayed low. This closeout uses the user-requested authored tool-call workload. The output shape is much shorter and more repetitive: each request emits one structured function call such as `read_file`, `exec_command`, `apply_patch`, or `write_file`, and the measured window includes four serial plus four concurrent Responses calls.

That makes suffix-cache reuse and accepted-token ratio materially better for this workload. The result is valid for the agentic tool-call workload and for exercising the Issue #40875 XML parser path; it should not be read as a new speed claim for the long text-only release-plan workload.

## Gate Fix

`scripts/run_track_b_tool_call_gate.py` now supports `--measure-throughput` and `--target-decode-tps`. It also distinguishes exact argument matching from structural matching: `--no-exact-arguments` still validates required argument content, but does not fail when serial and concurrent calls produce different valid patch text.

## Caveat

The long `release-note-to-plan-translation/v1-clean-baseline` text workload remains below target. Candidate `056` measured `15.148751` decode tok/s there and still emitted visible `Thinking Process` text under Responses JSON shaping. The 30 tok/s closeout is therefore for the authored tool-call Codex workload requested by the user, not for the long release-plan text-only prompt.
