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

## Root cause: vLLM Responses API forced tool_choice bypasses the parser

**Updated 2026-05-08 (post-investigation):** the initial root-cause
hypothesis (prompt length × parser whitespace handling) was wrong.
The actual cause is a vLLM Responses API bug isolated to **forced
`tool_choice = {"type": "function", "name": "..."}`**.

**The bug** is in `vllm/parser/abstract_parser.py:_parse_tool_calls`
(verified against the version shipped in the running container).
When `tool_choice` is forced, the function bypasses the configured
tool parser and stuffs the raw model output text into
`FunctionCall.arguments`:

```python
if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
    # Forced Function Call (Responses API style)
    assert content is not None
    function_calls.append(
        FunctionCall(name=request.tool_choice.name, arguments=content)  # <-- raw content
    )
    return function_calls, None
```

The same broken logic exists for `ChatCompletionNamedToolChoiceParam`.
Auto / required / unforced tool_choice paths run the parser
correctly. Upstream Issue #23227 ("Support tool_choice other than
auto") was closed as not-planned; the half-implemented forced path
is the artifact.

**Confirming the root cause:**
- Direct `/v1/responses` call, forced tool_choice, **short** prompt:
  fails identically (arguments contain raw XML).
- Direct `/v1/chat/completions` call, forced tool_choice, short
  prompt: parses correctly. The bug is Responses-API-specific.
- Direct `/v1/responses` call, **auto** tool_choice, long prompt:
  parses correctly. Prompt length is irrelevant.
- The qwen3_xml parser run standalone on the exact failing XML
  payload (`<tool_call>...<parameter=path>\n...AGENTS.md\n</parameter>...`)
  produces `arguments='{"path": "AGENTS.md"}'`. The parser is fine;
  the Responses API never invokes it under forced tool_choice.

**Production impact: zero.** Codex CLI 0.128.0 uses auto tool_choice
(or no tool_choice) when sending `/v1/responses` requests, which
hits the working path. The v2 Round 0 measurement's 12/13 trusted
task summaries are unaffected — every Codex turn parsed correctly.
Step 0d is the only consumer of forced tool_choice in our codebase,
which is why the bug only surfaces there.

## Recommendation: Option 4 — patch vLLM, not the model config

The earlier 3-remediation enumeration (re-enable thinking; switch
parser; fall back to ngram-PLD) all targeted the wrong layer. The
bug is the Responses API forced tool_choice path, not the model
output. The right fix is small and surgical:

**Option 4 — patch `vllm/parser/abstract_parser.py:_parse_tool_calls`**
to run `self._tool_parser.extract_tool_calls(content, request=request)`
on `content` even under forced tool_choice, then use the parser's
parsed `arguments` instead of passing raw content through. The
forced name still overrides whatever the parser thinks.

**Status: applied.** The patch is in place at the running
container's filesystem path, and codified in the prelaunch hook
(`scripts/run_track_b_loop.py:_track_b_runtime_prelaunch_shell`,
commit e67832c) so future container relaunches auto-apply it. The
running vLLM process loaded the old code at startup; activation
requires a relaunch. After the relaunch:
- Step 0d should pass (B-1/B-2/B-3 against live SuffixDecoding).
- v2 round 0 measurements are unaffected (production used auto
  tool_choice, which was always on the working path).
- No model-config regression: keep SuffixDecoding, keep thinking
  off, keep all v2 round 0 wins on tool-call regime.

## Status

- v2 round 0: trusted 12/13 task summaries → committed (6846ec8).
- Step 0d: **FAIL** → root cause isolated to vLLM Responses API.
- Patch applied (e67832c, prelaunch hook). Activation = next vLLM
  relaunch.
- Next-restart attempt blocked on GB10 unified-memory pool
  reclamation (only 13 GiB available after the prior vLLM exit;
  driver hasn't released the model's prior allocation yet).
  Operator-gated.
