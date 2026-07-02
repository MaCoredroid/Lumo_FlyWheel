# FR13 — `apply_patch` always fails ("unsupported call: apply_patch") — BANKED FINDINGS

**Date:** 2026-07-02. **Status:** root-caused; fix DESIGN in progress (research workflow); arms PAUSED until fix lands.

## Symptom
Every SWE task's codex trace shows repeated `codex_core::tools::router: error=unsupported call: apply_patch`
(e.g. 4× on astropy-12907). The model then falls back to shell (`bash -lc` + `sed`/`python3` heredoc) to
edit files. Present in EVERY run, including the 5/5 native-cache-OFF baseline — **constant + survivable**,
not a regression.

## Root cause (evidence-backed)
1. We invoke codex with `--model qwen3.6-27b`, `wire_api="responses"`, `model_provider=local-proxy` → our
   vLLM (`run_swe_bench_q36_a.py:71-86` CODEX_TEMPLATE).
2. `apply_patch` is **not a normal function tool** — per OpenAI docs it is a *native/built-in* tool
   (`tools=[{"type":"apply_patch"}]`, no input schema; "the model knows how to construct operation objects"),
   supported by **GPT-5.1/5.2/5.4/5.5 only**. Codex's `ToolsConfig::new()` picks the apply_patch tool FORM
   from the **model family**; for the GPT-5 family it registers the **freeform / `type:"custom"`** grammar.
3. Codex **injects the apply_patch instruction itself** (its baked agent prompt) — our chat template
   `docker/chat_templates/qwen3-openai-codex.jinja` has **ZERO** `apply_patch` mentions.
4. So codex tells Qwen3.6 to use apply_patch, but Qwen (not GPT-5) emits it as an ordinary function/tool
   call, which does NOT match the freeform grammar codex registered → router rejects: **"unsupported call:
   apply_patch"** → model degrades to shell edits.

## Exact config knob (verified against installed codex 0.139.0 via `codex exec --strict-config`)
- `apply_patch_tool_type`, `tools.apply_patch`, `include_apply_patch_tool` → **ALL "unknown configuration field"** (not valid keys in 0.139).
- `features.apply_patch_freeform` → **ACCEPTED** (valid key). This is the real toggle:
  - `true`  = GPT-5 freeform `type:"custom"` grammar (non-OpenAI endpoints reject / can't round-trip).
  - `false` = standard **function-calling** apply_patch (what Qwen3.6 CAN emit over our vLLM responses API).
- alienware `~/.codex/config.toml` sets `model = "gpt-5.5"` (overridden by our `--model qwen3.6-27b`), does
  NOT set `apply_patch_freeform` → codex uses its default (suspected freeform=on for the responses wire).

## Candidate fix (to be validated by the workflow + a live probe)
Add to CODEX_TEMPLATE: `-c 'features.apply_patch_freeform=false'` → register apply_patch as a routable
FUNCTION tool for qwen3.6-27b → the model's calls route to the real handler instead of "unsupported".
UNVALIDATED empirically (needs a codex probe against a live vLLM endpoint; not run yet — arms paused).

## Why it matters (link to char-8)
Because apply_patch is rejected, the model crams patch content **inline into shell args** — and those are
exactly the char-8 poison strings previously found (`{"cmd":"printf '*** Begin Patch…`,
`{"command":"*** Begin Patch…`, truncated → "Unterminated string char 8"). A working apply_patch channel
would likely REDUCE char-8, complementing the transcript-repair already shipped
(LUMO_PROXY_REPAIR_TOOLCALL_JSON). Also removes wasted turns (4× rejected calls before shell fallback).

**Scope honesty:** this is an EFFICIENCY + char-8-reduction lever, NOT the cause of resolve failures —
astropy-12907 failed with a coherent shell-based *wrong fix* (real 1000B patch, tests_failed), not because
apply_patch was rejected. But it is a harness confound worth removing BEFORE the big measurement runs, which
is why all arms are paused until it lands.

## Sources
codex #17899, #15642, #14046; OpenAI Apply Patch tool guide; Codex config reference.
