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

## CORRECTION (research workflow wnws53l2k, 2026-07-02) — candidate fix REFUTED, wrong binary
My earlier `--strict-config` probing hit the **host** codex 0.139.0. **The harness actually runs codex
0.128.0 INSIDE the `codex-runner:v1` container** (`run_swe_bench_q36_a.py:62` pins tag
`q36-a::codex-cli-0.128.0`; `--strict-config` isn't even a valid arg on 0.128 `codex exec`). The workflow
empirically captured the real `tools` list the 0.128.0 container sends (mock `/v1/responses`):
- baseline (current template) → **NO apply_patch tool** (11 tools, 45340 B).
- `-c experimental_use_freeform_apply_patch=false` → **byte-identical to baseline** (=false is the default → NO-OP).
- `-c apply_patch_tool_type="function"`, `-c include_apply_patch_tool=true`, `-c features.apply_patch_freeform=false` → **all silently ignored**, still NO apply_patch tool.
- `-c experimental_use_freeform_apply_patch=true` → registers apply_patch, but ONLY as `{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark"}}` (FREEFORM lark grammar), **never `type:"function"`**.
So **there is NO `-c` flag that yields a routable FUNCTION-form apply_patch.** The prior candidate
(`features.apply_patch_freeform=false`) is a **no-op** and must NOT be shipped.

**Why (deeper):** the `unsupported call: apply_patch` error is raised **client-side inside the container's
Rust codex**, validating the model's returned call against the tools *codex itself* registered — NOT
against what the proxy returns. So proxy/jinja tool-injection **cannot** fix routing. codex's base
`instructions` (20771 chars) direct the model to use apply_patch 5×, so Qwen emits it even with no tool
registered → registry miss → "unsupported call". Real trace: model emits
`{"type":"function_call","name":"apply_patch","arguments":"{\"cmd\":\"apply_patch\\n*** Update File...\"}"}`
→ codex writes `function_call_output "unsupported call: apply_patch"`.

## The two coherent routes (workflow verdict: fix_holds=true, must_probe=true, confidence=high)
- **ROUTE A — RECOMMENDED — do nothing.** Shell fallback is **proven survivable**: leafsrc runs logged
  **3774** `unsupported call: apply_patch` events yet still resolved (5/5 native baseline). apply_patch
  routing is NOT the char-8 blocker and NOT the resolve-cause. Ship no change; resume the campaign on the
  current harness. Lowest risk; matches measurement discipline.
- **ROUTE B — only if structured edits are genuinely wanted (multi-part, higher risk):**
  (i) CODEX_TEMPLATE `-c experimental_use_freeform_apply_patch=true` (registers the custom/lark tool
      client-side so the router stops rejecting a matching `custom_tool_call`);
  (ii) NEW proxy response-side branch (`inference_proxy.py` near `normalize_responses_response_payload` /
      the streaming emitters ~1583-1608/1747-1805, which today emit ONLY `type:"function_call"`):
      transform the model's `function_call` named apply_patch into a Responses `custom_tool_call` item,
      stripping the `{"cmd":...}` JSON wrapper into the raw `*** Begin Patch…*** End Patch` text (vLLM has
      ZERO custom_tool_call emission path, so the proxy is the only place to synthesize it). Env-gated,
      default OFF.
  Validate with PROBE 1 (1 astropy-separability task, live vLLM: router error DISAPPEARS + a
  patch_apply/function_call_output reports files changed + `git diff` produced via apply_patch not sed) AND
  a ≥8-task control showing no regression vs the shell-fallback resolve rate BEFORE any campaign use.
  Risks: moves the action distribution off the proven-survivable shell path (could resolve FEWER); re-opens
  char-8 exposure on truncated patch args; the new proxy path can corrupt EVERY tool call if buggy.

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
