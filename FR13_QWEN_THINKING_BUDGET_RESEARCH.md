# FR13 — How to cap Qwen3-Next thinking tokens in our stack (Codex → proxy → vLLM)

**Context:** `reasoning_effort` is INERT on Qwen3.6 (only flips thinking on/off, no depth dial). Thinking is
ON, with a single 32768 **total**-output cap (thinking + answer share it) — so over-thinking starves the edit
(→ give-ups + char-8 truncation). We need to bound the THINKING tokens specifically. (Workflow `wf_5574d2ce`.)

## A clean built-in EXISTS — but it's BROKEN on our exact stack
vLLM **`thinking_token_budget`** (top-level request field via `ThinkingTokenBudgetLogitsProcessor`, PR #20859,
v0.19.0+) force-injects `</think>` at the cap. **Two independent bugs void it on our config (both present):**
1. **MTP/spec-decode silently voids the budget** — per-step count runs on stale batched-accept state; fires
   late or never. Fixed only in **vLLM ≥ 0.21.0**. We run pinned `qwen3_5_mtp` spec. ([vllm#39573])
2. **Qwen3.x token-ID mismatch** — processor hardcodes `</think>` IDs that don't match the Qwen3.5/3.6
   tokenizer; overshoots ~7×. ([sglang#25536])

So treat the native knob as **unverified-until-canaried, not the answer**. (`reasoning_budget` from PR#37112
was closed unmerged — don't use that name.)

## Ranked options for OUR stack (lowest effort that *actually works*)
| rank | option | reliability | where |
|---|---|---|---|
| 1 | Prompt budget hint ("you have ~N thinking tokens, then edit") | **Low** (advisory; models don't self-count) | Codex/proxy prompt — complement only |
| **2** | **Proxy-side stream-count + inject `</think>`** (s1/Qwen budget-forcing) | **High — bug-immune** (enforced outside the decode loop, so MTP + token-ID bugs can't defeat it) | OUR offload-proxy (`$REPO/src`), new `LUMO_PROXY_THINK_BUDGET` |
| 3 | Proxy two-call split: A=`max_tokens=N`+`stop=["</think>"]`, B=re-prompt with the prefix | **Med-high**, spec-safe | OUR proxy — cheap stopgap of #2 |
| 4 | Native vLLM `thinking_token_budget` (extra_body field) | High in theory, **expected-broken here** (MTP + IDs) | vLLM launch; only if canary passes |
| 5 | Logit bias on `</think>` | **Low** (constant, ID-fragile) | assist only |

**Rank 2 injection** (Qwen official cutoff nudge — reproduce the framing exactly or you get empty `content`):
```
<think>
{reasoning_so_far}
Considering the limited time by the user, I have to give the solution based on the thinking directly now.
</think>

```

## Recommended: canary the built-in, then ship the proxy bound
- **Step A (10 min):** send an over-thinker with `extra_body={"thinking_token_budget":16}` (real top-level
  field, NOT `extra_args` which is silently ignored). Assert reasoning ≤ ~budget×1.2 + `</think>` appears.
  **Expect FAIL** on our pinned `qwen3_5_mtp` image (confirms the MTP bug) → go to Step B.
- **Step B:** implement Rank 2 in the offload-proxy behind `LUMO_PROXY_THINK_BUDGET=N`; start **N=1024** (Qwen's
  floor), sweep **512/1024/2048**.
- **Test on 12907** (the sanctioned 1-task proxy gate, serial/concurrency=1): baseline vs N, measure
  (a) mean thinking-tokens/turn, (b) give-up/no-edit rate, (c) resolve. Accept the smallest N that cuts thinking
  + give-up **without dropping resolve**. Commit raw artifacts incrementally; no analysis on the DGX; 12907 is a
  proxy gate not truth.

## Conflicts to respect
- MTP ⊗ native budget (primary) → proxy bound is the only spec-safe enforcement.
- Token-ID mismatch breaks native + logit-bias; proxy string-matching `</think>` sidesteps it.
- `</think>` can leak into `content` (scrub if using native). Empty-block hazard with thinking ON (reproduce the
  `<think>\n…\n</think>\n\n` framing). `stop` ends the WHOLE request unless paired with the two-call split.
- Keep `--reasoning-parser qwen3` — both native + proxy injection rely on its boundary strings.

Sources: vLLM reasoning-outputs docs · PR#20859 · vllm#39573 (MTP) · #39697 (leak) · sglang#25536 (token-ID) ·
neoteric MTP writeup · Qwen3 thinking_budget doc · s1 (arXiv:2501.19393).

---

## VALIDATED + SHIPPED (2026-06-30) — canaried on the live qwen3.6-27b spec/MTP server

**The cap is implemented in the proxy behind `LUMO_PROXY_THINK_BUDGET=N` (default OFF = byte-identical).**
Every assumption below was canaried against `127.0.0.1:9950 /v1/responses`, not assumed.

**What does NOT work in our infra (all canaried-dead):**
- **Native `thinking_token_budget`** — IGNORED (asked 64, got 1500 reasoning tokens to the hard cap). MTP void, confirmed.
- **`enable_thinking=false` / `/no_think` / `reasoning_effort`** — ALL inert via `/v1/responses`. Root cause found in
  source: `ResponsesRequest.build_chat_params` only forwards `reasoning_effort` + `add_generation_prompt` +
  `continue_final_message` to the template — it NEVER passes `enable_thinking`. Our `qwen3-openai-codex.jinja`
  keys its no-think branch on `enable_thinking`, so it's unreachable from the Responses API; and the template
  ignores `reasoning_effort`. So the easy levers cannot fire here.

**What DOES work — `continue_final_message` (Anthropic-style partial-assistant prefill), native in vLLM 0.19.2:**
`should_continue_final_message(input)` returns True when the LAST input item is a reasoning OR
`ResponseOutputMessage` with status `in_progress`/`incomplete`. The cap exploits this:
1. **Call A**: cap `max_output_tokens=N`. A runaway dead-ends as an incomplete reasoning item (no action).
2. **Call B**: re-issue with a partial assistant `ResponseOutputMessage(status=incomplete)` whose text is
   **`"<think>\n" + callA_reasoning + <terse cutoff>`** with **NO `</think>`** — the model GENERATES the close +
   the tool call, so the qwen3 parser (which only watches *generated* tokens) labels it as a message/function_call.
3. **Merge**: present call-A's full reasoning + call-B's action → one well-formed reasoning+action turn.

**Three gotchas that cost canary rounds (don't regress):**
- **Prefill an OPEN `<think>`, never a CLOSED one.** A closed `</think>` in the prefill mislabels the whole
  continuation as *reasoning* (parser never sees the close) → the tool call is LOST.
- **Cutoff must be TERSE/decisive** (`"Okay, I have analyzed enough. I will stop reasoning and act now."`). The
  verbose Qwen official framing ("Considering the limited time by the user…") let the model keep thinking — FAILED.
- **No trailing USER message.** It moves `last_user_index` past call-A's reasoning → the template HIDES it →
  breaks interleaved thinking. The prefill is an *assistant* turn; the nudge lives only inside it.

**Interleaved-safe** (Qwen3.6 reasons before each tool call, prior reasoning kept visible): the cap is per-turn,
never strips the prior chain, and the cutoff is ephemeral (only in the proxy re-issue; codex sees a normal
reasoning+action turn). Validated e2e with the proxy's OWN helpers: a capped over-thinker (N=512) was forced to a
correct text answer (`2413`) AND a correctly-parsed `report_product({"value":2413})` tool call.

**Env (proxy):** `LUMO_PROXY_THINK_BUDGET=N` (per-turn cap; unset=OFF), `LUMO_PROXY_THINK_CUTOFF` (override the
cutoff line). Start **N=8192**: caps the 32768 runaway (the char-8 / reasoning-only-dead-turn failure modes) while
rarely biting a legit per-turn think; tune down for more aggressive capping. NOTE this targets *flavor-2* (single
runaway turn); the *flavor-1* many-short-inspection-turns give-up is separate (operator-prompt territory, kept frozen
for SWE-Verified comparability).
