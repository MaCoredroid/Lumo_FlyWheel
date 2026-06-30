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
