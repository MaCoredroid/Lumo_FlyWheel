# FR13 — Is the high agent-nudge/give-up rate a Qwen3+Codex+vLLM misconfig? (online research, 2026-06-30)

**Verdict: NO — our core serving config is correct on every load-bearing axis. The give-up/nudge rate is NOT
a parser/template/sampling misconfiguration.** The two real, fixable levers are (a) `reasoning_effort="high"`
(should be medium/low for open models — high over-thinks and stalls) and (b) char-8 = mechanical *truncation*
surfaced as a 400, not a model give-up. (Workflow `wf_8cab6c5d`, 5 agents, online + our-config.)

## Config vs best-practice — what's RIGHT (✅), DELIBERATE (⚠️), WRONG (❌)
| Axis | best-practice | ours | |
|---|---|---|---|
| reasoning-parser | `qwen3` | `qwen3` | ✅ |
| tool-call-parser | `qwen3_xml` (robuster than `qwen3_coder`) | `qwen3_xml` | ✅ |
| tool_choice | `auto` (not `required` w/ thinking) | `auto` (`--enable-auto-tool-choice`) | ✅ |
| sampling temp/top_p/top_k/min_p | 0.6 / 0.95 / 20 / 0 | 0.6 / 0.95 / 20 / 0 | ✅ exact thinking preset |
| max_output_tokens | 32768 (thinking) | 32768 | ✅ |
| wire_api | `responses` | `responses` | ✅ |
| chat-template (multi-turn thinking) | strip prior `<think>` | **KEEP reasoning visible** | ⚠️ deliberate — our interleaved-thinking fix; strip-on-`last_user` caused ~100% attempt-1 give-up under Codex nudges. Correct for our harness. |
| presence_penalty | 0.0 (thinking row) | **1.0** | ⚠️ above rec; anti-runaway, shifts argmax (gated off for lossless A/B) |
| reasoning summaries | avoid unless server supports | force-send (`auto`) | ⚠️ tolerable — template consumes `message.reasoning`; verify no 400 |
| **reasoning_effort** | **low/medium** for open models | **`high`** | ❌ **mismatch — highest-value fix** |

## Per-root-cause verdict (why it's NOT a serving misconfig)
- **Tool-parser mismatch** → NO (qwen3_xml + matching XML template, internally consistent).
- **Thinking `<think>` never closes** → NO (deterministic open, `qwen3` parser detects close; one lane).
- **Reasoning-content round-trip loss across turns** → **WAS the culprit, ALREADY FIXED** by our preserve-
  reasoning template (the generic strip boundary let Codex's nudges reset the window → blind re-derive → give-up).
- **Wrong sampling** → NO (exact thinking preset).
- **`reasoning_effort="high"` over-thinking → stall/runaway** → **YES, most likely remaining cause.** Matches our
  observed "endless-reasoning grinds to ~80000 tok / ~83min" runaway the token cap exists to bound.
- **char-8 truncation → 400** → **YES, partly mechanical** (not a model give-up).

## char-8 = a KNOWN truncation bug (not a model-quality bug)
The model is still streaming a tool-call `arguments` JSON string (big diff / file contents) when generation hits
the length cap → **unterminated string** ("opens a string, never closes / char-8") → the parser throws on
incomplete JSON → 400 → our `RETRY_UPSTREAM_400`/auto-continue fires. Known fixes, priority: (1) raise the token
budget so args don't truncate mid-string [ollama#14570]; (2) check `finish_reason`; if `length` → continue, don't
hard-fail; (3) use `qwen3_xml` not `qwen3_coder` (**we already do** ✅); (4) avoid `tool_choice:"required"` w/
thinking (**we use `auto`** ✅). Watch: qwen3_xml multi-`<function>` invalid-JSON [vllm#43713], single-quote args.

## Ranked fixes (all cheap-testable on 12907, codex-side flag flips, no vLLM restart for 1/2/4)
1. **`reasoning_effort: high → medium`** (highest value, lowest risk). Set in `run_swe_bench_q36_a.py:72`,
   `launch_v4a_v2_round_4_alloff.py:52`, `run_live_codex_long_task.py:296`. Expect: less over-think, fewer
   runaways/truncations, lower nudge rate, faster turns. Test: 12907 high-vs-medium → give-up count, tok/turn,
   auto-continue fires.
2. **Classify `finish_reason:length` as continue-not-give-up** in the proxy (separate mechanical truncation from
   genuine give-up; stops the "you have enough context, edit NOW" nudge from firing on a mid-write turn).
3. **A/B `presence_penalty 1.0 → 0.0`** (Qwen thinking-row value) once Fix 1 reduces over-thinking.
4. **Verify reasoning-summary metadata doesn't 400** against the proxy; else `model_supports_reasoning_summaries=false`.
5. (if char-8 persists) raise per-request `max_tokens` headroom / continue-on-length for big `apply_patch` args.

Sources: Qwen vLLM deployment docs, vLLM reasoning/tool-calling docs, Qwen3 cards, Codex config-reference,
vllm#22975/#27118/#37079/#19051/#43713, ollama#14570, codex#13009/#2510.
