# FR13: anatomy of the codex+SWE-Verified request stream (clean B=1 evidence) — 2026-06-17

EVIDENCE for why per_request_decode_tps (per-request-equal) under-counts cat6's token-throughput, and WHAT the
requests actually are. Source: the 251 cat6root_b1 proxy_pair_dumps in the TRUE B=1 window (created_at
2026-06-17T04:26:06Z..07:11Z, the runner_metadata window; num_running~0). NOT the contaminated offload file.

## Output-length distribution (per codex turn, usage.output_tokens)
| output tokens | <=16 | <=64 | <=128 | <=256 | <=512 |
|---|---|---|---|---|---|
| % of 251 turns | 3.6% | 24.7% | 53.0% | 74.1% | 84.1% |
median = 115 tok, mean = 327 tok (right-skewed: a few long turns pull the mean up).

CORRECTION: an earlier note (deploy_tps_resolved_plus27.md, now fixed) said "39-45% of requests <=64 tok / most
are short." That was WRONG — it came from the coarse vllm:request_generation_tokens histogram buckets (50/75/100ms
class) over a metrics span that likely included warmup. The DIRECT per-turn dump measurement is 24.7% <=64, median
115. "Most requests" are short-to-MEDIUM (53% <=128, 74% <=256), not tiny.

## WHAT the requests are (the codex SWE-Verified agent loop)
Each turn: HUGE input (~11k-14k tokens = the GROWING repo context, re-sent each turn, cached_tokens=0 so NOT
KV-cached across turns) + a SMALL output = reasoning (chain-of-thought) + a function_call. Observed:
- SHORT turns (<=64 tok, 25%): tool call = exec_command (run a shell command: ls/grep/cat/run-tests to explore the
  repo + check results), short reasoning, mostly no user-text message. = 25% of turns but only 3% of OUTPUT TOKENS.
- LONG turns (>256 tok, 26%): exec_command (bigger) + apply_patch (the actual code edits) + long reasoning.
So the SWE-Verified workload = an explore/test LOOP (many short-to-medium tool-call turns) punctuated by a few
long edit/reasoning turns. Spec-decode's per-token benefit grows with output length, so cat6's advantage lands on
the LONG turns; the many short tool-call turns barely benefit.

## The throughput-vs-per-request gap, in plain terms (not "Jensen")
- TOKEN-weighted decode throughput = total_output_tokens / total_decode_time = cat6 23.9 vs E5 18.8 tok/s = +27%.
  This is what the GPU delivers; it's dominated by the long turns (where the tokens are).
- per_request_decode_tps = average over turns of (1 / per-turn TPOT), weighting every turn EQUALLY = cat6 18.51 vs
  E5 17.80 = +4%. The 25% short turns (3% of tokens) get 25% of the weight here, and on them cat6 ~= E5, so they
  pull the average down toward parity.
The two numbers measure different things; both are real. cat6's GPU throughput advantage is +27%.

## OPEN QUESTION (what the instrumented re-run would answer)
WHY is cat6 ~= E5 on the short turns (so they dilute the per-request average)? Two candidates, NOT yet separated:
 (a) the short turns are STRUCTURED tool calls (a shell command) -> high accept for BOTH cat6 and E5 -> cat6's
     extra tree width buys little there (only the harder long-reasoning turns reward it); OR
 (b) a FIXABLE cat6 per-request FIXED overhead (first decode step / tree setup) that dominates a short reply's TPOT
     and drags it to parity, which a kernel/wiring change could recover.
Distinguishing (a) vs (b) = the instrumented re-run: per-turn accept/event + TPOT split by output length (does
cat6's per-turn advantage GROW with length, and is there a fixed first-step cost?). If (a), +27% is the honest
ceiling at B=1 and there's nothing to fix; if (b), there's a real per-request lever. Banked timer design wb81uvy7w
(drafter+committer async timers) supports this. NOTE: input is NOT KV-cached across turns (cached_tokens=0) -> the
11k-token prefill is paid EVERY turn = the dominant deploy cost; prompt-cache reuse is a separate large lever.
