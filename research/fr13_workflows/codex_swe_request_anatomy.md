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

## Use TOKEN-weighting for throughput (not per-request)
Throughput = tokens per second is TOKEN-weighted by definition:
- TOKEN-weighted decode throughput = total_output_tokens / total_decode_time = cat6 23.9 vs E5 18.8 tok/s = **+27%**.
  This is the right "deploy decode throughput" number; it's where the tokens (and cat6's accepted-token win) are.
- per_request_decode_tps = average over turns of (1/per-turn TPOT), weighting every turn EQUALLY = cat6 18.51 vs
  E5 17.80 = +4%. That weights a 5-tok exec_command turn the same as a 500-tok edit -> a per-request LATENCY notion,
  NOT throughput. The 25% <=64-tok tool-call turns (3% of tokens) get 25% of this weight, dragging it to parity.
So the metric to report/optimize for deploy throughput is the TOKEN-weighted +27%; per_request_decode_tps is the
wrong basis for the throughput question. No open metric question remains.

## Honest observation (a SEPARATE, deployment question — NOT pursued here)
cat6's +27% is DECODE throughput. End-to-end on SWE-Verified the deploy is prefill-dominated: each turn re-sends
~11k-14k input tokens, and prefix caching is OFF (cache_config_info enable_prefix_caching=False — the vLLM DEFAULT
for this hybrid GDN/mamba model; prefix_cache_queries_total=0, prompt_tokens_cached_total=0 of 1.6M prompt tokens),
so the full ~11k prefix is re-prefilled EVERY turn even though the agent's context is append-only. That makes
PREFILL the dominant per-turn cost, so cat6's +27% decode shrinks total task wall by less than 27%. This is a
DEPLOYMENT/serving question (prefix-cache reuse, or B>1), DISTINCT from the cat6 speed question and NOT investigated
here (default prefix caching is off for hybrid GDN for correctness reasons; enabling it would need separate
verification it doesn't break the GDN recurrent state / tree spec-decode). Recorded for honesty; cat6's kernel is
already +27% on decode throughput regardless.
