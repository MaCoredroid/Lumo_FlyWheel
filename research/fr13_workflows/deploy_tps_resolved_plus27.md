# FR13: cat6 deploy decode-throughput is +27%, NOT +4% — the "+4%" was a diluted metric (2026-06-17)

User: "deploy TPS cat6 should get what it deserved -- not 4% definitely." Investigated on the CLEAN true-B=1
data (committed brackets). RESOLVED: cat6's real decode-throughput advantage is +27%; the +4% headline was a
per-request-equal metric diluted by short requests. NO kernel throttle.

## Two unit errors corrected along the way (both caught by the user)
1. "+28ms cat6-SLOWER" = a DERIVATION artifact (committed/per_request_decode_tps). DIRECT measure
   (request_decode_time/draft): cat6 step 201.9ms < E5 218.8ms — cat6 is SHORTER, forward equal.
2. "realized ~4 tok/s throttle" = WRONG: inter_token_latency is recorded PER ITERATION (ITL_count = drafts:
   E5 28105, cat6 20990), so 1/ITL ~= 4 is 4 STEPS/sec, not 4 tok/s. No throttle.

## The two TPS bases (clean B=1, gen/decode-time)
| basis | cat6 | E5 | cat6 vs E5 | what it is |
|---|---|---|---|---|
| per_request_decode_tps = 1/mean(per-request TPOT) | 18.51 | 17.80 | **+4%** | per-request EQUAL-weighted; the merged headline |
| derived/aggregate = gen_tok / request_decode_time | 23.88 | 18.80 | **+27%** | TOKEN-weighted decode throughput; B=1-comparable |

cat6 is faster on BOTH (TPOT percentiles: cat6 p25=50ms/p99=75ms vs E5 p25=75ms/p99=100ms — faster at every
percentile, no slow tail). The +4%-vs-+27% gap is JENSEN: per_request_decode_tps weights every request equally,
and **39% (E5) / 45% (cat6) of requests are <=64 tokens** where spec-decode barely helps (fixed first-step
overhead dominates a short reply) -> cat6 ~= E5 there. The token-weighted aggregate ignores those and captures
cat6's win on the long replies, where the extra accepted tokens accumulate. So cat6's GPU genuinely produces
decode tokens +27% faster; the +4% under-counts it because of the short-reply-heavy workload, NOT a cat6 defect.

## VERDICT CORRECTION
The merged b1_depth5 verdict headlined per_request_decode_tps (cat6 18.51 = +4%). The CORRECT decode-throughput
advantage at true B=1 is **+27%** (derived/aggregate gen/decode-time, E5-comparable since num_running~0). cat6
deserves the +27% decode credit. NO throttle/kernel fix exists — cat6 already delivers +27%.

## Honest caveat (end-to-end)
The SWE-Verified deploy is PREFILL+AGENT-heavy (~11k-token prompts, ~260-token replies), so DECODE is a small
fraction of each request's wall. cat6's +27% DECODE shrinks total task time by less than 27% (≈ decode-fraction x
27%). Realizing more of cat6's GPU advantage end-to-end needs B>1 (overlap one stream's prefill/agent idle with
another's decode) — a deployment lever, not a kernel change. At B=1, +27% decode throughput is the deserved,
honest number.
