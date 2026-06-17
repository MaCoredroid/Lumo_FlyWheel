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

cat6 is faster on BOTH (TPOT percentiles: cat6 faster at every percentile, no slow tail). The +4%-vs-+27% gap is
a WEIGHTING difference: per_request_decode_tps averages 1/TPOT over turns EQUALLY, while the aggregate weights by
tokens. Clean B=1 evidence (codex_swe_request_anatomy.md): ~25% of the 251 codex turns are <=64 output tokens
(median 115), and those <=64 turns are only 3% of output tokens but get 25% of the per-request-equal weight; they
are codex exec_command tool calls (explore/test) where spec-decode helps little, so cat6 ~= E5 on them. The
token-weighted aggregate captures cat6's win on the LONG turns (where the tokens are). So cat6's GPU genuinely
produces decode tokens +27% faster; the +4% under-counts it. (Earlier "39-45% <=64 tok" was from the coarse
bracket histogram and is superseded by the per-turn dumps.)

## VERDICT CORRECTION
The merged b1_depth5 verdict headlined per_request_decode_tps (cat6 18.51 = +4%). The CORRECT decode-throughput
advantage at true B=1 is **+27%** (TOKEN-weighted = total_output_tokens / total_decode_time, E5-comparable since
num_running~0). THROUGHPUT IS TOKEN-WEIGHTED BY DEFINITION; per_request_decode_tps (per-request-EQUAL, +4%) answers
a different question (is the average REQUEST faster, weighting a 5-tok tool-call == a 500-tok edit) = a per-request
LATENCY notion, NOT throughput. So the deploy-throughput headline should be the +27% token-weighted number; the
+4% was simply the wrong basis for "throughput". cat6 = +27%, no remaining question on the metric.

## Honest caveat (end-to-end)
The SWE-Verified deploy is PREFILL+AGENT-heavy (~11k-token prompts, ~260-token replies), so DECODE is a small
fraction of each request's wall. cat6's +27% DECODE shrinks total task time by less than 27% (≈ decode-fraction x
27%). Realizing more of cat6's GPU advantage end-to-end needs B>1 (overlap one stream's prefill/agent idle with
another's decode) — a deployment lever, not a kernel change. At B=1, +27% decode throughput is the deserved,
honest number.
