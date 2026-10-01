# Distribution diagnostic protocol v2 (frozen 2026-10-01 before reading any candidate distribution data)

Supersedes analyze_dist.py (v1), which retokenized parsed text without an END category, aggregated
dependent positions with Fisher's method, and had a same-seed null replicate. v1 is not used for claims.

Purpose: a sampling-level diagnostic that LumoTree's served outputs are consistent with the target
model's sampled outputs at the deployed settings. It is NOT full-model qualification, NOT a proof of
joint/conditional distribution preservation, and NOT a correctness certificate.

Collection (dist_sample.py): 20 fixed recorded agent requests (every 2nd of the sorted 43-request corpus),
40 continuations each, max_tokens 24, temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0 / presence 1.0,
no per-request seed, streaming chat endpoint, identical request bodies across arms.

Arms (all vLLM, same image, reasoning parser qwen3, tool parser qwen3_xml, same chat template):
- REF  : plain decoding, server seed 0  (dist/ar-sampled-20261001T150739Z, collected before this protocol)
- NULL : plain decoding, server seed 1  (independent noise-floor replicate)
- NEG  : plain decoding, server seed 2, temperature 0.65 (planted small distribution shift; sensitivity control)
- CAND : LumoTree deployed route (hydra27 fixed32, split-K), server seed as deployed
- SEC  : vLLM MTP-5, server seed 0 (secondary candidate)
- SGLang EAGLE arms are descriptive only (different parsers; 146-token prompt-count difference unresolved).

Analysis (analyze_dist_v2.py): observable = first 16 tokens of the parsed output stream (fields joined by
U+001F, model tokenizer), END-padded. T = sum over requests and positions of total-variation distance between
the two arms' marginals. Null by permuting whole continuations within request (1000 permutations, fixed RNG).
Report T, null mean/sd, excess = T - null mean, p_perm. One global test per comparison; no per-position claims.

Decision rules (fixed now):
1. Validity: NULL vs REF must have p_perm >= 0.01. Otherwise the diagnostic is invalid and nothing is concluded.
2. Sensitivity: NEG vs REF must have p_perm < 0.01. Otherwise the diagnostic lacks power for a 0.6->0.65
   temperature shift and candidate results are reported as uninformative.
3. CAND is "consistent with REF at this resolution" iff p_perm >= 0.01 AND excess <= 0.25 x excess(NEG vs REF).
   Otherwise report "inconsistent" or "inconclusive" (p >= 0.01 but excess above margin). SEC is judged the same way.
All runs, including failed or superseded ones, are retained.
