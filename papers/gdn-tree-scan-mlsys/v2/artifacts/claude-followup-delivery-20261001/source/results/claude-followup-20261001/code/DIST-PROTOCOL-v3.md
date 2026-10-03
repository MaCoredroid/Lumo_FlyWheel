# Distribution diagnostic protocol v3 (frozen 2026-10-01 before any v3 collection)

Why v3: v2 (DIST-PROTOCOL.md) was invalid under its own rule 1 — two plain-decoding boots (seed 0, seed 1)
differed (p=0.001), at a magnitude larger than a 0.6->0.65 temperature shift. The likely source is boot-to-boot
numerical variation (KV block count follows free memory at boot). The deployed LumoTree route enforces an unset
KV size and server seed 0 at batch 1, so this variation cannot be pinned for the candidate without changing the
deployed configuration. v3 therefore measures each arm against the observed boot-to-boot spread of plain decoding.

Collection (same request bodies as v2: 20 requests, 40 continuations, 24 tokens, temp 0.6/top_p 0.95/top_k 20/
min_p 0/presence 1.0, no per-request seed), order = request-outer (all samples of one request consecutively),
every boot fresh, same images and parsers as v2:
- AR   x4: plain decoding, server seeds 21, 22, 23, 24
- TREE x3: deployed LumoTree route (server seed fixed at 0 by the route)
- MTP5 x2: vLLM MTP-5, server seeds 31, 32 (positive control: exact rejection sampling)
- NEG  x1: plain decoding, server seed 41, temperature 0.65 (sensitivity control)
No v1/v2 runs are reused.

Statistic per pair of boots: excess = T - mean(T under within-request continuation permutation), with T as in v2
(sum over requests and first 16 parsed-stream token positions of total-variation distance, END-padded).

Decision rules (fixed now):
1. E_AA = max excess over the 6 AR-AR boot pairs (boot-to-boot noise scale).
2. Sensitivity: mean over AR boots of excess(NEG, AR) must exceed E_AA; otherwise every verdict is "uninformative".
3. For an arm X with boots x_k: m_k = mean over AR boots of excess(x_k, AR).
   consistent   iff all m_k <= E_AA;
   inconsistent iff any m_k > mean excess(NEG, AR);
   inconclusive otherwise.
4. Positive control: if MTP-5 is not "consistent", the LumoTree verdict is reported as inconclusive (rule too strict).
p-values are reported but not used for decisions. The diagnostic concerns parsed-stream token marginals only;
it is not full-model qualification or a joint-distribution proof. All runs are retained.
