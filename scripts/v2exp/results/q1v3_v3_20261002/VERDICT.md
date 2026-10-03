# q1v3 v3 — full-model continuation correctness (frozen protocol v3; run 2026-10-02 6:06–7:03 PDT)

**Frozen verdict: EQUIVALENT.** All gates pass: R1 integrity, R2 prompt parity, R3 O0 import, R5 native categorical, R6 discrimination. No candidate cell fails, and there are no structural failures.

**Rules, frozen before collection.**
- Each cell (case, cycle) is compared with the native arms at the same cell. The bound is `2.0 × max(B, V, P distance to A, floor)`, with floor 0.02 on state surfaces and 0.15 nats on next-token KL.
- The rule was calibrated on the v2 data. The v3 prefixes are fresh, so this is an out-of-sample test.

**Data.**
- Six fresh prefixes (23k–46k tokens) from four unseen SWE-bench tasks.
- Natural tree paths harvested from the deployed vehicle: real MTP and Arctic drafts, the accepted paths it actually chose, and its pending tokens.
- Each prefix appears three ways: imported O0, unsalted repeat, and cache reuse (`o0_mode=none`).
- 126 cycles, plus a trailing root-only cycle per case that consumes the final pending token.
- Arms A, B, V, P and CAND, all 18/18 valid; CAND 29/29 including the 10 negative controls. Every candidate tree forward consumed exactly the forced rows.

| Surface | Worst candidate ratio (bound 2.0) | Worst absolute distance |
|---|---|---|
| GDN state | 1.08 | 0.436 |
| conv taps | 1.22 | 0.304 |
| KV, new positions | 1.22 | 0.375 |
| KV, history | 1.17 | 0.234 |
| next-token KL | 0.74 | 0.111 nats |

| Negative control (mutation cell, natural depth-2 path) | p1 ratio | p3 ratio |
|---|---|---|
| NC_CONV (conv) | 10.9 | 12.3 |
| NC_GDN (GDN) | 7.1 | 8.0 |
| NC_KV (KV new) | 11.4 | 7.3 |
| NC_STALE (next KL) | 42.2 | 60.3 |
| NC_SIB (conv / GDN / KV) | 8.2 / 3.4 / 8.8 | 6.6 / 4.9 / 8.3 |

The smallest fault ratio is 3.4, against a clean maximum of 1.22.

**Cache reuse was exercised.** Every `Nr` observation in all five arms started with at least 90% of the prefix cached (for example A 22528/23376 and CAND 21504/23376 on p1). The cache-reuse cells pass under the same rule.

**Native reference.**
- B, the same configuration in a new process, is bitwise equal on 34/126 cells.
- V, run at the candidate's KV capacity, is equal on 84/126.
- P, the packed GDN kernel, is equal on 0/126.
- The candidate's within-process repeat is bitwise identical.

**Scope.** This tests that, after LumoTree commits the paths it actually selects, the GDN, convolution and attention state it publishes and the next-forward logits are within the variation between native vLLM implementations at the same point. It covers 7 consecutive cycles, including prefix-cache reuse and consumption of the final pending token. It does not test sampling: the presence-penalty history defect is upstream of the walk and is reported in `sampling_correctness_20261002.md`. Batch size is 1, and MTP state is not compared.
