# q1v3 v1 — full-model state-equivalence run (2026-10-02, 04:43–05:48 PDT)

**Frozen verdict: INCONCLUSIVE.** Gates R1–R5 pass; R6 (negative-control discrimination) fails: 9 of 10 negative-control observations were not detected.

**Candidate observations are invalid (harness defect, found afterwards).** The engine's drafter runs after every prefill chunk, and the v1 hook indexed forced tree rows by the raw drafter-call count.
- Cold prefixes: the forced rows went into discarded prefill-chunk drafts.
- Prefix-cached cases: the forced rows arrived one cycle early.

The tree forwards therefore verified tokens other than the forced ones, while the harness forced the commit products. A GPU diagnostic (`q1v3diag`, 2026-10-02 06:11 PDT) captured the 32 inputs of every tree forward: 0 of 25 steps carried the forced rows. All the candidate distances below come from this defect, not from LumoTree:
- candidate-vs-native cycle-0 verify-row KL of 2–27 nats;
- K/V, convolution and GDN relative errors of about 1–1.6.

v1 checked only the root input, so the defect went undetected. The negative controls ran on the same broken forcing and are invalid too.

**Valid parts of v1 (native arms only):**

| Pair | Result |
|---|---|
| A vs B (same configuration, separate process) | bitwise identical on all 147 cells |
| V vs A (candidate's KV capacity) | identical on 104/147 cells; elsewhere state rel ≤ 0.28, next-token KL ≤ 0.25 |
| P vs A (stock packed GDN decode kernel) | state rel 0.2–0.4 (max over layers), next-token KL ≤ 0.27 |

- Root-only candidate trees (no forced drafts involved): KL ≤ 0.13, state rel 0.1–0.2, about the same distance from native as P.

The resulting tolerances τ (4 × native envelope) are 1.0–1.6 on state surfaces. Two unrelated vectors of equal norm sit at about √2 ≈ 1.41, so the state-surface negative controls may stay undetectable even with correct forcing.

Fix, regression test and v2 amendment: `scripts/v2exp/q1v3/PROTOCOL.md` (v2 section).
