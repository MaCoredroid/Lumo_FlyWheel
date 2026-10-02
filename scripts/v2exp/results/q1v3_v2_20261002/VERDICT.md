# q1v3 v2 — full-model state-equivalence run (2026-10-02, 06:25–07:30 PDT)

**Frozen verdict: INCONCLUSIVE.** R1–R5 pass; R6 (negative-control discrimination) fails: only 1 of 10 negative-control observations exceeds τ.

**Integrity.** All arms are complete and valid: A 33, B 33, V 33, P 33, CAND 44. The new check confirms that every candidate tree forward consumed exactly the forced rows.

**Candidate (descriptive; no candidate cell fails, but R6 prevents an EQUIVALENT verdict):**

| Surface (max over 147 cells) | LumoTree vs A | native P vs A | native B vs A | τ (4 × envelope) |
|---|---|---|---|---|
| next-token KL | 0.49 | 0.49 | 0.16 | 1.96 |
| GDN state rel | 0.30 | 0.28 | 0.24 | 1.05 |
| conv rel | 0.37 | 0.34 | 0.26 | 1.16 |
| KV new rel | 0.35 | 0.40 | 0.32 | 1.59 |
| KV hist rel | 0.34 | 0.33 | 0.24 | 1.32 |

- 0 of 147 candidate cells fail, with no structural failures.
- The largest cycle-0 verify-row KL (on identical imported state) is 0.43, within the native tolerance.
- The candidate repeat is bitwise identical.
- Distances do not grow over cycles 0–6.
- MTP top-1 matches native greedy on 87 of 147 post-commit steps.

**Native noise.**
- A vs B (same configuration, separate boots): bitwise equal on 101 of 147 cells (SPREAD).
- V vs A: bitwise equal on 147 of 147 cells. The v1 pattern was the reverse; native nondeterminism is boot-level.

**Negative controls at their mutation cell (cycle 1), vs the clean candidate at the same cell:**

| Fault | Surface | cal-short (clean → fault) | cal-medium (clean → fault) | τ |
|---|---|---|---|---|
| NC_CONV (conv gather skipped) | conv | 0.21 → 1.21 | 0.20 → 1.06 | 1.16 |
| NC_GDN (replay skipped) | GDN | 0.22 → 0.61 | 0.15 → 0.54 | 1.05 |
| NC_KV (off-by-one remap) | KV new | 0.26 → 1.15 | 0.23 → 1.16 | 1.59 |
| NC_STALE (stale root input) | next KL | 0.075 → 1.66 | 0.048 → 0.78 | 1.96 |
| NC_SIB (sibling rows committed) | conv/GDN/KV | 0.21/0.22/0.26 → 0.61/0.35/0.65 | 0.20/0.15/0.23 → 0.57/0.30/0.77 | — |

Every injected fault is 1.6–22× the clean candidate distance at the same cell. Native P sits at about the clean candidate's distance there; for example, GDN 0.216 vs 0.223 on cal-short. The pre-registered τ (4 × the worst-case native envelope over all calibration cells) is too loose to flag them, so under the frozen rules the comparator cannot certify equivalence. Any per-cell or tighter tolerance would be a new, disclosed, post-hoc protocol version.
