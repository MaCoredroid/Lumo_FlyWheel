# FR13 — Raising accept from 3.56 to >5 (suffix TAIL + complement + 32-node horizon)

**Status:** DESIGN (2026-07-15). Grounded in the MEASURED per-depth acceptance of cat33333 (below).
Goal: push accept-per-forward from **3.56** (measured) toward and **past 5** — which is impossible inside
the depth-5 tree, so this doc combines a **32-node horizon**, a **suffix tail** past depth-5, **suffix
complement branches**, and **MTP-guided suffix decoding**. Every piece is lossless-by-committer and
never-regress by construction. Companion to FR13_SMART_MATCHING_DRAFTER_DESIGN.md and the Front-2 close
in FR13_CAT6_CAT8_ACCEPT_INVESTIGATION.md.

---

## 1. Where accept is lost — MEASURED, not estimated

cat33333 baseline (MTP-only, merge16d baseline arm, n=26,556 draft events, native
`vllm:spec_decode_num_accepted_tokens_per_pos` metric):

| depth d | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **survival** P(chain reaches d) | 0.972 | 0.839 | 0.695 | 0.575 | 0.479 |
| **conditional** P(hit d \| reached d) | 0.972 | 0.863 | 0.829 | 0.828 | 0.833 |

**accept = Σ survival = 3.56** (max possible 5.0; loss 1.44).

**The decisive finding: the conditional hit rate is FLAT ~0.83 after d0** — it does *not* decay with depth
(my earlier estimate of a decaying head was wrong). So:
- MTP's per-depth quality is *uniform* — at **every** depth, ~17% of the time the true next token is not in
  MTP's {spine top-1 + 2 branches}.
- Accept caps at 3.56 purely by **compounding**: 0.972 × 0.83⁴ ≈ 0.48 survives to d4. Not a weak-deep-head
  problem — a multiplicative-attrition problem.

**Two independent barriers to >5, and we must break BOTH:**
- **Barrier A (attrition):** the ~0.83 conditional caps accept *below* 5 even at infinite depth
  (Σ 0.83^d → 5.9 in the limit, but only ~4.4 by d10). Raising the conditional is §3.
- **Barrier B (geometry):** the depth-5 spine caps accept *at* 5. Exceeding 5 needs spine depth >5 → more
  than 16 nodes → §4 (32-node horizon) + §5 (suffix tail).

---

## 2. The two levers, mapped to the two barriers

| Lever | Attacks | Mechanism | Lossless? |
|---|---|---|---|
| **Suffix COMPLEMENT branches** | Barrier A | add suffix candidates alongside MTP's top-3 at each depth → raise the 0.83 conditional | YES — committer is `accept=p(S)` **monotone** (adding candidates never lowers accept, Gate1 32/32) |
| **Suffix TAIL (depths 6+)** | Barrier B | MTP has only 5 heads; fill depths 6..D with suffix → chain can exceed 5 | YES — committer is source-agnostic; tail nodes are just more candidates on a longer path |

Both are **free on the verify side** *if* the tree fits one forward — which is why §4 (32-node horizon at
the same register budget) is the enabling infrastructure.

---

## 3. Lever 1 — Suffix COMPLEMENT branches (raise the 0.83 conditional)

At each depth ~17% of tokens miss MTP's {top-1 + 2 branches}. Replace/augment one branch slot with a
**suffix candidate** at that depth. Because the committer is monotone+lossless, this can **only raise
accept**; the empirical question is the *complementarity rate*: of the 17% MTP misses, what fraction does
the suffix trie's candidate contain? (**This is exactly the Q2 measurement — task #33.**)

- Arithmetic: if suffix covers fraction *f* of the misses → conditional 0.83 → 0.83 + 0.17·*f*. Even
  0.83→0.86 (f≈0.18) compounds to **accept ≈ 3.56 → 3.7** (+0.14, free).
- **Honest EV (from Front-2 measurements):** arctic's *standalone* deep-accept is ~0.5, and complementarity
  is *stricter* (needs the specific missed token) — so *f* is realistically single-to-low-double-digit %,
  worth **+0.1..0.3** accept. Real but modest. **Not step-change; measure before build (Q2/task #33).**
- Where it helps most: the miss rate is uniform, so complementarity helps at **every** depth (not just
  deep) — but it's most valuable on repetitive/templated spans (identifiers, imports, boilerplate,
  punctuation) that MTP top-k under-ranks.

---

## 4. Enabling infra — 32-node horizon at the SAME register budget (cheap verify)

The 16-node wall is the GDN tree kernel `h_cache=[N_PAD≤16, BV=16, DIM_K=128]` fp32 = 128 KB/CTA (half the
SM register file); N_PAD=32 spills to HBM. **Shrink-BV hypothesis:** BV is BLOCK_V (the value tiling). Halve
**BV 16→8** → the per-node h_cache footprint halves → **N_PAD=32 fits the same 128 KB** (32·8·128·4 =
16·16·128·4). The kernel processes DIM_V in more, smaller tiles (2× the DIM_V loop trips) — extra *compute*,
which is nearly free since decode is HBM-bound.

- **Direction matters:** the prior BV experiment *widened* BV 16→32 and spilled (harmful, refuted). *Shrinking*
  to buy nodes is the untested, opposite direction. [[reference_bv_geometry_refuted_and_harmful]] [[reference_fr13_tree_size_16_register_wall]]
- **Cheap verify:** the verify is ONE forward; if its cost is weight-read-floor-bound (HBM), 32 nodes ≈ 16
  nodes in wall-clock. *(Feasibility + scaling: workflow w9j8r0rbf — §4 to be finalized from its verdict:
  shrink-BV bit-exactness, verify O(nodes) vs O(nodes²), conv per-depth loop count.)*
- **Cost-gates (before build):** (a) BV-shrink **bit-exact** — the GDN scan under the re-tile must be
  byte-identical (same reduction order); gate same-boot vs BV=16. (b) verify wall-clock 32 vs 16 nodes must
  stay ~flat (dfwd timer). If either fails → fall back to a **2-graph pipelined verify** (deeper on a 2nd
  forward only when the first fully accepts — amortized) or stay at 16 nodes with complement-only (§3).

---

## 5. Lever 2 — Suffix TAIL past depth-5 (the only path to >5)

MTP produces 5 tokens; beyond depth-5 there is **no MTP head**. Fill depths 6..D with a **suffix chain**.
The tail only matters when the MTP chain **reaches depth-5** (survival[4] = 0.479, so ~48% of forwards) —
i.e. we're already on a highly-predictable span (the model followed MTP for 5 tokens straight). On such
spans the suffix continuation has its **best** shot (predictable ⇒ repetitive ⇒ trie-covered), which is
*conditionally* far better than arctic's unconditional ~0.5 deep-accept.

- **Accept > 5 comes from here and only here:** on a repetitive span, MTP carries depths 1-5 and the suffix
  tail carries 6, 7, 8… as long as the exact repeat continues. A 12-token verbatim repeat → accept ≈ 12.
- **Lossless by construction:** the committer is source-agnostic; tail nodes are extra candidates on a
  longer path. Cold/novel span → tail nodes don't accept → accept falls back to ≤5 (never-regress).
- **Assembly change:** `fr13_mtp_suffix_assembly.py` hard-codes N_DEPTH=5; extend to N_DEPTH=D (needs the
  32-node room from §4). *(Losslessness + exact change: workflow w9j8r0rbf tail lens.)*
- **Honest EV:** the tail's expected contribution = P(reach d5) × E[suffix run length | predictable span].
  On SWE agentic code the repetitive fraction is modest, so typical gain is **+0.1..0.5**, with occasional
  large spikes (>5) on long verbatim repeats. The *average* may be modest; the *tail of the distribution*
  is where >5 lives.

---

## 6. MTP-GUIDED suffix decoding (make the suffix precise)

Don't let suffix speculate from raw context — **seed the suffix trie walk from MTP's confident prefix**. The
merged drafter already builds `pattern = _COMMITTED[req] + near-MTP-tokens` (fr13_merged_drafter.py), which
conditions arctic on the committed suffix. Strengthen it: seed the walk from the **MTP-drafted spine**
(depths 1..k) so the tail *continues MTP's own confident prediction* rather than a raw context match. This:
- raises the tail's precision (the trie continues exactly where MTP is confident), directly countering the
  Front-2 failure mode (arctic diverging from the model after ~1 token — here MTP anchors the first k),
- is the design's answer to "use the MTP head to guide suffix decoding, then use suffix to add the tail."

Concretely: **mtp_k = 5** (full MTP spine) seeds the suffix walk; suffix extends depths 6+. This is the
*inverse* of the refuted Front-2 config (mtp_k=1, suffix replaces the deep spine) — here MTP keeps its
strong 5 depths and suffix only **adds** beyond, never replaces.

---

## 7. The combined 32-node architecture

```
depth:  1    2    3    4    5    | 6    7    8   ...  (up to ~10-15 within 32 nodes)
source: MTP  MTP  MTP  MTP  MTP  | SUFFIX tail (MTP-guided), fires when chain reached d5
        +suffix-complement branch candidates at each MTP depth (raise the 0.83 conditional)
```
- **Spine 1-5:** MTP (accept ~3.56, unchanged, strong).
- **Branches:** MTP top-3 **∪ suffix candidate** (monotone → raise conditional toward 0.86+ → +0.1-0.3).
- **Tail 6+:** MTP-guided suffix (fires ~48% of forwards; >5 on repetitive spans).
- **Budget:** 32 nodes via shrink-BV (§4). Node allocation between deeper tail vs wider branches is a
  per-step or static tuning (like cat33333's shape choice).
- **Never-regress:** every added candidate/tail node is lossless; cold spans fall back to the 3.56 baseline.

**Target:** typical accept 3.56 → ~4.0-4.5 (complement + short tails), with **>5 on repetitive spans**
(long suffix tails). Speed follows accept directly (more committed tokens per verify forward, at ~flat
verify cost if §4 holds) — a lossless TPS gain, unlike the Front-2 skip (which traded accept for drafter time).

---

## 8. Honest EV & why this differs from the closed Front-2 result

Front-2 closed because arctic **replacing** the MTP spine loses (arctic is a weaker deep drafter). This
design does the **opposite**: MTP keeps all 5 strong depths, and suffix only **adds** (branches + tail) via
the **monotone lossless committer** — so it can **never regress**, only add. The open question is *magnitude*,
not *direction*:
- Complement branches: +0.1-0.3 (gated on Q2/task #33 complementarity rate).
- Suffix tail: +0.1-0.5 average, >5 spikes on repeats (gated on live tail-accept measurement).
- 32-node infra: enables both at ~flat verify (gated on shrink-BV bit-exactness + verify-cost).

Realistic combined: **3.56 → ~4.0-4.5 typical, >5 on the repetitive tail of the distribution.** Whether that
beats the drafter/committer overhead of building a wider tree is the deliverable gate (dfwd-inclusive TPS).

---

## 9. Cost-gate ladder (each before the next)

1. **Q2 complementarity** (task #33, ~free): commit-trace offline join — suffix_covers_miss[d]/mtp_miss[d].
   If ~0 at all depths → complement branches dead; if materially >0 at d≥1 → proceed.
2. **Shrink-BV bit-exactness** (1 GPU boot): re-tile BV 16→8, N_PAD=32; assert GDN scan byte-identical vs
   BV=16 same-boot. If not bit-exact → 2-graph pipelined verify fallback OR complement-only at 16 nodes.
3. **32-node verify cost** (same boot): dfwd 32 vs 16 nodes must stay ~flat (HBM-bound). If ~2× → not cheap.
4. **Tail accept** (1 GPU boot): live per-depth survival for depths 6+ (MTP-guided suffix tail) — does the
   tail actually accept on repetitive spans?
5. **Live B=4 A/B**: 32-node MTP+suffix-tail+complement vs cat33333 baseline — accept, dfwd-inclusive TPS,
   resolve/give-ups/garble. DELIVERY = accept up AND TPS same-or-better AND lossless/garble-clean.

**Losslessness is free throughout** (committer Gate1 + source-agnostic); the risk is entirely *magnitude*
(does accept rise enough to beat the wider-tree overhead) and *shrink-BV bit-exactness*. All gated, all
measured before commit — no repeat of assuming a win.
