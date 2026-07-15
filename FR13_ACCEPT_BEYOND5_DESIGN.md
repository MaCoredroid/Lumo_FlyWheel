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
- **CONFIRMED feasible (workflow w9j8r0rbf, cited):** `h_cache=[N_SPAN, BLOCK_V, DIM_K]` fp32
  (fr10_gdn_tree_kernel.py:721). N_PAD=16,BV=16 = 128 KiB; **N_PAD=32,BV=8 = 131072 B = byte-identical
  footprint**; N_PAD=32,BV=16 = 256 KiB = the spill. BV is a pure DIM_V *tiling* constexpr tiled by the
  launch grid `(num_vh, cdiv(dim_v, BV))` (:2005) — **no inner DIM_V loop**, so BV=8 just **doubles grid.y**
  (cdiv(128,8)=16). And a knob is **already wired**: `FR13_TREE_GDN_GEOM_OVERRIDE="BV=8"` (:1972-2005),
  value-neutral → GATE 1 is runnable **today** with zero deploy change.
- **Bit-exact risk (adversarial):** every reduction is over DIM_K (axis=1, :616/:619) or N_SPAN (axis=0,
  :741-744), **never over BV**, so the [16,128]→[8,128] tile-shape change leaves per-channel math untouched.
  BUT the kernel itself flags (:577-581) that Triton reduction-tree codegen identity is *not* spec-guaranteed
  under a shape change, and the prior D16=D32=0.0 A/B was a *widen*, not a *shrink*. → **GATE 1** below.
  `num_warps` MUST stay pinned at 8 (w8→4 measured harmful = spill).
- **Verify is ~FLAT, not 2×** (HBM weight-read-floor ~98.6 ms, weights load once regardless of 16 vs 32
  tree tokens): GDN scan runs on a **node-independent grid**; extra nodes pile into the same CTAs as an
  O(N) `static_range` loop (:722) with an O(N²) ancestor-gather (small base, ~8× ALU 16→32); tree-attn
  intra-tree block is O(tree_n²) but tiny (256→1024); the fused conv is **one pass with zero extra
  per-depth launches**. Net: node-dependent work is sub-few-percent of the weight floor → verify stays flat.
  Confirm with GATE 3.
- **Re-tile change (bounded, mechanical, cited):** set BV=8 (:33 or the override); lift the three hard
  `n_pad>16` raises (:185, :1160, :1629) + patch:236; add ONE 32-node `NODE_FAMILY` entry (:26) + make_tree
  topology + parent-tuple + Triton warmup (a 32-node tree does not exist yet — families cap at 14).
- **Fallback if GATE 1/3 fail:** 2-graph pipelined verify (deeper on a 2nd forward only when the first fully
  accepts — amortized), or stay at 16 nodes with complement-only (§3).

---

## 5. Lever 2 — Suffix TAIL past depth-5 (the only path to >5)

MTP produces 5 tokens; beyond depth-5 there is **no MTP head**. Fill depths 6..D with a **suffix chain**.
The tail only matters when the MTP chain **reaches depth-5** (survival[4] = 0.479, so ~48% of forwards) —
i.e. we're already on a highly-predictable span (the model followed MTP for 5 tokens straight). On such
spans the suffix continuation has its **best** shot (predictable ⇒ repetitive ⇒ trie-covered), which is
*conditionally* far better than arctic's unconditional ~0.5 deep-accept.

- **Accept > 5 comes from here and only here:** on a repetitive span, MTP carries depths 1-5 and the suffix
  tail carries 6, 7, 8… as long as the exact repeat continues. A 12-token verbatim repeat → accept ≈ 12.
- **Lossless by construction — NO committer change (workflow w9j8r0rbf, cited):** the committer walks every
  leaf's root→leaf path, scores LCP vs parent_targets, keeps the longest, appends the bonus, truncates to
  `max_spec_len+1` (fr13_gpu_committer_kernel.py:124-159) — **nothing in the accept logic reads depth or
  source**. A tail node past depth-5 is lossless the instant it's a verified node (its parent_targets come
  from the same generic parent[]-built verify mask, patch:224-248), and `max_spec_len = num_speculative_tokens`
  ≫ any single-path depth, so the cap doesn't bite. Cold/novel span → tail nodes don't accept → falls back
  to ≤5 (never-regress).
- **Code changes (all OUTSIDE accept logic):** generalize `N_DEPTH=5` in the 3 drafter modules
  (merged_fill.py:26, merged_drafter.py:31, mtp_suffix_assembly.py:21); drop the 15-node
  `assert len(nodes)==len(CAT33333_ORDER)` and PAD-fill the tail's empty branch slots (merged_fill.py:79-90,
  PAD is Gate1-lossless p[pad]~0); break the n_pad>16 wall via §4's BV re-tile; widen `assert mtp_k in (1,2)`
  (assembly:69) if seeding from more MTP heads.
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

## 6b. PRE-WARMED harness-aware trie (fixes the Front-2 cold/task-local weakness — biggest EV lever)

Front-2's suffix decoder was weak in large part because the arctic trie is **COLD at every task start** and
only ever learns *this task's* repetition (task-local) — so early tokens have no coverage and cross-task
structure is invisible. But the SWE agentic harness is **highly structured and repeats ACROSS tasks**:
tool-call XML (`<tool_call>…</tool_call>`), the qwen-code system prompt + system reminders, JSON arg
formats, diff/patch headers, file paths, ubiquitous imports (`import numpy`, `from astropy…`), edit-echoes,
and boilerplate. **Pre-warm the trie with a hand-picked corpus of prior completed trajectories** — selected
for high *cross-task structural* frequency — so the suffix decoder is strong on the harness-repetitive
fraction **from token 1**.

- **Mechanism (uses arctic's EXISTING cross-request cache):** arctic `SuffixDecodingCache` keeps a
  cross-request corpus (`cached_requests`); pre-load it at boot via `add_active_response` over the selected
  prior-trajectory token sequences. The per-request trie then **unions** the pre-warm (harness/cross-task
  patterns) + the live request suffix (task-local repetition). No new arctic code — just a boot-time seed.
- **Why it's the biggest EV lever:** the suffix TAIL (§5) fires on repetitive spans, and harness-structural
  spans are a **large, predictable fraction** of agentic decode — a cold task-local trie misses them until
  they recur *within* the task, but a pre-warmed trie covers them **deeply and immediately**. This turns
  ">5 as a rare windfall" into ">5 as a **regularity on harness spans**," and also strengthens the §3
  complement branches on those spans → lifts the whole EV (§8) toward a *higher average*, not just spikes.
- **Corpus selection (harness-aware, bounded):** mine prior completed SWE trajectories already on disk
  (proxy request-dumps `/tmp/lumo_proxy_request_dumps`, commit-traces, per-task outputs) → extract
  high-frequency n-grams, dedupe, **weight toward cross-task structural tokens** (tool-call scaffolding,
  system reminders, common imports/idioms) over task-specific content. Keep it under the trie's memory cap
  (a few MB of the highest-frequency patterns; arctic evicts by policy).
- **Never-regress preserved:** pre-warm only ADDS candidates through the monotone committer — a pre-warm
  pattern that doesn't match the live model output simply doesn't accept. Lossless throughout.
- **Measurement (new gate):** suffix-tail + complement accept **cold vs pre-warmed** on the same SWE tasks —
  does the pre-warm raise the tail hit-rate / repetitive-coverage? This is the cheapest high-EV experiment
  and can be run *independently* of the 32-node infra (pre-warm helps the 16-node complement too).

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

**Quantified honest EV (workflow w9j8r0rbf, adversarial):** on GENERIC decode, accept>5 is NOT reachable.
With conditional 0.83 held forever the infinite-depth ceiling is only ~5.9 (and 0.83 *will* decay); with the
tail sourced from arctic (deep-accept ~0.5) the geometric tail past d5 sums to 0.479·0.5/(1−0.5) ≈ **+0.48**
→ a 32-node deep build lands at **accept ≈ 4.0 on generic prose** — short of 5. **>5 is a BIMODAL,
workload-conditional windfall**: only on repetitive/boilerplate spans where the suffix trie has long *exact*
matches (per-token accept ~0.95+), the depth-6..10 tail adds ~2.0 → that span's accept ≈ **7-9**. Blending:
to *average* >5 you need ~27% of steps on repetitive spans (5 = 4.0·(1−f) + 7.6·f → f≈0.27). For agentic
**code** (repeated identifiers, imports, boilerplate, edit-echoes) that fraction is plausible; for prose it
is not.

**So budget the *cold-trie* build as: accept 3.56 → ~4.0 baseline everywhere, with >5 as a repetitive-span
windfall — NOT a flat average.** BUT the **pre-warmed harness-aware trie (§6b) is the lever that moves the
average**: it makes the *harness-structural* fraction (tool calls, boilerplate, system reminders, common
imports — a large slice of agentic decode) trie-covered *from token 1*, so those spans hit the >5 tail
*regularly* rather than only after intra-task recurrence. That raises the effective repetitive fraction
toward/past the ~27% needed to average >5. The go/no-go is GATE 4 (tail accept) measuring the real
repetitive fraction **cold vs pre-warmed**, plus whether the accept-per-verify lift beats the wider-tree
overhead (dfwd-inclusive TPS).

---

## 9. Cost-gate ladder (workflow-refined, cheapest-first — commit nothing until each passes)

**STEP 0 / GATE 1 — BV-shrink bit-exact (runnable TODAY, zero deploy risk).** Same-boot **BV=8 vs BV=16 byte
A/B at n_pad=16** via `FR13_TREE_GDN_GEOM_OVERRIDE="BV=8"` (already wired, value-neutral) — require **0.0** +
`ptxas -v`/nsight regs-per-thread confirming no spill. If nonzero → Triton reduction-tree identity broke
under the shrink → **shrink-BV dead, 32-node path blocked** (fall back to §4 2-graph verify or complement-only).
*This is the first thing to run and it needs no new topology.*

**GATE 2 — register/occupancy at N_PAD=32.** Actual N_PAD=32,BV=8 launch + `ptxas -v`/nsight proves spill-free
(h_cache 128 regs + the *larger unrolled scan's* other live registers < 255). The 128 KiB arithmetic alone
does NOT guarantee this — the bigger `static_range` scan grows other live pressure.

**GATE 3 — 32-node verify cost.** 32-node verify-forward A/B vs the ~98.6 ms weight floor confirms the scan
(~8× ALU on a small base) + doubled-CTA increment stays **sub-few-percent (flat)**, not 2×.

**GATE 4 — tail accept, live.** Per-depth survival for depths 6+ (MTP-guided suffix tail) on a REAL agentic
**code** workload @ temp 0.6, depth-matched. This measures the repetitive-span fraction — where ~all the gain
past 3.56 lives. Go/no-go: does accept actually rise enough to matter?

**GATE 0.5 — PRE-WARM effect (§6b, pivotal, 16-node, NO BV work):** live A/B of suffix-complement (16-node
cat33333 + one suffix branch slot) **cold trie vs harness-aware pre-warmed trie** on real SWE tasks —
does the pre-warm raise the suffix accept/coverage on harness-structural spans? This is the **cheapest test
of whether the suffix source is strong enough to justify the 32-node tail build at all** — if pre-warm
doesn't lift it, the whole suffix direction stays weak (Front-2 territory) and we stop; if it does, the
tail build is justified. Run early, independent of the BV work.

**Q2 (task #33, parallel/free) — complement-branch complementarity:** commit-trace offline join
suffix_covers_miss[d]/mtp_miss[d]. Gates §3 (branches) independently of the tail; run alongside GATE 0.5.

**GATE 5 — live B=4 A/B (deliverable):** 32-node MTP-spine + MTP-guided suffix-tail + suffix-complement
branches vs cat33333 baseline — accept, **dfwd-inclusive TPS**, resolve/give-ups/garble. DELIVERY = accept up
AND TPS same-or-better AND lossless + garble-clean.

**Build order (each behind its gate):** STEP 0 override A/B (GATE 1) → lift n_pad>16 + BV=8 + one 32-node
`NODE_FAMILY`+topology+warmup (GATE 2/3) → generalize `N_DEPTH` in the 3 drafter modules + drop the 15-node
assert + PAD-fill tail branch slots (committer unchanged) → wire the arctic tail seeded by the existing
`pattern=_COMMITTED[req]+near-MTP` (GATE 4) → live A/B (GATE 5). **Losslessness is free throughout** (committer
Gate1, source-agnostic, depth-blind); the risks are entirely *shrink-BV bit-exactness* (GATE 1) and
*magnitude* (GATE 4/5) — both measured before commit. No assuming a win.
