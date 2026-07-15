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

## §6b UPDATE (2026-07-15, GATE 0.5 MEASURED): pre-warm = NO measurable benefit (refuted as top lever)
Clean same-config A/B (pw1, merged, 4 real SWE tasks, only diff = pre-warm sidecar; 132/132 seeded confirmed):
PREWARM accept 1.902 / fullstep 15.65 / resolve 3/4 / match_full ~19.6% == COLD accept 1.896 / 15.66 / 3/4 /
~20.0%. IDENTICAL. Coverage did NOT lift. ROOT: the PER-REQUEST arctic trie already self-warms from the
task's OWN repetition (tool-calls/imports recur within the task), so a thin cross-task pre-warm is redundant;
its only unique value is the early-task cold-start, a small fraction of aggregate decode. => §6b DOWNGRADED
from "biggest EV lever" to "modest/negligible at this corpus". A richer corpus (long real generations) MIGHT
help more but the flat COVERAGE says the addressable gap is small. The TAIL (§5, 32-node) is the real >5
mechanism and does NOT depend on pre-warm. Pre-warm machinery kept (lossless, gated OFF) for a later richer-
corpus retest; not on the critical path.

## GATE 2 UPDATE (2026-07-15, IN PROGRESS): n_pad=32 live-boot (BV=8, 25-node MTP-native tree)
Boot test = `scripts/fr13_gate_32node_boot.sh` (25-node wide depth-5 tree `7-6-5-4-3` -> n=26 -> n_pad=32;
`FR13_TREE_GDN_GEOM_OVERRIDE=BV=8`, plain qwen3_5_mtp spec so MTP-native tree fill isolates the KERNEL, not
the merged drafter). **Config-parse cap lift VALIDATED LIVE**: `num_speculative_tokens=25` + 25-node tree spec
parsed with NO `NotImplementedError` (patch:236 raised 16->32); model loaded 82%+ shards; SpeculativeConfig
accepted. **Attempt 1 CRASHED at warmup/graph-capture (~6.5min, container exited on its own — NOT oom_guard,
NOT the 900s cap)** but the engine log was discarded by the boot script's own container-rm before I read it.
FIX (committed): boot script now streams `docker logs -f` to `$RUN/engine.log` so the crash cause survives
removal. Re-run in flight (run_20260715T152303Z, monitor bggdut877). OPEN QUESTION the re-run answers:
register spill (BV=8 h_cache=[32,8,128]fp32=131072B is byte-identical to [16,16,128]=131072B, so spill would
be from an n_pad-scaled index/accum array, NOT h_cache -> needs a different fix) vs graph-capture OOM
(gpu_util tuning: lower 0.8->0.7 or fewer cudagraph sizes -> NOT a fundamental block). Verdict gates the whole
horizon-expansion. If PASS -> build tail (below); if register-spill -> localize the n_pad-scaled consumer.

### GATE 2 ATTEMPT 2 (2026-07-15, crash cause = SOFTWARE CAP, not a spill; committed fix)
Captured engine log pinned the crash: `fr10_gdn_tree_kernel.py:1862 ValueError: n_pad must be a power of two
<=16, got 32` in `launch_tree_gdn_prepared` -- a SECOND cap (the runtime kernel launcher), deeper than the
patch:236 config-parse cap I'd already lifted. **The kernel never launched -> the register question was still
unanswered.** Found FOUR identical `n_pad>16` guards: `padded_nodes:185`, `launch_tree_gdn_replay:1161`,
`launch_tree_gdn_replay_all_layers:1630`, `launch_tree_gdn_prepared:1862`. Kernel-register audit: only the
FORWARD `_tree_gdn_kernel` (via prepared) carries `h_cache=[N_PAD,BV,DIM_K]fp32` (the wall); BOTH replay
kernels carry NO h_cache (one `[BLOCK_V,DIM_K]` tile), so they are n_pad-independent register-wise. FIX
(committed): replay caps + padded_nodes lifted to <=32 unconditionally; the prepared cap lifted to <=32 GATED
on effective BV<=8 (reads FR13_TREE_GDN_GEOM_OVERRIDE -> [32,8,128]==[16,16,128] byte-identical; BV=16 fails
loud, no silent HBM spill). Also confirmed the patcher ring/descriptor alloc (patch:434-534) is already
n_pad-parameterized (scales to 32 automatically) and `_FR13_N_FIXED=16` only bites under FR13_NPAD_INVARIANT
(OFF here). Re-boot in flight (run_20260715T153701Z, monitor b1s1z6ygb) -> now the kernel ACTUALLY launches at
n_pad=32/BV=8, so this run finally measures the register budget (the true GATE 2). Deployed cat9 path
(n_pad=16, no override) is byte-identical -- every guard still allows <=16 exactly as before.

### ✅ GATE 2 PASS (2026-07-15, run_20260715T153701Z) — 32-node horizon infra VIABLE
The n_pad=32/BV=8 forward kernel launched **spill-free** through BOTH memory profiling (the exact step that
crashed on the cap) AND cudagraph capture (PIECEWISE 8/8 + FULL-decode 4/4, "Graph capturing finished in 5
secs, took 0.35 GiB"); server healthy; **zero spill/register signals** ("(none)"). Generation smoke on
`def fibonacci(n):` returned COHERENT code (`if n==0: return 0 ... else: return fibonacci`) -> a mini
end-to-end garble sanity on n_pad=32 tree decode also passed. **=> the shrink-BV thesis is confirmed live: the
32-node register budget is byte-identical (BV=8) and the register wall is NOT a blocker.** GATE 1 (bit-exact
BV8-vs-BV16) and GATE 3 (verify-cost flat) will be measured on the deliverable merged-tail server; the smoke's
coherent output is a preliminary GATE-3 sanity (n_pad=32 verify decodes at normal speed, no catastrophic tax).
Note: boot health window extended 900->1500s (n_pad=32 warmup ~13min, was right-censoring at 900s).
**NEXT: build the depth-6+ arctic TAIL (the only path to >5) on the now-unblocked 32-node horizon.**

## LADDER STEP 1 MEASURED (2026-07-15, ctrace1 live SWE, n=25,080 node-records — NOT a probe)
Ran `fr13_analyze_branch_topp.py` on the legacy-walk commit-trace (real SWE task, argmax_prob emitted):
- **argmax_prob >0.9 at 94.9% of nodes** (0.7-0.9: 2.4%, 0.5-0.7: 1.7%, <0.5: 1.0%). The agentic-coding
  token stream is EXTREMELY peaked/near-deterministic. => strongest signal FOR the TAIL: a near-deterministic
  stream means the arctic suffix trie predicts depth-6/7/8+ tokens with high accuracy on repetitive spans ->
  tail accepts -> the >5 windfall. (Magnitude still = GATE 4, not proven here.)
- **Where the model's argmax falls among the 3 tree children: spine 84.5%, BRANCH 10.2%, missed 5.3%.**
  Conditioned on confidence: at argmax_prob>0.9 branch STILL catches 8.9% (spine 87.1%, missed 4.0%); at
  0.7-0.9 branch 31.8%; at 0.5-0.7 branch 36.1%. **This REFINES/partly-refutes the design's "high-p branches
  are wasted -> reallocate to tail" claim**: argmax_prob is the MODEL's confidence, not the MTP DRAFTER's
  accuracy -- the spine draft (rank0) is wrong ~9% even when the model is confident, and the branch rescues it.
  Branches are NOT free to reallocate.
- **overlap_mass>0.9 at 93.5%; missed only 5.3%** -- the 3 candidates already cover the nucleus; a WIDER tree
  has limited upside (the 5.3% miss is usually a genuinely-novel token, not a rank-4/5 the model nearly took).
- Per-depth 1..5: branch coverage FLAT 9.3/11.4/9.8/10.6/10.1%, missed 3.4->6.4% (rises slightly with depth)
  -- consistent with the Q1 ~0.83 conditional hit.

**DESIGN CONSEQUENCE (honest):** favor **pure ADDITION** for the 32-node horizon -- keep the 15-node depth-5
tree (branches intact, they earn ~10%) and ADD ~17 tail nodes at depth 6+. The "top-p-adaptive shrink branches
to fund the tail" reallocation is DOWNGRADED to a secondary optimization: only worth it if the 32-node budget
is exhausted AND GATE 4 shows the displaced tail out-earns the ~9% high-p branch coverage it costs. Never-regress
still holds (committer is source/depth-blind); this only reprioritizes WHICH nodes fill the extra budget.

### ✅ TAIL ASSEMBLY CORE DONE (2026-07-15, CPU-validated, committed)
`fr13_mtp_suffix_assembly.assemble_tail_tree` + `tail_tree_order` built: cat33333 head (depths 0-4, pure MTP
at mtp_k=5 == baseline byte-identical) + deep Arctic-filled spine CHAIN (16 nodes, depths 6-21) = 31 nodes ->
n_pad=32. PURE chain tail (no branches) per ladder-step-1 (94.9% peaked -> deep chain maximizes reach/node).
`tail_tree_order` is ONE topology source (drives spec + packer + assembly). NEVER-REGRESS proven in CPU tests
(`fr13_tail_tree_test.py`): mtp_k=5 cold -> head==baseline cat33333, tail=pad(repeat) that never matches past
the head -> accept==baseline; warm -> 16 arctic chain tokens; partial -> fills then pads. Arctic adapter
already tail-ready (`max_rel` parameterized -> caller passes max_rel=head_depth-mtp_k+tail_len=16).
### ✅ FULL TAIL WIRING DONE (2026-07-15, seam map agent a4516b7c) — GATE 4 boot in flight
Seam map finding: the packer/injection/attn-bias side is 100% topology-agnostic (auto-derives from
SPEC_CONFIG tree_choices: parent[], masks, _fr10_wide_plan, n_pad); the ONLY hard tripwire is the wide
packer's `len(_fr10_spine_tokens) == _fr10_wide_D` check (must return wide_D=21 spine tensors). Built:
- `build_tail_columns` (fill.py): 16 pure-chain tail tensors (rk==0, no wide_topk), OOB bounds-guard.
- `decide_tail` (drafter.py): head=100% native MTP (byte-identical); produces ONLY the Arctic chain
  (pattern=_COMMITTED[req]+native head; max_spec_tokens=32). tail_on() sidecar + TAIL_HEAD_DEPTH=5.
- Patcher (3 edits in the injected EAGLE-propose string, validated to compile): cap native spine_steps at
  head_depth-1=4 (else ~20 slow autoregressive MTP forwards); DISABLE the incompatible mtp_k=1 skip seam
  in tail mode; after the head loop, append decide_tail's 16 tensors (spine_tokens 5->21 == wide_D).
- Launcher: /logs/fr13_tail_mode.arm on FR13_TAIL_MODE=1 (worker strips FR13_* env). Needle now logs
  TAIL[fired hit cold] to ASSERT engagement before trusting any accept number.
- SPEC: tail_tree_order() -> 31 nodes (n_pad=32, wide_D=21); num_speculative_tokens=31.
NOTE: pad token = row-0 root (scalar, cross-row for cold rows) -> lossless (committer verifies; a wrong
pad never commits). Refine to per-row pad only if it costs measurable accept.
GATE 4 boot: scripts/fr13_gate_tail_boot.sh (run_20260715T161934Z, monitor bbm05q9rc) -- boots the tail
tree in TAIL+merged mode, drives a repetitive workload (arctic self-warms), verifies TAIL[fired>0 hit>0],
reads per-position accept (pos 6-20 non-zero => the tail accepts => >5 signal). Then the real GATE 4/5 =
live B=4 SWE-Verified A/B (accept UP AND dfwd-TPS same-or-better AND lossless/garble-clean).

### GATE 4 ATTEMPT 1 (depth-21) = graph-capture STALL, NOT a crash (2026-07-15, run_20260715T161934Z)
The 31-node depth-21 tail tree booted CLEANLY through model-load + conv-fusion (path_cols=32, width=4,
state_len=34, NO n_pad/packer/spill error) and reached KV-cache sizing (217,120 tokens) -- BUT profiling took
12min (vs GATE-2's ~6min) and then it STALLED entering graph capture: last engine line 16:36:25, then ZERO
"Capturing CUDA graph" lines for 8min -> killed at the 1800s window. No Traceback. So: correctness path fine,
but the very deep chain is pathologically slow/hung at graph capture. HYPOTHESIS (under test): depth-driven
(the tree spine is a depth-21 sequential dependency; even though the GDN kernel loops over n_pad=32 not depth,
graph-capturing a depth-21 chain -- or the tail-mode drafter path during profiling -- blows up). CONFOUNDS
still open: depth(21) vs node-count(31) vs tail-mode-drafter vs chain-vs-fan. **PIVOT (honest): the tail does
NOT need depth-21 to clear 5 -- a depth-8..11 tail already exceeds 5 (accept up to 8-11 on repetitive spans).**
Re-boot at tail_len=6 (depth-11 tree, 21 nodes, still n_pad=32; run_20260715T164344Z, monitor bu8qx0x9p w/
stall detector). If depth-11 boots -> depth/node-count was it -> shallow tail works -> measure accept. If it
STILL stalls -> not depth magnitude -> isolate tail-mode-drafter (boot topology-only) or chain-capture (eager).
This is a SPEED/capture cost, NOT a losslessness issue (never-regress holds regardless of tail depth).

### GATE 4 ATTEMPT 2 (depth-11) = BOOTS + GRAPH-CAPTURES CLEAN; tail-append bug found (2026-07-15, run_20260715T164344Z)
depth-11 tail (tail_len=6, 21 nodes, n_pad=32) profiled in 4.3min (vs depth-21's 12min -- cost scales steeply
with tree size) and **graph-captured + became HEALTHY** -> the shallower tree operationally works (BV=8 + n_pad=32
+ deep chain + tail mode all boot). BUT the first LIVE completion raised `RuntimeError: FR13_RESHAPE_WIDE:
collected 5 spine tokens, expected D=11` (packer tripwire): the tail-append block SILENTLY SKIPPED (its
try/except:pass swallowed the reason), so _fr10_spine_tokens stayed 5 (head only) while the wide packer required
wide_D=11. TWO flaws: (1) NO graceful fallback -- tail mode serves a wide_D tree so a skipped tail MUST crash
(need pad-to-wide_D); (2) silent skip violates fail-loud. Likely cause: _LUMO_FA_SPEC_ROW_REQ_IDS stale/None
AFTER the head loop's MTP forwards (the merged skip seam reads it BEFORE the loop). FIX (in flight, workflow
w574b0v8i mapping req-id lifecycle + red-teaming): capture req_ids at the valid pre-loop point + pad-to-wide_D
(lossless: repeated deep token ~never matches past head) + log the skip reason.

### USER REDIRECT (2026-07-15): measure on REAL SWE-Verified tasks, NOT the synthetic repetitive probe.
The fr13_gate_tail_boot.sh embedded config-dict probe is a PROXY; drop it. After the crash fix, run the tail
tree on the REAL SWE-Verified gate (fr13_b4_campaign_driver.sh, B=4, temp 0.6) as a tail arm vs a baseline
cat33333 arm -> GATE 4/5 = accept UP AND dfwd-TPS same-or-better AND lossless/garble-clean, on live agentic
coding tasks (where the repetitive-span windfall actually lives). Workflow w574b0v8i is mapping the b4 driver's
arm/env structure + which harness is the real SWE-Verified gate.

### TAIL-APPEND FIX + REAL-SWE WIRING (2026-07-15, workflow w574b0v8i mapped + red-teamed)
Root cause (req-id agent, DEFINITIVE): NOT stale req_ids -- _LUMO_FA_SPEC_ROW_REQ_IDS has ONE writer
(_prepare_inputs, once/outer-forward) and is NEVER reset, so it's valid post-head-loop. The real bug is the
MISSING FALLBACK: tail caps spine_steps=4 but serves wide_D, so ANY guard-skip/exc left _fr10_spine_tokens at
5 -> packer crash (merged mode survives the identical miss by falling through to full MTP; tail cannot). Likely
live skip = merged_off (needs BOTH sidecars) OR ids_ne_B (mixed prefill/decode batch: _LUMO_FA_SPEC_ROW_REQ_IDS
is the spec-row SUBSET vs the drafter row count). FIX (red-team FINDING 1, make-or-break): pad _fr10_spine_tokens
to wide_D by repeating the last [batch] spine tensor OUTSIDE the try + explicit skip reasons + throttled
fail-loud log. Red-team: NONE of 5 concerns break losslessness (committer is a per-row target-verified backstop);
scalar row-0 pad is lossless + magnitude-neutral (FINDING 3); head accept is WITHIN-FLOOR not exact vs the
15-node baseline (n_pad 16->32 ~1-ULP drift, FINDING 2 -- don't gate on exact head-accept equality).
Real-SWE wiring: added `tail6` arm to fr13_bigdenom_swe_serve_variant.sh (TAIL6_TREE 21 nodes, XFLAGS=
FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8, EXPECT_RATIO=21) + fr13_tail_g4_seq.sh.
GATE 4 LAUNCHED: real SWE-Verified B=4 via fr13_b4_campaign_driver.sh (SUBSET=subset_b4_four, monitor bin0rrcpj)
-> reads TAIL[fired/hit] engagement + [FR13_TAIL] skip reasons + accept_per_event + derived_tps_gpu from
deploy_speed_tailg4.json. If it engages + accept>5 + TPS ok -> GATE 5 full A/B (tail6 vs t33333 baseline).

### Scoped tail-build (post-GATE-2, each edit committed behind the gate)
The arctic substrate is ALREADY deep-capable: `fr13_merged_drafter.py:get_cache(max_tree_depth=24)` holds
up-to-24-deep committed patterns. Missing pieces (all depth-5-locked today):
1. `fr13_merged_fill.py`: `N_DEPTH=5` + `for d in range(N_DEPTH)` device column-builder -> generalize to tree depth.
2. `fr13_mtp_suffix_assembly.py`: `assert len(mtp_spine)>=N_DEPTH`; fills branch slots WITHIN depth-5 only
   (complement) -> **no depth-6+ tail node machinery yet = the missing >5 piece**. Add tail nodes fed by arctic
   (MTP has no head past depth-5; arctic walk `pattern=_COMMITTED[req]+near-MTP` produces depth-6+ spine).
3. `fr13_merged_drafter.py`: `N_DEPTH=5`, `need=N_DEPTH-mtp_k` -> generalize; extend CAT33333_ORDER node map to
   the wider/deeper 32-node topology; PAD-fill any unfilled tail slots (Gate-1 lossless).
Losslessness stays free (committer is source/depth-blind); only magnitude (GATE 4/5) is open.
