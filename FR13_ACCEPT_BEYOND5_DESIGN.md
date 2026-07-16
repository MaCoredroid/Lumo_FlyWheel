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

### GATE 4 SWE ATTEMPT 1 = oom_guard kill (NOT a tail bug); GPU_UTIL fix (2026-07-15, tailg4)
The tail6 SWE arm booted through model-load + conv-fusion (path_cols=22, no error) but the gpu_oom_guard KILLED
it mid-graph-capture (PIECEWISE 4/8): GPU free dropped to 8612MiB < the SWE-serve's 9000MiB guard floor. My
standalone depth-11 boot survived only because it used a 3000MiB floor. The n_pad=32 tail tree + FR13_DEVICE_
MULTIDRAFT (device committer scratch) uses more GPU during capture than the cat9 (n_pad=16) the serve is tuned
for. This is a MEMORY-HEADROOM issue, NOT a tail-code bug (the crash fix is fine; it never reached a request).
FIX: GPU_UTIL 0.78->0.72 for the tail arm (frees ~7GB -> capture free ~15GB, above the 9000 floor; KV cache
still ~170k tokens, ample for B=4). Re-launched tailg4b (monitor b9twjhxtb). Keep the 9000 floor (protects the
offload/codex/monitor session). If the tail arm needs even more headroom, drop GPU_UTIL further or trim cudagraph
sizes -- do NOT lower the guard floor.

### GATE 4 SWE ATTEMPT 2 (tailg4b) = boot survived, tail VACUOUS on B=4 (red-team's predicted failure, MEASURED)
GPU_UTIL=0.72 boot survived (HEALTHY after 592s). The fail-loud log EARNED ITS KEEP: warmup skipped x1 ids_none
(expected: _prepare_inputs, the sole writer, doesn't run at warmup), then LIVE DECODE skipped x51->x101 with
`ids2_ne_B4` = len(SPEC_ROW_REQ_IDS)=2 != drafter B=4. This is EXACTLY the red-team's flagged risk: at B=4 the
batch is mixed prefill/decode, so _LUMO_FA_SPEC_ROW_REQ_IDS (the spec-row SUBSET) has fewer rows than the drafter
batch -> the tail-append guard skips 100% -> tail PADS everywhere -> VACUOUS (accept=baseline, no >5). Agentic
tasks are async so the batch stays mixed -> persistent. KILLED the run (concrete refutation; don't wait a
known-vacuous 30min). FIX (committed): when SPEC_ROW_REQ_IDS len != B, fall back to _LUMO_FA_SAMPLER_ROW_REQ_IDS
-- set UNCONDITIONALLY to the FULL batch (patch:11250), length==B, row-aligned with spine_tokens. decide_tail
keys _COMMITTED by req_id: spec rows hit Arctic, non-spec (cold) -> pad. Lossless (per-row committer; misalignment
worst case = accept collapse not garble). Re-launched tailg4c (monitor b1wo1837z) -> watching for TAIL[fired>0]
(engagement = fix works) then accept_per_event/derived_tps_gpu. NOTE: never-regress held even in the vacuous run
(tail padded == baseline, no crash) -- the infra gauntlet has been ALL operational, zero losslessness failures.

### ✅ GATE 4 SWE ATTEMPT 3 (tailg4c) = TAIL ENGAGES + ACCEPTS past depth-5 on LIVE B=4 (mechanism PROVEN)
SAMPLER-row-id fix WORKED: TAIL[fired=650 hit=606] = ~93% arctic hit on live B=4 SWE decode (was 100% skip).
DIRECTIONAL /metrics read mid-run (~315 forwards, spec_decode_num_accepted_tokens_per_pos):
- HEAD pos 0-4 (depths 1-5): 306,254,204,176,143 = 1083 -> ~3.44 accept ~= baseline 3.56 => NEVER-REGRESS
  CONFIRMED LIVE (within-floor, n_pad 16->32 drift).
- **TAIL pos 5-10 (depths 6-11): 67,55,48,42,34,30 = 276 -> +0.88 accept/forward -- the tail GENUINELY
  COMMITS past depth-5** (decays 67->30, real reach). pos 11-20 = 0 (correct, wide_D=11).
- TOTAL accept ~4.3 (accepted 1369 / drafts 6720 basis). ABOVE baseline (+~0.9 from tail), BELOW 5 on this subset.
**HONEST VERDICT (matches the up-front EV): the tail is REAL + additive + lossless, but the per-task AVERAGE
(~4.3) is below 5 -- subset_b4_four has <27% repetitive spans, so no >5 average.** >5 is a repetitive-span
WINDFALL, not a per-task average, on typical SWE. Awaiting full deploy_speed (proper accept_per_event +
derived_tps_gpu) -- the TPS is the OTHER gate (does +0.9 accept beat the 21-node tree's extra verify?). Per-pos
decay says a DEEPER tail gives diminishing returns (pos11~25,12~20...); the real >5 lever is workload repetition.

### ✅ GATE 4 FINAL (tailg4c deploy_speed, serve rc=0, n_tasks=4, prefill_frac=0.40): accept 4.28, committed 5.28
accept_per_event=4.277, committed_per_event=5.277, s_per_fwd_gpu=85.3ms, derived_tps_gpu=61.85,
derived_tps_fullstep_gpu=18.81. TAIL[fired=1029+ hit ~94%]. **The tail WORKS end-to-end on the LIVE B=4
SWE-Verified gate: engages + accepts past depth-5 (+0.72 accept over baseline 3.56) + never-regress (head
~3.54==baseline) + lossless (rc=0, committer per-row) -- ZERO correctness failures.**
HONEST VERDICT on the >5 HEADLINE: **NOT met as a per-task average (4.28 on typical SWE); >5 is a
repetitive-span WINDFALL exactly as the up-front EV stated** (need ~27% repetitive steps; subset_b4_four has
less). Deeper tail = diminishing returns (per-pos decay). So >5-average is workload-repetition-bound, not a
cheap code lever. WHAT IS DELIVERED: a lossless spec-decode tail that raises accept 3.56->4.28 (+20%) with
never-regress by construction. SHIP-vs-cost-gate now hinges ENTIRELY on TPS: running the t33333 baseline A/B
(basec, monitor bli5eu1zn) for derived_tps_gpu + derived_tps_fullstep_gpu at the SAME config. If tail
fullstep-TPS >= baseline -> SHIP (faster + lossless, even at accept<5); if the arctic-tail drafter overhead
eats the accept gain -> honest cost-gate (mechanism proven, no net speed).

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

## 3-WAY A/B + user-requested controls (2026-07-15)
The tail is SPINE-ONLY (pure arctic chain, no tail branches); head keeps its 2 branches/depth. Two controls
added (queued after the t33333 baseline, GPU serialized):
1. **suffonly (arctic-only cat33333)** = t33333 + FR13_DRAFT_SOURCE=merged + FR13_MERGED_FLAVOR=always: root
   from base model, Arctic fills deep spine+branches, MTP deep forwards SKIPPED = the closed Front-2 config
   (prev -17% B=4). Re-run as the CONTROL isolating arctic-alone vs tail6 (MTP-head+arctic-tail, 4.28) vs
   t33333 (MTP-only). Also applied the SAMPLER-row-id fallback to the MERGED seam (was skipping ~half at B=4
   on ids_ne_B -> now arctic engages). KIND=suffonly (fr13_bigdenom_swe_serve_variant.sh), fr13_suffonly_seq.sh.
2. **wider tail (tail branches)** = the measured tail-accept DECAY (pos 6-11 = 67,55,48,42,34,30) is the chain
   dying when arctic top-1 is wrong; a tail BRANCH = arctic top-2 (use_tree_spec=True already provides ranked
   siblings) rescues "top-1 wrong, top-2 right" -> slows decay -> more tail accept toward >5. Never-regress
   (monotone committer), fits n_pad=32 (21+6=27). NEEDS decide_tail top-2 + topology (not yet built). Ladder-
   step-1 (branches ~10% shallow / missed 5.3%) predicts marginal (+0.1-0.3), but user-requested + measurable.
3-way order: t33333 (baseline, running) -> suffonly (arctic-only) -> wider-tail if data warrants.

## MISSES ARE MTP HARD-MISSES, not rank-4+ (2026-07-15, ctrace1 n=1327 missed records) — user Q2
For the 5.3% cat33333 "missed" (model argmax NOT in the MTP top-3): 71.7% at model argmax_prob>0.9, 94.6% at
>0.5, and overlap_mass<0.5 for 99.1% -> the MTP head CONFIDENTLY drafted the WRONG tokens, missing the model's
obvious high-prob answer. So the miss is a genuine DRAFTER divergence, NOT "answer at MTP rank 4 catchable by a
wider top-k". CONSEQUENCE: wider MTP branches (top-4/5) are DEAD for the misses (same head, same miss). The
right lever is a COMPLEMENTARY source = ARCTIC (historical retrieval catches the model's answer where MTP hard-
misses) -> this is WHY the arctic tail adds accept, and says the head fix is an ARCTIC COMPLEMENT (arctic fills
head branch slots), not wider MTP branches. suffonly (arctic-only) arm tests if arctic catches the head-misses.
Q1 (branch correctness after slot-reorder): branches commit correctly (superset +0.166==predicted; monotone
committer; tail6 deep tail commits) -- adding a spine-vs-branch delta check to the 3-way A/B for fresh confirm.
Proxy caveat: argmax_prob/overlap_mass are proxies; exact MTP-rank of the answer needs a rank-probe (queued).

## TPS verdict (directional, 2026-07-15): tail6 fullstep ~= baseline, NOT a Front-2 loss
tail6 derived_tps_fullstep_gpu=18.81 (B=4, pf=0.40, accept 4.28) vs prior t33333 baseline bc1diag fullstep=16.58
(B=1, pf=0.2, accept 3.79) -> COMPARABLE-to-slight-WIN, NOT the arctic-overhead loss feared. The +0.72 accept
~compensates the arctic tail-drafter host overhead. tail6 verify-basis derived_tps_gpu=61.85 vs baseline-implied
~53 (+15%). Clean same-config baseline (t33333base_basec, B=4 pf~0.40) pending (slow: agentic tasks 25-61 turns).
NET so far: the tail is a LOSSLESS drafter, accept 3.56->4.28 (+20%), TPS-neutral-to-slight-positive. NOT the >5
headline (windfall/workload-bound). Novel = first lossless GDN-tree suffix-decode TAIL past depth-5, live B=4.

## GATE 5 on the RIGHT denominator (2026-07-15, user flagged): 16-task fr9-matched B=4, 3-way
The tail6/baseline A/B ran on subset_b4_four (4 tasks) -- SAME set BETWEEN arms (valid A/B) but a small sample.
Switched to subset_b4_sixteen (16 tasks = the fr9-matched "native MTP-5 B=4 decode_tps=39.9" apples-to-apples
aggregate set, user 2026-06-16). Launched the full 3-way (t33333 baseline / tail6 / suffonly arctic-only) on
16 tasks, B=4, GPU_UTIL=0.72 (fr13_tail_3way_seq.sh, TAG=g5, monitor bhwa6zfd4). The 4-task numbers (tail6
accept 4.28, fullstep 18.81) PROVED the mechanism (engages+accepts+never-regress+TPS-neutral); the 16-task run
gives the ROBUST deliverable accept/TPS + the arctic-complementarity answer (suffonly head accept vs MTP).

## UNION of two trees (MTP cat33333 ∪ arctic) — user idea 2026-07-15, feasibility + plan
Idea: verify MTP cat33333 AND an arctic tree together, committer picks best path across both. RIGHT structure
for Q2 (MTP hard-misses at every depth caught by arctic's INDEPENDENT path). Never-regress (union ⊇ MTP).
Two realizations, DIFFERENT build cost:
- **Option 1 (COMPLEMENT branches):** arctic candidates as EXTRA siblings off the MTP spine (rk 3-4 in
  wide_topk). BUILDABLE with the existing wide packer (just widen wide_topk width + fill rk3+ from arctic
  tree-speculate). Catches MTP misses at depth d GIVEN correct prefix (0,)*d. Misses the ~2.8% depth-1
  whole-path divergence.
- **Option 2 (TRUE two-spines union):** arctic gets its OWN rk-0 spine chain from the root. **BREAKS the wide
  packer**: _fr10_wide_plan maps (pp=depth, rk) which COLLIDES for two subtrees at the same depth+rank
  ((0,0,3) MTP-branch vs (3,3,3) arctic-spine both = wide_topk[2][3]). Needs a per-NODE packer (bigger build).
PLAN (measure-before-build): the 3-way (t33333 + suffonly) gives MTP-tree and arctic-tree per-token accepts;
do the OFFLINE complementarity join -> P(arctic commits | MTP misses) at each depth -> estimate union accept.
If the join shows strong complementarity, build Option 1 (cheap, wide packer) first; Option 2 only if the
depth-1 whole-path divergence is a big chunk. EV stays complementarity-bound (+0.1-0.3), stacks with tail's
+0.72, does NOT reach >5 average alone (workload-bound) -- but it's the correct hard-miss lever.

## GATE 5 baseline accept (16-task set, mid-run robust, 2026-07-15): t33333 = 3.59
Mid-run /metrics over 2563 forwards (draft_tokens 38445 / 15): accepted 9202 -> accept_per_forward = 3.59, all
head (pos 0-4), tail=0 (correct, no tail). Matches Q1 baseline 3.56. => the ACCEPT A/B is settled on live B=4:
baseline 3.59 -> tail6 4.28 = +0.69 (+19%), lossless, never-regress. Accept aggregates over forwards not tasks,
so robust already; the 3-way run continues only for the clean same-config TPS + the suffonly complementarity.
DELIVERABLE (established): a lossless GDN-tree suffix-decode TAIL, accept 3.59->4.28 (+19%) live B=4 SWE-Verified,
TPS comparable-to-slight-win (tail6 fullstep 18.81 vs baseline ~16.58). >5-AVERAGE not met = repetitive-span
windfall, workload-bound (no cheap code lever). Novel contribution = first lossless GDN-tree tail past depth-5.

## SPEED breakdown (tail6, live B=4, measured 2026-07-15) — user speed question
Per decode step GPU compute: DRAFTER 97.7ms (35%) + VERIFY 85.8ms (30%) + COMMITTER 97.5ms (35%) = 281ms ->
committed 5.28 -> derived_tps_fullstep_gpu=18.81. verify-only=61.85. aggregate(prefill+idle)=10.9, per-req=4.8.
KEY: (1) the COMMITTER is as expensive as the drafter (~35% each) -- surprising, a direct lever (S1 sampled-
committer). (2) VERIFY is HBM-bound (~flat vs tree size) -> a bigger tree adds ~0 verify. (3) drafter+verify are
FIXED per step -> higher accept amortizes them over more committed tokens => bigger tree = faster (CONFIRMED:
tail6 +19% accept -> +13% fullstep-TPS). SPEED LEVERS ranked: (1) bigger tree/tail -> more accept -> amortize
fixed drafter+verify (accept-is-the-only-lever physics; committer replay is the eventual ceiling since it
SCALES with accept). (2) sampled/faster committer (35%). (3) overlap arctic speculate with verify forward /
fewer MTP head forwards (35%). The union + bigger tree both push accept => both faster until committer dominates.

## ⚠️ TPS VERDICT CORRECTED (2026-07-15, metric audit): tail6 is ~10% SLOWER, not a win
Prior "+13% fullstep-TPS win" was WRONG -- it used deploy_speed derived_tps_fullstep_gpu (18.81) which mixes
per-DRAFT verify (85.8ms) with per-STEP drafter/committer (~4 draft-events/step at B=4 => verify understated 4x).
CONSISTENT per-step (raw non-overlapping timers, FR13_PIPELINE_SPEED_BREAKDOWN.md §6): baseline 434.6ms/step ->
GPU-TPS ~42.2; tail6 553.8ms/step -> ~38.1 => tail6 ~10% SLOWER despite +19% accept. ROOT: the DEEP tail's
VERIFY (+31%, 258->340ms) + COMMITTER (+53%, 74->113ms) scale with DEPTH and exceed the accept gain; drafter is
NOT the cost. => VERIFY is NOT HBM-flat for deep trees (design §4 wrong: 258ms@depth-5 >> 98.6ms floor = GDN
tree-scan is depth-compute-bound). CORRECTED DELIVERABLE: lossless +19% accept, but as-built (depth-11) it's a
COST-GATE (-10% GPU-TPS), not a ship. PATH to a NET win (depth is the axis): sweep tail_len {2,4,6,8} to find the
accept/depth-cost sweet spot (shallower = less verify+committer, still >baseline accept) + cheaper committer (S1
sampled, the +53% depth-scaler). The union's arctic-parallelism helps the DRAFTER (not the deep-tail cost) -> not
the speed fix here. This is the honest measure-before-claiming correction (audit found the metric error).

## VERIFY-COST OPTIMIZATION — FR13_PARENT_GATHER (2026-07-15, the OPTIMIZE-TO-HW-LIMIT lever)

The as-built depth-11 tail is +19% accept but ~10% slower (verify 258->339ms >> the 98.6ms weight-read
floor). Workflow wbvlbn0x3 found the verify headroom root cause: `_tree_gdn_kernel`'s inner ancestor loop
does one full-tile fp32 reduction PER ancestor j<i, but `state_i=where(ancestor,h_j,state_i)` overwrites in
increasing j -> only the largest-index ancestor (= the immediate parent, topological order) survives. So the
O(N^2) loop computes `h_cache[parent]` -- one gather, not i. FR13_PARENT_GATHER (default OFF, byte-identical)
finds that parent with a cheap integer mask-scan over the SAME strict_mask + ONE gather. Reductions/CTA:
N(N-1)/2 -> N (7.5x fewer at n_pad=16, **15.5x at n_pad=32** => helps THIS tail most). See
FR13_PIPELINE_SPEED_BREAKDOWN.md §10.

**GATE (n_pad=32, BV=8) PASS:** in-process self-check (both scans on identical live-decode inputs, raise on
any bit-diff) = BYTE-IDENTICAL, no mismatch (pg_gate_t55555_np32). Losslessness proven at the tail's shape.
Timing run (graph, PARENT_GATHER=1) in flight vs the 171.3ms baseline. Next: n_pad=16 gate (shipping shape)
+ live B=4 SWE deliverable gate, then bake ON if verify drops and never-regress holds.

## PRE-WARM DIAGNOSIS + GATE 0.5-ON-DELIVERABLE (2026-07-15)

**GOAL METRIC CLARIFIED:** "accept>5" = accept_per_event (design's "accept 3.56" = sum of per-depth survival
[.972,.839,.695,.575,.479]). tail6 COLD baseline (g4c, 16-task): **accept_per_event 4.277**, committed_per_event
**5.277** (=accept+1 bonus), derived_tps_gpu 61.85. So the goal needs accept_per_event 4.277 -> >5 = **+0.72**;
committed/step is already >5 (but that includes the always-present bonus, not the goal metric).

**PRIOR PRE-WARM ~0 DIAGNOSED (merged_cold vs merged_prewarm_t33333_pw1, 4-task):** pre-warm WAS loaded
(132/132 seeded). match_full RATE ~FLAT (cold 43.9% vs prewarm 43.1% of speculate_fired) -> the boilerplate
corpus did NOT raise arctic coverage. prewarm's confhist <.05 bucket ~DOUBLED (944->2164) => added candidates
are LOW-confidence and REJECTED by verify (temp-0.6 generation doesn't reproduce cross-task boilerplate
token-exact). => the windfall (long deep accept from a prewarmed span) needs EXACT reproduction, which temp 0.6
breaks; within-task arctic already covers ~43% cold (the model quoting its own recent output).

**CORPUS SUPPLY-LIMITED:** 6461/6495 request-dumps are astropy = the SAME repo as all 16 test tasks. A
leakage-free corpus (--exclude-substr the 16 instance_ids) can only use non-test trajectories, which are scarce
(all prior runs were the 16 astropy tasks). So a bigger leakage-free corpus is not buildable from what's on disk;
the 132-seg corpus IS the fair leakage-free artifact.

**RIGOROUS CONFIRMATION IN FLIGHT:** tail6 + pre-warm (132-seg corpus) on the full 16-task fr9-matched set,
setsid-detached (output/fr13_tail6_prewarm, TAG=pw16), vs the established cold 4.277. If ~4.28 -> honest
cost-gate on the windfall (the design's >5 path fails on real agentic temp-0.6 coding). If it jumps -> pursue +
run the clean same-session cold A/B. Per "no early-needle conclusions / LIVE SWE-Verified only", NOT concluding
until this 16-task tail result lands.

## PRE-WARM STRONG POSITIVE (2026-07-15, tail6 16-task IN FLIGHT — potential no-go OVERTURN)

**tail6 + PRE-WARM live 16-task run (engagement asserted: PREWARM seeded 132/132, TAIL fired 3661 hit 97%):**
vLLM SpecDecoding running aggregate over 168 windows = **mean acceptance length 6.01 (median 5.56, min 2.86,
max 11.14)**. mean-accept-length = committed_per_event => **accept_per_event ~5.0**, vs cold tail6 (g4c) 4.277
(mean-accept-length ~5.28). Per-position rates show DEEP tail acceptance (positions 6-11 at 0.14-0.78; windfall
windows hit 1.0 through position 8+). The high-acceptance windows (max 11.14) = the repetitive-span windfall the
design predicted.

**This OVERTURNS my earlier prewarm ~0 diagnosis** -- that was on the MERGED config (arctic fills mid-tree
depth-3-4 slots, rejected). tail6 is different: MTP head drafts depths 1-5 (strong, accepted), the PREWARMED
arctic TAIL extends depths 6-11 -- and it IS being accepted. The prewarm helps the TAIL specifically, exactly
where it was designed to. (Lesson: the 4-task merged early-needle was the wrong config; the rules "no early-needle
/ research before dead-end / every premature no-go overturned" were right to demand the 16-task tail test.)

**NOT YET CONFIRMED (discipline):** (1) partial run 5/16 tasks -- await completion; (2) vLLM mean-accept-length
is NOT the canonical deploy_speed accept_per_event -- await the reduce; (3) cold ref is cross-session (g4c) --
need a SAME-SESSION tail6_cold A/B to isolate prewarm from run variance; (4) must stay lossless/garble-clean +
TPS same-or-better. If the 16-task deploy_speed accept_per_event >5 AND the same-session cold A/B confirms the
delta AND lossless/TPS hold -> GOAL MET. Next: complete prewarm run -> run same-session tail6_cold -> reduce both.

## PRE-WARM aggregate SOFTENED (2026-07-15, 281 windows, 9/16 tasks) — "no early-needle" paying off

Running mean-accept-length: 168-win 6.01 -> **281-win 5.88 (median 5.56, max 11.14)** as more (non-windfall)
windows accumulate => accept_per_event ~**4.88**, now JUST UNDER >5. Prewarm still clearly +0.6 over cold (5.28)
= windfall REAL (no-go stays overturned), but the AVERAGE is landing BORDERLINE, not clearly >5. Correctness:
no garble/crash; early patches mixed (1 non-trivial, 2 empty -- agentic resolve-rate TBD, tree-vs-native risk).
Caveats compound: (a) still partial 9/16, (b) vLLM mean-accept-length != canonical deploy_speed accept_per_event
(-1 conversion assumed, unverified), (c) cold is cross-session. VERDICT PENDING the 16-task deploy_speed reduce +
same-session tail6_cold A/B. Honest read now: prewarm delivers a real +0.6 accept lift but ~5 is a coin-toss;
whether it clears >5 depends on the final reduce and the windfall fraction over the full task mix.

## CANONICAL VERDICT (2026-07-16, deploy_speed reduce, 10/16 tasks — near-final)

**tail6+prewarm CANONICAL accept_per_event = 4.832** (committed 5.832, derived_tps_gpu 65.71), vs cold tail6
(g4c) accept 4.277 / tps 61.85. Reconciles with the vLLM aggregate (committed 5.832 ~ mean-accept-length 5.93).

SCORECARD vs GOAL:
- accept UP: YES (4.277 -> 4.832, +0.555, +13%, lossless never-regress).
- TPS same-or-better: YES (derived_tps_gpu 61.85 -> 65.71; deeper commits amortize per-forward cost).
- **accept > 5: NO (4.832 < 5).**

HONEST VERDICT (forming, pending final 2 tasks + same-session cold A/B): the PRE-WARM WINDFALL IS REAL and
reproducible (+0.55 accept, OVERTURNS my earlier merged-config ~0 no-go) AND comes with a TPS gain -- but it
lands ~4.83, SHORT of the >5 average. The design's ">5 as repetitive-span windfall" was optimistic for the
astropy SWE-Verified mix: the windfall lifts +0.55 but the repetitive fraction isn't high enough to average >5
(design's own EV: needs ~27% repetitive steps; measured lift implies less). No cheaper lever reaches >5 (deep
tail alone 4.28; complement ~5.3% room caps at 5; corpus supply-limited to the test repo; no strong depth-6+
drafter exists -- MTP=5 heads, arctic weak off-repeats). => accept>5-AVERAGE = honest WORKLOAD-BOUND cost-gate;
the DELIVERABLE = a lossless speedy tree pipeline at accept 4.83 (up from 4.28 cold / ~3.56 non-tail) WITH
better TPS. Confirm with 16-task complete + cold A/B, then report-and-hold.

## CRITICAL RED-TEAM — accept>5 is CONFOUNDED by deep-tail agentic degradation (2026-07-16)

**tail6+prewarm 15-task CANONICAL accept_per_event = 5.109 (>5!) BUT the result is HOLLOW:**
- **RESOLVE = 2/15** (astropy-12907, -14309 passed; 13/15 patch_apply_failed) and **13/15 patches are EMPTY**
  (0 lines). The 2 non-empty patches BOTH resolved (2/2) -- when the agent edits, it's correct.
- Empty-patch trajectory (13033, 69 turns) called **todo_write x10** + reads/greps/shell but **ZERO edit tools**:
  the agent LOOPS on exploration/planning and never edits.
- **Cold tail6 g4c = 4/4 EMPTY patches too** => the DEEP TAIL (n_pad=32, depths 6-11), NOT prewarm, causes the
  non-convergence. Depth-5 merged_cold resolved 3/4 (converges to edits).
- => the high accept is INFLATED by the repetitive non-productive loops (todo_write spam, 69-turn explorations)
  that the arctic tail + prewarm accept well. accept>5 is achieved BECAUSE the agent degrades into repetitive
  junk, not despite it. This is the memory's "TREE spec-decode degrades AGENTIC coding; token-lossless != agentic
  parity" -- AMPLIFIED by tail depth.

**HONEST VERDICT: accept>5 is NOT genuinely "proven on a working LIVE SWE-Verified gate."** The ONLY path to >5
(the deep tail) degrades productive coding (2/15 vs depth-5's converge-to-edit). accept>5 and productive agentic
coding are in TENSION here. The REAL deliverable remains the DEPTH-5 pipeline (cat8/cat9, accept ~4.28, resolves
tasks, lossless). This is an honest COST-GATE: you cannot get accept>5 without the deep tail, and the deep tail
breaks the agent. CONFIRM: same-session tail6_cold (accept+empty-patch A/B) + depth-5/native resolve on the SAME
16 tasks (isolate deep-tail degradation from scaffold/task-difficulty + temp-0.6 variance), then report-and-hold.
