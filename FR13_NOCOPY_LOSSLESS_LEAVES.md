# FR13 — NO-COPY / NO-HBM-TAX ROUTE TO LEAVES + LOSSLESS-WITHIN-FLOOR

Date 2026-06-15. CPU-only, READ-ONLY research (a GPU committer fork-margin probe `fr13_fork_margin_boot_capture.sh`
booted 04:18 into `output/fr13_fork_margin_probe/`; no code/boot touched). vLLM source read DIRECTLY from the
pinned image `vllm/vllm-openai@sha256:3dbe092e` (= 0.19.2rc1.dev134) via `scripts/vllm_src.sh`, NEVER a /tmp
cache. Kernel lines from the repo working tree (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`,
`scripts/fr10_phase4_patch_vllm_tree_gdn.py`, `scripts/fr13_patch_fa2_tree_bias.py` — the served patches).
Online SOTA searched first (STree 2505.14969, MARS 2601.15498, Traversal Verification 2505.12398, Mamba
drafters, PackMamba/SpecMamba). MEASURED/CODE-READ vs INFERRED/LITERATURE labelled throughout.

**HEADLINE (the skeptic's bottom line): the task's central premise is REFUTED by MEASURED data.** The no-copy
path-isolated GDN tree-scan (the WY/STree/recompute direction) was ALREADY BUILT and RUN; it made the GDN node
state **bit-exact to native (int-view 0.0)** and **removed leaf co-residency entirely** — and e2e clear-margin
flips **ROSE 23→32, did not drop toward native-3** (FR13_SCAN_NOT_E2E_CARRIER_BIND, run w7wr68z06). Therefore
the GDN co-residency leak is **REAL but NON-CAUSAL** for the cat9 flip gap. The dominant carrier is the
**LCP-committer trajectory fork (d)** — a near-tie verify-logit nudge moving the longest-common-prefix boundary,
forking the served stream into a different (sometimes degenerate) trajectory. The cheapest no-copy/no-HBM route
is therefore **NOT a fancier scan** — it is **K1 (compute-only, verified partial) + committer margin-damp
(committer-only, free), with the GDN/FA2 floors accepted as irreducible-small and topology reshape as the
amplification lever.** We have a PARTIAL pattern, with a precisely-named gap.

---

## 1. LEAK DECOMPOSITION — pinned ops (CODE-READ)

### 1a. FULL-ATTENTION (16 layers): TRULY ancestry-isolated, NO co-residency leak (CONFIRMED, code-read)

The served full-attn tree decode runs through the **forked FA2 additive `-inf` ancestry bias**
(`scripts/fr13_patch_fa2_tree_bias.py`, helper `apply_tree_bias` L40-73, served decode wiring
`_patch_tree_attn` L569-604). The EXACT op:

```
// flash_fwd_kernel.h apply_tree_bias, AFTER QK, BEFORE mask+softmax:
if (bias == -INFINITY) tensor(...) = -INFINITY;          // hard-mask non-ancestor keys
else                   tensor(...) += bias / scale;       // ancestors: zero bias
```
`tree_bias[q,k] = 0` iff k is an ancestor of q (or q itself), else `-INFINITY`
(project_fr13_fa2_fork_nocopy_floor). After softmax, `exp2(-inf)=0` exactly, so **every sibling/non-ancestor
key contributes ZERO to a node's attention output**. Each node sees ONLY its path-to-root. This is SpecInfer
Def 4.1 / STree Eq.4-6 ancestry isolation, realized in ONE FA2 call over the whole tree (no per-node copy, no
extra HBM).

**The exact residual (MEASURED, not narrative):** 14/16 full-attn decode calls are whole-tree byte-exact 0.0;
**2 single-bf16-ULP elements in ~983k comparisons** (max 0.0039, ~15x below the E5 self-noise floor ~0.059),
root-caused to irreducible no-copy MMA fp32 fragment-grouping (`gemm_rs`, post-QK so the bias cannot change
lane assignment), a ~2e-6 PROBABILISTIC tie-break with NO depth growth. The mask is provably exact `-inf`.
**Verdict: full-attn is ancestry-isolated; its only non-zero is a sub-ULP rounding floor, NOT a leak.** (CODE-READ
mask + MEASURED floor, project_fr13_fa2_fork_nocopy_floor.)

### 1b. GDN LINEAR-ATTN (~48 layers): the shared-tile scan — where the per-node state IS path-isolated by mask, but realized in a leaf-shared tile (CODE-READ)

The served GDN tree scan = `_tree_gdn_kernel` (`fr10_gdn_tree_kernel.py:508-667`). The shared tile and the EXACT
ops:

- **Shared tile:** `h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K))` (L581) — ONE register tile holds ALL N_PAD
  node states.
- **Parent read (the reduction that COULD leak):** for node i, ancestor j, it reconstructs the parent state by
  `h_j = tl.sum(tl.where((offs_n == j)[:,None,None], h_cache, 0.0), axis=0)` (L586-590), gated by
  `ancestor = (strict_mask[i,j] != 0) & (j < N_ACTUAL)` (L585) and `state_i = tl.where(ancestor, h_j, state_i)`
  (L590).
- **Own write:** `h_cache = tl.where((offs_n == i)[:,None,None], state_i[None,:,:], h_cache)` (L651).

**Is the per-node state path-isolated or does it leak across siblings?** It is **algebraically path-isolated**:
node i reads ONLY ancestor rows (the `strict_mask[i,j]` ancestry gate, L585), writes ONLY its own row (L651).
A sibling never reads another sibling's state; the `tl.where(offs_n==j)` select masks every non-ancestor lane
to 0 before the reduce. So **there is NO cross-sibling state read — the mask is correct.** This matches the
full-attn ancestry isolation.

**The EXACT leak op is therefore NOT a cross-sibling read; it is a CO-RESIDENCY REALIZATION diff:** the shared
tile's SIZE and the `tl.sum`/`tl.where` reduction over `offs_n` range over **N_PAD lanes** (L587-589). Adding
leaves moves `N_PAD = 1<<(n-1).bit_length()` (L159-163) from 4→8, which **recompiles the scan** (the
`tl.static_range(0,N_PAD)` unroll depth doubles, the register tile doubles, the reduction tree changes) — so
even the SPINE nodes (i=0,1,2) get a different FMA schedule / accumulation order under the larger N_PAD. This is
a **bug-class #10 (shared-source ≠ shared-SASS codegen-identity)** realization nudge, NOT a mask/state leak. The
served scan state-vs-native-packed gap is **MEASURED 0.0289 max_abs** (FR13_SCAN_NOT_E2E_CARRIER_BIND STATE
gate, OFF int_view=False).

**The cited leak op, precisely:** the N_PAD-keyed `tl.sum(tl.where(offs_n==j, h_cache, 0.0), axis=0)`
reduction (L586-590) over a leaf-sized tile + the per-node fp32-carry vs native's bf16 store-boundary (K1, L503-504)
— a per-node REALIZATION difference, not a sibling state-bleed. **This is the entire substance of "co-residency"
on the GDN side: a codegen/realization seam, not an algorithmic leak.**

### 1c. The DECISIVE MEASURED refutation: removing the GDN co-residency made e2e WORSE

`_tree_gdn_recompute_kernel` (L670-846) is the no-copy path-isolated alternative (STree/WY direction): it DROPS
the h_cache entirely (L708-722), replays each node's full ancestry chain from h0 into ONE `[BLOCK_V,DIM_K]` fp32
tile (L765-835), at native BV32/w1/s3 geometry. It makes the per-node state **bit-exact to native packed-decode
(int-view 0.0)**, removing co-residency at the state level.

**MEASURED e2e (FR13_SCAN_NOT_E2E_CARRIER_BIND, recurrent-oracle frame, same oracle all arms):**
| arm | clear flips | accept/event | state vs native |
|---|---|---|---|
| native-E5 (BAR) | **3** | 3.076 | (incumbent) |
| cat9 OFF (deployed h_cache scan) | 23 | ~3.15 | 0.0289 (REAL gap) |
| cat9 RECOMPUTE (path-isolated, no co-residency) | **32 (ROSE)** | — | **0.0 (bit-exact)** |

**The strongest possible form of the no-copy path-isolated scan made flips RISE, not fall.** So the GDN scan
state-feed is NOT the dominant e2e carrier; the kernel-level state gap (0.0289) is REAL but NON-CAUSAL for the
flips. **This is the load-bearing fact that reframes the entire task.** (MEASURED, triple-non-vacuity-proven.)

---

## 2. NO-COPY OPTION SURVEY — each tagged ALLOWED (compute/committer) vs BANNED (copy/HBM)

| option | what it touches | tag | what it reaches (MEASURED unless noted) |
|---|---|---|---|
| **(a) margin-damp** (committer LCP near-tie gate) | committer only, zero memory traffic | **ALLOWED (committer-only, FREE)** | bounds ONLY the sub-1-nat near-tie forks; the FORK-MARGIN GPU probe (running NOW) classifies how many of the residual 12 forks are fixable-near-tie vs confident; LITERATURE = MARS, but MARS is explicitly LOSSY — see note |
| **(b) WY / STree no-copy path-isolated scan** (= our `_tree_gdn_recompute_kernel`) | compute only, one SRAM tile, no extra HBM | **ALLOWED by the copy/HBM rule, but REFUTED on outcome** | **MEASURED: makes state bit-exact 0.0 AND removes co-residency, yet e2e flips ROSE 23→32 + produced a DIFFERENT non-lossless stream (~369 tok diffs). Does NOT clear the within-floor bar.** within-floor does NOT revive it. |
| **(c) extend K1/in_proj_ba batch-invariance to remaining diffuse ops** (per-node bf16 store-boundary K1) | compute only (constexpr-dead default-OFF) | **ALLOWED (compute-only)** | **MEASURED PARTIAL: K1 cut de-cascaded gap 18→12 (~33%) WHILE holding accept 3.004** (FR13_K1_STORE_BOUNDARY_BIND). The remaining diffuse ops (K2-K5) are provably ~0 (FR13_REALIZATION_AGREEMENT §4). So (c) reaches ~1/3 of the gap, not native. |
| **(d) literature: STree / SpecInfer / Traversal Verification / MARS / Mamba-drafters** | varies | mixed | STree = exactly (b), math-lossless not bit-exact, 1.5-1.7x. Traversal Verification = proven-lossless BOTTOM-UP tree commit (replaces top-down LCP — the ACTUAL carrier). MARS = (a) but LOSSY. See §2d. |
| copy-recurrent multi-spine | per-node state copy | **BANNED** (NOT lossless, CLOSED_NON_SHIP) | — |
| per-leaf isolated forward | re-reads weights | **BANNED** (HBM tax on 273 GB/s; FR9 block-pool surgery) | — |
| dense GDN substitute / forced-spine / splice | — | **BANNED** (reward-hack; oracle-only) | — |

### 2a. margin-damp (ALLOWED, committer-only, FREE) — the on-carrier lever, but losslessness is conditional
The committer is `_lumo_tree_path_lcp_max_greedy_sample` (patch L6680-7134). LCP per path =
`drafts[node]==parent_targets[node]` (L6909), longest-LCP wins (L6919-6923), commit = accepted `drafts` +
one correction (L6954-6976). The leaf forks the stream when a leaf path's LCP ≥ the spine's BECAUSE a near-tie
`parent_targets` argmax (`parent_targets = target_logits.argmax`, L8370) moved across the accept boundary by a
sub-ULP verify nudge. **Margin-damp = require a leaf to beat the spine LCP by a real margin (>floor), not a ULP,
before forking.** Compute-free, committer-only, keeps genuine leaf wins. **The within-floor-losslessness
condition (CRITICAL):** it is lossless ONLY if it suppresses forks whose deciding `parent_targets` top-2 margin
is BELOW the native realization floor (sub-ULP near-ties the realization could flip either way) — i.e. it is a
DETERMINISTIC TIE-BREAK toward the spine at margins the incumbent itself cannot resolve, NOT a quality-accepting
relaxation. If it accepts runner-up tokens at real margins (MARS-style) it is LOSSY (§2d). The FORK-MARGIN probe
labels each fork (A) fundamental margin>1.0 / (B) fixable margin<1.0 (`fr13_fork_margin_classify.py` L45-49).

### 2b. WY / STree no-copy path-isolated scan (ALLOWED by rule, REFUTED on outcome) — re-evaluated vs the WITHIN-FLOOR bar
- **What it is:** `_tree_gdn_recompute_kernel` (built, L670-846) = the STree packed-tree / WY direction realized
  no-copy in-SRAM. Lineage table FR13_GDN_KERNEL_LINEAGE.md:24 has WY archived as last-resort (different
  summation tree, never byte-exact); recompute is the deployable no-copy form already on a branch.
- **Why it was NO-SHIP — CORRECTNESS, not cost (this is the re-evaluation answer):** the within-floor bar does
  NOT revive it. It already CLEARS the kernel-state correctness target (int-view 0.0, bit-exact to native) — the
  abs-0.0 it "failed" was a per-node state target it actually PASSED. The reason it is NO-SHIP is **e2e
  outcome**: at the within-floor e2e bar it scored **32 flips (worse than the 23 it replaced) and produced a
  DIFFERENT deterministic stream (not byte-lossless, 369 tok diffs).** It re-rolls WHICH trajectory the LCP
  committer walks (different per-node verify logits → different LCP-max path), and that re-rolled trajectory has
  comparable-or-more near-tie crossings. **So within-floor does NOT revive WY/recompute; the bar was never the
  blocker — the carrier is downstream of the scan.** (MEASURED, FR13_CARRIER_REOPEN §2.) LITERATURE caveat:
  STree's "lossless" = math-correct vs a per-path serial recurrence, which is a WEAKER claim than "bit-exact to
  the incumbent native packed-decode SASS"; we have the stronger property AND it still doesn't help e2e.
- **HBM/compute cost (for completeness):** recompute is actually CHEAP — spill-free single tile, the sibling
  replay route is **0.86x native HBM** (36→6 row-touches/layer, FR13_WY_CHASE_PLAN_BIND). Cost was never the
  blocker; correctness-of-outcome (worse + non-lossless stream) is.

### 2c. extend the K1/in_proj_ba batch-invariance grind (ALLOWED, compute-only) — the verified partial
- **Which ops:** K1 = per-node `state_i.to(bf16).to(fp32)` store-boundary (L503-504, the SCAN_ALIGN MODE=body
  seam) — the ONLY GDN sub-op with depth growth (FR13_REALIZATION_AGREEMENT §3). Body seams d (l2norm
  div-by-sqrt, L477-478) + e (beta bf16 round-trip, L466) are also under SCAN_ALIGN. in_proj_ba M-invariance
  (LUMO_FB pad) + fp8/gate M-invariance already DONE.
- **Reach (MEASURED):** K1 cut de-cascaded gap 18→12 (~33%) holding accept 3.004 (FR13_K1_STORE_BOUNDARY_BIND,
  verify HOLDS). Remaining K2-K5 ≈0 (exhausted). So (c) reaches ~1/3, leaving ~2/3 = the committer fork (on the
  (a) lever). K1 is compute-only, constexpr-dead default-OFF (reward-hack-clean).
- **Are they compute-only fixable?** YES — K1/d/e are all op-order/cast-boundary aligns in `_gdn_node_step`,
  zero extra memory traffic. But they are bounded: K1 is the only one with depth-growth and it's already
  measured at ~1/3.

### 2d. Literature (INFERRED/LITERATURE)
- **STree (2505.14969):** packed-tree execution via accumulated log-state-transition matrices
  `A_tree = (L · A_log)`, `y_t = C_t(exp{A_tree_t}∘x_0) + Σ L_{t,s} exp{A_tree_t − A_tree_s}∘(C_t B_s u_s)`
  (Eq.6). No per-node state storage; in-SRAM `TreeScan` kernel, "avoids instantiating SSM states off the fast
  shared memory." Lossless = generalizes Mamba2's causal matrix (math-exact, not bit-exact-to-incumbent). 1.5-1.7x.
  **= exactly our recompute kernel; same direction we already measured worse e2e.** Does NOT address the
  committer-fork carrier.
- **Traversal Verification (2505.12398):** PROVEN-lossless BOTTOM-UP tree verification — sequence-level accept,
  "once a parent is rejected child nodes are [no longer] discarded." **This is the lever that targets the ACTUAL
  carrier (the top-down LCP committer that forks at near-ties).** INFERRED-promising: replacing our top-down
  LCP-max commit (L6909-6923) with a bottom-up traversal-verified commit could remove the fork mechanism while
  keeping leaves and staying lossless. UNTESTED here; changes accept logic (needs user sign-off, class #12).
- **MARS (2601.15498):** margin-aware verification = accept runner-up if `z_(2)/z_(1) > θ` (θ=0.9). **Explicitly
  LOSSY** ("not lossless," 0.2-0.5 BLEU drop). So margin-damp ALA MARS breaks within-floor; our margin-damp must
  be the narrower sub-floor tie-break (§2a), NOT MARS's quality relaxation.
- **PackMamba/SpecMamba/Mamba-drafters:** position-index packing-invariance for SSM scan/conv (batch-invariance
  flavor) — supports the (c) direction (make the scan reduction N-independent) but is the per-layer floor, which
  the MEASURED recompute result shows is NON-CAUSAL e2e.

---

## 3. THE CHEAPEST ROUTE (ranked by within-floor reach × compute cost × risk)

**The honest answer is the hybrid, NOT a fancier scan:**

> **K1 (compute-only, ~1/3 of the gap, accept-holding) + committer margin-damp at the sub-floor tie (committer-only,
> free, targets the ~2/3 fork carrier) + accept the irreducible small GDN/FA2 per-forward floor (native-E5=3 has
> it too) + topology reshape as the amplification control.**

- **Named route:** `K1 + LCP-margin-damp` (config the running probe boots: `FR13_SCAN_ALIGN=1 MODE=body` =
  K1 ON, + `FR13_FORK_MARGIN_DUMP` classifier).
- **The op:** K1 = `_gdn_node_step` L503-504 (`state_i.to(bf16).to(fp32)`, compute-only); margin-damp = a
  deterministic gate in `_lumo_tree_path_lcp_max_greedy_sample` L6909-6923 requiring a leaf to beat the spine LCP
  by > the per-node verify-margin floor before forking (committer-only).
- **Expected residual flips:** K1 takes 18→12 (MEASURED). Margin-damp removes the (B)-fixable fraction of the 12
  (the FORK-MARGIN probe gives the exact count; from the carrier-reopen split, ~5-7 are sub-floor near-ties
  vs ~3-5 genuine leaf wins / fork progeny). **Expected residual ≈ native-3 to ~7**, i.e. within or near the
  within-floor bar — IF the probe finds the residual 12 are mostly (B)-fixable. If the probe finds they are
  mostly (A)-fundamental (genuine confident leaf wins), the lossless+fast tension is FUNDAMENTAL and the route
  relaxes to accept/event-parity (cat9 fast, lossy-but-superset) or ships chain3 (lossless-slow). The probe is
  the discriminator.
- **Compute/HBM cost:** K1 = zero extra memory traffic (one cast per node, in-SRAM). Margin-damp = zero memory
  traffic (one comparison per fork in the committer). **No copy, no HBM tax, no extra forward.** This is the
  cheapest possible route on a 273 GB/s part — it adds essentially zero bandwidth.
- **Why NOT the WY/recompute route:** MEASURED worse e2e (32 vs 23) + non-lossless stream; the within-floor bar
  does not revive it because the scan was never the carrier (§2b). Do NOT bake recompute.

**Secondary (if K1+margin-damp under-reaches): topology reshape** (shallower / root-sibling tree, drafter-packing
only, no copy/dense) reduces the depth-accumulation that turns the ~1-ULP floor into a flip AND reduces
basin-amplification (FR13_CARRIER_REOPEN H-FORK-AMPLIFICATION; chain5→2 precedent). Arbiter = e2e accept/event vs
E5, NOT raw flip count (fork-inflated, length-sensitive, class #12).

---

## 4. DO WE HAVE A PATTERN — verdict: PARTIAL (with the precise gap)

**PARTIAL.** We have a no-copy/no-HBM pattern that reaches ~1/3 of the gap with certainty and plausibly reaches
within-floor, but the closing step is gated on one unrun classification:

- **HAVE (MEASURED):** (i) full-attn is ancestry-isolated (no leak; sub-ULP floor only). (ii) the GDN "leak" is a
  codegen/realization seam, not a state bleed, and is NON-CAUSAL e2e (recompute bit-exact → flips rose). (iii)
  K1 (compute-only) closes ~1/3 holding accept. (iv) the carrier is the LCP-committer trajectory fork, which a
  committer-only margin-damp (free) directly targets.
- **GAP (the one open question):** are the residual ~12 (post-K1) committer forks (A) genuine confident leaf-LCP
  wins (margin > 1 nat → margin-damp would reject a real accept → FUNDAMENTAL, relax) or (B) sub-floor near-ties
  (margin < 1 nat → margin-damp stops the spurious fork losslessly → FIXABLE, route closes within-floor)? **This
  is the entire gap between "partial" and "yes."** The MARS literature says low-margin rejections are a
  recognized inefficiency (supports (B)-heavy), but our bar is within-floor-lossless not MARS-lossy, so only the
  sub-floor subset of (B) is touchable losslessly.
- **NO (ruled out):** a fancier no-copy scan (WY/STree/recompute) is NOT the pattern — MEASURED worse e2e. The
  scan-seam framing is exhausted (it was the strongest per-forward candidate and is dead e2e).

---

## 5. MINIMAL VALIDATING EXPERIMENT

**It is ALREADY RUNNING — and it is the right one.** `scripts/fr13_fork_margin_boot_capture.sh` →
`output/fr13_fork_margin_probe/` boots the LOCKED cat9 server at the candidate config (`FR13_SCAN_ALIGN=1
MODE=body` = K1 ON) with `FR13_FORK_MARGIN_DUMP=1` (read-only per-fork committer classifier) + ENFORCE_EAGER.
Phase 2 = `fr13_fork_margin_classify.py` joins the dump to the recurrent-oracle rescore of the SAME stream and
labels each clear-margin fork (A) fundamental margin>1.0 vs (B) fixable margin<1.0.

**Decision rule (no GPU beyond this one boot):**
- **B-heavy** (most residual forks margin<1.0): margin-damp at the sub-floor tie closes the route within-floor
  WITH K1 → `K1 + margin-damp` is the lossless+fast no-copy/no-HBM ship. Pattern = YES.
- **A-heavy** (most margin>1.0 genuine leaf wins): the lossless+fast tension is FUNDAMENTAL on cat9 → relax to
  accept/event-parity (lossy-superset) or chain3 (lossless-slow); margin-damp would cost real accepts. Pattern
  = NO for strict within-floor.

**Non-vacuity (already armed in the harness):** flag-live bridge-needle on worker /proc/environ (FR13_FORK_MARGIN
_DUMP + FR13_SCAN_ALIGN + MODE), tok/draft==9 engagement gate, within-boot det [T,T,T,T], dump-non-empty
with per-node margins (bug-class #9). The classifier asserts every fork joins a real dump step (#12).

**If the probe is inconclusive,** the SINGLE follow-on is the FR13_CARRIER_REOPEN topology A/B: one boot, cat9
reshaped to a shallower root-sibling tree, seed 1313, same 4 prompts, re-scored vs the same recurrent oracle —
varies ONLY topology (no copy/dense/forced-spine), isolating amplification. Predict de-cascaded flips drop toward
native if H-FORK-AMPLIFICATION holds.

---

## Playbook rows quoted (FR13_BUG_CLASS_PLAYBOOK)
- **#12 Measurement traps / co-residency-trajectory** — the +16 is a forked trajectory (16 independent forks),
  NOT N un-aligned per-forward ops; recompute "32 vs 23" is partly a length/denominator artifact (rate 52.9 vs
  62.5 = 1.18x not 1.39x) + a committed-path re-roll; raw counters only, no hand-rolled TPS÷accept.
- **#10 Shared-source ≠ shared-SASS (codegen identity)** — the GDN "leak" = N_PAD 4→8 recompiles the scan
  (different unroll/register tile) even with identical grid/num_warps/BLOCK_V; int-view only, never atol.
- **#9 Silent fallback / vacuous instrument** — the running probe's non-vacuity asserts (flag-live needle,
  engagement gate, dump-non-empty) before any fork label; the WY/recompute claim is MEASURED non-vacuously
  (triple-proven engaged), not a streamed-logprob proxy.

## MEASURED/CODE-READ vs INFERRED/LITERATURE
- **MEASURED:** recompute bit-exact 0.0 yet flips rose 23→32 (carrier bind); K1 18→12 holding accept 3.004;
  FA2 floor 14/16 calls 0.0, 2 ULP/983k; GDN state gap 0.0289; the 16-vs-1 leaf fork count.
- **CODE-READ:** FA2 ancestry `-inf` mask (apply_tree_bias L40-73); GDN h_cache ancestry gate (L585-590) +
  own-write (L651); recompute path-isolated tile (L708-722); LCP committer (L6909-6976); K1 seam (L503-504).
- **INFERRED/LITERATURE:** STree=our recompute (math-lossless≠bit-exact); Traversal Verification = the
  bottom-up commit that could remove the fork mechanism losslessly (untested, prime literature lead); MARS
  margin-aware = LOSSY, so our margin-damp must be the narrower sub-floor tie-break.

Links: [[reference_multispine_not_lossless_closed_nonship]], [[gdn_tree_superset_routes]],
[[project_fr13_fa2_fork_nocopy_floor]], [[reference_gdn_tree_branch_oracle_losslessness]],
[[reference_diffuse_gdn_accumulation_explained]], [[reference_scalar_metric_per_token_blindspot]],
[[project_fr13_tree_reshape_unifying_lever]].
Sources (repo): FR13_SCAN_NOT_E2E_CARRIER_BIND.md, FR13_CARRIER_REOPEN.md, FR13_LEAF_CORESIDENCY_PATH.md,
FR13_K1_STORE_BOUNDARY_BIND.md, FR13_DIFFUSION_DEEP_DIVE.md, FR13_WY_CHASE_PLAN_BIND.md,
FR13_GDN_KERNEL_LINEAGE.md; kernel `fr10_gdn_tree_kernel.py`, patch `fr10_phase4_patch_vllm_tree_gdn.py`,
FA2 fork `fr13_patch_fa2_tree_bias.py`; probe `fr13_fork_margin_boot_capture.sh` + `fr13_fork_margin_classify.py`.
Sources (online): STree arXiv 2505.14969, MARS arXiv 2601.15498, Traversal Verification arXiv 2505.12398.
