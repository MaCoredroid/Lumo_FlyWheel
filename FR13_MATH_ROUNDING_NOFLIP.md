# FR13 — MATH/ROUNDING ROUTES TO ARGMAX-INVARIANCE (no-copy / no-HBM compute-only)

Date 2026-06-15. CPU-ONLY, READ-ONLY (an apple-to-apple committer workflow runs concurrently; no code/boot
touched). vLLM source read DIRECTLY from the pinned image `vllm/vllm-openai@sha256:3dbe092e`
(= 0.19.2rc1.dev134+gfe9c3d6c5) via `scripts/vllm_src.sh`, NEVER a /tmp cache. Kernel lines from the repo
working tree `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` (the served scan). Online SOTA searched FIRST
(Thinking Machines batch-invariance, vLLM batch_invariance docs, Kahan/compensated summation, delta-rule /
DeltaNet numerics) — MEASURED/CODE-READ vs INFERRED/LITERATURE labelled throughout.

**Scope guard:** this doc is the MATH/NUMERICS survey ONLY. It does NOT propose the spine-bonus (user rejected),
WY (parked), copy/dense/multi-spine/HBM-tax (banned). It builds on (does not redo) the two binds:
FR13_NOCOPY_LOSSLESS_LEAVES (no-copy leak decomposition) and FR13_DIFFUSION_DEEP_DIVE (the per-layer growth
model). The decode oracle = native recurrent (FR13 compare target), never a serial-torch / fallback proxy.

---

## 0. TL;DR (the skeptic's bottom line)

1. **The flip is a floating-point realization difference, and the dominant per-layer floor source is the GDN
   recurrent scan's bf16 store-boundary fed recurrently + the N_PAD-dependent reduction order** (CODE-READ +
   MEASURED). It is **geometrically amplified ~1.166x/layer over 64 layers** (MEASURED, not the old "32x"), so
   the residual is correlated/compounding, not a single paddable spike.

2. **The N_PAD-invariance fix (lever a) is the one MATH lever that is BOTH compute-only AND untested in pure
   form.** The refutation on the books (recompute → flips ROSE 23→32) is **confounded**: the recompute kernel
   `_tree_gdn_recompute_kernel` ALSO changed geometry to native BV32/num_warps=1/num_stages=3 (L708-709,717)
   AND still loops `tl.static_range(0,N_PAD)` (L765,771) so it did not even isolate the N_PAD reduction order.
   A **same-geometry, N_PAD-canonicalized** reduction order on the EXISTING h_cache scan is genuinely UNTESTED
   and is the prime, cheapest math lead.

3. **Honest diffuse verdict:** the per-layer floor is genuinely diffuse-from-L0 (MEASURED two ladders), so NO
   single reorder is guaranteed to clear the 1.875-nat clean margin. But the diffuseness is **correlated**
   (geometric, ratio ~1.1–1.3x/layer), and the N_PAD seam is the *same* realization knob at every GDN layer —
   so a per-layer order-canonicalization attacks all ~48 instances of the seam at once (it is one constexpr,
   not 48 fixes). That is the difference between "diffuse → only compensated-accum can help" and "diffuse but
   one knob." **Compensated/fp32-carry accumulation (lever b) is the BACKSTOP** with quantified-but-bounded
   reach (it removes the bf16-store ULP but NOT the N_PAD reorder, so it is necessary-not-sufficient).

4. **Cheapest route:** `N_PAD-invariant canonical reduction order on the h_cache scan (same geometry)` + keep
   K1 ON. Compute-only, zero extra HBM, zero copy, reward-hack-clean (constexpr-dead default-OFF). Expected
   reach is the prime open question the minimal experiment answers; it is the ONLY lever that targets the
   pinned rounding-order leak without the geometry confound.

---

## 1. CHARACTERIZE THE FLIP AS A FLOATING-POINT PHENOMENON (build on the binds)

### 1a. Which ops contribute the rounding drift (CODE-READ + MEASURED, from the diffusion bind)

The verify forward and the recurrent-decode oracle are two float realizations of the SAME recurrence. Ranked by
contribution (FR13_DIFFUSION_DEEP_DIVE §2c, §3, MEASURED per-stage at the onset layer):

| op | where | drift mechanism | depth growth? | dtype regime |
|---|---|---|---|---|
| **GDN scan bf16 store-boundary** (K1) | `_gdn_node_step` L503-504 (carried state `.to(bf16).to(fp32)`); native `fused_recurrent.py:336` stores `b_h` bf16, L303 reloads | per-token state rounded to bf16, fed recurrently → the ONE op with depth growth | **YES (recurrent)** | fp32 compute, **bf16 store** |
| **N_PAD-dependent reduction order** | scan `tl.static_range(0,N_PAD)` L582; parent read `tl.sum(tl.where(offs_n==j, h_cache, 0.0), axis=0)` L586-589 over N_PAD lanes | N_PAD 4→8 (L159-163) recompiles the unroll + changes the FMA/accumulation schedule for the SAME spine nodes | reorder per layer, compounds via residual | fp32 reduce |
| **conv1d anchor row** | native `causal_conv1d`; our bf16-tap + ex2.approx silu | 1 bf16 ULP (0.000977) on the value-dependent anchor row | seeds, no recurrence | bf16 tap |
| **gate 1/rms** | `RMSNormGated.forward_native` layernorm.py L455-503 | `rsqrt(mean(x^2)+eps)` is fp32 but data-dependent → AMPLIFIES the inherited diff ~1.1–1.3x/layer (NOT 32x) | amplifier, per-layer | **fp32 internal**, bf16 store |
| **per-layer residual add** | hidden_states += block_out | correlated diff rides the residual stream; ~1.166x/layer geometric | **YES (compounding)** | bf16 residual |
| **lm-head GEMV** over verify rows | final logits | spreads the accumulated ~2.5-nat hidden diff into the argmax-deciding logit gap | terminal | bf16 GEMM |
| **fp8 in_proj_qkvz / o_proj** | block-scaled, BLOCK_SIZE_M=64 constexpr, no split-K | **~0 on the spine data path (M-invariant)** | no | fp8, ALREADY clean |
| **FA2-fork** (full-attn tree-bias) | additive -inf mask | 2 single-ULP in ~983k, max 0.0039, NO depth growth | **NO** | amplifier only |

### 1b. Where the argmax becomes sensitive (MEASURED)

The flip is NOT uniform — it crystallizes at small clean-margin structural-boundary tokens. node5 capture
(MEASURED): the accepted-prefix clean teacher-force margin is **1.875 nats** (` ``` ` −0.158 vs `Let` −2.033);
the accumulated residual reaches **~2.5 nats** at `final_norm` (max_abs 7.59) and flips by COLLAPSING the
` ``` ` logit (live 15.94 vs clean 26.60, ~10.7-nat deficit on ONE token). The flip first appears L60, locks
L61. node7-p2: margin 0.5, smaller absolute residual, same shape. **So the sensitivity threshold is ~0.5–1.9
nats of accumulated logit drift on structural-boundary tokens** — exactly where a confident fork (margin > the
~0.11 bag-TV floor but small) lives. This is the regime a spine-bonus/margin-damp CANNOT touch losslessly
(they would suppress real ≥1-nat leaf wins), which is why the user asks for a MATH (rounding-invariance) route
instead.

### 1c. The pinned rounding-order leak (CODE-READ, the prime math lead)

Re-confirmed against the working tree (FR13_NOCOPY_LOSSLESS_LEAVES §1b). The GDN per-node state is
**algebraically path-isolated** (node i reads only ancestor rows via `strict_mask[i,j]` L585; writes only its
own row L651) — there is NO cross-sibling state bleed. The ENTIRE "co-residency" substance on the GDN side is a
**bug-class #10 codegen-identity / reduction-order seam**: adding leaves moves
`N_PAD = 1<<(n-1).bit_length()` (L159-163) from 4→8, which:
- doubles the `tl.static_range(0, N_PAD)` unroll depth (L582), and
- changes the `tl.sum(tl.where(offs_n==j, h_cache, 0.0), axis=0)` reduction tree (L586-589) over the N_PAD lane
  range,

so the SAME spine nodes (i=0,1,2) get a DIFFERENT FMA / accumulation order under the larger tile. The kernel
state-vs-native gap is **MEASURED 0.0289 max_abs** (FR13_SCAN_NOT_E2E_CARRIER_BIND STATE gate). **This is a pure
rounding-order diff, not a mask/state leak.** It is the canonical Thinking-Machines failure (see §2c): "the
reduction order for a given token depends on how many other tokens are processed concurrently."

---

## 2. MATH/NUMERICS LEVER SURVEY (each tagged COMPUTE-ONLY=allowed vs copy/HBM=banned)

### 2a. (a) FIXED REDUCTION ORDER / N_PAD-INVARIANCE — COMPUTE-ONLY (ALLOWED), prime lead

**Goal:** make the scan's reduction order INDEPENDENT of N_PAD/tree-size so the spine nodes get the SAME FMA
order with or without leaves (the pinned-leak fix). **CODE-READ feasibility** (L159-163, L578-651):

- **The unroll source (L582):** `for i in tl.static_range(0, N_PAD)` — the OUTER loop length is N_PAD, so the
  number of unrolled node-steps differs (4 vs 8). For the spine nodes the BODY is identical, but the surrounding
  unrolled schedule and register allocation differ (codegen-identity). **Clean fix:** loop over a FIXED canonical
  bound `N_FIXED` (= the max warmed family, 16) with `i < N_ACTUAL` masking — the spine node-steps then live at
  the same unrolled position regardless of how many leaves are active. This is a pure reorder (no copy, no HBM),
  identical to the recompute kernel's structure (L771 already loops to N_PAD), but applied to the h_cache scan
  WITHOUT the geometry change.
- **The reduction (L586-589):** `tl.sum(tl.where(offs_n==j, h_cache, 0.0), axis=0)` reduces over `offs_n =
  tl.arange(0, N_PAD)`. The `axis=0` reduction TREE over N_PAD lanes changes shape when N_PAD doubles. **Clean
  fix:** make `offs_n = tl.arange(0, N_FIXED)` (fixed lane count) and zero the inactive lanes — the reduction
  tree is then canonical; the extra lanes contribute exact `0.0` (the `tl.where` already masks). This is the
  direct analogue of the batch-invariant "fixed split-size, not fixed number of splits" rule (§2c).

**Is it a clean reordering? YES** — both are constexpr-bound changes (N_PAD → N_FIXED) with `< N_ACTUAL` masking
that already exists. No new buffer, no copy, no extra HBM (the h_cache tile grows to N_FIXED lanes in SRAM only —
register-resident, no global traffic). **Caveat (bug-class #10):** the larger fixed tile may force a different
num_warps / spill at N_FIXED=16 (FR13_CACHE_SCALING_FUTURE: h_cache spills at N_PAD=16 → num_warps=8) — so the
fix must pin num_warps too, or it re-introduces a geometry seam. This is the one real risk and the experiment
must control it.

**Distinguish from the refuted recompute (CRITICAL):** the recompute "32 vs 23" result is CONFOUNDED on TWO
axes — (i) it changed geometry to native BV32/num_warps=1/num_stages=3 (L708-709,717), and (ii) it still loops
`tl.static_range(0,N_PAD)` (L765,771), so it never canonicalized the N_PAD reduction order at all. It also
re-rolled WHICH LCP trajectory the committer walks (different per-node logits → different path, 369 tok diffs).
**A same-geometry, N_PAD-canonical reduction order on the h_cache scan is UNTESTED.** It is the only lever that
isolates the pinned rounding-order leak from the geometry/trajectory confounds.

### 2b. (b) COMPENSATED / HIGHER-PRECISION ACCUMULATION — COMPUTE-ONLY (ALLOWED), bounded backstop

**Which ops, what precision (CODE-READ):**
- The scan already computes in fp32 (`_gdn_node_step` works on fp32 `state_i`). The drift ENTERS at the **bf16
  store boundary** (K1, L503-504) and at the **N_PAD reorder** (§2a). So Kahan/Neumaier on the scan's internal
  fp32 sums is LOW-VALUE — the internal sums are already fp32 and the error is at the cast boundary, not the
  accumulation precision.
- **The lever that IS available:** DROP the bf16 store-boundary (K1 OFF = pure fp32 carry) — but this makes us
  MORE precise than the native oracle, which INCREASES the verify-vs-decode gap (we are scored against native's
  bf16-carried recurrence). K1 ON (match native's bf16 carry) is the right direction and is the MEASURED ~1/3
  reach (18→12). **So compensated-accumulation does not help the bf16-store axis — alignment to native does
  (K1), and that is already the verified partial.**
- **Where compensation COULD help:** the **per-layer residual add** and the **lm-head GEMV** are bf16
  accumulations that the gate 1/rms amplifies. A fp32-accumulated residual stream / fp32 lm-head reduction would
  remove the ~1.166x/layer compounding's bf16-rounding component. BUT: (i) native ALSO uses bf16 residual / bf16
  lm-head, so a fp32 verify residual makes us DIVERGE from the oracle, not converge — it is the K1 trap again at
  the residual level. (ii) The MEASURED per-layer ratio is geometric (signal-proportional), i.e. the diff scales
  with the residual MAGNITUDE, not with per-step rounding noise — so fp32 accumulation removes only the small
  additive rounding component, not the dominant correlated amplification.

**Quantified reach (from the MEASURED growth model):** resid_L2 grows 0.012 (L0) → 178.5 (L63), geometric mean
**1.166x/layer**. The bf16-rounding-per-layer additive component is ~1 ULP of the residual magnitude at each
layer; compensated accumulation removes O(n·eps) → O(eps), i.e. it caps the ADDITIVE error at machine-eps. But
the dominant term is the MULTIPLICATIVE 1.166x propagation of the L0-born diff, which compensation does NOT
touch (it is a correct-but-different realization riding the residual, not accumulated rounding noise). **So
compensated-accumulation reach ≈ remove the per-layer bf16-residual ULP ≈ a few % per layer at most, and ONLY
if we are NOT scored against a bf16-residual oracle (we are).** Verdict: lever (b) is a **necessary-not-
sufficient backstop**, dominated by lever (a) for the N_PAD seam and by K1 (alignment, not extra precision) for
the store boundary. It is NOT the primary route. **Honest:** the genuinely-diffuse part (the 1.166x
correlated amplification of the L0 diff) is NOT reachable by any compensation, because it is not rounding noise
— it is a different-but-equally-valid float realization propagating. The only way to kill it is to make the L0
realization itself match native (levers a + K1), not to accumulate more carefully downstream.

### 2c. (c) BATCH-INVARIANCE (#42960 / Thinking-Machines / vLLM batch_invariance) — partly applies

**State of the art (ONLINE, LITERATURE):** Thinking Machines "Defeating Nondeterminism in LLM Inference"
(thinkingmachines.ai, Sep 2025) — the core principle is verbatim **"the reduction order for a given token does
not depend on how many other tokens from its sequence are being simultaneously processed"** and **"the reduction
order for each element must be fixed regardless of the batch-size."** Applied to:
- **matmul:** data-parallel by chunking the OUTPUT into tiles (batch-invariant); split-along-K and
  batch-size-keyed tensor-core instruction selection BREAK it → use a **fixed kernel configuration** (~20%
  loss). [Our fp8 GEMMs already satisfy this: BLOCK_SIZE_M=64 constexpr, no split-K, MEASURED M-invariant.]
- **attention:** use a **fixed split-SIZE for Split-KV, NOT a fixed number of splits**, so the reduction order
  is invariant to how many tokens are processed. [Our FA2-fork is already byte-exact 14/16, 2 ULP/983k.]
- **RMSNorm:** split reductions when cores > batch elements break invariance; recommendation is to ignore the
  small-batch case. [Our gate is fp32-internal, not the floor source.]
- Cost: vLLM deterministic mode 26s → 42–55s (1.6–2.1x), "performance is not disastrous."

**Does it apply to the GDN delta-rule scan? PARTIALLY, and this is the load-bearing finding.** The Thinking
Machines work makes **NO mention of state-space models / Mamba / linear attention / recurrent scans** (verified,
WebFetch). vLLM's batch_invariance feature covers matmul/attention/RMSNorm, NOT the GDN/DeltaNet scan. BUT the
PRINCIPLE transfers EXACTLY: our N_PAD-keyed reduction (§2a) is the identical failure mode — the spine node's
reduction order depends on how many LEAVES (other tree nodes) are co-resident, which is precisely "depends on
how many other tokens are processed concurrently." So **lever (a) IS the batch-invariance fix specialized to the
GDN tree scan** — we are applying the Thinking-Machines fixed-split principle ("fix the size, not the number of
splits") to the scan's N_PAD lane reduction. This is novel (no published SSM batch-invariant scan kernel exists;
the lit confirms the GAP) but the principle is proven. **#42960 / deterministic-inference does not ship a GDN
scan fix — we would build it, which is exactly lever (a).**

### 2d. (d) ROUNDING-MODE / OP-ORDER alignment beyond K1 — COMPUTE-ONLY (ALLOWED), exhausted

K2-K5 (l2norm div-vs-rsqrt L477-481, beta bf16 round-trip L466, gate-order, conv-tap) are CODE-READ aligned
under MODE=body and MEASURED ~0 incremental (FR13_REALIZATION_AGREEMENT §4). Triton already uses RN (round-to-
nearest-even) consistently; there is no rounding-MODE mismatch with native (both RN). So (d) beyond K1 is
exhausted — K1 is the only op-order seam with depth growth, and it is the verified ~1/3 partial. **No remaining
rounding-mode lever.**

### Summary table (each tagged)

| lever | op | tag | reach (MEASURED unless noted) |
|---|---|---|---|
| **(a) N_PAD-invariant canonical reduction order** | scan L582/L586-589, N_PAD→N_FIXED + `<N_ACTUAL` mask, pin num_warps | **COMPUTE-ONLY (ALLOWED)** | UNTESTED in pure form; targets the pinned 0.0289 rounding-order leak directly; refutation was confounded by geometry+trajectory |
| **(b) compensated / fp32 residual+lm-head accumulation** | residual add, lm-head GEMV | **COMPUTE-ONLY (ALLOWED)** | BOUNDED: removes per-layer bf16 ULP (additive O(eps)) but NOT the 1.166x correlated amplification; AND diverges from bf16-residual oracle → necessary-not-sufficient backstop |
| **(c) batch-invariance principle → GDN scan** | = lever (a) specialized | **COMPUTE-ONLY (ALLOWED)** | principle proven (Thinking Machines), no published SSM scan kernel — we'd build it = lever (a) |
| **(d) rounding-mode / op-order (K1)** | `_gdn_node_step` L503-504 | **COMPUTE-ONLY (ALLOWED)** | MEASURED ~1/3 (18→12) holding accept 3.004; K2-K5 exhausted ~0 |
| K1+fp32 internal scan sums (Kahan) | scan internal | COMPUTE-ONLY but LOW-VALUE | internal sums already fp32; error is at cast boundary not accumulation → ~0 |
| recompute / WY / multi-spine / per-leaf forward | — | **copy or geometry-change or HBM (BANNED/refuted)** | recompute confounded (geometry+trajectory), flips ROSE 23→32 |

---

## 3. THE CHEAPEST MATH ROUTE (ranked by argmax-flip reduction × compute cost × risk)

**Ranked:**

1. **`N_PAD-invariant canonical reduction order on the h_cache scan, SAME geometry, num_warps pinned` + K1 ON.**
   - **Op:** scan L582 `tl.static_range(0, N_PAD)` → `tl.static_range(0, N_FIXED)`; L550/L586-589
     `offs_n = tl.arange(0, N_PAD)` → `tl.arange(0, N_FIXED)`; both masked by the existing `< N_ACTUAL`; pin
     num_warps to the cat9-deployed value so N_FIXED=16 does NOT change the geometry (the recompute confound).
   - **Expected effect on the confident-fork flips:** makes the spine nodes' FMA/accumulation order identical
     with or without leaves → removes the 0.0289 rounding-order component of the per-layer floor at EVERY GDN
     layer simultaneously (one constexpr, not 48 fixes). This is the ONLY lever that attacks the pinned leak
     without the geometry/trajectory confound, so it is the only one that can reduce the **confident-fork** flips
     (the ones margin-damp/spine-bonus cannot touch losslessly). Whether it reaches native-3 is the open
     question; it is the cheapest test of the math hypothesis.
   - **Cost:** compute-only, zero copy, zero extra HBM (SRAM tile grows in registers only); risk = the N_FIXED=16
     tile may spill / force num_warps (the ONE real risk, controlled by pinning num_warps + accepting the spill
     or staying at the deployed N_PAD per family with a fixed canonical INNER order). Reward-hack-clean
     (constexpr-dead default-OFF, bug-class #10).

2. **K1 ON** (already verified): MEASURED 18→12 holding accept 3.004. Keep it; it is orthogonal to (1) (store
   boundary vs reduction order) and free.

3. **(backstop) fp32-accumulated lm-head GEMV ONLY** (not residual): the lm-head is the terminal reduction into
   the argmax-deciding logit gap; fp32 accumulation there is compute-only and does NOT diverge from native IF
   native's lm-head is already fp32-accumulated (CHECK: vLLM lm-head is typically fp32-reduced). Low reach but
   zero risk; only if (1) under-reaches.

**vs the refuted recompute:** recompute is BANNED-as-refuted not on cost (it is actually 0.86x native HBM) but
on confounded outcome (geometry change + trajectory re-roll → 32 flips + non-lossless stream). Route (1) is
strictly cheaper-and-cleaner: it changes ONE constexpr on the DEPLOYED kernel, no new kernel, no trajectory
re-roll if num_warps is pinned.

---

## 4. HONEST: SINGLE FIXABLE OP, OR GENUINELY DIFFUSE?

**The honest answer is BOTH, split by component:**

- **The rounding-ORDER component (the 0.0289 N_PAD seam) is a SINGLE fixable knob** — it is one constexpr
  (N_PAD → N_FIXED) applied to all ~48 GDN layers at once. It is NOT 48 separate fixes; it is the same codegen-
  identity seam realized 48 times. Lever (a) attacks it in one change. **This is the part that is "one fixable
  rounding-order op," and it is UNTESTED in pure form.**

- **The propagation/amplification component (the 1.166x/layer geometric growth of the L0-born diff) IS
  genuinely diffuse over ~48 GDN + 16 full-attn layers** (MEASURED, two ladders, smooth monotone, no single
  dominant layer). This part is NOT a reduction-order artifact — it is a CORRECT-but-different float realization
  of the L0 state riding the residual stream. **Compensated accumulation is the only math lever that touches
  it, and its reach is BOUNDED to the additive bf16-ULP-per-layer component (a few %), NOT the multiplicative
  amplification** — because the amplification is signal-proportional propagation of the L0 diff, not accumulated
  rounding noise. The dominant cure for the diffuse part is to make the L0 realization itself match native
  (levers a + K1 at the source), so the diff that propagates is smaller to begin with — NOT to accumulate more
  carefully downstream.

So: **if lever (a) drives the L0 scan realization to match native (small kernel-state gap → smaller L0 diff to
amplify), the diffuse 1.166x amplification has less to amplify and the argmax may stop crossing the 0.5–1.9-nat
margin.** That is the hypothesis. If lever (a) makes the kernel state bit-exact (like recompute did, int-view
0.0) but flips DON'T drop (like recompute, once geometry/trajectory confounds are removed), then the residual is
**irreducibly diffuse** and NO compute-only math lever suffices — the lossless+fast tension is fundamental on
cat9 and the route relaxes to topology reshape (drafter-packing, the amplification control, FR13_CARRIER_REOPEN)
or accept/event-parity. The minimal experiment is designed to give exactly this verdict cleanly.

---

## 5. MINIMAL VALIDATING EXPERIMENT (the EXACT test of the top route)

**One GPU boot, no copy/dense/forced-spine, varies ONLY the reduction order at fixed geometry:**

- **Build:** add a constexpr `FR13_NPAD_INVARIANT` to `_tree_gdn_kernel` that sets the scan's loop bound and
  `offs_n` lane count to a FIXED `N_FIXED` (= the deployed family's N_PAD, e.g. 16) for ALL tree sizes, masked
  by the existing `i < N_ACTUAL` / `j < N_ACTUAL`, with **num_warps PINNED to the cat9-deployed value** (do NOT
  inherit recompute's BV32/w1/s3). Default-OFF = constexpr-dead = byte-identical locked path (bug-class #10,
  reward-hack-clean). Combine with `FR13_SCAN_ALIGN=1 MODE=body` (K1 ON).
- **Gate 1 (kernel-state, the mechanism check):** byte-A/B the spine node states (i=0,1,2) with leaves ON vs
  the leaf-free spine-only tree, int-view equality (NEVER atol, bug-class #10). PASS = the N_PAD seam is closed
  (spine FMA order now leaf-independent). This is the DIRECT test that lever (a) did what it claims — the
  recompute confound is excluded because geometry is held.
- **Gate 2 (e2e, the verdict):** re-score the served cat9 stream vs the **native recurrent oracle** (FR13
  compare target, same oracle all arms — NOT a serial-torch / streamed-logprob proxy) on the pinned 4 prompts,
  de-cascaded clear flips (FR13_PLUS2 cluster-collapse) + accept/event. Compare to: native-E5 (BAR, 3 flips,
  3.076), cat9 OFF (18 de-cascaded), cat9+K1 (12 de-cascaded).
- **Decision rule:**
  - **Gate 1 PASS + Gate 2 flips drop toward native (≤~7 de-cascaded) holding accept:** lever (a) is the math
    fix → `N_PAD-invariant order + K1` is the lossless+fast no-copy/no-HBM ship. The diffuse amplification had a
    smaller L0 diff to amplify.
  - **Gate 1 PASS + Gate 2 flips do NOT drop (stay ~12 or rise):** the rounding-order seam is closed but
    NON-CAUSAL e2e (the recompute verdict, now de-confounded) → the residual is irreducibly diffuse;
    compute-only math is exhausted; relax to topology reshape / accept-parity. **This is the clean refutation
    the recompute run could not give (it confounded geometry).**
  - **Gate 1 FAIL (spine state still differs leaf-on vs leaf-off):** num_warps was not actually pinned / the
    N_FIXED tile forced a spill → fix the geometry pin and re-run (do not conclude from a confounded boot).

- **Non-vacuity (bug-class #9):** flag-live needle on worker /proc/environ (FR13_NPAD_INVARIANT + FR13_SCAN_ALIGN
  live in PID 1/175/556), tok/draft==9 engagement gate, within-boot det [T,T,T,T], Gate-1 A/B non-empty with
  per-node int-view. The e2e rescore asserts every fork joins a real oracle step (bug-class #12, no hand-rolled
  TPS÷accept, raw counters only, de-cascade arithmetic re-derived).

**This is the minimal experiment because it changes ONE constexpr on the DEPLOYED kernel at FIXED geometry —
the smallest possible isolation of the pinned rounding-order leak from the geometry/trajectory confounds that
muddied the recompute run.**

---

## Playbook rows quoted (FR13_BUG_CLASS_PLAYBOOK)

- **#10 Shared-source ≠ shared-SASS (codegen identity / reduction order):** the GDN "leak" = N_PAD 4→8
  recompiles the scan (different `tl.static_range` unroll + `tl.sum`/`tl.where` reduction tree) even with
  identical grid; the N_PAD-invariance fix canonicalizes it; gate = int-view byte-A/B, NEVER atol; the N_FIXED
  tile must pin num_warps or it re-introduces a codegen seam.
- **#12 Measurement traps / co-residency-trajectory:** the recompute "32 vs 23" is partly a forked trajectory +
  geometry change, NOT a clean order-invariance test; raw counters only; de-cascade arithmetic re-derived; the
  e2e verdict is flips-vs-the-same-recurrent-oracle, not a TPS÷accept hand-roll.

## MEASURED/CODE-READ vs INFERRED/LITERATURE

- **MEASURED:** K1 18→12 holding accept 3.004; recompute bit-exact int-view 0.0 yet flips rose 23→32 (CONFOUNDED
  by geometry+trajectory); GDN scan state gap 0.0289; per-layer growth 1.166x/layer geometric (0.012→178.5);
  clean margin 1.875 nats, flip = ` ``` ` logit collapse ~10.7 nat; FA2 floor 2 ULP/983k; fp8 GEMMs M-invariant.
- **CODE-READ (pinned image + working tree):** scan N_PAD unroll L582, parent reduction L586-589, own-write
  L651, `padded_nodes` L159-163, K1 seam L503-504, l2norm/beta seams L466/L477-481; recompute geometry comment
  L708-709,717 + its N_PAD loops L765,771 (still N_PAD-keyed); gate fp32-internal layernorm.py L455-503.
- **INFERRED/LITERATURE:** Thinking Machines batch-invariance principle (fixed split-SIZE not number; "reduction
  order must not depend on concurrent tokens") — proven for matmul/attn/RMSNorm, **NO SSM/scan kernel published**
  (the gap we'd fill = lever a); Kahan/compensated summation caps additive error O(n·eps)→O(eps) at ~3-4x cost
  but is LOW-VALUE here (internal sums already fp32; error at cast boundary); vLLM batch_invariance docs (fixed
  split-KV); DeltaNet/EFLA chunked-scan numerics (no batch-invariant scan kernel).

Sources (repo): FR13_NOCOPY_LOSSLESS_LEAVES.md, FR13_DIFFUSION_DEEP_DIVE.md, FR13_K1_STORE_BOUNDARY_BIND.md,
FR13_SCAN_NOT_E2E_CARRIER_BIND.md, FR13_CARRIER_REOPEN.md, FR13_REALIZATION_AGREEMENT.md,
FR13_CACHE_SCALING_FUTURE.md, FR13_GATEA_DEEP_DIVERGENCE.md, FR13_BUG_CLASS_PLAYBOOK.md; kernel
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` (scan L508-667, recompute L670-846, `_gdn_node_step`
L450-505, `padded_nodes` L159-163).
Sources (online): Thinking Machines "Defeating Nondeterminism in LLM Inference"
(thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference); vLLM Batch Invariance docs
(docs.vllm.ai/en/latest/features/batch_invariance); Kahan summation (en.wikipedia.org/wiki/Kahan_summation_algorithm);
DeltaNet / Error-Free Linear Attention numerics (arXiv 2512.12602).
