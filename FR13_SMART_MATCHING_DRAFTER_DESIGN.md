# FR13 Smart-Matching Drafter — Adaptive-Geometry, Load-Aware Speculation (design)

**Status:** DESIGN (2026-07-15). Prereq: the merged MTP-k + Arctic-suffix drafter (Path B, feeds OUR
multidraft committer). This doc proposes the *next* lever after the merged A/B lands: make the tree
**geometry** a per-step function of the suffix-match signal and the serving load, instead of a fixed
cat33333. Grounded in our measured two-ceiling analysis and positioned against the 2025–26 adaptive-
drafter literature (DFlash, FailFast, EAGLE-2, OPT-Tree, DISCO, Nightjar).

---

## 1. Motivation — the two ceilings (measured)

cat33333 (depth-5 spine, 2 branches/level, 15 nodes) beats native MTP-5 by only **+0.166 accept/fwd**
(cat8+fix **3.500** vs native-5 **~3.34**; E5 non-MTP bar **3.076**). That small gap is not a bug — it
is two ceilings pressing together:

- **Ceiling #1 — GEOMETRY.** Accept = longest verified root→leaf path ≤ depth = **5 (+1 bonus) ≈ 5–6
  tokens/fwd, maximum, for any drafter.** Set by the **16-node BV register wall**
  (`fr10_gdn_tree_kernel.py:408` `h_cache=[N_PAD, BV=16, DIM_K=128]` fp32 = 128 KB/CTA at n_pad=16 =
  half the SM register file; n_pad=32 spills to HBM — see [[reference_fr13_tree_size_16_register_wall]]).
  Native MTP-5 shares the exact same depth-5 cap.
- **Ceiling #2 — DRAFTER QUALITY.** Within depth-5, accept = Σ_d P(all ancestors + node-d correct within
  top-3). Native MTP nearly saturates it; the cat33333 branches only rescue rank-2/3 near-misses (the
  +0.166). Hard misses (true token rank ≥ 4) truncate regardless of branching. S2 (task #27) proved the
  **depth lever dead *for MTP*** — the k-th MTP head decays fast, so a deeper MTP spine earns ≈ 0.

**Key realization that re-opens the game:** S2's "depth is dead" is **MTP-specific**. On a *verbatim*
suffix match the per-depth hit rate **does not decay** — if the trie holds an exact 15-token repeat and
the model is actually repeating it, all 15 verify. Agentic coding is saturated with such repeats
(re-emitted file bodies in edits, import blocks, signatures, paths, error strings, boilerplate). So:

> **Suffix decoding cannot break Ceiling #1 *inside* cat33333 (depth-5 wastes long matches), but it
> re-opens the depth lever — and a depth-15 pure chain (15 nodes) FITS UNDER THE 16-NODE WALL.**

That is the whole thesis of this doc: **reallocate the fixed 16-node budget by matchability** — deep
chain when the trie has a long exact match ("win big"), cat33333 width when it doesn't ("stay safe"),
shrink when cold or when load is compute-bound ("fail fast").

---

## 2. Prior art — where this sits (and how we differ)

| Work | Mechanism | Adaptive? | Trained drafter? | Lossless verify | Relevance to us |
|---|---|---|---|---|---|
| **DFlash** (arXiv 2602.06036) | block-diffusion drafter, 1 parallel fwd, target-latent KV-injection | **fixed** block (4/8/16) | **yes** (diffusion) | AR verify | accept-len **4.0–4.2**, **2.2–3.3×** vs EAGLE-3's 1.4–2.2×. Proves a single-pass parallel drafter reaches accept-len ~4; but *fixed* depth + *heavy training* |
| **FailFast** (2512.20573) | dLLM drafter, "fail fast" (min compute, hard regions) / "win big" (long drafts, easy regions) | **length** | yes (dLLM) | AR rejection, lossless | **4.9× vanilla / 1.7× EAGLE-3, accepts up to 70 tok** in easy regions. Directly validates deep-when-easy; corroborates depth-5 wastes the easy region |
| **EAGLE-2** (2406.16858) | confidence-scored **dynamic draft tree**, context-dependent accept | **structure** | yes (EAGLE head) | AR, lossless | static trees (Medusa/Sequoia/cat33333) leave accept on the table; confidence-guided expansion is SOTA |
| **OPT-Tree** (2406.17276) | adaptive draft-tree structure per step | **structure** | yes | AR, lossless | closest structural precedent for adaptive geometry |
| **DISCO** (2405.04304) | classifier picks speculation length per step, provably lossless | **length** | classifier | AR, lossless | **+10.3% / +31.4%**; precedent for runtime length, quality-preserving |
| **Nightjar / SGLang roadmap / BanditSpec** (2512.22420 / gh#23705 / 2505.15141) | **load-aware**: shrink/disable spec at high batch (compute-bound), expand at low (memory-bound); bandit selection | length+on/off | varies | — | the load overlay; BanditSpec ties to our MAB heritage |

**Our four differentiators** (none of the above has all of these):

1. **Training-free.** All the diffusion/dLLM/EAGLE drafters *train* a draft model. Ours is a **suffix
   trie (exact-match lookup) + native MTP heads** — zero training. Accept on a matched span is by
   *construction* (an exact repeat verifies deterministically), not by a learned confidence.
2. **Register-constrained optimization.** The papers grow *unbounded* trees / lengths. Our lever is a
   **partition of a FIXED 16-node budget** between depth and width — a constrained allocation, not
   growth. This is a GB10-hardware-specific problem the literature does not address.
3. **Lossless by the committer, for free.** Every geometry feeds the **same Gate-1 multidraft committer**
   (source-agnostic: output=p lossless + accept=p(S) monotone + garble-safe — 32/32). So we can switch
   geometry per step **without re-deriving correctness** — unlike trained dynamic trees that must re-prove
   losslessness per structure. [[feedback_fr12_subkernel_zero_gate]]
4. **Verify-cost invariance on GB10.** Decode is **HBM-bound** (~98.6 ms weight-read floor,
   [[reference_decode_hbm_bound_accept_is_the_lever]]); the tree-verify is **one forward, flat for
   n_pad ≤ 16 nodes** regardless of shape. So a depth-15 chain costs the **same verify** as cat33333 —
   deeper accept on matched spans is ~free upside. (On a compute-bound GPU a deep chain and a wide tree
   differ; GB10's memory-bound regime makes them equal. The lever is *especially* attractive here.)

---

## 3. Design — the Smart-Matching Drafter

Per request, per step, pick the geometry `G ∈ {DEEP, CAT33333, SHRINK/OFF}` from two signals already in
hand — the Arctic **match_len / prob** (matchability) and the scheduler **Running count** (load) — then
fill the fixed 16-node tree accordingly and verify through the unchanged committer.

### 3.1 Policy (threshold, not DP — keep it off the critical path)

```
signal:  m   = arctic best-match length at this step (match_len)         # matchability
         p   = arctic match prob / score                                  # confidence
         R   = scheduler Running count (effective batch)                  # load
                                                                          #   GB10 agentic ~1.3 (memory-bound)

if R >= R_compute_bound:                 # high load: verify competes -> FAIL FAST
    G = SHRINK    (or OFF)               #   fewer nodes / disable spec  (Nightjar/DISCO/SGLang)
elif m >= L_deep and p >= P_deep:        # long confident exact repeat -> WIN BIG
    G = DEEP      depth = min(m, 15), width = 1     # spend all 16 nodes on depth
elif m >= L_mid:                         # partial match -> keep branches
    G = CAT33333  depth = 5, width = 3   # current shape: MTP spine + trie/MTP branches (+0.166)
else:                                    # cold / novel -> never-regress
    G = CAT33333  (pure MTP)             # == today's baseline, byte-identical when cold
```

- `L_deep` ≈ 6–8, `P_deep` tuned so DEEP fires only when the repeat is near-certain (else it truncates
  and wastes the draft — never-regress, but no win). Start conservative; the match_len histogram (live)
  sets it.
- `SHRINK/OFF` guarded by `R`: on GB10 agentic R is usually ~1 ([[reference_b4_effective_batch_agentic_sparsity]]),
  so DEEP/CAT33333 dominate; the load arm matters for higher-concurrency serving and is cheap insurance.
- **Never removes branches** ([[feedback_chain5_reshape_not_a_solution]]): CAT33333 stays the default;
  DEEP is an *additional* mode selected only on long exact matches. Branches are the deliverable on the
  novel-token majority; the deep chain is bonus reach on the repetitive minority.

### 3.2 The constraint math (why 16 nodes, and the partition)

The kernel budget is **n_pad ≤ 16 nodes** (hard, register wall). Geometry = choose (depth d, per-level
width w_1..w_{d-1}) s.t. total nodes `1 + Σ w_i ≤ 16`. Two useful corners:

- **CAT33333**: d=5, w=[1,2,2,2,2]-ish → 15 nodes. Max accept ≈ 5–6. Best when misses are rank-2/3.
- **DEEP chain**: d=15, w=1 → 15 nodes. Max accept ≈ 15. Best when a verbatim repeat exists.
- (Optional **DEEP+1branch**: d=8, root-siblings → hedge one alternative on a semi-confident repeat.)

The optimal corner depends only on the **match_len distribution** at that step — a per-step threshold,
O(1), no DP (S2 already showed DP over shapes is near-flat *for MTP*; the new signal is *match_len*,
which the threshold reads directly).

### 3.3 Losslessness & garble (unchanged — the free lunch)

All geometries are just different candidate sets into the **same committer**. By Gate-1 the committer is
output=p lossless + garble-safe for *any* candidate set (proven 32/32, and the FR13_ATTN_KV_REMAP fix
made the branched tree garble-clean 15/15 → [[project_fr13_garble_attn_kv_remap_fix]]). So DEEP inherits
losslessness with **no new proof** — the deep chain's tokens are verified per-depth-argmax exactly like
the spine. This is the structural advantage over trained dynamic-tree methods.

---

## 4. Implementation plan (gated, CPU-first — mirrors the merged-drafter ladder)

All behind a new sidecar-gated flag `FR13_SMART_GEOMETRY` (default OFF → cat33333 untouched). Reuse the
merged-drafter seam (`fr13_merged_drafter.decide_and_fill`) — it already owns the Arctic draft object and
the assembled-node output; geometry selection is a new step *between* speculate and assemble.

- **(a) geometry-select helper + CPU test.** `select_geometry(match_len, prob, running) -> (depth, widths)`
  pure function; unit-test the threshold table (DEEP on long match, CAT33333 on partial, SHRINK on load,
  cold==baseline). No torch. Deterministic.
- **(b) DEEP assembler path.** Extend `assemble_cat33333` → `assemble_shape(nodes, depth, widths)` so the
  same Arctic suffix_rel fills a depth-d chain (d>5) when G=DEEP. CPU test: depth-15 chain from a 15-long
  match == the 15 trie tokens; cold==pure-MTP byte-identical (unchanged invariant).
- **(c) KERNEL SHAPE RISK — the one real unknown.** Does the GDN tree kernel + `_prepare_tree_attn_bias`
  accept a **variable tree shape at runtime**, or is cat33333 baked into the compiled graph? Two outcomes:
  - shape-parametric (tree passed as data / attn-bias) → DEEP is a data change, no recompile → cheap.
  - shape-baked (graph captured for cat33333) → DEEP needs a **second captured graph** (2 shapes) or a
    shape-generic kernel. **Probe this FIRST** (read the kernel + bias builder; a 1-node A/B of a d6 chain
    vs cat33333 same-boot tells us if it recompiles). This gates the whole effort's cost.
- **(d) seam wire + patcher self-test.** Byte-identical when `FR13_SMART_GEOMETRY` off; DEEP marker fires
  when on. Reuse the merged self-test harness.
- **(e) live A/B** (after the merged A/B): `smart` vs `merged-cat33333` vs `MTP-only`, B=4 16-task —
  accept / dfwd / derived_tps_fullstep_gpu / garble / resolve. **Gate:** DEEP must *raise* accept on the
  repetitive fraction WITHOUT dfwd regression (verify is flat ≤16 nodes, so the risk is host-side geometry
  selection + any recompile from (c), not the forward).

---

## 5. Risks & the honest measurements that decide it

1. **Kernel recompile on shape switch (§4c)** — the dominant cost risk. If cat33333 is graph-baked and DEEP
   forces a second capture or eager fallback, the per-step overhead could erase the accept win. **Measure
   before building the full path** (the dfwd timer is the gate, as with the merged skip).
2. **DEEP fires too rarely to matter** — the win is only on steps with a long *confident* exact match. The
   live **match_len histogram** (add a needle: bucket match_len 0/1–2/3–5/6–10/11+) sizes the addressable
   fraction. If long matches are <5% of steps, DEEP is not worth the shape-switch complexity → report and
   hold at merged-cat33333.
3. **Mid-chain divergence** — model follows the repeat for k<d tokens then diverges → chain truncates at k
   (never-regress via committer, but wasted draft compute). `P_deep` thresholding + `min(m,15)` bound this;
   the accept metric measures it directly.
4. **Sync on the host path** — geometry selection adds a host branch per step; must stay off the critical
   path. Same D2H/H2D concern flagged for the merged skip; same dfwd gate.
5. **Load signal noise** — `Running` count is bursty in agentic serving; use a short EMA, and keep SHRINK
   conservative (only at clearly compute-bound R) so it never hurts the common R≈1 case.

**Decision rule (honest cost-gate):** build (a)(b) CPU-cheap unconditionally; run the §4c kernel-shape
probe + §5.2 match_len histogram BEFORE committing to the full seam. Proceed to the live A/B only if
(shape-switch is cheap OR two-graph capture is affordable) AND (long-match fraction is materially >5%).
Otherwise record the measured verdict — merged-cat33333 remains the deliverable; smart-geometry is a
documented, quantified next-lever, not a mandate.

---

## 6. Relationship to the merged drafter (this run)

The merged MTP-k + Arctic drafter (running now, merge16c) is the **prerequisite and the measurement
instrument**: it already produces the Arctic draft object with `match_len`/`prob`, already routes through
the committer, and its `[FR13_MERGED ENGAGED]` needle already reports match_full/match_partial. Smart-
geometry is the *consumer* of that signal. Concretely: the merged A/B answers "does suffix break Ceiling
#2 (drafter quality) inside cat33333?" (expected: real but small, ~+0.1–0.3). Smart-geometry answers the
bigger question — "can suffix break Ceiling #1 (geometry) by spending the 16-node budget as depth on
matched spans?" — which is where FailFast's "win big" (70-token easy-region drafts) says the real
speedup lives.

**Sources:** [DFlash 2602.06036](https://arxiv.org/abs/2602.06036) ·
[DFlash/Spec-V2 LMSYS blog](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/) ·
[FailFast 2512.20573](https://arxiv.org/abs/2512.20573) ·
[EAGLE-2 2406.16858](https://arxiv.org/html/2406.16858v1) ·
[OPT-Tree 2406.17276](https://huggingface.co/papers/2406.17276) ·
[DISCO 2405.04304](https://arxiv.org/html/2405.04304v1) ·
[Nightjar 2512.22420](https://arxiv.org/abs/2512.22420) ·
[BanditSpec 2505.15141](https://arxiv.org/pdf/2505.15141) ·
[SGLang adaptive-spec roadmap gh#23705](https://github.com/sgl-project/sglang/issues/23705)
