# FR13 — WIDTH / LEAF CO-RESIDENCY: ranked mechanism hypotheses + A/B signature + isolate-fix

Date 2026-06-14. READ-ONLY locate task (no GPU). Author: red-team agent. Builds on (does NOT re-derive):
`FR13_BRANCH_FLIP_LOCALIZED_BIND` (spine-perturbation, 11/11 ch2 on spine, commit 2fe2c567),
`FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND` (our `_tree_gdn_kernel` D16=D32=0.0 at N_PAD=1 AND 16 on identical
inputs — DECISIVE: the tree-scan KERNEL is M-invariant), `FR13_CONV_NOT_CARRIER_SCAN_STATEFEED_BIND`
(conv row-occupancy M-invariant), `FR13_BV_SPILL_VERDICT` (scan single-forward M-invariant),
`FR13_PLUS2_NOT_WALL_ORACLE_FRAME_BIND` (+2 spine floor = oracle-frame, SEPARATE from +17).

The open number: cat9 (5-spine + 4 leaves, tree_n=10) = **22 flips** vs chain5/chain3 (pure spine) = **5**
=> **+17 LEAF CO-RESIDENCY**; cat3w (depth-3 + 2 shallow leaves) = **25** vs chain3 = **5** => **+20**. The
leaf rows perturb the COMMITTED-SPINE output even when the leaf is attention-invisible (strict -inf mask) and
even when shallow. The carrier enters through the GDN block, not attention. This doc settles the MECHANISM;
the next-GPU-slot `FR13_GDN_SUBOP_MAB` A/B measures WHICH sub-op.

---

## THE DECISIVE REFRAME (what the prior bind missed)

`FR13_BRANCH_FLIP_LOCALIZED_BIND:21` lists **"GDN scan/conv/gate/o_proj/in_proj (bit-exact)"** as RULED OUT.
That "in_proj bit-exact" claim rests on the `FR13_GDN_SUBOP_MAB` **`pre_conv` arm**, which is
**STRUCTURALLY BLIND to in_proj M-dependence**:

- The harness stash (`fr10_phase4_patch_vllm_tree_gdn.py:1864`) is
  `_fr12_pre_conv_spec = mixed_qkv_spec.detach().clone()` — captured AFTER `in_proj_qkvz` already ran the
  SINGLE GEMM at **M = tree_n = 10**.
- `a`/`b` handed to the harness are likewise the **M=10 `in_proj_ba` outputs**.
- The harness `pre_conv`/`scan`/`conv` arms then `index_select` rows out of those M=10 tensors
  (`:1655 pc_m5 = pre_conv0.index_select(0, idx_spine)`; `:1592 a0.index_select(0, idx)`).
  Slicing rows out of an already-computed M=10 result **cannot reveal whether running in_proj at M=5 changes
  the spine row's bits.** The current 3 sub-ops (`pre_conv`, `conv1d_out`, `scan_out`) all live DOWNSTREAM of
  the projection; they share the SAME upstream M=10 GEMM output.

So "in_proj is bit-exact" was never measured. The +17 carrier is exactly where the harness does not look:
the **batched input projection GEMM at M=tree_n**, specifically its **bf16 sub-GEMM**.

And `FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND` proves the converse half: given **identical** q/k/v/g/beta inputs,
our scan kernel's spine output is **bit-exact at N_PAD=1 vs 16** (D16=D32=0.0). A bit-exact kernel fed
M-variant inputs still produces an M-variant output. **The leaf co-residency therefore cannot originate in
the scan/conv kernels (proven M-invariant); it must originate in what PRODUCES their inputs — the in_proj
GEMM whose tiling/reduction is keyed on M=tree_n.**

---

## GEMM dtype + kernel + M-invariance (checkpoint-confirmed)

Model = `/models/qwen3.6-27b-fp8`. `config.json:quantization_config`: `quant_method=fp8`,
`weight_block_size=[128,128]`, and **`modules_to_not_convert`** explicitly lists, per layer,
`linear_attn.conv1d`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b` (371 entries). So:

| GDN GEMM | dtype/method | kernel | spine row M-invariant? |
|---|---|---|---|
| `in_proj_qkvz` (q,k,v,z) | **fp8** block-scaled w8a8 | `w8a8_triton_block_scaled_mm` (custom op) | **YES** — default cfg `BLOCK_SIZE_M=64, GROUP_SIZE_M=32, BLOCK_SIZE_K=128 pinned`; M=10 and M=5 both fit ONE M-tile (M≤64), so `accumulator += tl.dot*a_s*b_s` over `range(cdiv(K,128))` runs identical tile-count + order. **No split-K at these tiny M.** fp8 M-keyed cfg lookup is DEAD on GB10 (`get_w8a8_block_fp8_configs`→None, no Spark JSON). REFUTED as carrier (2fe2c567). |
| `in_proj_ba` (→ a, b = the gate raw_a/raw_b) | **bf16** (in_proj_a/_b in `modules_to_not_convert`) | `UnquantizedLinearMethod` → cuBLAS/cuBLASLt `mm` | **SUSPECT — likely M-VARIANT.** Plain bf16 GEMV/GEMM: cuBLASLt picks kernel + split-K heuristically by (M,N,K). At skinny M, M=10 vs M=5 can select different tensor-core tile / split-K → different fp32 reduction order → ~1 bf16-ULP shift in a/b on the spine row. **Existence proof it is M-variant: `LUMO_FB_BATCH_INVARIANT_BA_PROJ` (gdn_linear_attn.py:553-601)** was built specifically to pad the ba projection to a fixed row group "so the BA projection shape is independent across active K." (It is OFF in the locked launcher — see below.) |
| `out_proj` | **fp8** block-scaled | `w8a8_triton_block_scaled_mm` | **YES** (same single-M-tile argument); plus `LUMO_FB_BATCH_INVARIANT_GDN_OUT_PROJ` exists but is OFF and out_proj is post-scan so it cannot feed the recurrent state. |
| gate (`RMSNormGated`) | bf16 elementwise (`norm(core_attn_out, z)`) | per-row RMSNorm + SiLU gate | row-local → M-invariant per row. Not a GEMM. |
| conv1d | **bf16** depthwise | `causal_conv1d_update` (+ our fused bf16-tap) | **YES** per-channel, no cross-row reduction (`FR13_CONV_NOT_CARRIER`). |

**Locked-launcher state** (`fr13_launch_locked.sh`): `BATCH_INVARIANT=0`, and **`LUMO_FB_KERNEL_ROWS` is
UNSET** → both ba-proj and out-proj batch-invariance workarounds are **OFF**, and the fp8 GB10 cfg patch is
OFF. So in the deployed path the **bf16 `in_proj_ba` runs in its default M-keyed cuBLAS kernel.**

Why a 1-ULP a/b shift becomes a flip: `a`/`b` feed the scan gate
`b_g = -exp(A_log)*softplus(a + dt_bias)`, `b_beta = sigmoid(b)` (`_gdn_node_step`, tree kernel :365-373;
identical native :143-150). A perturbed `b_g`/`b_beta` enters the spine node's recurrent state at EVERY node
of the accepted chain, is amplified ~32× by the gate `1/rms`, and compounds over ~48 GDN layers
([[reference_diffuse_gdn_accumulation_explained]]) until argmax flips. This is the SAME amplification path
the standing "diffuse GDN accumulation" note describes — now with a concrete M-dependent BIRTHPLACE.

---

## RANKED HYPOTHESES (most→least likely)

### H1 (PRIME): bf16 `in_proj_ba` GEMM is M-keyed → a/b shift on the spine row. ★★★★★
Evidence: (a) bf16 GEMM is the ONLY GDN GEMM not proven M-invariant; (b) checkpoint puts in_proj_a/_b in
`modules_to_not_convert` (bf16); (c) the `LUMO_FB_BATCH_INVARIANT_BA_PROJ` pad-to-fixed-rows workaround was
purpose-built for THIS GEMM's batch-variance and is OFF in the deployed path; (d) Thinking Machines
"Defeating Nondeterminism": at small M/N libraries switch to Split-K, altering reduction order AND
tensor-core tile selection — exactly a 10-row vs 5-row regime; (e) the current harness is blind to it (it
slices the M=10 ba output), so the prior "in_proj bit-exact" ruling is vacuous for this GEMM. **This is the
only candidate that is simultaneously (i) M-dependent, (ii) upstream of the proven-M-invariant scan, and
(iii) on the spine row's data path, and (iv) covered by vLLM BI's `mm`/`linear` override (the C1 lever).**

### H2: bf16 conv-tap or fused-conv MAC picks up an M-keyed input from H1. ★★☆☆☆ (dependent, not independent)
The conv emulation is row-occupancy M-invariant (`FR13_CONV_NOT_CARRIER`), so conv adds no NEW M-dependence,
BUT it consumes `mixed_qkv` (the fp8 q/k/v, M-invariant) — conv1d does NOT consume a/b. So conv cannot carry
H1's a/b perturbation. Rank low as an INDEPENDENT carrier; it is a clean control (expected ~0 in the A/B).

### H3: fp8 `in_proj_qkvz` M-dependence (q/k/v shift). ★☆☆☆☆ (REFUTED-ON-GB10, keep as A/B control)
fp8 block GEMM single-M-tile + no Spark cfg = M-invariant (2fe2c567). Listed so the A/B q/k/v arm is the
explicit fp8-control: predict ~0. If it is NONZERO, the single-M-tile assumption is wrong (M crosses a
GROUP_SIZE_M swizzle boundary or a split-K kicks in) and fp8 re-enters — but that needs a measured surprise.

### H4: cross-branch mask bleed in OUR tree-scan ancestry accumulator. ★☆☆☆☆ (REFUTED by code + A/B)
See "cross-branch bleed" section: the strict-mask select-by-where folds NO leaf state into a spine node, and
`FR13_BV_GEOMETRY` measured the scan output bit-exact across N_PAD on identical inputs. The leaf rows DO
participate in one shared reduction (`tl.sum(tl.where(offs_n==j, h_cache,0), axis=0)` over all N_PAD slots),
but the masked-to-zero leaf lanes contribute exact 0.0 and the +0.0 ordering is N_PAD-fixed (static_range),
so the reduction tree does not change with which slots are *populated*. Kept on the list only because it is
the textbook STree failure mode; the code shows it is closed here. Predict scan_out M10-vs-M5 == 0.0 on
identical inputs.

### H5: chunk-boundary M-shift. ★☆☆☆☆ (not applicable)
Neither path chunks by tree_n: our tree-scan is a per-node `static_range(N_PAD)` (N_PAD = pow2 ceil, fixed
per family, not a reduction chunking); native verify is sequential `for i_t in range(T)` (no chunk). The
fp8 K-loop chunks K (=128, fixed), not M. No M-dependent chunk reduction exists. Predict null.

### H6: state-feed / bank-layout (leaf rows change which banks/cols the spine reads). ★☆☆☆☆ (refuted)
`FR13_CONV_NOT_CARRIER` + the bank geom-fix (8cdda4c4/d30755c8) confirmed h0_state_in BYTE-EXACT; the spine
reads its own committed-path bank (`spec_state_indices[:,0]` / `accepted_len-1` column) independent of leaf
bank ids. Leaf presence does not move the spine's bank/column reads. Predict null.

---

## CROSS-BRANCH BLEED — does the ancestry mask isolate spine from leaves? (cite lines)

YES — perfectly, in OUR tree-scan `_tree_gdn_kernel` (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`):

- Parent-state read (`:459-467`): `for j in static_range(0,i): ancestor = (strict_mask[i*N_PAD+j]!=0) &
  (j<N_ACTUAL); h_j = tl.sum(tl.where((offs_n==j), h_cache, 0.0), axis=0); state_i = tl.where(ancestor,
  h_j, state_i)`. A spine node `i` adopts a node `j`'s checkpoint ONLY when `strict_mask[i,j]==1`
  (= j is a true ancestor of i). For a spine node, its ancestors are only the prior spine nodes; off-spine
  leaf slots have `strict_mask[spine,leaf]==0` → never selected. **No leaf state is folded into a spine
  node's state.** Mask built from `Tree.ancestors` (`:43-51`), correct by construction.
- The ONE place all rows co-mingle is the gather reduction `tl.sum(tl.where(offs_n==j, h_cache, 0.0),
  axis=0)` (`:463-466`) — a reduction over the FULL `N_PAD` slot axis. Non-`j` slots (incl. all leaves)
  are masked to exact 0.0; the add tree is over a STATIC `N_PAD` extent (`static_range`), so it does NOT
  change with which slots are populated. Adding leaf rows changes the *values in non-selected lanes* but
  those are forced to 0.0 before the add → no bit change to `h_j`. This is the "select-by-mask folds no
  branch into spine" the prior bind asserts, now line-pinned.
- DECISIVE empirical confirmation: `FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND` — our `_tree_gdn_kernel` replayed on
  the captured spine inputs is **0.0 vs native at N_PAD=1 AND N_PAD=16** (cat9 size). The kernel does not
  leak leaves into the spine.

CONCLUSION: cross-branch bleed (H4) is CLOSED in our kernel. The leaf perturbation is NOT a mask bug; it is
an **upstream input-projection M-dependence (H1)** that makes the spine row's q/k/v/**a/b** themselves
M-variant BEFORE the bit-exact scan consumes them.

---

## GEMM M-DEPENDENCE (per GEMM, restated for the schema)

- `in_proj_qkvz`: fp8 `w8a8_triton_block_scaled_mm`, spine row **M-INVARIANT** (single M-tile, K-loop fixed).
- `in_proj_ba`: **bf16** `mm`/`linear` (unquantized; in `modules_to_not_convert`), spine row
  **M-VARIANT (suspected carrier)** — cuBLASLt heuristic kernel/split-K selection differs M=10 vs M=5.
  **This is the bf16-batch-variant GEMM.**
- `o_proj` (out_proj): fp8, M-INVARIANT, AND post-scan (cannot feed recurrent state) → cannot be the carrier.
- gate: bf16 elementwise RMSNormGated, row-local → M-invariant per row (not a GEMM).

---

## CHUNK BOUNDARY

No. Neither the tree forward nor native verify forwards rows in an M-dependent chunking that moves the spine
row. Our scan = per-node `static_range(N_PAD)` (N_PAD fixed per family by `padded_nodes`, not data chunking);
native verify = sequential `for i_t in range(T)`. fp8 GEMM chunks only K (=128 fixed). The spine row's chunk
membership / reduction order does not move when leaves are added. (The only M-keyed reduction-order change is
the bf16 in_proj_ba GEMM internal split-K — that is H1, not a row-chunk-boundary effect.)

---

## STATE-FEED GEOMETRY

No. Leaf rows do NOT change the spine row's cache bank/column reads. The spine reads its prior recurrent /
conv state from the committed-path bank (`spec_state_indices[b,0]`, or column `accepted_len-1`), which is
node-indexed and independent of leaf bank ids; the bank geom-fix (8cdda4c4) confirmed the deep-row prior
read is in-range and `h0_state_in` is BYTE-EXACT across the carrier event. Leaf bank ids occupy OTHER columns
the spine never dereferences.

---

## A/B SIGNATURE PREDICTION (so the next-GPU-slot `FR13_GDN_SUBOP_MAB` discriminates immediately)

CRITICAL HARNESS GAP: the present `FR13_GDN_SUBOP_MAB` (`:1295`) measures M10-vs-M5 on `pre_conv`,
`conv1d_out`, `scan_out` — ALL three slice the captured **M=10** in_proj outputs, so ALL THREE are
**predicted ~0.0** even if H1 is true. The current harness CANNOT see H1. **It must be extended with an
`in_proj_ba` (and `in_proj_qkvz`) RE-RUN arm** that, instead of slicing, re-invokes the projection at
reduced M from a captured **pre-projection `hidden_states`** (or the per-spec hidden-states span). Then:

Per-hypothesis deep-spine-row M10-vs-M5 delta that should be NONZERO:

- **H1 (bf16 in_proj_ba carrier) ⇒** the new **`in_proj_ba` re-run arm M10-vs-M5 is NONZERO** (~1e-3, ~1 bf16
  ULP on a/b for the deep-spine row), while `in_proj_qkvz` re-run, `conv1d_out`, and `scan_out` (re-run on
  identical inputs) stay 0.0. *Decisive positive for H1.*
- **H3 (fp8 in_proj_qkvz) ⇒** the new `in_proj_qkvz` re-run arm M10-vs-M5 NONZERO; ba arm may be 0. (Predict
  NULL — fp8 is M-invariant; a nonzero here is a surprise that re-opens fp8.)
- **H4 (mask bleed) ⇒** `scan_out` M10-vs-M5 NONZERO **even when scan is fed bit-identical sliced inputs**
  (the current harness IS competent for this one). Predict 0.0 (refuted by FR13_BV_GEOMETRY).
- **H2/conv ⇒** `conv1d_out` M10-vs-M5 NONZERO. Predict 0.0 (control).
- **H5/H6 ⇒** no sub-op crosses on M10-vs-M5; a NONZERO would appear only if a re-run arm changes bank
  geometry. Predict 0.0.

SHORTCUT A/B (zero new capture, decisive, flag-only): the prior bind's **C1 = cat9 + BI-on**
(`BATCH_INVARIANT=1`). vLLM `enable_batch_invariant_mode` overrides `aten::mm/addmm/matmul/linear/bmm` —
which **covers the bf16 `in_proj_ba`** but NOT the fp8 custom op. So:
- **22 → ~5 under BI-on ⇒ carrier IS a bf16 `mm`/`linear` (= H1 in_proj_ba).** This is the predicted result
  if H1 holds. (Reinterpret the prior bind's "BI-coverable" as specifically the bf16 in_proj_ba, not fp8.)
- **stays 22 under BI-on ⇒** carrier is the fp8 custom op (re-opens H3, needs a custom-op BI override) OR
  diffuse-non-GEMM. Then the in_proj_ba re-run arm above disambiguates.
Run **C2 = chain5 + BI-on** as the no-branch control (expect ~5 unchanged).

The two tests are complementary: C1 (BI-on) is the cheap discriminator; the in_proj re-run arm is the
mechanism-pinning instrument that names the exact GEMM (and survives if C1 is ambiguous because GB10 takes
the reduced BI override branch, `FR13_BRANCH_FLIP_LOCALIZED_BIND:25-26`).

---

## ISOLATE-FIX DIRECTION (make the spine row's GDN output bit-invariant to leaf co-residency; NO reward-hack)

Goal: leaves give accept WITHOUT perturbing the spine. NOT a copy/dense/splice/route-around. The fix is to
make the **input-projection M-invariant** so the spine row's a/b (and q/k/v) are bit-identical regardless of
how many leaves co-reside — then the proven-bit-exact scan produces a bit-exact spine output.

If A/B confirms **H1 (bf16 in_proj_ba)** — the likely outcome — three legitimate, ranked fix directions:

1. **Batch-invariant bf16 in_proj_ba (PREFERRED, already-built scaffolding).** Turn on
   `LUMO_FB_BATCH_INVARIANT_BA_PROJ` (`LUMO_FB_KERNEL_ROWS=1`, with a `LUMO_FB_PROJ_PAD_ROWS` that fixes the
   ROW count) so the ba GEMM always runs at a fixed M (pad spine+leaves to a constant tree_n-independent
   row group, scatter the real rows back). This pins the cuBLASLt kernel/split-K selection to one shape →
   M-invariant a/b. NOTE the existing impl pads across the **spec-decode batch (nspec)** dimension; for the
   B=1 / single-spec-decode width problem it must instead pin the **tree_n (within-spec row)** dimension to a
   constant (e.g. always pad to N_PAD=16 rows). This is lossless by construction (the extra padded rows are
   discarded; the real rows' math is unchanged-shape), not a reroute. Verify: in_proj_ba re-run arm 0.0 +
   e2e flips 22→≈5 BI-OFF.
2. **vLLM batch-invariant `mm`/`linear` for in_proj_ba only.** Wrap just `in_proj_ba.forward` under
   `enable_batch_invariant_mode` (or a targeted fixed-config cuBLASLt heuristic) so its bf16 GEMM uses a
   batch-invariant kernel with a fixed reduction. Narrower blast radius than global BI; fp8 GEMMs untouched
   (they are already M-invariant). This is the "C1 made surgical" fix if global BI proves the carrier.
3. **Compute a/b for the committed spine at a fixed M (isolated spine projection), genuinely isolated — NOT
   a copy.** Run `in_proj_ba` on the spine rows padded to a constant N_PAD, and on the leaf rows separately,
   so the spine's a/b never share a GEMM shape with a variable leaf count. This is "genuinely-isolated spine
   compute," not a route-around: the leaves still get their own (M-invariant-padded) projection and still
   verify; only the GEMM SHAPE the spine sees is frozen. Most code; use only if 1/2 are infeasible under
   capture.

If A/B instead implicates **H3 (fp8)** — only on a measured surprise — the lossless fix is the fp8 custom-op
BI override (fixed BLOCK_SIZE_M / GROUP_SIZE_M / num_warps independent of M, K-loop already fixed), the
`FR13_GB10_FP8_GEMV_CFG`-class patch made M-pinned. If **H4** (it won't, per code) — fix the ancestry-mask
reduction, but FR13_BV_GEOMETRY already shows it is 0.0.

BANNED here (per [[feedback_no_reroute_reward_hacking]]): routing the spine through native's
non-tree GEMM, copying the spine's a/b from a separate clean run, dense recompute, or splice. The fix must
make OUR co-resident projection bit-invariant, not avoid co-residency.

---

## ONLINE RESEARCH (what it confirms)

- **Thinking Machines, "Defeating Nondeterminism in LLM Inference"** (#42960 family): batch-invariance breaks
  in matmul because at small M/N libraries switch to **Split-K** (parallel reductions along K), which
  "alters the reduction order and even instruction selection (tensor-core tile sizes)." Fix = **fix kernel
  config regardless of batch shape**, sacrificing some perf. EXACTLY our M=10-vs-M=5 bf16 in_proj_ba regime,
  and their fix = fix direction #1/#2 above. They ship `batch-invariant-ops` (drop-in mm/RMSNorm/softmax),
  integrated with vLLM deterministic mode → 1000 runs bit-identical. vLLM's `enable_batch_invariant_mode`
  is the same lever, but it overrides only ATEN ops (covers bf16 `mm`/`linear` = in_proj_ba; does NOT cover
  the fp8 custom op) — which is why C1's outcome cleanly distinguishes bf16-carrier from fp8-carrier.
- **STree (arXiv 2505.14969)** "Speculative Tree Decoding for Hybrid SSMs": packs tree nodes into one
  sequence + attention mask; the named risk is **cross-branch contamination of the shared recurrent
  accumulator** if all branches update one state — they enforce **per-path state accumulation** so leaves
  cannot retroactively alter spine state. In OUR kernel this is already enforced by the strict-mask
  select-by-where (H4 closed); our +17 is NOT the STree shared-accumulator bug — it is a numeric
  (sub-ULP, M-keyed-GEMM) input perturbation, a class STree's algebraic isolation does not address (their
  isolation is structural/exact, ours is structurally exact but the INPUTS to the structure are M-variant).
- **Mamba Drafters (2506.01206) / SpecMamba (2509.19873)**: SSM decode compresses history into a fixed-size
  state, impeding backtracking; tree candidates organized to raise accept. No statement contradicting the
  above; reinforces that the recurrent state is the amplifier (small input perturbation → state drift).
- **Gated DeltaNet (ICLR 2025)** / Yang 2406.06484: chunked vs recurrent realizations are ℝ-equal, not
  bit-exact (fp non-associativity) — the SEPARATE +2 oracle-frame gap, not the +17 (kept distinct per
  directive).

GB10 BANDWIDTH CONTEXT: 273 GB/s LPDDR5X unified memory; B=1 decode is bandwidth-bound. The bf16 in_proj_ba
weight DMA is the same pool decode saturates; the batch-invariant-pad fix (#1) adds a few discarded rows to a
FIXED-shape GEMM — negligible extra bandwidth vs the decode weight streaming, so the M-invariance fix is
effectively free on the bandwidth budget (the cost is a slightly larger fixed M-tile, not more weight
traffic). This matters because the speed bar is cat9 > native E5 at B=1; a fixed-M ba GEMM does not add
weight bandwidth, only a constant pad.

---

## SUMMARY / NEXT GPU SLOT

1. Carrier ranking: **H1 bf16 `in_proj_ba` GEMM M-keyed (a/b shift)** ≫ H3 fp8(refuted) ≈ H4 mask-bleed
   (refuted) ≈ H2/H5/H6 (null/dependent).
2. The current `FR13_GDN_SUBOP_MAB` 3 sub-ops are **blind to H1** (slice the M=10 in_proj output) — extend
   with an **in_proj_ba / in_proj_qkvz RE-RUN-at-reduced-M arm** (capture pre-projection hidden_states).
3. Cheap discriminator first: **C1 cat9 + BATCH_INVARIANT=1** (flag-only). 22→~5 ⇒ bf16 `mm`/`linear`
   carrier = H1 (BI covers in_proj_ba, not fp8). Stays 22 ⇒ fp8 or diffuse → use the re-run arm.
4. Isolate-fix (non-reward-hack): pin the **bf16 in_proj_ba to a fixed M** (within-spec tree_n pad to N_PAD)
   so the spine's a/b are M-invariant → the bit-exact scan yields a bit-exact spine output. Reuse the
   already-built `LUMO_FB_BATCH_INVARIANT_BA_PROJ` scaffolding, re-scoped from the nspec-batch dim to the
   tree_n row dim.
