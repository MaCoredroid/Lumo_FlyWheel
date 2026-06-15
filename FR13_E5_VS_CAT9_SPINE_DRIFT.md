# FR13 — E5-spine (3 flips) vs cat9-spine (~17 flips): the op-by-op attribution, the precise WHERE, and the lever verdict

Date 2026-06-15. CPU-only, READ-ONLY (a big-denom GPU serve runs concurrently; no boot, no edit). vLLM source read
DIRECTLY from the pinned image via `scripts/vllm_src.sh` (0.19.2rc1.dev134, sha 3dbe092e). Banked captures:
`output/fr13_node5_ladder/` (deep-accept input-aligned per-layer ladder, holds=TRUE), `output/fr13_node7_ladder/`
(p2 confirming ladder), `FR13_GATEA_DEEP_DIVERGENCE.md` (post-fork spine-vs-native ladder),
`output/fr10_native_mtp5_same8_20260604T210257Z` (E5 native MTP-5 capture), `output/fr13_shape_sweep/`
(chain5/chain3 flips). Compare target throughout = the deployment-correct RECURRENT decode oracle
(`scripts/fr13_recurrent_decode_oracle.py`: single-token decode → `_forward_core_decode_non_spec` →
`causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`), NOT a serial-torch ref, NOT a
chunked-prefill, NOT a backend name. int-view, never atol.

Playbook rows quoted (**FR13_BUG_CLASS_PLAYBOOK**): **#12 Measurement traps** (raw flip counts are length- and
cascade-inflated; the arbiter is per-event/accept-event, not the raw 17/3); **#10 Shared-source ≠ shared-SASS /
codegen identity** (byte/int-view A/B, never atol; the tree rank-1 scan and native FLA scan are the same math, a
realization identity question).

---

## 0. TL;DR — the answer to the user's precise puzzle

The puzzle: E5-SPINE drifts ~3, cat9-SPINE drifts ~17 vs the SAME recurrent decode oracle, both verifying the
SAME MTP-5 spine. **Where does the ~14 excess come from, is it a kernel, and is there a lever?**

**Answer (MEASURED, decisive):**

1. **The ~14 excess is NOT either of cat9's two extra kernels in isolation.** The cleanest existence test is
   already banked: **chain5** = cat9's EXACT two kernels (forked-FA2 tree-bias + GDN tree-scan) running the 5-spine
   ALONE with NO branches → de-cascades to **2 independent clear-margin flips, AT-OR-BELOW native's 3**
   (`FR13_PLUS2_DECASCADE`, MEASURED, same de-cascade rule on all arms). So the forked-FA2 and the GDN tree-scan,
   computing the spine by themselves, land at the E5 floor. **The kernels are not the ~14 source.**

2. **The ~14 excess is CO-RESIDENCY: running the spine rows in the SAME batched tree-forward as the 4 branch rows
   perturbs the spine's own realization.** MEASURED: cat9 22 flips, chain5 5 raw → "the branches add ~17 flips via
   co-residency" (`FR13_FA2_CARRIER_OVERTURNED_BIND`); commit `2fe2c567` = **11/11 channel-2 flips land ON the
   spine, 0 on the leaves = SPINE_PERTURBATION**. The spine token's GDN recurrent state / conv window / attention
   tiling are computed differently when 4 interleaved branch tokens share the forward, even though the cross-token
   ops are tree-masked.

3. **The WHERE is precise and MEASURED, not hand-waved:** the per-forward floor is **BORN in the GDN-layer COMPUTE
   at L0** (node5/node7: first-nonzero L0 `linear_attention`, ~2 bf16-ULP, from a byte-exact 0.0 input), and is
   **AMPLIFIED by the inter-layer RESIDUAL-STREAM CONNECTION and disproportionately by the deep FULL-ATTN layers.**
   Decomposing the 1.166x/layer geometric growth: **GDN layers amplify ~1.075x/layer (geomean), full-attn layers
   amplify ~1.236x/layer (geomean)** — the full-attn connections are the dominant amplifier, the GDN compute is the
   originator.

4. **LEVER VERDICT: FIXABLE, named seam.** The lever is **make cat9's spine rows M-invariant** (compute
   bit-identically regardless of how many branch rows co-reside). chain5 = the M=5 (spine-only) realization at the
   native floor; cat9 = the M=10 (spine+4 branches) realization that drifts. Align the M=10 spine-row computation to
   the M=5/M=1 one and cat9-spine → chain5-spine (~2-5 flips ≤ native 3), while branches still supply the accept
   edge (3.198). Expected reach: **17 → ~5** (the residual ~5-vs-3 is the separate, smaller chain5-vs-native
   intrinsic, well within the per-event superset gate). This is NOT a "tree-batched verify intrinsically drifts
   more" fundamental — chain5 is the existence proof that our tree kernels CAN spine at the floor.

**One honest caveat carried from the bank:** a per-NODE scan-state recompute (`FR13_SCAN_NOT_E2E_CARRIER_BIND`)
made the scan state bit-exact yet e2e flips did NOT drop (rose, via trajectory change). So the M-invariance lever
must be validated by an A/B that holds the trajectory fixed (§4 names the exact minimal experiment).

---

## 1. THE 3-vs-17 OP-BY-OP (E5-spine vs cat9-spine, same recurrent decode oracle)

### 1a. What is literally different between the two spine verify forwards (CODE-READ)

| stage | E5 (native MTP-5) | cat9 (our tree verify) | divergent? |
|---|---|---|---|
| backend | `FLASH_ATTN` (CUTLASS FA2) | `TREE_ATTN` + **forked FA2** (`scripts/fr13_patch_fa2_tree_bias.py`) | SAME KERNEL FAMILY + added bias (see 1b) |
| full-attn op | stock `flash_fwd_kernel.h` | **same kernel** + `apply_tree_bias` at `acc_s` post-QK, pre-mask | bias-only delta |
| GDN scan (decode rows) | native MTP-5 multi-token scan (`fused_sigmoid_gating_delta_rule_update`) | our tree rank-1 scan (`fr10_gdn_tree_kernel.py::_tree_gdn_kernel`, ancestor-replay) | realization diff (1e-6, near-bit-exact, K1) |
| in_proj / o_proj (fp8) | block-scaled fp8 GEMM | **same** | M-invariant, ~0 (CODE-READ) |
| conv | native `causal_conv1d_update` | tree conv (`use_fr10_tree_conv`, source-by-width) | 1 bf16-ULP anchor row |
| **batch shape (M)** | **M = spine rows only** | **M = spine rows + interleaved branch rows** | **THE decisive difference (1d)** |

The only NEW kernels in cat9 vs E5 are (a) the forked-FA2 tree-bias and (b) the GDN tree-scan. Everything else
(fp8 GEMMs, gate, RMSNorm) is identical code.

### 1b. (a) forked-FA2 vs FLASH — NOT the ~14 source (CODE-READ + MEASURED)

The fork is the **SAME FlashAttention-2 CUTLASS kernel** as native `FLASH_ATTN`. The ONLY edit
(`fr13_patch_fa2_tree_bias.py` L26-78, `apply_tree_bias`): after QK and the softcap, before `mask.apply_mask` and
softmax, add a dense ancestry bias to `acc_s`:
```
if (bias == -INFINITY) tensor(...) = -INFINITY;   // non-ancestor → masked
else                   tensor(...) += bias / params.scale_softmax;  // ancestor → bias is 0.0 → no-op
```
So for an ANCESTOR (q,k) pair (the spine sees only its own ancestors) the added value is **0.0** — the exp2
softmax, the fp32 accumulator, the online-rescale, the qk cast, the MMA grouping are **byte-identical to native
FLASH** for the spine. There is no divergent op (softmax-scale / accum-dtype / online-rescale / qk-cast) to align:
they are the same instructions. MEASURED (`project_fr13_fa2_fork_nocopy_floor`): the fork is byte-exact 14/16 calls
on the whole tree, **2 single-bf16-ULP in ~983k**, max 0.0039, **NO depth growth** — a probabilistic MMA-grouping
tie-break, not a per-row floor. `FR13_FA2_CARRIER_OVERTURNED_BIND` (QPAD null result) is decisive: padding the
forked-FA2 query to an M-invariant 64-tile drove the named FA2 carrier (L31 deep_spine_raw 3.9e-3 → 0.0) yet e2e
flips did NOT drop (24, the same diffuse signature). **The forked-FA2 is a downstream AMPLIFIER at the deep
full-attn layers, not the originator of the ~14.** (This is the literature's known point: tree masks break the
standard FA causal path — [vLLM #18327], hybrid-tree-attention — but our fork keeps the same kernel and adds a 0.0
ancestor bias, so the spine path is the standard FA path.)

### 1c. (b) GDN tree-scan vs native MTP-5 scan — the per-layer FLOOR SOURCE, near-native (MEASURED, K1)

The GDN recurrent scan is the per-LAYER floor leading edge: it is fp32-compute / **bf16-STORE of `b_h` per token**,
fed recurrently, so it compounds. But it is NEAR-bit-exact: K1 mechanism proof measured the deployed scan STATE at
**rel-err 2.2e-4** vs native; the one-layer trace (`FR13_DIFFUSION_DEEP_DIVE` §2c, MEASURED) shows `gdn_scan_out`
= **1e-6** vs native — the scan op itself is essentially native. So the tree-scan is the *origin* of the diffuse
floor but its per-forward magnitude is ~1e-6, far below the ~14-excess. By itself (chain5) the scan + fork spine at
the native floor (1a). **The tree-scan is not the ~14 source either.**

### 1d. THE ~14 SOURCE = co-residency (the M-shape), not a kernel (MEASURED, decisive)

The decisive existence test is banked. Run cat9's EXACT two kernels but change ONLY the batch shape:

| arm | kernels | topology (M) | raw flips | de-cascaded | accept/event |
|---|---|---|---|---|---|
| native E5 | FLASH + native scan | 5-spine, M=spine | 3 | **3** | 3.076 |
| **chain5** | **forked-FA2 + tree-scan** | **5-spine, NO branches, M=spine** | 5 | **2 (≤ native)** | 2.664 |
| cat9 | forked-FA2 + tree-scan | 5-spine + 4 leaves, **M=spine+branches** | 22 | ~14-18 | 3.198 |

(chain5/cat9 from `FR13_FA2_CARRIER_OVERTURNED_BIND`/`FR13_PLUS2_DECASCADE`, MEASURED.) chain5 holds the kernels
fixed and removes the branches → **the spine lands at native floor**. cat9 adds the 4 branch rows into the same
batched forward → the spine drifts ~17. Commit `2fe2c567` pins the mechanism: **11/11 channel-2 (clear-margin)
flips land ON THE SPINE, 0 on the leaves = SPINE_PERTURBATION** — the leaves do not flip; the spine flips because
the branch rows co-reside in its forward. The cross-token GDN ops ARE tree-masked (scan ancestor-replay
`_tree_gdn_kernel` L278-289; conv `use_fr10_tree_conv` source-by-width — CODE-READ `FR13_GATEA_DEEP_DIVERGENCE`),
so this is NOT a mask leak; it is the **realization** of the spine row changing with M (reduction grouping /
tile occupancy / state-bank column geometry at num_accepted>1 — class #10 codegen identity, class #11
batch-composition).

**Dominant source of the ~14 excess: co-residency M-dependence of the spine rows** (forked-FA2 query-tile
M-dependence + GDN deep-accept state-feed at num_accepted>1), NOT the kernels' intrinsic math. The kernels at M=5
spine at the floor; the excess is M=10 vs M=5.

---

## 2. THE WHERE (diffuse, precise — MEASURED from the node5/node7 ladders)

### 2a. BORN in the GDN-layer COMPUTE at L0 (not the residual connection, not full-attn)

Both same-boot input-aligned ladders enter L0 BYTE-EXACT (`input_maxabs=0.0`) and the FIRST nonzero is **L0
`linear_attention` (GDN)**: node5 resid_max_abs 0.00122 / resid_L2 0.01205; node7-p2 max_abs 0.0078 (2 bf16-ULP).
There is no clean→broken spike at any single later layer. So the floor is **born in the GDN compute**, specifically
(one-layer trace, MEASURED `FR13_DIFFUSION_DEEP_DIVE` §2c): `pre_conv`(in_proj)=0.0 → **`conv1d_out`=0.000977 (1
ULP, anchor row)** + **`gdn_scan_out`=1e-6 (bf16-store recurrent)** → gate amplifies to 0.000488 → o_proj spreads
to 0.00195. in_proj/o_proj fp8 = ~0 (M-invariant). **WHERE-born = conv anchor-row + GDN scan bf16-store, inside
the GDN block.**

### 2b. AMPLIFIED by the RESIDUAL-STREAM CONNECTION — and the connection is dominated by the deep FULL-ATTN layers (MEASURED decomposition)

The user asks: is the 1.166x/layer a layer-OUTPUT effect or a residual-stream CONNECTION effect, and is it the
gate 1/rms reading a drifted residual? Decomposing the node5 resid_L2 ladder (the residual stream is what
propagates; `resid_L2` is the per-layer residual-stream magnitude, `hid_L2` is the layer's own block-output diff):

- Overall geometric mean = **1.166x/layer** (0.012 L0 → 178.5 L63 over 63 steps).
- **Split by layer type (L4+, excluding signal-birth L1-L3):**
  - **GDN (linear_attention) layers: geomean 1.075x/layer** (median 1.053, max 1.289, n=45).
  - **FULL-ATTN layers: geomean 1.236x/layer** (median 1.212, max 1.615, n=15).
- The 4-5 LARGEST single jumps are ALL full-attn: **L35 1.61x, L51 1.34x, L47 1.32x, L62 1.29x (GDN-adjacent to
  L63 full), L19 1.24x** — vs the GDN median 1.05x.

**Reading (MEASURED, distinguishes the user's options):**
- It is **(b) AMPLIFIED in the inter-layer residual-stream connection**, NOT (a) born-and-constant. Each layer's
  output diff is ~proportional to the residual it receives (the network is locally near-linear in the residual) →
  the diff rides and scales WITH the residual stream = geometric, not additive. `hid_L2` (the layer's own output
  diff) tracks `resid_L2` at a roughly constant ratio (~0.5-0.9x), confirming the per-layer diff is
  signal-proportional — a **CONNECTION effect** (the residual carrying a proportional perturbation forward), not a
  fresh per-layer additive ULP.
- The dominant amplifier is the **deep FULL-ATTN connection** (1.236x vs GDN 1.075x). The full-attn layers read a
  drifted residual through their input-norm (1/rms) and Q/K/V, and the attention over a longer drifted context
  multiplies the residual harder than the local GDN recurrence does.
- The "gate 1/rms amplifies 32x" figure is **REFUTED / INFERRED-OVERSTATED**: the gate is computed in fp32 (CODE-READ
  `RMSNormGated.forward_native`, layernorm.py L455-503), so it is not itself a bf16 source; and if it multiplied a
  per-layer diff by 32x we would see ~32x GDN-layer jumps — we measure **GDN-layer ratios of 1.0-1.29x, never 32x**.
  The gate is ONE contributor to the ~1.075x GDN-layer amplification (it makes a 1e-6 scan diff a larger o_proj-input
  diff WITHIN the layer), not a 32x event.

### 2c. CRYSTALLIZES at the deep layers (MEASURED)

final_norm max_abs 7.59 / L2 103. The sustained final-token argmax flip first appears L60 and locks L61
(node5 `per_depth_argmax`, MEASURED). The flip is a **logit COLLAPSE on ONE boundary token** (node5: ` ``` `
collapses live 15.94 vs clean 26.60, ~10.7 nat) crossing a small clean margin (` ``` ` vs `Let` = 1.875 nat). The
biggest `B-T gap` steps are L63/L62/L59 (full-attn-adjacent, SUMMARY.json), confirming the deep full-attn
connections are where the accumulated residual finally tips the argmax.

**WHERE verdict:** (a) BORN in GDN compute at L0 (conv anchor + scan bf16-store); (b) AMPLIFIED through the
residual-stream connection, signal-proportionally, **dominated by the deep full-attn layers (1.24x vs GDN 1.08x)**,
NOT by a 32x gate; (c) crystallizes L60/L61 at small-margin structural boundaries. For cat9 specifically, the
co-residency makes the L0-GDN birth-amplitude bigger and adds extra divergent spine forwards that ride this same
amplification — same WHERE, larger seed and more of them.

---

## 3. THE LEVER VERDICT — FIXABLE, named seam (research-before-deadend)

### 3a. The lever exists and is named: make cat9's SPINE ROWS M-INVARIANT

The per-component numbers say the excess is co-residency M-dependence, and chain5 is the existence proof that the
SAME kernels spine at the native floor when M=spine. So the lever is: **align the M=10 (spine+4 branches) spine-row
computation to the M=5/M=1 spine-row computation, op-by-op, so the spine is M-invariant.** Then cat9-spine →
chain5-spine (~2-5 flips) while the branches keep supplying accept/event 3.198.

Named seams to align (each is the divergent op, alignable not algorithmic):
- **forked-FA2 query-tile M-dependence** (CODE-READ: `apply_tree_bias` row offset
  `m_block*kBlockM + (tidx/32)*16 + (tidx%32)/4`, M-dependent tile assignment; QPAD probed it M5 pad=59 vs M9
  pad=54 suffix-KV layouts slip). Align the spine query to a fixed N_PAD_Q tile across M (QPAD direction) — but
  QPAD's e2e null says this is downstream, fix it AFTER the L0-GDN seam.
- **GDN deep-accept state-feed at num_accepted>1** (the L0-GDN co-residency carrier, `FR13_FA2_CARRIER_OVERTURNED`
  prime suspect = conv1d prior-window / state-bank column geometry, "fixable wiring at
  `fr10_phase4_patch_vllm_tree_gdn.py:797-818`", `project_fr13_conv_priorwindow_root`). This is the FIRST-nonzero
  sub-op and the highest-value alignment.

Expected reach: **17 → ~5** (chain5 level). The residual ~5-vs-3 (chain5 vs native) is the separate, smaller
intrinsic chunk-vs-recurrent realization diff (`FR13_PLUS2_DECASCADE`: chain5 de-cascades to 2 ≤ native 3 anyway),
and the per-event superset gate already passes at cat9 (+15 net lossless, `FR13_PEREVENT_SUPERSET_GATE_RESULT`), so
17→5 clears the bar comfortably.

### 3b. Why it is NOT fundamental (with numbers)

"Tree-batched verify intrinsically drifts more" is REFUTED by chain5: the tree-batched verify of the spine ALONE
(forked-FA2 + tree-scan, M=spine) = **2 de-cascaded flips ≤ native 3**. The extra drift is NOT intrinsic to the
tree math; it is the co-residency M-shape, which is an alignable realization (class #10/#11) — the spine row's
reduction grouping / tile occupancy / state-bank columns change with M, but the MATH is identical (the cross-token
ops are tree-masked, CODE-READ). native has no tree, so there is no "align our kernel to native's kernel"; the
alignment target is **OUR OWN M=5 spine** (chain5), which we already have at the floor.

### 3c. The honest caveat (carried from the bank — why §4's experiment is needed)

`FR13_SCAN_NOT_E2E_CARRIER_BIND` (MEASURED): a per-NODE scan recompute made the scan STATE bit-exact (int-view 0.0)
yet e2e clear-margin flips ROSE 23→32 — because the recompute changed the committer's near-tie trajectory (a
different ~369-token stream with its own boundaries). So a fix that achieves M-invariance must be validated by a
**trajectory-fixed A/B**, not a free-running re-score (class #12: free-running streams fork at near-ties and the
raw count is trajectory-confounded). The lever is sound (chain5 proves it); the validation must control the
trajectory.

---

## 4. THE EXACT MINIMAL GPU EXPERIMENT TO VALIDATE THE LEVER

**M-invariant-spine in-process MAB A/B (decoherence-free, trajectory-fixed), 1 GPU boot, ~minutes.** This is the
decisive test of "align the L0-GDN spine sub-op → cat9-spine drops to chain5."

1. ONE boot, eager, locked pipeline, the 4 pinned prompts (`output/fr13_acceptance_ladder/prompts_swe4.json`),
   temp 0.0 seed 1313.
2. **In-process A/B at the SAME captured tree-forward input** (decoherence-free, like the FA2 QPAD MAB and the
   L0-GDN sub-op A/B the carrier-overturned bind specifies): for each verify event capture the deep-spine row's
   **L0 GDN sub-ops pre_conv → conv1d_out → scan_out → gate_out → o_proj_out at M=10 (cat9 tree) vs M=5
   (spine-slice) vs M=1 (decode)** on the IDENTICAL input. First-nonzero sub-op vs M = the co-residency carrier;
   apply the M-invariant fix (fixed-tile forked-FA2 spine query + conv prior-window / state-bank column geometry
   keyed to the spine path not the co-resident M) and confirm the spine sub-op → 0.0 across M=10/5/1.
3. **Validation (trajectory-fixed):** with the fix ON, re-score cat9 vs the SAME `fr13_recurrent_decode_oracle`
   AND require the served stream to be byte-identical to fix-OFF on the accepted-spine positions (hold the
   trajectory; the fix must be lossless on regular decode and not fork the spine), then count clear-margin
   spine-flips. PREDICT: spine-flips 17 → ~5 (chain5 level), leaves unchanged (+15 net lossless holds),
   accept/event ≥ 3.198. If spine-flips do NOT drop with the spine sub-ops M-invariant AND the trajectory held →
   the drift is a depth-intrinsic per-forward seam not the M-shape, and the verdict reverts toward the topology
   (reshape) lever (`project_fr13_tree_reshape_unifying_lever`).

This varies ONLY the spine M-invariance with kernels/seed/prompts/trajectory fixed → it isolates the co-residency
lever from the trajectory confound that defeated the scan-recompute A/B. It is the single highest-value GPU minute.

---

## 5. MEASURED vs INFERRED (honest split)

| claim | status | evidence |
|---|---|---|
| chain5 (our kernels, spine-only) = 2 de-cascaded ≤ native 3 | **MEASURED** | `FR13_PLUS2_DECASCADE` per_prompt, same de-cascade rule all arms |
| branches add ~17 flips via co-residency; 11/11 ch2 flips ON spine | **MEASURED** | `FR13_FA2_CARRIER_OVERTURNED_BIND`, commit 2fe2c567 |
| forked-FA2 spine = same FA2 kernel + 0.0 ancestor bias (no divergent op) | **CODE-READ** | `fr13_patch_fa2_tree_bias.py` L40-78 |
| forked-FA2 = 2 ULP/983k, no depth growth; QPAD e2e null | **MEASURED** | `project_fr13_fa2_fork_nocopy_floor`, `FR13_FA2_CARRIER_OVERTURNED_BIND` |
| GDN scan op near-bit-exact (1e-6 / state rel-err 2.2e-4, K1) | **MEASURED** | `FR13_DIFFUSION_DEEP_DIVE` §2c, K1 |
| floor born L0 GDN (conv anchor 1-ULP + scan bf16-store) from 0.0 input | **MEASURED** | node5/node7 ladders, sub-op trace |
| GDN layers amplify 1.075x/layer, full-attn 1.236x/layer (geomean) | **MEASURED** | computed from node5 `per_layer_maxabs.json` (this doc §2b) |
| 1.166x/layer is a residual-stream CONNECTION effect (signal-proportional), not additive | **MEASURED** | hid_L2 ~proportional to resid_L2 across the ladder |
| "gate 1/rms amplifies 32x" | **REFUTED/INFERRED** | gate is fp32; measured GDN-layer ratios 1.0-1.29x, never 32x |
| flip = boundary-token logit collapse crossing a 0.5-1.9 nat margin, L60/L61 | **MEASURED** | node5 `per_depth_argmax` + SUMMARY |
| lever = make spine M-invariant → 17→~5 | **INFERRED** (from chain5 existence proof + co-residency mechanism); §4 validates | chain5 floor + 2fe2c567 spine-perturbation |
| scan-state recompute rose e2e flips (trajectory confound) | **MEASURED** | `FR13_SCAN_NOT_E2E_CARRIER_BIND` |

---

## 6. Reward-hack / hygiene

CLEAN: pure read of banked artifacts + committed vLLM source via `scripts/vllm_src.sh` (no /tmp); no GPU boot; no
served-path splice; no new kernel; this doc is the only write (pathspec). The recurrent decode oracle is the
deployment-correct compare target (A/B oracle only, no adoption). No self-declared pass/fail; the arbiter remains
e2e cat9-vs-E5 per-event superset gate + accept/event, brought to the user. NOT proposing K1/N_PAD (done), WY
(parked), bonus (rejected), or copy/dense.

Pairs with FR13_DIFFUSION_DEEP_DIVE.md, FR13_FA2_CARRIER_OVERTURNED_BIND.md, FR13_PLUS2_DECASCADE.md,
FR13_PEREVENT_SUPERSET_GATE_RESULT.md, FR13_GATEA_DEEP_DIVERGENCE.md, FR13_SCAN_NOT_E2E_CARRIER_BIND.md,
[[project_fr13_fa2_fork_nocopy_floor]], [[reference_diffuse_gdn_accumulation_explained]],
[[project_fr13_tree_reshape_unifying_lever]], [[feedback_math_correct_vs_bitexact]],
[[reference_scalar_metric_per_token_blindspot]].

## Sources (online corroboration of the realization-floor framing)
- vLLM tree-attention for spec decode (FA causal-path vs tree-mask compatibility): https://github.com/vllm-project/vllm/issues/18327 , https://github.com/vllm-project/vllm/issues/3960
- SpecInfer (tree branch-path oracle losslessness): https://arxiv.org/pdf/2305.09781
- BF16 rounding compounds with sequence length (depth-accumulation mechanism): https://arxiv.org/pdf/2510.26788
- SSM recurrent-state init across chunk boundaries (chunk-vs-recurrent realization diff): https://arxiv.org/pdf/2308.16369
