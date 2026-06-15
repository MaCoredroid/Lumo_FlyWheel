# FR13 — DIFFUSION DEEP DIVE: a MEASURED per-layer account of the cat9-vs-native flip carrier (replacing the hand-wave)

Date 2026-06-15. CPU-only, READ-ONLY. This doc replaces the recited "diffuse per-layer ~1-bf16-ULP GDN
accumulation over ~48 layers, amplified ~32x by gate 1/rms, crystallizing at L60/L61" sentence with the
quantitative per-layer evidence that exists in the bank, an HONEST split of measured-vs-assumed, a growth
model derived from the captured numbers (not "~32x"), a kernel attribution, and the e2e-flip mechanism.

Grounding: vLLM source read DIRECTLY from the pinned image via `scripts/vllm_src.sh`
(0.19.2rc1.dev134+gfe9c3d6c5). Banked captures: `output/fr13_node5_ladder/` (the decisive same-boot
input-aligned per-layer ladder), `output/fr13_node7_ladder/` (p2 ladder), `FR13_GATEA_DEEP_DIVERGENCE.md`
(the post-fork spine ladder), `FR13_SCAN_NOT_E2E_CARRIER_BIND.md` + `FR13_CARRIER_REOPEN.md` (the
scan-ruled-out e2e + de-cascade). Playbook rows quoted: **#12 measurement traps**, **#10 codegen identity**.

Skeptic's posture: the "diffuse" claim has been invoked for weeks largely UNMEASURED. Where a number is a real
GPU capture it is labelled MEASURED; where it is a narrative figure ("~32x", "~1 ULP per layer") it is labelled
ASSUMED and either re-derived from data or flagged as un-backed.

---

## 0. TL;DR (the corrections this dive forces)

1. **The per-layer floor is MEASURED and it is genuinely diffuse-from-L0** — two independent same-boot
   input-aligned ladders (node5, node7-p2) both show the verify-vs-clean residual is born inside L0's GDN
   compute from a BYTE-EXACT input (input_maxabs=0.0) and grows monotonically + smoothly to L63. There is no
   isolated clean→broken spike at any single layer.
2. **The growth is NOT "~32x at the gate"** — it is a near-constant geometric compounding of ~**1.05–1.3x
   residual-L2 per layer** (measured), i.e. a ~5–30% multiplicative gain per layer, with the LARGEST single
   jumps at the deep full-attention layers (L35, L47, L51, L59, L62/63), not at the GDN gate. The "32x via gate
   1/rms" figure is ASSUMED and not supported by the captured per-layer residual ratios.
3. **The dominant kernel is the GDN recurrent SCAN realization (tree rank-1 scan vs native sequential
   roll-slot), NOT the FA2-fork and NOT the gate** — but with a sharp caveat: the per-NODE scan STATE was made
   bit-exact via recompute and e2e flips did NOT drop (they rose), so the scan op is the per-layer FLOOR SOURCE
   while the e2e CARRIER is tree-topology amplification of that floor. These are two different questions and the
   bank answers them differently.
4. **FA2-fork is NOT the per-layer floor** — it is byte-exact 14/16 calls, 2 single-bf16-ULP in ~983k (2e-6),
   max 0.0039, with NO depth growth (MEASURED). It is an AMPLIFIER at the deep full-attn layers, not the
   originator. The L63 full-attn 30.0 spike (p2) is amplification of an already-large inherited residual, not a
   fresh fork seam.
5. **Fixable-or-diffuse verdict: genuinely diffuse at the per-layer level (no single dominant alignable
   layer/kernel), BUT not irreducible** — native-E5=3 at the same model/fp8/frame is the existence proof. The
   actionable lever is TOPOLOGY (tree reshape), not a kernel seam, because the carrier is amplification of a
   small per-forward floor, not one paddable op.

---

## STEP 1 — THE MEASURED PER-LAYER DIVERGENCE TABLE

### 1a. The decisive capture: `output/fr13_node5_ladder/` (MEASURED, holds=TRUE)

Same-boot, input-aligned. Live tree-verify node-5 row (deep-spine, num_accepted=4, the carrier event = the
`Let`(9764)-vs-` ``` `(71093) p3 flip) vs the clean teacher-forced single-forward of the accepted prefix
`[0,1,3,5]`. The two arms **enter L0 BYTE-EXACT** (`per_layer_maxabs.json` input_maxabs=0.0). This is the
apples-to-apples requirement (capture-once pinned, class #12 avoided). Residual-L2 is the carrier signal (it is
what propagates through the residual stream); hid_max_abs is the per-layer block output diff.

| L | type | resid_max_abs | resid_L2 | jump-ratio | note |
|---:|---|---:|---:|---:|---|
| input | embed | — | **0.0** | — | **byte-exact (same token), apples-to-apples** |
| **0** | **GDN** | 0.00122 | **0.01205** | — | **FIRST nonzero — born inside L0 GDN from a 0.0 input** |
| 1 | GDN | 0.00183 | 0.02209 | 1.83 | |
| 2 | GDN | 0.0625 | 0.07340 | 3.32 | signal-birth, trivial absolute mag |
| 3 | full | 0.0161 | 0.28839 | 3.93 | signal-birth, trivial absolute mag |
| 4–6 | GDN | ~0.02 | 0.315→0.364 | 1.09,1.18,0.98 | settles to slow compounding |
| 7 | full | 0.125 | 0.58852 | 1.62 | full-attn step |
| 8–18 | GDN/full | 0.25–0.375 | 0.63→1.45 | **1.0–1.24** | smooth geometric growth |
| 19 | full | 0.5 | 1.80431 | 1.24 | |
| 23 | full | 0.141 | 2.03416 | 1.06 | |
| 27 | full | 0.25 | 2.51962 | 1.13 | |
| 31 | full | 0.5 | 3.46605 | 1.21 | |
| 35 | full | 1.0 | 6.80199 | **1.61** | deep full-attn jump |
| 39 | full | 2.25 | 9.20172 | 1.15 | |
| 43 | full | 0.614 | 12.0679 | 1.21 | |
| 47 | full | 0.793 | 17.1285 | **1.32** | deep full-attn jump |
| 50 | GDN | 1.594 | 25.895 | **1.26** | |
| 51 | full | 3.0 | 34.7048 | **1.34** | deep full-attn jump |
| 52–54 | GDN | 9→18.75 | 39.7→56.3 | 1.14,1.18,1.20 | residual now O(50) |
| 55 | full | 24.75 | 68.3508 | 1.21 | |
| 58 | GDN | 25.5 | 88.7943 | 1.07 | |
| **59** | **full** | 23.25 | 100.019 | 1.13 | deep full-attn |
| **60** | **GDN** | 18.0 | 113.781 | 1.14 | **clean reaches ` ``` `; final-token argmax starts crystallizing** |
| **61** | **GDN** | 12.5 | 126.992 | 1.12 | **live locks `Let`** |
| 62 | GDN | 18.5 | 163.693 | **1.29** | |
| **63** | **full** | 11.0 | 178.546 | 1.09 | last full-attn |
| final_norm | RMSNorm | — | 103.086 (max_abs 7.59) | — | **argmax flipped 71093→9764** |

**Per-layer final-token early-exit argmax** (`per_depth_argmax.json`, MEASURED): live==clean on the FINAL
TOKEN through L0–L59. The flickers at L34/39/45/46/48/50/52/53/56 are TRANSIENT early-exit projection artifacts
(those layers don't predict the final token and re-converge — class #12 trap, explicitly corrected in the
node5 bind). **The decisive, sustained final-token flip first appears at L60 (clean argmax reaches `71093`) and
locks at L61 (`9764`), holding L61→L63.** The flip mechanism: `Let`(9764) is essentially MATCHED in both arms
(live 25.38 / clean 24.80); the argmax flips only because the **` ``` `(71093) logit COLLAPSES live (15.94)
vs clean (26.60)** — a ~10.7-nat deficit on ONE token. The clean teacher-force margin is ` ``` ` −0.158 vs
`Let` −2.033 = **1.875 nats** (drive_result.json clean_reps), so the accumulated residual must move the
relative logit by ≥1.875 nats to flip — and it moves it by ~2.5 (final_norm max_abs).

### 1b. Confirming capture: `output/fr13_node7_ladder/` p2 (MEASURED)

Different prompt (p2, forward-row 4, the ` code`(1970)-vs-` files`(3425) flip, margin 0.5), same structure.
input_max_abs=0.0; **first nonzero L0 GDN max_abs 0.0078**; cos stays >0.995 through L58; then **L59 full_attn
cos 0.996→0.971** (first sharp directional drop) and **L63 full_attn max_abs 1.69→30.0** (catastrophic
amplification, Δ+28.3); final_norm cos 0.987, argmax flips. Same diffuse-from-L0 + deep-full-attn-amplification
shape, smaller absolute mag because the flip margin is 0.5 not 1.875.

### 1c. The post-fork SPINE ladder: `FR13_GATEA_DEEP_DIVERGENCE.md` (MEASURED, different geometry)

This is the tree-SPINE-vs-NATIVE ladder (not vs the clean teacher-force). It is **0.0 through full_attn L23**,
first nonzero at the FIRST GDN layer after a clean full-attn (branched L24 GDN 0.035; spine-only L45 GDN
0.0195), then compounds 0.05→0.125→…→1.9 (L58)→5.25 (L63). The onset op was pinned offline to the
**causal_conv1d 1-bf16-ULP on the anchor row** (conv1d_out 0.000977, h0_state_in byte-exact). KEY: this ladder
diverges LATER (L24/L45) than node5/node7 (L0) because the SPINE input here is byte-exact to native's chain by
construction, whereas the node5/node7 clean reference is a teacher-force of the ACCEPTED prefix at
num_accepted=4 — a co-resident deep-accept state that differs from L0. So the two ladders are NOT the same
experiment; they bracket the floor: spine-vs-native onsets at the conv anchor row; deep-accept-vs-clean onsets
at L0 GDN scan.

### 1d. MEASURED-vs-ASSUMED split (the honest accounting)

| claim | status | evidence |
|---|---|---|
| input byte-exact (0.0), so the ladder is apples-to-apples | **MEASURED** | node5 input_maxabs=0.0; node7 p2 input_max_abs=0.0 |
| first nonzero is L0 GDN (deep-accept) / L24-L45 GDN (spine) | **MEASURED** | node5/node7 L0 0.0078; GATEA L24/L45 |
| residual grows monotonic + smooth to L63 (no spike) | **MEASURED** | node5 jump-ratios all <1.7x for L4+; 0/64 re-derive mismatch |
| final-token flip crystallizes at L60/L61 | **MEASURED** | node5 per_depth_argmax (sustained L60→63) |
| the flip is a ` ``` `-logit COLLAPSE (~10.7 nat), not a `Let` rise | **MEASURED** | node5 drive_result + final logits |
| FA2-fork = 2 ULP in ~1M, no depth growth | **MEASURED** | FR13_FLOOR_WORKFLOW_VERDICT (v2 packed oracle, 983k elems) |
| GDN scan op bit-exact (recompute int-view 0.0) yet e2e flips ROSE | **MEASURED** | FR13_SCAN_NOT_E2E_CARRIER_BIND (run w7wr68z06) |
| native-E5 = 3 flips at same probe | **MEASURED** | per_prompt [0,0,2,1], FR13_PLUS2_DECASCADE native row 3→3 |
| **"~1 ULP per GDN layer, ×48 layers"** | **ASSUMED** | the L0 step is ~1 ULP (0.0078) but per-layer ADDED diff is NOT re-measured as ~1 ULP at every layer; it is a *ratio* (1.05–1.3x), not a fixed additive ULP |
| **"amplified ~32x by gate 1/rms"** | **ASSUMED / UN-BACKED** | the per-layer residual ratios are 1.05–1.3x; no single layer multiplies by ~32x; "32x" traces to an old FR12 narrative, not these captures |
| **"crystallizing at L60/L61"** | **MEASURED** (this one is real) | node5 per_depth_argmax |

---

## STEP 2 — THE GROWTH MODEL (quantified from the captured residual ratios, NOT "~32x")

### 2a. It is (b)-AMPLIFIED but in a specific, measured way: near-constant GEOMETRIC compounding

The node5 residual-L2 ladder is the cleanest series (smooth, input-aligned). Fit:
- resid_L2 goes 0.012 (L0) → 178.5 (L63). Total gain ≈ **14,800x over 64 layers**.
- Geometric-mean per-layer ratio = 178.5/0.012 ^ (1/63) ≈ **1.166x per layer** (a ~17% multiplicative gain
  per layer, averaged).
- The DISTRIBUTION of per-layer ratios (L4+): median ≈ 1.10, almost all in **[1.0, 1.34]**, EXCEPT the
  signal-birth L1–L3 (1.8/3.3/3.9, trivial absolute magnitude) and the deep full-attn jumps L35 (1.61), L47
  (1.32), L51 (1.34), L62 (1.29). **No layer multiplies by anything near 32x.**

So the model is **NOT (a) constant-per-layer additive** (an additive ~1 ULP would make resid_L2 roughly
linear in depth; it is exponential) and **NOT a single ~32x gate event**. It is option **(b) amplified =
geometric compounding ~1.17x/layer**, with the amplification distributed across all 64 layers and
concentrated extra at the deep full-attn layers. This is exactly what a CORRELATED per-layer realization diff
running through a residual network does: each layer's output diff is roughly proportional to the residual it
receives (the network is locally near-linear in the residual), so the diff scales with the signal — geometric,
not additive.

### 2b. WHERE the gate actually enters (read from source via vllm_src.sh)

`RMSNormGated.forward_native` (layernorm.py L455-503, MEASURED source):
```
x = x.float()                                   # gate compute is FP32 internally
if z is not None and not norm_before_gate: x = x * silu(z)
variance = x_group.pow(2).mean(dim=-1)          # per-group (group_size) RMS
x_normed = x_group * rsqrt(variance + eps)      # 1/rms factor, data-dependent
out = x_normed * weight  (then .to(orig_dtype)) # bf16 store boundary
```
The `1/rms = rsqrt(mean(x^2)+eps)` factor IS data-dependent and CAN be large when the group's RMS is small —
that is the kernel basis of "gate amplifies." BUT: (i) it is computed in **fp32**, so the gate is not itself a
bf16-ULP SOURCE; it amplifies the diff already present in `x`. (ii) The captured per-GDN-layer residual ratios
(node5: the GDN layers are L0,1,2,4,5,6,8,9,10,12... and their jump-ratios are 1.0–1.29) show NO systematic
~32x multiply at the GDN layers — if the gate multiplied a per-layer diff by 32x we would see ~32x jumps at
every GDN layer, and we do not. The gate's 1/rms is a real per-element amplifier WITHIN a layer's GDN→o_proj
chain (it makes a 1-ULP scan diff into a larger o_proj-input diff INSIDE the layer), but the LAYER-OUTPUT
residual ratio it produces is ~1.1x, not 32x. **So "amplified ~32x by gate" overstates by ~30x; the measured
per-layer amplification is ~1.1–1.3x and the gate is one contributor to it, not a 32x multiplier.**

### 2c. One layer traced end-to-end (in_proj → conv → scan → gate → o_proj)

From `FR13_GATEA_DEEP_DIVERGENCE` (offline per-stage diff at the onset layer, MEASURED) + the sub-op
M-invariance evidence (`FR13_RESIDUAL13_RESOLVED_DEPTH_INTRINSIC_BIND`):
- `pre_conv` (in_proj output) = **0.0** (in_proj_qkvz fp8 GEMM is M-invariant, block-scaled, no split-K → no
  realization diff on the spine data path).
- `conv1d_out` = **0.000977 (1 bf16 ULP)** ← the FIRST nonzero at the onset layer (the value-dependent anchor
  row; bf16-tap + ex2.approx silu rounding edge — NOT the tap dtype, proven by the fp32-tap regression).
- `h0_state_in` = **0.0** (byte-exact bank row; state-handoff RULED OUT).
- `gdn_scan_out` = **1e-6** (scan op near-bit-exact; native `fused_sigmoid_gating_delta_rule_update_kernel`
  computes b_h in **fp32**, l2norm `rsqrt(sum(b_q*b_q)+1e-6)` in fp32, then **stores b_h.to(bf16)** — the bf16
  STORE boundary is where the per-token ULP enters and feeds the next token recurrently).
- `gate_out` = **0.000488** (the 1/rms amplifies the 1e-6 scan diff + the conv seed within the layer).
- `o_proj_out` = **0.00195** (bf16 GEMM spreads it across channels).

So the ULP ENTERS at the conv anchor-row bf16/silu rounding (onset layer) and at the GDN scan bf16-store
(every layer), and GROWS by (i) the gate 1/rms within the layer and (ii) the residual-stream compounding
across layers. The o_proj is a SYMPTOM (fix the input → it cascades), consistent with the FR12 finding.

---

## STEP 3 — WHICH KERNEL DOMINATES (attribution)

| kernel | per-layer floor contribution | depth growth? | deterministic? | verdict |
|---|---|---|---|---|
| **GDN recurrent SCAN** (tree rank-1 `_tree_gdn_kernel` vs native `fused_sigmoid_gating_delta_rule_update`) | **DOMINANT floor source** — fp32 compute, **bf16 STORE of b_h** each token, fed recurrently → compounds | **YES (recurrent)** | yes (Triton, fixed grid) | the leading edge of the diffuse residual; born at L0 in the deep-accept arm |
| **causal_conv1d** (anchor row) | 1 bf16 ULP at the onset spine layer (0.000977) | seeds, then conv is per-row (no recurrence) | value-dependent edge | the SPINE onset seed; not the deep-accept L0 source |
| **gate** (RMSNormGated) | AMPLIFIER within a layer (1/rms, fp32) — NOT a bf16 source | n/a (per-layer) | yes | amplifies the scan/conv diff ~1.1–1.3x/layer at the layer output, NOT 32x |
| **fp8 in_proj_qkvz / o_proj** | **~0** on the spine data path (M-invariant, BLOCK_SIZE_M=64 constexpr, no split-K) | no | yes | NOT a per-layer floor source; o_proj only SPREADS an inherited diff across channels |
| **in_proj_ba** (the ONE bf16 GEMM) | the banked ~8-flip co-residency fix (M-keyed) | no | yes | a real ALIGNABLE seam for the leaf co-residency component — already FIXED (LUMO_FB pad) |
| **FA2-fork** (full-attn tree-bias) | 2 single-ULP in ~983k (2e-6), max 0.0039, **NO depth growth** | **NO** | **probabilistic tie-break** (not a per-row floor) | AMPLIFIER at deep full-attn (L59/L63), not the originator |

### The decisive nuance the bank forces (scan: floor-source ≠ e2e-carrier)

The scan is the per-layer FLOOR SOURCE (fp32-compute / bf16-store recurrent realization diff, born at L0 in the
deep-accept arm). **BUT the recompute A/B (FR13_SCAN_NOT_E2E_CARRIER_BIND, MEASURED, non-vacuous triple-proven)
made the per-node scan STATE bit-exact (int-view 0.0) and the e2e clear-margin flips ROSE 23→32, not dropped.**
Reconciliation:
- The scan-STATE bit-exactness removes the LEAF co-residency state diff, but the cat9 committer then walks a
  DIFFERENT trajectory (different verify logits at near-ties → different LCP-max path → ~369 token-diff stream)
  with its OWN high-entropy boundaries — comparable-or-more flips.
- So the scan op is the per-FORWARD floor's leading edge, but it is NOT the dominant e2e CARRIER. The e2e
  carrier is **the tree-topology amplification of the small per-forward floor**: cat9 runs MORE divergent
  forwards than native's single spine, and a near-tie crossing can fork into a degenerate basin (the p3
  `<ctrl:token>` block) that spawns a downstream cluster.

**Attribution verdict:** per-LAYER floor = GDN scan bf16-store recurrent realization diff (dominant) + conv
anchor-row 1-ULP (onset seed) + FA2-fork (amplifier only). e2e CARRIER = tree topology amplifying that floor +
the LCP committer walking a forked trajectory. The two are different and the bank measures both.

---

## STEP 4 — FIXABLE-OR-GENUINELY-DIFFUSE VERDICT (with numbers)

**It is genuinely diffuse at the per-layer level — there is NO single dominant alignable layer/kernel whose fix
lands native — BUT it is not irreducible.** The numbers:

1. **No dominant layer.** The node5 per-layer residual contributions are COMPARABLE and NUMEROUS: ~50+ layers
   each contribute a 1.05–1.34x multiplicative step; the 4–5 largest (L35,L47,L51,L59,L62) are 1.29–1.61x —
   only ~2–4x above the median 1.10x, not 100x above. Zeroing any ONE layer's step changes the L63 residual by
   its ratio (~1.1–1.6x) — never enough to recover a 1.875-nat margin. So no single-layer or single-op patch
   recovers native. This is the genuine sense of "diffuse": ~50 comparable contributions.
2. **No alignable single KERNEL seam (the few-vs-many question RESOLVED).** Every per-forward sub-op was driven
   to bit-exact and re-measured (FR13_RESIDUAL13_RESOLVED_DEPTH_INTRINSIC): conv = row-occupancy M-invariant;
   GDN scan op = RAW 0.0 at both BV geometries (BV is a reduction over DIM_K, geometry-invariant —
   `FR13_BV_GEOMETRY_NOT_THE_SEAM`); fp8 in_proj/o_proj = M-invariant; gate = M-invariant per-row. The ONLY
   M-keyed bf16 GEMM was in_proj_ba (the +8 co-residency seam) — already FIXED. After that, batch-invariance is
   EXHAUSTED → the residual is the chunk-vs-recurrent realization diff with no paddable op.
3. **But NOT irreducible** (native-E5=3 is the existence proof, MEASURED at the same probe). The
   chunk-vs-recurrent diff is a REALIZATION difference (our deep-accept builds node-5's state via rank-1
   tree-scan over [0,1,3,5] seeded from b_h0; clean builds it via a 1687-token chunked-prefill scan) — same
   logical state, different fp realization (bf16-store accumulation order). Match native's realization bit-for-
   bit and we accumulate like native (3-flip floor). Native does NOT have a counterpart tree-scan, so there is
   no "align our kernel to native's kernel" for the TREE forward — the alignment lever is TOPOLOGY.
4. **The actionable lever is therefore TOPOLOGY, not a kernel fix** (quantified): a shallower / root-sibling
   tree cuts the co-resident accept depth (fewer deep nodes → less state-feed realization drift) AND reduces
   basin amplification. Precedent (FR13_PLUS2_DECASCADE, MEASURED): deep 5-spine chain de-cascades to **2
   independent flips ≤ native 3**; the shallow-but-deep chain3 stayed at 5 dispersed. So reshape moves the
   per-forward-floor-induced flips toward native. The binding arbiter is e2e accept/event vs E5, NOT the raw
   flip count (cascade- and length-inflated, class #12).

**One caveat to the "no seam" verdict (skeptic's flag):** the per-NODE scan recompute was bit-exact yet did
not help e2e — this CHALLENGES "fixing the scan realization is the lever." It is consistent with diffuse +
topology-carrier, but it also means a kernel alignment that makes the TREE forward's deep-accept scan match the
clean chunked-prefill scan has NOT been demonstrated to reduce flips. If a future bit-exact tree-vs-clean scan
were built and STILL didn't drop flips, that would confirm topology-only; if it DID drop them, the "diffuse"
label would partially revert to "one un-aligned recurrent seam." The recompute result leans toward
topology-only but is not the same experiment (recompute changed the trajectory).

---

## STEP 5 — CONNECT TO E2E: how the per-layer floor becomes the ~16 structural-boundary flips

### 5a. The per-token mechanism (MEASURED at node5/node7)

At a flip position the verify forward accumulates the diffuse residual to final_norm (node5 max_abs 2.5 / L2
103; node7-p2 max_abs 2.5). The lm-head GEMV over the verify rows then produces a logit vector whose argmax
differs from clean ONLY when the accumulated residual exceeds the token's clean margin:
- node5 (p3): clean margin ` ``` ` vs `Let` = 1.875 nat; the residual collapses ` ``` ` by ~10.7 nat → flip.
- node7 (p2): clean margin ` code` vs ` files` = 0.5 nat; residual moves it → flip.

The flip is at a HIGH-ENTROPY STRUCTURAL BOUNDARY (codefence ` ``` `, prose `Let`, tool-call, JSON brace) —
exactly where the clean margin is small (1–2 nat) so the ~2.5-max_abs accumulated residual can cross it. At
FORMAT-FIXED positions the clean margin is huge (many nat) and the same residual does NOT flip (the oracle
agrees at dev=0). This is why the flips cluster at boundaries: it is a margin × residual interaction.

### 5b. Reconciling the per-layer floor (~ULP/layer) with the "dev up to 10" boundary flips

The carrier-reopen says the ~16 confident flips have **dev up to 10 (NOT sub-ULP near-ties)**. This is NOT a
contradiction with a per-layer ~ULP floor — it is the COMPOUNDING resolving it: the per-layer diff is ~1 ULP
at L0 (0.0078) but compounds geometrically ~1.17x/layer to **final_norm max_abs ~2.5 and L2 ~103** (MEASURED).
A 2.5-max_abs / 103-L2 final-norm perturbation, pushed through the lm-head GEMV, easily produces a 10-nat
logit swing on a single token (the ` ``` ` collapse was ~10.7 nat). So:
- The per-layer FLOOR is ~1 ULP (sub-noise) — TRUE.
- The ACCUMULATED divergence at the flip is dev~10 (clear-margin, structural) — ALSO TRUE.
- They are the SAME phenomenon at two depths: the floor AMPLIFIES (geometric compounding ×~14,800 over 64
  layers + the lm-head GEMV) to dev~10 at the boundary. **The floor claim is NOT revised — it is reconciled:
  ~1 ULP at L0 → ~2.5/103 at L63 → ~10 nat post-lm-head, and the flip needs only to beat a 0.5–1.9 nat
  margin.** The "structural-boundary, dev up to 10" is the boundary's SMALL clean margin being crossed by the
  fully-amplified floor, not a separate large per-forward seam.

### 5c. Why ~16 (cat9) vs 3 (native) — the topology multiplier (MEASURED de-cascade)

- native MTP-5: ONE linear spine forward, FLASH_ATTN, no tree → 3 flips (per_prompt [0,0,2,1]).
- cat9: a 9-node caterpillar with TWO extra divergent kernels (forked-FA2 tree-bias + GDN tree-scan) and more
  forwards per step → ~13–15 genuine in-topk near-tie crossings of the SAME boundaries native crosses, PLUS
  ~5 fork-progeny (the p3 ctrl-basin = 1 root fork + 5 downstream) → raw 23, de-cascaded ~14–18.
- The excess over native = (more divergent forwards) × (tree amplification into basins). Reduce either
  (topology reshape) and the excess drops toward native — the lever, not a kernel.

---

## STEP 6 — MINIMAL GPU RE-CAPTURE TO CLOSE THE REMAINING GAP (if needed)

The banked node5/node7 ladders are RECENT (2026-06-13/14), input-aligned, holds=TRUE, and 0.19.2-keyed
(the captures ran on the pinned image). They are NOT stale for the per-layer growth model. The ONE gap the bank
does NOT close is whether the diffuse per-layer floor, if driven to per-layer bit-exact in the TREE forward,
reduces e2e flips (the recompute A/B changed the trajectory, so it tested a confound, not the floor). Two
minimal captures, in priority:

1. **(decisive, ~minutes, 1 GPU boot) Tree-reshape A/B at fixed kernels/seed/prompts.** Run cat9 with the tree
   reshaped to a shallow root-sibling tree (depth-3 spine + 2 root siblings) at temp 0.0 seed 1313 on the SAME
   4 prompts, re-score vs the SAME `fr13_recurrent_decode_oracle`. PREDICT: de-cascaded independent flips drop
   toward native (chain5→2 precedent) AND accept/event holds/improves. This isolates the TOPOLOGY multiplier
   (varies ONLY topology). If flips do NOT drop, the carrier is a depth-independent per-forward seam and the
   FA2-fork "no depth growth" finding must be re-opened as deterministic. This is the carrier-reopen's proposed
   test and it is the single highest-value GPU minute.

2. **(closes the floor-source gap, ~minutes) Per-layer ladder at a SHALLOW node** (num_accepted=1 spine node)
   vs clean, SAME boot, input-aligned. The banked ladders are at deep-accept (num_accepted=4). A num_accepted=1
   ladder would show whether the L0-GDN birth is the deep-accept chunk-vs-recurrent diff (predict: shallow node
   onsets LATER, like the GATEA spine L24/L45) — confirming the floor source is the DEEP-ACCEPT recurrent
   state-feed, not a generic per-layer GDN seam. Capture: `FR10_LAYER_HIDDEN_CAPTURE` + `_ROWS` at the spine
   row, `NUM_TOKENS` gated, vs the clean teacher-force of the 1-token-accept prefix. Reuse the node5 reduce.py.

Do NOT re-capture the basic per-layer growth model — it is measured (node5/node7). Do NOT bake recompute (it is
a different deterministic stream, NOT lossless, and WORSE).

---

## Playbook rows quoted (FR13_BUG_CLASS_PLAYBOOK)

- **#12 Measurement traps** — applied throughout: the input byte-exact (0.0) requirement makes the ladder
  apples-to-apples; the per-depth early-exit argmax flickers (L34/39/45...) are the "per-pos counter indexing
  the wrong projection" trap and were excluded; the raw 23/32 flip counts are length- and cascade-inflated and
  are NOT the arbiter (accept/event is). The "~32x gate" figure is exactly the kind of un-backed estimate this
  row warns against — re-derived to ~1.1–1.3x/layer from the captured ratios.
- **#10 Shared-source ≠ shared-SASS (codegen identity)** — the per-layer floor is a codegen/realization
  identity question: our tree rank-1 scan and native's `fused_sigmoid_gating_delta_rule_update` are both
  fp32-compute/bf16-store sequential scans but compile/accumulate differently (tree-mask ancestor replay vs
  single roll-slot). The bf16 STORE boundary (`b_h.to(element_ty)`) is the int-view A/B point; the diff is a
  realization (rounding/order) identity gap, not a math bug. The recompute int-view 0.0 is the byte-A/B that
  proved the scan STATE alignable — but per #10 that did not transfer to e2e (different SASS path → different
  trajectory).

## Reward-hack / hygiene
CLEAN: pure read of banked artifacts + committed vLLM source via `scripts/vllm_src.sh` (no /tmp cache); no GPU
boot; no served-path splice; no new kernel; this doc is the only write. The recurrent oracle and native
packed-decode are A/B oracles only (no adoption — that would be the ORACLE_FRAME reward-hack). No self-declared
pass/fail; the arbiter remains e2e cat9-vs-E5 (per-depth-argmax + bag-TV ≤ floor + accept/event ≥ native),
brought to the user.

Pairs with FR13_CARRIER_REOPEN.md, FR13_SCAN_NOT_E2E_CARRIER_BIND.md, FR13_NODE5_LADDER_DIFFUSE_BIND.md,
FR13_NODE7_LADDER_BIND.md, FR13_GATEA_DEEP_DIVERGENCE.md, FR13_RESIDUAL13_RESOLVED_DEPTH_INTRINSIC_BIND.md,
FR13_PLUS2_DECASCADE.md, [[reference_diffuse_gdn_accumulation_explained]],
[[project_fr13_fa2_fork_nocopy_floor]], [[reference_scalar_metric_per_token_blindspot]],
[[project_fr13_tree_reshape_unifying_lever]], [[feedback_math_correct_vs_bitexact]].
