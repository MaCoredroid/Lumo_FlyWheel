# FR10 No-Copy GDN Tree Verify — Resolution Pass (math + bug + literature)

**Date:** 2026-06-05 · Combines two adversarially-verified investigations: a code/math resolution workflow (mine git history + map both kernels + CPU repros + adversarial synthesis, 17 agents) and a deep-research literature pass on the closeout's cited papers (97 agents, 3-vote claim verification). Supersedes the "diffuse drift, no fixable seam, no-go on evidence" framing of `FR10_CLOSEOUT.md` — **not by overturning the no-go, but by showing it was under-investigated.**

---

## TL;DR — combined verdict: **NEEDS-ONE-GPU-PROBE** (no-copy NOT proven dead)

The closeout closed no-copy on two claims: (1) the literature makes it hopeless, and (2) the 0.0156 layer-0 drift is "diffuse, no locus, no fixable seam." **Both claims are weaker than stated.**

1. **Literature: no theoretical no-go exists.** No cited paper proves impossibility. STree (the central paper) is *itself* no-copy and *refutes* the "shared state degrades path0" belief. The scary 0.038 (Component-Aware) is orthogonal self-speculation, and that paper *endorses* our exact direction. Full detail in `FR10_PAPER_NOGO_RESEARCH.md`. → **NOT-CATASTROPHIC.**
2. **Code: a real, previously-missed source seam was found** (conv tap dtype), the GDN scan was exonerated to 7.45e-9, and **two decisive boot-free GPU probes were never run** — one of which (state-handoff) could flip the whole verdict to BUG-FIXABLE.

Net: no-copy stays **provisionally banked** (the acceptance deficit 1.77 vs 3.076 is large and no cheap lossless path is yet proven), but the precision-floor-vs-fixable-bug question is **genuinely open** and decided by ~2 hours of boot-free GPU measurement, not by anything currently known.

---

## What the literature settles (see FR10_PAPER_NOGO_RESEARCH.md)

- **STree (2505.14969)** is no-copy (one shared state + tree mask; post-accept by recompute) and preserves path0 exactly *under a diagonal state transition*. Its losslessness construction is **validated only on diagonal Mamba2/S6 and never covers the gated delta rule** ("delta" never appears in the paper). GDN's rank-1 `(I−βkkᵀ)` update is non-diagonal — **but our scan already solves that case directly** (exonerated to 7.45e-9 below), so STree's diagonal limitation is not our blocker. STree covers only the scan, **not conv, not the drafter** — which is exactly where our seam turned out to be.
- **Component-Aware (2605.01106):** the 0.038 is component self-speculation off-distribution, not external-MTP + tree-verify; LayerSkip gets 12× higher on the same model; the paper explicitly endorses tree-verify (STree) as the way forward. Orthogonal.
- **GDN-2 (2605.22791):** pure architecture, no decoding content. Orthogonal.
- **No impossibility theorem** for fixed-recurrent-state tree verify exists anywhere surveyed; the "superset-only" claim is an engineering observation, not a theorem.

## What the code investigation found (adversarially corrected)

### SOLID
1. **A real source seam the byte-native gates missed — conv tap-multiply dtype.**
   - Native Triton `causal_conv1d_update` multiplies width-4 taps as a **bf16 product** then fp32-accumulates (`acc += matrix_x * matrix_w`, both bf16; live `causal_conv1d.py:442`). Triton `bf16*bf16` rounds the product to bf16 *before* the fp32 add.
   - FR10 tree conv **upcasts taps to fp32** before multiplying (`src/lumo_flywheel_serving/fr10_tree_conv.py:59-61,68,72`; live patch `scripts/fr10_phase4_patch_vllm_tree_gdn.py:576,608-611`). Window/index are byte-correct (tree-vs-serial 0.0); the *only* divergence is arithmetic dtype.
   - **Why the gates missed it:** the scan gate compared tree-vs-update kernels on *shared post-conv inputs* (conv baked in identically to both sides); the tree-conv parity test used *fp32 inputs + fp32 serial reference* — both structurally blind to a bf16-vs-fp32 tap product. This is a genuine, useful finding.
   - **Cheap candidate fix:** drop the `.to(torch.float32)` upcast on the taps → tree conv matches native **bit-for-bit (0.0)**.
2. **The GDN scan is exonerated.** Tree kernel matches the native update primitive to **7.45e-9**; recurrent-step / softplus-g / beta seams all 0.0 at T=1; scale & l2norm placement algebraically identical (~1e-9). The hard, literature-open piece (non-diagonal tree scan) is solved and banked.
3. **Precision floors (measured, faithful CPU refs):** all-fp32 chunk-vs-recurrent = **7.45e-9**; faithful native bf16-matmul single-chunk vs dense fp32 = **6.7e-5**. The 5–11-token spine fits in **one** FLA chunk (size 64), so **0.0156 is NOT a cross-chunk algebra floor** (~230× above 6.7e-5). This kills the strong "chunked-vs-dense algebra floor" reading.
4. **Softmax washout:** at T=0.6 over the 12 aligned spine rows, **0/12 argmax flips** despite the drift — consistent with a precision-class (marginal) effect on acceptance.

### RETRACTED / DOWNGRADED by adversarial review (honesty)
- ❌ "The conv seam exactly reproduces 0.0156 at layer-0." **Tensor confusion.** The 0.0156 the conv reproduces lives in **conv-output / q-k-v space** (mag ~1). The live 0.0156 lives in **residual-hidden space** (mag ~1.9). Faithfully propagated, the conv seam reaches only **~1.2e-4 at the GDN output** — 30–130× *below* the observed 0.0156. **No probe propagates the deterministic seam through `o_proj`+residual to a 0.0156 residual drift.**
- ⚠ "Single-seed amplification, slope 1.21." The slope **exists in no artifact** and is uncomputable from the data (no per-layer native-magnitude series). There are **5** layers with >3× ratio (incl. L58 5.94× unexplained), not 2. Growth from 0.0156→53 is **equally consistent** with accumulating per-layer bf16 rounding.
- ⚠ "ULP-lattice fingerprint = proof of precision." **False discriminator.** *Any* bf16-stored tensor lands on the ULP lattice regardless of cause; it confirms only bf16 storage (never in dispute), not precision-vs-logic-bug.

## The two decisive measurements (boot-free GPU; neither was run)

Both replay captured tensors — **no model load**. Until both run, PRECISION-FLOOR is unproven and BUG-FIXABLE is not excluded.

**Probe α — faithful conv-seam → residual propagation.** Replay one captured bf16 `mixed_qkv_spec`/`conv_weights`/`bias` through (a) native `causal_conv1d_update` vs (b) fp32-product tree conv, then push **both through the real `o_proj` + RMSNormGated + residual** to the layer-0 residual-hidden output.
- `|residual(a) − residual(b)| ≈ 0.0156` → conv-dtype seam is **causally sufficient** → PRECISION-FLOOR, and the one-line conv fix is worth testing for acceptance recovery.
- `≈ 1.2e-4–5e-4` (what every faithful probe so far suggests) → conv seam is **NOT** the source of the live 0.0156 → locus is elsewhere; closeout's "no single-layer locus" partly reinstated.

**Probe β — live event-0 state-handoff byte-compare (the one that could flip to BUG-FIXABLE; never run).** At decode-event-0 dump the spine's initial recurrent state `h0[ssm_state_indices[i_n, num_accepted_tokens-1]]` and the conv prior-state window (`conv_state_token_offset = num_accepted_tokens-1`) loaded by the native MTP-5 forward, byte-compare to what the FR10 tree spine loads.
- **Match** → #39273/#40738 wrong-state class excluded for the seed (prior evidence favors this: `ssm_next_vs_native 2.86e-6`, `conv_next_vs_native 0.0`).
- **Differ by ≥1 column** → **#40738-class wrong-`initial_state` bug → verdict flips to BUG-FIXABLE**; fix = port #40738's `conv_state_token_offset`/`num_accepted_tokens` threading, then confirm accept recovers toward 3.076 in one B=4 metrics-off run.

Estimated flip probability ~0.2 on priors, but it was **never directly differenced on the live spine** — and the original closeout shipped a no-go while naming this exact unmeasured fact as the only thing that could change it. That gap must be closed before no-copy is closed for good.

## Decision

- **No-copy: provisionally banked, not dead.** Run Probe β first (cheap, decisive, could flip to BUG-FIXABLE), then Probe α. Total ≈ a couple hours of boot-free GPU replay. If both come back precision-floor + handoff-match, *then* close no-copy with a real evidentiary basis (which the closeout lacked) and proceed to copy-recurrent multi-spine. If β differs or α hits 0.0156 with the conv fix recovering acceptance → no-copy revives at near-zero cost.
- **Dead-ends NOT to retry** (already measured): M-RoPE explicit broadcast (→1.20); TREE_ATTN backend (1.49 < 1.77); big-tree kernel speed opt (can't beat 135µs FLA flat); byte-exact-vs-MTP-5-baseline (wrong bar).
- **Cheap aligned-numerics win regardless:** tree conv should match native's bf16 tap product (drop the fp32 upcast) so the verify distribution matches what acceptance is measured against — correct-direction and one line, independent of α/β.

## Artifacts
- CPU probes + results: `output/fr10_nocopy_resolve/probe_{0,1,1b,2,2b,3,3b,4}*.{py,json}`
- Live source seam: `causal_conv1d.py:442,859-953`, `fused_sigmoid_gating.py:106`; tree seam `src/lumo_flywheel_serving/fr10_tree_conv.py:59-72`; patch `scripts/fr10_phase4_patch_vllm_tree_gdn.py:576,608-611`
- Per-layer drift: `output/fr10_match0_layer_compare_*/aligned12_layer_compare.json`
- **To write before any final verdict:** `output/fr10_nocopy_resolve/gpu_conv_seam_replay.py` (Probe α) + a Probe β state-handoff dump.
- Literature: `FR10_PAPER_NOGO_RESEARCH.md`
