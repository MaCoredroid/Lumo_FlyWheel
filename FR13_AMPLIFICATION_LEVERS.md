# FR13 — Amplification-reduction / "keep drift in the argmax floor" levers (user 2026-06-15)

> **SUPERSEDED AS THE PRIMARY LEVER (2026-06-15, by wsvy4vn5k / FR13_E5_VS_CAT9_SPINE_DRIFT.md, verify HOLDS).**
> The study answered the gate-keeper question below: the 1.166x/layer amplification is **shared with E5** (E5
> rides the SAME residual-stream compounding — GDN 1.075x/layer, full-attn 1.236x/layer), so amplification-
> reduction **does NOT close the cat9-vs-E5 gap**. The ~14 excess (cat9-spine 17 vs E5-spine 3) is **co-residency
> M-dependence** at the **L0 GDN birth-amplitude** (existence proof: `chain5`, cat9's exact kernels on the spine
> alone at M=5, de-cascades to 2 ≤ native 3). The PRIMARY lever is now **make the spine rows M-INVARIANT** (reach
> 17→~5), pursued by the math-expert design workflow (wdq98mpeo) + the GPU sub-op A/B. Amplification-reduction
> survives only as an **orthogonal SECONDARY** for the residual 5→3 (it shrinks the shared compounding for both
> arms equally, so at best it lowers the absolute floor, it does NOT differentially fix cat9). Keep the levers
> below on the bench for that secondary use; do NOT pursue them as the path to native-3.

Captured thoughts to spin out AFTER the E5-vs-cat9 spine-drift study (wsvy4vn5k) lands (it decides whether the
amplification is reducible). A SECOND class of lever beyond the bit-exact-alignment grind we'd over-focused on.

## The reframe (the key idea)
A flip happens iff **accumulated drift > the argmax margin** at the lm-head. So you do NOT have to drive the
per-layer drift to 0 (bit-exact = hard/diffuse). You can instead **keep the drift BELOW the margin (in the
"floor")** so it never crosses. Often cheaper than bit-exact. The 17 cat9-spine flips are the positions where
the accumulated drift crossed a small (1-2 nat) structural-boundary margin; the diffusion deep-dive measured the
drift growing ~1.166x/layer (14,800x over 64 layers) riding the residual stream, biggest jumps at deep full-attn
L35/47/51/62 + the gate 1/rms.

## Lever class A — amplification-reduction (keep drift in the floor)
- **Targeted fp32 at the amplification HOTSPOTS only** (NOT whole-model — that's an HBM/bandwidth tax on a
  273 GB/s part): the deep full-attn layers L35/47/51/62 (the biggest jumps) + the **gate 1/rms** (RMSNormGated:
  small activation -> large 1/rms -> it MULTIPLIES the drift). Compute just those in fp32 so they stop
  amplifying. Cheap, compute-only, no copy.
- **Clamp/floor the gate rms denominator** so a near-zero activation can't blow up 1/rms (a classic numerical-
  stability move) -> caps the per-layer amplification of the drift.
- **Periodic residual re-anchoring** so the drift doesn't compound 14,800x over 64 layers.

## Lever class B — unification (make the two forwards agree where it matters)
- Make cat9's verify forward and the decode forward share IDENTICAL numerics at the drift-CRITICAL points only
  (deep full-attn + gate), not everywhere.
- **Margin-aware near-tie targeting**: the flips live at structural-boundary near-ties; a targeted higher-
  precision pass ONLY at those positions keeps them in the floor without a full alignment.

## The honest catch (the study wsvy4vn5k must resolve FIRST)
The diffusion deep-dive called the 1.166x/layer growth **"signal-proportional"** (drift ~ a fixed fraction of
the residual signal). IF strictly true, amplification-reduction is HARD (the drift grows with the signal, which
you can't shrink). BUT the gate 1/rms blow-up + the deep-full-attn jumps are exactly where it might be a
**reducible numerical** component, not pure proportionality. **Which it is decides whether these levers fire** -
that is what wsvy4vn5k decomposes (GDN-compute vs residual-stream-connection vs full-attn; is the 1.166x
reducible).

## The spin-out experiment (after wsvy4vn5k, IF reducible)
GPU: apply the top amplification-reduction lever (likely fp32 the gate + the 4 deep-full-attn hotspots, default-
OFF flag), re-score cat9 vs the recurrent oracle -> how many of the 17 spine flips drop BELOW margin (toward
E5's 3)? Combine thoughts into ONE focused experiment informed by the study's WHERE/reducibility answer.
CONSTRAINTS: keep cat9 leaves (the per-event superset is net-positive +15, FR13_PEREVENT_SUPERSET_GATE_RESULT);
no copy/HBM tax; targeted not whole-model fp32; default-OFF byte-identical; NOT K1/N_PAD/WY/bonus (done/parked/
rejected). Gate = does it drive the 17 spine flips toward 3 while holding accept/event + the superset net.
