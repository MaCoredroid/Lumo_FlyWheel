# FR13 garble — BRANCH-node seed investigation (2026-07-12, frontier re-open)

User reopened the garble drive (the 2026-07-10 "accept within-floor" close is REFERENCE, not a
wall): native GDN with **real batching** has 0% garble, so our machinery has a specific, findable
defect that accepts an impossible token. This doc breaks down the numbers and localizes it.

## The numbers (all measured, our instruments)

### The 16 localized garbles (output/fr13_dbg/commit_trace/localizer.jsonl) split 2 / 14
- **Class B (14/16) = "impossible token accepted in verify":** the TRUE (teacher-forced) forward
  gives the committed garble a **median e^-12.4 (~1/240,000)** prob — down to **1.6e-8 at rank 15** —
  with the CORRECT token at ~0.9999. Our tree committed it anyway.
- Class A (2/16): genuine argmax flips (forward really preferred the garble; margin_nats < 0).

### The 64-layer drift ladder (output/fr13_node5_ladder/per_layer_maxabs.json) — why it's accepted
- Input embedding into layer 0: **bit-identical (0.0)**.
- Layer-0 GDN (linear_attention) output: **0.012 L2** — the seed enters HERE, from the recurrent
  state, NOT the token.
- Amplifies **×~1.166/layer** (full-attn layers dominate at ×1.2–1.6) → **178 L2 at layer 63** =
  **~14,800×**. That logit kick lifts an e^-17 token into the accept nucleus.

## Who we compare against, and why native-batched is immune
- **no-spec decode** = ground truth (gives the near-neighbor ~1e-6). bf16 sequential.
- **native MTP-5 (E5)** = native's own linear spec-decode; ~0 garble; the RIGHT reference for our
  SPINE (same kernel family).
- **branch-path oracle** (fr12_branch_path_oracle_probe.py = SpecInfer/STree path-rerun) = the RIGHT
  reference for a BRANCH node (native MTP has no branches to compare against).
- **Native-batched immune because:** independent requests each carry their OWN exact recurrent
  state, and the near-neighbor is NEVER offered as a candidate. Cross-request coupling = ~1 ULP GEMM,
  never accept-flipping (empirical: native+cache+realB8 Running=8 = 0.00%). Our tree does two things
  native never does: OFFERS the near-neighbor (a branch) AND seeds that branch's forward wrong.

## Localization — the seed is on BRANCH nodes (spine is clean)
- Spine conv prior-window was FIXED (3a9039cc: call2 conv1d_out 18.375→0.0) and chain5 (spine-only)
  kills the garble — so the SPINE path is clean.
- node5 = (0,1) = a BRANCH node still shows the 0.012 layer-0 drift.
- Every compute-only lever that attacked the SHARED/spine path is refuted: geometry (scan proven
  bit-exact both geometries, AND reverting warps reintroduces the spill = HBM tax), fp32-conv
  precision (≤2× vs 5.71× needed), M-invariance, committer (abandoned as a class). See
  FR13_GARBLE_FIX_DECISION.md, FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND.md.
- **UN-REFUTED, compute-only lever:** branch-node forward CORRECTNESS vs the path-rerun oracle,
  never tested on the current (post-conv-fix, stateless-baked) build.

## Running experiment (scripts/fr13_branch_seed_localize.sh)
Capture served cat8 per-node GDN stages (body-only; scan/recompute modes deleted), then diff each
node vs native per-path FLA replay (fr12_branch_path_oracle_probe.py), split spine vs branch.
- **BRANCH drifts, SPINE ~0.0** → confirmed: the branch-path forward is the seed → targeted
  compute-only fix (which state a branch reads/replays) → gate temp-0.6 garble + native-B8 + live SWE.
- Branch ~0.0 too → re-aim.
