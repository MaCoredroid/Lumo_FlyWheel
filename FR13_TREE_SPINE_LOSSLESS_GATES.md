# FR13 — Gate ladder to confirm the tree-GDN-kernel carrier + validate the fix, on BOTH tree & spine

**Date:** 2026-07-02. Workflow wtlxfpoil (6 agents) + adversarial verdict (sufficient=False → made runnable).
Carrier established (wkrqdt1gl): cache-seeded ssm_state consumed by `_tree_gdn_kernel`/`_gdn_node_step`
(fp32-carry, fr10_gdn_tree_kernel.py:693) = different rounding than native chunked-FLA (patch:4189-4207);
spine breaks too (mode-gated dispatch). Fix = align rounding OR recompute kernel (FR13_SCAN_ALIGN_MODE=recompute).

## The ladder (ordered; every gate runs BOTH spine=chain5 AND tree=cat8)
**G1 — KERNEL-CONFIRM (deterministic, seed-fixed, cheapest, RUN FIRST).** Capture ONE decode-step tree payload,
diff `gdn_scan_out` per node across 3 arms on the SAME seed: (A) default h_cache `_tree_gdn_kernel`, (B) recompute
kernel (SCAN_ALIGN_MODE=recompute), (C) native FLA replay per root→node path. Split spine-nodes vs branch-nodes.
**PASS:** default-vs-FLA > ~1e-3 on spine AND branch nodes (kernel is the carrier) AND recompute-vs-FLA == 0.0
bit-exact on all nodes (recompute is the fix) → confirms `_gdn_node_step` locus + **exonerates FA2/TREE_ATTN by
elimination** (byte-identical across all 3 arms). REUSE `scripts/fr12_branch_path_oracle_probe.py` (already does
A-vs-C with spine/branch split, probe:236,360). No temp 0.6, no SWE agent, one capture + offline diff.

**G2 — DEPLOYMENT LOSSLESS (temp 0.6, real prompts, no teacher-force).** Sig-0 (ADDED per verdict): temp-0.6
same-prompt q-vs-p TV, default-tree vs native — the one *sufficient* distributional gate every prior conclusion
skipped. Sig-1: carrier_locator_grid cells N / D0,D(spine) / B0,B(tree) same-prompt malformed + accept. Sig-2:
SWE qwen-code resolve on 13453. PASS: D and B each match N. (Sig-1/2 are non-inferiority = necessary-not-sufficient;
Sig-0 TV is the confirmer.)

**G3 — FIX-VALIDATION.** 7 boots cache-ON: {chain5,cat8}×{body,recompute}+native, real SWE. Three legs BOTH
topologies: KERNEL (G1 flips==0 under recompute), BEHAVIOR (Fisher resolve recompute-vs-native p>0.05, accept≥3.2),
SPEED (recompute TPS ≥ 0.90×default). Reuse fr13_apc_e2e_lossless_gate.sh + fr13_measure.py; build thin driver.

## Adversarial verdict (sufficient=False as written) — fixes REQUIRED before first run
1. **BLOCKER / build item 0:** G1 cannot boot — `patch:4346 _fr13_replay_route_on = True` is BAKED, and
   patch:4353-4366 RAISES the moment FR10_TREE_GDN_CAPTURE_PAYLOAD is set. **Must un-bake REPLAY_ROUTE (env-gate
   patch:4346) first.** This is the true blocker, not the probe extension.
2. **Oracle identity:** G1's reference is `fr12_branch_path_oracle_probe._native_path_scan` (probe:246, a fresh
   FLA call on captured q/k/v) — NOT `FR12_NATIVE_SPINE_ORACLE` (a different in-serving substitution that also trips
   the REPLAY_ROUTE raise). Use the probe's FLA-replay.
3. **G1 spine arm is redundant as a second boot:** the probe splits spine-vs-branch nodes WITHIN one cat8 payload
   (spine_nodes = the root→leaf chain), so ONE cat8 capture already proves "spine nodes route to _gdn_node_step and
   break." chain5 becomes a cheap config check (does MODE-gating reach the kernel), not a second capture.
4. **G2/G3 behavior floors only fail-to-reject** — a lossy kernel can resolve 13453 by luck at temp 0.6. Hence
   Sig-0 (the q-vs-p TV gate) is mandatory for the deployment lossless *claim*.

## Focused build order (reuse-first, GPU-free first)
0. **Un-bake REPLAY_ROUTE** (patch:4346 → env-gated) — GPU-free source edit, the blocker.
1. **Extend fr12_branch_path_oracle_probe.py** with arm-B (recompute payload diff) — GPU-free, ~1 arg + 1 branch.
2. **Run G1** on one cat8 capture (default vs recompute vs FLA) — the cheapest decisive gate: confirms the locus
   AND the recompute fix's bit-exactness before any temp-0.6/SWE compute. If G1 fails, G2/G3 are moot.
3. Only then G2 (add the q-vs-p TV Sig-0 + a vs-native reducer + a recompute grid cell) and G3 (lift the
   variant:304-305 hard-block on FR13_SCAN_ALIGN=1 + thin driver).
