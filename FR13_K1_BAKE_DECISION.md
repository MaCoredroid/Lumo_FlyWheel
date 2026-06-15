# FR13 — K1 bake decision: PENDING a same-boot mechanism proof (user wants to bake; queue after N_PAD)

Date 2026-06-15. User: "K1 we should bake — are we sure it's reducing drift in theory AND by measure?"
Decision: **bake is QUEUED, gated on one cheap same-boot mechanism proof, to run AFTER the N_PAD test
(wtuyrq24t) settles** (GPU serialized).

## Where K1 stands
- **THEORY: sound.** K1 = our tree-scan carries the GDN state through native's exact per-token bf16 store→
  reload boundary (instead of full fp32). The lossless oracle IS the recurrent decode, so K1 aligns our
  verify's state realization to the oracle's = the CORRECT realization (authorized numerics-alignment,
  [[feedback_no_reroute_reward_hacking]]). It addresses ~1/3 of the gap (the only depth-growth op; the rest =
  diffuse amplification + committer forks).
- **MEASUREMENT: suggestive, NOT proven above noise.** Banked K1 = de-cascaded 18→12 / raw 23→20, accept 3.004
  held (FR13_K1_STORE_BOUNDARY_BIND). BUT: (1) it is a SINGLE CROSS-BOOT comparison (K1-boot vs a different
  OFF-boot); raw −3 is INSIDE the ±9 autotune cross-boot floor ([[feedback_no_cross_boot_byte_gate]]); de-
  cascaded −6 is audit-fuzzy + cross-trajectory. (2) K1's OWN state-toward-native effect was NEVER isolated —
  we measured OFF-state-vs-native = 0.0289 and full-recompute = 0.0, but NOT K1's partial position between.

## The QUEUED proof (run after N_PAD, before baking)
Reuse the PROVEN same-boot int-view instrument scripts/fr13_gdn_scan_warp_gate.py (hardened neg-control
native_norm>0, int-view NEVER atol):
1. **MECHANISM (decisive, boot-noise-immune):** int-view the carried GDN scan state with K1-ON (FR13_SCAN_ALIGN
   =1 MODE=body) vs K1-OFF, each vs native-packed-decode state. PASS = K1-ON state-vs-native < OFF's 0.0289
   (K1 moves the state TOWARD native). This proves the mechanism same-boot, immune to the cross-boot floor.
2. **ACCEPT neutrality (same boot):** accept/event with K1 ON ~ OFF (no speed regression).
DECISION: K1-state closer to native (< 0.0289) AND accept neutral → BAKE K1 (flip it default-ON in the locked
path, behavior = the proven realization). K1-state NOT closer to native → do NOT bake (it isn't doing what
theory says). The e2e 18→12 stays a supporting (not primary) datum given the cross-boot caveat.

## Status
- Workflow staged: scripts/fr13_k1_mechanism_proof_workflow.js (ready; launch AFTER wtuyrq24t frees the GPU).
- Nothing baked yet; cat9 LOCKED, K1 is a default-OFF flag (b91c1bc0), deployed path byte-identical.
