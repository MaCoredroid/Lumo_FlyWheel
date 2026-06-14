# FR13 — FA2-fork is M_DEPENDENT on the spine row (in-process A/B); carrier LOCALIZED + fixable (FR13_FA2_QPAD)

Date 2026-06-14. Workflow `wob0t2y8v` (in-process FA2 M=10-vs-M=5 A/B), Verdict **holds=True = M_DEPENDENT**.
Raw: `research/fr13_workflows/fa2_minvariance_ab_wob0t2y8v.raw.json`. This is the DEFINITIVE, decoherence-free
localization the free-running ladder couldn't be.

## The controlled A/B (no stream decoherence)
In ONE cat9 boot at the p3 deep-accept carrier event, re-call OUR forked-FA2 (apply_tree_bias post-QK,
contiguous-KV oracle) TWICE on the SAME captured K/V: M=10 (full tree) vs M=5 (spine-slice = spine rows
[0,1,2,4,6] + the 5x5 spine sub-bias, restricted spine-ancestor KV). The ONLY varied factor is M
(query-row occupancy). Compare the deep-spine row (flat row 6 = node5) attn_out, RAW max_abs per full-attn layer.

## Result: M_DEPENDENT (RAW != 0)
- Carrier event: 15/16 full-attn layers RAW=0.0 bit-exact; **L31 = 3.90625e-3 = exactly 1 bf16-ULP** (single
  channel, mean_abs 6.36e-7). NOT a bit-identity.
- 14-event sweep: 26/224 cells nonzero (~12%), recurs across **14 of 16 full-attn layers**, every value an
  EXACT power-of-2 (bf16 quanta). Divergence MONOTONE in spine depth (depth-0/1/2 bit-identical) =>
  pure QUERY-OCCUPANCY (kBlockM=64 MMA fragment tile + Is_even_MN=false predication + tree_bias lane
  offsets q/k_offset=max_seqlen_q-rows) — NOT a KV-slicing artifact (red-team neutralized).
- => the forked-FA2 query-tile is THE M-dependent carrier (co-residency batch-variance); GDN scan PROVEN
  M-invariant; the ~1-ULP/full-attn-layer compounds over 16 layers + the deep stack to the argmax flip.

## FIX: FR13_FA2_QPAD (pad query to fixed N_PAD_Q, flag-gated default-OFF, keeps FULL ACCEPT)
Pad the query (+ the MxM tree_bias) to a fixed N_PAD_Q in the fork's tree-bias decode dispatch so the
spine row's kBlockM tile / Is_even_MN / tree_bias offsets are M-invariant; padded rows = -inf-masked filler
(contribute 0, sliced [:M]). Lossless-by-construction (value-preserving), targeted (only the tree-verify
forked-FA2 call, NOT global BI -> avoids the cat9+BI=34 + +13GB blowup), our kernel still computes.
GATE SEQUENCE: (1) re-run THIS MAB A/B with QPAD -> carrier L31 RAW -> 0.0 + the 26/224 sweep cells -> 0;
(2) e2e per-token argmax-vs-clean-decode flip count -> does 22 -> ~3? (the DECISIVE gate; my red-team
concern = N_PAD_Q makes verify M-invariant but the gate is verify-vs-DECODE, so gate 2 is load-bearing).
A/B instrument banked: scripts/fr13_fa2_mab_replay.py + the FR13_FA2_MAB hook (default-OFF). Pairs with
[[reference_diffuse_gdn_accumulation_explained]], [[feedback_no_reroute_reward_hacking]], [[project_fr13_fa2_fork_nocopy_floor]].
