# FR13 GDN tree-verify kernel status (canonical — read before touching the GDN path)

Two DIFFERENT kernels are easy to conflate. This pins which is which and the policy.
Verified from source on HEAD (40e46ae0), 2026-06-13.

## 1. The VERIFY SCAN — always runs
`_tree_gdn_kernel` (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:387`). Sequential
rank-1 per-ancestor-path scan, shared `_gdn_node_step` body, h_cache registers, strict-mask
path isolation, bit-exact-to-native-on-spine, batch-invariant by construction. It produces the
per-node logits used to verify each candidate. It has a `STORE_NODE_STATES` constexpr (:421)
toggled by the replay route (below).

## 2. The REPLAY ROUTE — DEFAULT-ON, the shipped path, STAYS ON (user 2026-06-13)
`_tree_gdn_replay_kernel` (`:546`) + `launch_tree_gdn_replay` (:709) + the all-layers sibling
`_tree_gdn_replay_all_layers_kernel` (:848). Gate flag `FR13_REPLAY_ROUTE` **defaults to "1"**
(`scripts/fr10_phase4_patch_vllm_tree_gdn.py` `os.environ.get("FR13_REPLAY_ROUTE","1")`;
launcher `${FR13_REPLAY_ROUTE:-1}`).
- ON (default): the scan runs with `store_node_states = not _fr13_replay_route_on` = **False**
  (`:4115`) → it does NOT export all node states to HBM; after the committer knows the accepted
  path, the replay kernel **re-executes ONLY the accepted chain** from h0 and publishes its
  durable state to the bank's linear columns. = the accept-only / no-copy mechanism. Speed:
  36→6 row-touches/layer = 0.86× native HBM, spill-free at any tree width.
- Gate-4 live-fail (accept 2.024→1.521) was a **conv-remap PAGE-STOMP**, FIXED pure-wiring at
  `02b1627a` (`fr13_replay_conv_remap.py:41-122`); post-fix determinism 4/4, token-identical to
  legacy. (See FR13_GDN_KERNEL_LINEAGE.md:27, FR13_WY_CHASE_PLAN_BIND.md.)
- **DEPENDENCY (be careful):** FIX-3 conv-fusion (`FR13_TREE_CONV_FUSED`, default-ON) **requires**
  `FR13_REPLAY_ROUTE=1` (raises at `:827`); FIX-2 eager-pack (`FR13_EAGER_PACK`) also has a
  replay-coupled requirement (`:5464`). So `FR13_REPLAY_ROUTE=0` is NOT a clean single flag — it
  forces conv-fusion OFF + eager-pack rework = the full legacy path. **Do not flip it for the
  chase.** The shipped path is replay-ON; chase there.

## 3. The WY KERNEL — PARKED, not on HEAD
`_tree_gdn_wy_kernel` — a DIFFERENT kernel: whole-tree Gram-matrix + UT/Householder solve
(chunked-WY math, a different summation tree → ~6e-5 chunk-vs-recurrent gap, **never bit-exact**
to native sequential verify). **NOT on HEAD** (grep count 0); present only on branch
`fr13-wy-archive` (c0448bd7 + state-fix 8a975837, count 2). It is the **last-resort fallback**,
triggered ONLY on a hard replay-route failure (FR13_GDN_KERNEL_LINEAGE.md:24). Re-plugging it =
a PORT onto HEAD's dispatch, not a cherry-pick.

## POLICY (user 2026-06-13)
- **ALWAYS use the replay route (keep current default-ON status). WY stays parked.** All gates,
  chases, and measurements run on the shipped replay-ON path.
- The active work is two SEPARATE tracks, BOTH chased on the replay-ON path (WY out of scope):
  - **(A) 22-flip lossless drift = VERIFIER side** (committer row-mapping OR verify-forward
    argmax): the node-7 per-sub-op ladder + the committer-row argmax gate (FR13_COMMIT_ARGMAX_GATE)
    to split channel-1 (served != argmax(verify logits)) vs channel-2 (verify argmax != clean).
    INCLUDE: **why the previous gate did not catch it** — the prior SCALAR superset/accept gate is
    blind to a ~4.8% per-token argmax flip (22/457) that barely moves the average; the binding
    instrument is the per-token argmax-vs-clean-oracle probe (FR13_GATE_BLINDSPOT, 30d749a4;
    reference_scalar_metric_per_token_blindspot). The fix must ship WITH a gate that would have
    caught it.
  - **(B) drafter −28 accept = DRAFTER side** (co-residency: the cat10 root sibling, added to the
    drafter packing order only, perturbs deep-spine PROPOSED tokens): the spine-only-drafter A/B
    discriminator (co-residency vs BI-asymmetry vs state-rebuild), BI pinned both arms.
