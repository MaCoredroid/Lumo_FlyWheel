# FR13 PIGGYBACK — eliminate the committer replay by folding the accepted-path SSM advance into the next forward

## Why (measured, FR13_TREE_VS_NATIVE_VERDICT.md)
The committer replay is **78% of the native>tree throughput gap** (committer 6.6ms native vs 100ms tree;
drafter is EQUAL, forward +22ms). The replay = 48 tiny latency-bound per-layer kernels (1.386ms each = 66-72ms)
re-deriving the accepted-leaf GDN state. Native is cheap because it advances the accepted-path state INSIDE the
verify forward's one fused high-occupancy scan and selects h_k. Attack math: kill the replay => tail6 ~PARITY
(loses 2%); + trim the +22ms forward tax => tree WINS +10%.

## The mechanism (native-style)
Today: verify forward -> committer rejection-walk -> **REPLAY (48 kernels, 72ms)** advance accepted path ->
committed leaf state -> next forward scans the new tree with h0 = committed leaf.
Piggyback: committer records the accepted-path nodes + sets next-scan h0 = PRE-STEP state; NO replay. The NEXT
forward's GDN scan processes **[accepted_path nodes] ++ [new tree]** — advancing through the accepted path
(deriving the committed leaf state INLINE, folded into the layer's existing scan, ~free HBM-bound) then the
tree. Committer -> ~walk+publish ~16ms (native-like).

## The seam (verified)
`launch_tree_gdn_prepared` (fr10_gdn_tree_kernel.py:1936) forward-scan takes `h0` + per-node input +
`spec_state_indices`/`prev_lens`/`h0_use_accepted_column`. Today h0 = replayed committed leaf. Piggyback:
h0 = pre-step running state (col-0 BEFORE this step); node input = accepted-path prefix ++ tree; the scan's
parent structure gets the prefix as a chain feeding the tree root.

## Build phases (each gated, commit per step)
1. **Correctness contract (offline, NO GPU):** prove that scanning [accepted_path ++ tree] from h0=pre-step
   yields byte-identical tree-node states vs today's [replay-to-committed-leaf then scan tree from h0=leaf].
   The GDN recurrence is associative along a path, so this MUST hold; verify with a CPU/fp32 oracle
   (fr13_native_committer_validate.py-style) before touching the live forward.
2. **Forward input assembly:** prepend the accepted-path nodes (from the committer, prev step) to the tree
   node/parent tensors fed to launch_tree_gdn_prepared; h0 <- pre-step col-0; extend the parent map so the
   prefix chain feeds the tree root. Flag-gated (FR13_PIGGYBACK, default 0 => today's replay path, byte-id).
3. **KV / conv1d handling:** the accepted-path tokens' attention KV is ALREADY cached (committed positions);
   the conv1d recurrent window needs the same prefix-advance as SSM (the design's conv col-0 machinery,
   FR13_APC_COMMIT_TO_RUNNING_ROW, is the hook). Ensure the re-processed prefix does NOT double-write KV.
4. **Drop the replay:** when FR13_PIGGYBACK on, the committer skips launch_tree_gdn_replay entirely (just
   records accepted nodes + pre-step h0 handle). Committer CFWD should drop 100->~16ms.
5. **GATES (live B=4 subset_b4_sixteen):** (a) accept BYTE/accept-identical vs replay path (piggyback is a
   pure re-association, MUST be lossless); (b) committer CFWD 100->~16ms (CF2 timer); (c) s_per_fwd_gpu:
   forward +a few ms (the +k prefix nodes) but committer -84ms => net step down ~90ms; (d) derived_tps_gpu +
   fullstep vs native — does tree reach parity/win? (e) no garble, resolve unchanged.

## Risks / red-team
- The +k prefix nodes push the forward scan past n_pad=32 (tree is already 21-32); may need n_pad headroom or
  the prefix rides a separate short scan segment. MEASURE the forward delta (GATE 5c) — must stay << 72ms saved.
- Conv1d prefix-advance parity is the delicate half (the SSM half is clean re-association); gate byte-exact.
- Batch/position management for the variable-length accepted prefix at B>1 (the SAMPLER-row-id lifecycle that
  bit the tail build) — reuse the sidecar/row-id fixes.
- Correctness-critical: this changes the deployed forward. Everything flag-gated (default 0 = today's replay);
  the offline contract (phase 1) + byte-identical accept gate (phase 5a) are the make-or-break, provable
  BEFORE trusting any speed number. LIVE GPU validation required (env was blocking; GPU now clean).

## Status: DESIGNED + seam-verified + feasibility-confirmed. Ready to build phase 1 (offline contract, no GPU).

## REFINED APPROACH (2026-07-18): EXTENDED-TREE (cleanest — reuses existing tree machinery)
Instead of a bespoke prefix mechanism, have the DRAFTER build the verify tree as:
    tree = [ prev-accepted-path as a CHAIN ]  ++  [ new speculation subtree rooted at the committed leaf ]
and scan it from h0 = PRE-STEP running state (the state BEFORE prev step's accepts), NOT the replayed leaf.
- The chain prefix re-advances the GDN state through the accepted tokens INSIDE the forward's fused per-layer
  scan (occupancy-free), landing the committed-leaf state at the chain's end = the speculation subtree root.
- The committer walks ONLY the speculation subtree (offset past the prefix); the prefix nodes are committed
  context (auto-consumed), never counted as new accepts.
- REUSES: the tree scan (_tree_gdn_kernel), masks (strict/visible), n_pad (prefix 5 + base 16 = 21 -> 32,
  already the tail size), and the committer walk (offset). NO replay call when FR13_PIGGYBACK on.
- Correctness: PROVEN by composition — the chain-advance == fr13_native_committer_validate.py's committed
  state (1.19e-7); the subtree scan from that state == today's forward. So byte-lossless-by-construction on
  the state carry (kernel-consistent: same forward kernel does both, unlike replay's separate kernel).

### Build order (extended-tree), each flag-gated (FR13_PIGGYBACK default 0 = today's replay path):
1. FR13_PIGGYBACK sidecar (launcher, worker-env-drop-proof) + read helper.  [plumbing — SAFE]
2. Drafter: when on, prepend the prev-accepted chain to the tree topology (parent map + tokens); grow
   wide_D/n_pad by the prefix length. The committed leaf becomes the subtree root (already is).
3. Forward: h0 <- pre-step col-0 (state before prev accepts); scan the extended tree (existing machinery).
4. Committer: walk offset past the prefix (commit only the subtree); the prefix is context.
5. Drop the replay when on (committer records prev-accepted for the next drafter step; no launch_tree_gdn_replay).
6. GATES (live B=4 subset_b4_sixteen): accept-identical vs replay, CFWD 100->~16ms, s/fwd forward-delta
   (the +k prefix nodes) << 72ms saved, tps vs native, no garble.
This is a DRAFTER+FORWARD+COMMITTER change but each piece reuses existing tree machinery -> lower risk than a
bespoke prefix kernel. Still correctness-critical + needs live GPU validation.

## STATE-FLOW RESOLVED (2026-07-18) — how col-0 updates WITHOUT the replay (the crux)
Key: the prev-accepted chain end is a FIXED, KNOWN position (last chain node) — unlike the spine-commit's
unknown accepted-leaf. So the forward exports THAT one node's state to col-0 (one cheap write, reuses the
FR13_APC_COMMIT_TO_RUNNING_ROW col-0 machinery), and col-0 carries with a one-step defer:

  Define S_k = committed GDN state after step k's accepts.
  Step N:  col-0 = S_{N-1}  (exported at end of step N-1).
    tree = [ path_N chain (prev-accepted tokens) ]  ++  [ subtree_N (new spec from committed leaf) ].
    forward scans from h0=col-0=S_{N-1}: chain re-applies path_N -> S_N at the chain-end node; subtree scans
    from S_N. Forward EXPORTS the chain-end node state (S_N, fixed position) -> col-0.
    committer walks subtree_N only (offset past the chain), commits path_{N+1}; NO replay.
  => the chain does in the forward's fused scan what the 48-kernel replay did (occupancy-free); the export is
     ONE state at a known column. col-0 lags one step and the chain re-derives -- self-consistent recursion.

The prev-accepted chain TOKENS are ALREADY available to the drafter via `_COMMITTED[req]` (the merged drafter
already builds pattern=_COMMITTED[req]+near-MTP) -> no new committer->drafter buffer needed.

### Two variants for the chain's ACTIVATIONS (phase-2/3 implementation choice):
- (A) RE-PROCESS: drafter puts the committed tokens as chain nodes; forward re-computes their activations +
  scans. Standard forward over +k tokens (~free HBM-bound) BUT re-writes their KV (must not double-write /
  land at committed positions). Simplest topology.
- (B) REUSE-RINGS: forward GDN scan input = [prev-accepted RINGS (stored) ++ subtree activations]; chain rides
  prev rings, NO re-processing, NO KV touch. More surgical; requires carrying prev rings + mixing into the scan.
Pick (B) if KV re-write is problematic; (A) if ring-carry is. Both land the same chain-end S_N -> col-0.

### Next implementation step: OBSERVE-ONLY validation (non-destructive, live).
Before the coordinated drafter+forward+committer change, add FR13_PIGGYBACK_VALIDATE: during today's forward,
ALSO scan [_COMMITTED-chain ++ tree] and assert the chain-end state == the replay's committed col-0 (byte or
within bf16-store floor). Proves the mechanism live with ZERO deployed-path change. Then flip to destructive.
