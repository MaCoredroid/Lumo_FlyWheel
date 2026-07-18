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

## SEAM MAP DONE (2026-07-18, agent acab2867) — build is FEASIBLE with 2 constraints
GREEN (clean reuse, no new machinery):
- Forward ALREADY seeds h0 from col-0 (h0_use_accepted_column=False via RUNROW_INIT) -> "h0=pre-step" hook exists (patcher:5159-61).
- col-0 export machinery EXISTS: RUNROW_COMMIT store (kernel:1058-72) + _fr13_conv_commit_to_col0 (patcher:7268-7326).
  Reuse verbatim, source the FIXED chain-end index instead of the accepted leaf.
- Replay is cleanly isolated -> ONE-condition drop (sampled patcher:9936/10052/10124; greedy:9080/9110; conv:9816/8879).
- _COMMITTED[req] carries the chain TOKENS (drafter:208/384) -> no new committer->drafter token buffer.
- Read helpers _fr13_piggyback_on/_cap ADDED (kernel:22+). [DONE]

CONSTRAINT 1 (n_pad=32 hard cap, kernel:1970-84): tail6=31 nodes has NO room for a K-chain (31+K>32 -> n_pad=64 -> spill).
  => BUILD ON cat9 FIRST (9 nodes; 9+K<=32 fits with BV<=8) to validate mechanism+win; the deep tail (accept 5.2)
  needs a subtree shrunk to (32-K) nodes -> slightly lower accept, but the committer 100->16ms dominates (math: still wins).
CONSTRAINT 2 (static masks -> FIXED K): masks/parent baked once from static SPEC_CONFIG (patcher:224-254) into the
  captured graph, so chain length K must be CONSTANT (the sidecar prefix-cap). Short prev-accepts (L<K) pad the chain
  to K with IDENTITY nodes (beta=0 => h_t=h_{t-1}); chain-end index K-1 then still holds the committed state S_N.
  => identity-padding is the "delicate half" -> gate byte-exact (a beta=0 node must be a pure no-op in the GDN scan).

## BUILD ORDER (revised, cat9-first): seam0 read-helper[DONE] -> topology prepend+identity-pad (seam1, cat9) ->
## forward extended-tree + kernel chain-end export (seam2/3) -> committer offset (seam4) -> replay drop (seam5)
## -> GPU gate on cat9 (accept-identical, CFWD 77->16ms, tps>native) -> then subtree-shrunk tail for max accept.

## SEAM 1 DESIGN RESOLVED (2026-07-18) — the variable-chain + identity-padding is the real crux
The chain = the PREVIOUS step's accepted path (length L varies 1..~depth), NOT the last-K-committed. Worked
through why: col-0 must lag exactly ONE STEP (col-0=S_{N-1}; chain=step N-1's accepts advances S_{N-1}->S_N;
forward exports chain-end=S_N->col-0). Fixed-K-committed-chain is WRONG — its export creates a col-0 vs
next-chain-start mismatch (double-processing). So the chain is variable-length.

But the mask/topology is STATIC (baked into the captured graph) => the chain slot must be a FIXED K, and a
short prev-accept (L<K) must PAD nodes L..K-1 so that:
  (a) node K-1 (the static subtree-root) still holds S_N  -> padding must be GDN-IDENTITY (state-preserving),
  (b) the export at fixed CHAIN_END_IDX=K-1 reads S_N.
GDN identity = beta=0 (no delta-rule update) at the padding nodes. beta is model-computed (beta_tree, patcher
:5146); forcing beta=0 at padding positions is a PACKER/forward override (zero beta_tree[L..K-1]) — this is
the delicate, must-gate-byte-exact half.

### Seam 1 is therefore a SUB-PROJECT, not a one-shot edit:
1a. Static extended tree_choices: {(0,)^j : j=1..K} ∪ {(0,)^K + p : p in base} -> masks/parent/n_pad auto (patcher:224-254). [config]
1b. Packer fills chain-node tokens = prev-accepted tokens (L real) + repeat-committed-leaf for L..K-1. [merged_fill]
1c. Packer zeroes beta_tree at padding positions L..K-1 (GDN identity) — GATE byte-exact that a beta=0 node is a pure no-op. [delicate]
1d. Plumb the prev-step accepted_len L (committer -> drafter); tokens are already in _COMMITTED, but L is not (agent Seam-6 gap).
CHAIN_END_IDX=K-1 (constexpr, kernel export DONE). Committer walk offsets to start at K-1 (seam 4).

## HONEST BUILD-SIZE UPDATE: seams 0+3 (read-helper + kernel export) DONE. Seam 1 is the bulk — a coordinated
## packer + accepted-len-plumbing + beta-identity sub-project needing its own byte-exact gate (1c), then the
## forward caller (2), committer offset (4), replay drop (5), then live cat9 GPU gates. This is a multi-session
## engineering build; the crux (kernel state-export) is landed. Recommend building 1a-1d as a focused unit.

## SEAM 1c DE-RISKED (2026-07-18, red-team of _gdn_node_step:616) — identity padding is BYTE-EXACT, clean
_gdn_node_step: a = exp(b_g), b_g = -exp(A_log)*softplus(b_raw_a + b_dt_bias); b_beta = sigmoid(b_raw_b).
Set raw_a = raw_b = -1e9 at padding positions L..K-1:
  softplus(-1e9) = log(1+exp(-1e9)) = log(1+0) = 0  (exp underflows to EXACTLY 0.0 in fp32)  => b_g = -exp(A_log)*0 = 0 => a = exp(0) = 1  (EXACT)
  sigmoid(-1e9) = 1/(1+exp(1e9)) = 1/(1+inf) = 0  (EXACT)
  => h_t = a*h_{t-1} + beta*(delta) = 1*h_{t-1} + 0 = h_{t-1}  = pure IDENTITY, BYTE-EXACT, independent of q/k/v.
(SCAN_ALIGN sigmoid->bf16->fp32 of 0.0 is still 0.0.) So the "delicate half" (1c) is a CLEAN targeted override
of raw_a/raw_b (=a[..],b[..] tensors at patcher:5147-48) at padding positions -- a scatter, not a kernel change,
byte-exact-provable offline. This significantly DE-RISKS seam 1: the whole piggyback state-carry is byte-exact
by construction (chain re-association = replay committed state at 1.19e-7; padding = exact identity).
=> seam1 = clean coordinated build (config 1a + packer token-fill 1b + raw-override 1c + accepted_len plumb 1d),
no delicate correctness gamble. Remaining risk is purely INTEGRATION (wiring 5 seams) + the live accept/tps gates.
