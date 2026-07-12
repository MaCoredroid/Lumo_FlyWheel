# FR13 garble fix — per-path native recurrence (branch-preserving, bit-exact)

## Why (validated, deterministic, no boot noise)
- Garble seed = the tree's custom `_gdn_node_step`/`_tree_gdn_kernel` differs from native
  `fused_recurrent`/`fused_sigmoid_gating` by an **unbiased ~9e-4** (codegen/FMA order, NOT the 3
  documented seams which are only ~1e-5). Confirmed vs ACTUAL native: `fr13_tree_vs_native_bias.py`
  (pos_frac 0.5027, signed_mean -5.4e-6). SCAN_ALIGN does NOT close it (9.256 vs 9.258e-4).
- The GDN token-recurrence is CONTRACTIVE (decay gate) — the seed does not amplify within a layer;
  the ~492x is CROSS-LAYER. So the only way to kill garble is to make the tree's per-node GDN output
  bit-identical to native (nothing to amplify).
- **`fr13_native_spec_vs_nospec.py`: native `fused_sigmoid_gating` per-chain (cu_seqlens) is
  BIT-IDENTICAL (max|d|=0.0) to native `packed_decode` no-spec.** So routing each tree node's
  ancestor path through `fused_sigmoid_gating` yields bit-exact-to-native node outputs.

## The fix
Replace the custom tree GDN scan (for the verify) with a **per-node-path** call to the native spec
kernel. Each node i runs its root-to-node ancestor path as its own cu_seqlens sequence; node i's output
is the LAST token of its sub-sequence. This is exactly native MTP/E5 (bit-exact-to-no-spec) applied per
path, so node i sees ONLY its path-ancestors (not branch siblings) — which is the whole reason the
current one-sequence custom kernel + ancestor-mask exists (and why it carries the 9e-4 seed).

### Layout construction (per layer, per decode step)
Given `parent: list[int]` (tree topology) and per-node post-conv `q_n,k_n,v_n,a_n,b_n` (computed ONCE
by the existing shared in_proj+conv):
- `paths = tree_paths_from_parent(parent)`  # paths[i] = [root..i] node ids (already exists)
- Concatenate, for every node i, the tokens `paths[i]` -> flat q/k/v/a/b of length `sum(len(paths[i]))`.
  Each token is a GATHER (index_select) of the once-computed per-node q/k/v — NO weight re-read.
- `cu_seqlens = [0, len(p0), len(p0)+len(p1), ...]`  # one sub-sequence per node
- `ssm_state_indices` -> every path starts from the committed prefix state h0 (same bank row).
- One `fused_sigmoid_gating_delta_rule_update(q,k,v,a,b,A_log,dt_bias,scale,initial_state=h0,
  cu_seqlens,ssm_state_indices,use_qk_l2norm_in_kernel=True)` call.
- Node i output = `out[cu_seqlens[i+1]-1]` (last token of its path). Scatter to the node grid the
  verify consumes.

### Cost (compute-only, no HBM tax)
Duplicated tokens = sum of node depths+1. cat8 node-paths lengths [1,1,2,2,3,3,4,5]=21 (vs 8 unique).
~2.6x cheap rank-1 recurrence; in_proj/conv/weights computed once. Recurrence is a tiny fraction of the
forward (tree TPS is overhead-bound). Optimize later to root->LEAF paths (11 vs 8) + dedup if needed.

## Open items / red-team before shipping
1. **Conv**: the tree conv (`FR13_TREE_CONV_FUSED`) is ALSO custom. The per-path fix feeds POST-conv
   q,k,v to fused_sigmoid_gating, so the conv output must ALSO be native-exact. MEASURE the tree conv
   vs native `causal_conv1d_update` (like the scan bias test). If it carries a seed, route conv per-path
   through native too (causal_conv1d_update per path token, conv_state carry).
2. **Committer/replay**: the accepted-path replay (`_tree_gdn_replay_kernel`) also uses `_gdn_node_step`.
   The committed running state must match native too — replay the committed path through the same native
   per-path call so the carried state is bit-exact.
3. **Cache ON (APC)**: the ship gate is cache-ON. Confirm h0 (committed prefix state) under APC is the
   native state; the per-path call reads it as initial_state.
4. **Overhead**: measure s/fwd delta vs the custom kernel on the speed bank; must stay overhead-bound.

## Gate (unchanged ship gate)
temp-0.6 garble gate + live SWE-Verified WITH cache ON, cat8 AND cat6, same-boot A/B vs native 0%,
same config as fr13_launch_locked. Never chain5/reshape (branches are the deliverable).

## Multi-path calling convention (from fused_sigmoid_gating.py kernel, red-teamed)
`fr13_perpath_realistic_validate.py`: Test A (single chain from NON-ZERO committed h0) == packed_decode,
max|d|=0.0 — bit-exactness holds with real state. Test B (naive batched multi-path) CRASHED because the
kernel's spec/continuous-batching convention is:
- `cu_seqlens` [N_seq+1] segments the flattened q/k/v (batch dim must be 1).
- `ssm_state_indices` is **2D [N_seq, tokens]**; initial state for seq i_n is read from the BANK at
  `state_idx = ssm_state_indices[i_n, num_accepted_tokens[i_n]-1]` (spec) or `[i_n, 0]` (non-spec).
- **State slot 0 = NULL_BLOCK_ID: `if state_idx <= 0: return`** (skips the sequence). Real slots are >=1.
- `num_accepted_tokens` [N_seq] is REQUIRED for spec decoding (selects init-read + write-back slot).
- `initial_state` is the full state BANK (not a per-seq stack); indexed by state_idx.
My naive Test B (1D arange incl. slot 0, no num_accepted, stacked state) violated all of this => crash.
NOT a fix failure — this is exactly how native MTP already drives the kernel.

## Overhead (resolved favorably)
Native MTP presents ALL spec sequences in ONE fused_sigmoid_gating call per layer. Presenting the tree
PATHS as the sequences => still ONE call per layer = **native-MTP launch parity** (viable). N separate
single-seq calls (~8 paths x 48 layers = 384/step) would be too heavy — do NOT do that; use the batched
call. So the batched multi-path call is both NECESSARY (overhead) and SUFFICIENT (native-exact).

## Status
Mechanism bit-exact-validated incl. non-zero committed h0 (Test A). Batched multi-path convention now
understood (2D indices, NULL slot 0, num_accepted). The tree ALREADY builds spec_state_indices_tensor +
num_accepted_tokens for its custom kernel (patcher ~L1813/1847/1893) — the fix should REUSE/adapt that
metadata PATH-decomposed, not rebuild from scratch. Next: (1) study the tree's spec_state_indices builder,
(2) redo Test B with the correct convention to fully validate the batched call, (3) wire into the patcher
behind a flag, (4) measure conv residual, (5) gate temp-0.6 + live SWE cache-ON.
