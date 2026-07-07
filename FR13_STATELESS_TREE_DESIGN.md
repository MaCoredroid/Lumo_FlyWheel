# FR13 — Stateless Tree: tree caches like native (design, verified)

**Governing principle (user 2026-07-07, decisive, no-setbacks/no-leak):**
> The tree pipeline maintains **ZERO** persistent state. Every step is a pure function:
> init tree scratch **from the native running rows** → draft + verify in scratch →
> **copy** the committed leaf → native running rows → **zero (burn) all tree scratch**.
> "MAKE ALL DATA the tree has beyond native MTP throw-away (zero out / mem collection);
> we don't maintain anything in life span; simplicity wins; HBM copy is fast."

Native's stock, req-keyed mamba cache then owns snapshot/restore/concurrency for **both**
recurrent states per GDN layer (SSM + conv1d) — exactly as it does for native MTP. This
eliminates carrier A′ + carrier B **by construction** (no tree-owned cross-step / cross-req
state to go stale, be reused-dirty, or bleed between agents). Same bug class as the mamba
`ZERO_MAMBA_ON_ALLOC` fix, generalized to the whole tree.

The per-step state is a few MB of GDN recurrent + conv state across 48 layers; an HBM→HBM
copy is microseconds against a ~98 ms decode forward. Avoiding the copy (SNAP_FIX redirect /
keep-alive) was a false economy that cost correctness. Copying is free; simplicity wins.

---

## Root cause — CASE B (verified in source)

**Node-bank col 0 IS native's running row.** `spec_state_indices_tensor = block_table[spec_mask, :num_spec+1]`
(gdn_attn.py:899-901/920-922); restore reads `block_table[:,0]` (gdn_attn.py:851/923). So
`spec_state_indices[b,0] == block_table[b,0] ==` the req-owned running block; cols `1..num_spec`
are ephemeral spec-draft blocks. `block_ids[cur_block_idx]` (bias 0) is the same physical block as col 0.

**The tree deposits its committed leaf one column over, and reads it back positionally:**
- `_tree_gdn_replay_kernel` dst_col map (fr10_gdn_tree_kernel.py:1182-1198): t=0→col0, t=1→col0 (overwrite),
  t=k→col k-1. So after replay **col 0 = post-FIRST-accepted state (STALE)**, **col nacc-1 = committed leaf**.
- Next step seeds h0 from col `max(nacc-1,0)`: served scan `h0_use_accepted_column=True` (patch:5352;
  kernel:634-639); replay h0 `h0_col=max(prev_len-1,0)` (kernel:1094-1096).
- So for `nacc>=2` the running row is stale and both readers re-init **positionally off the ephemeral
  node-bank at col nacc-1**. That is why `tree+no-cache` has 0 give-ups (col nacc-1 persists within one
  continuous decode) yet carrier B appears under CONC>1 + cache (the ephemeral spec block + the persistent
  `leaf_map`/SNAP_FIX redirect are not req-owned across reallocation/boundary). It is exactly **why SNAP_FIX
  exists**: the stock snapshot reads `block_ids[src+accept_bias]` (neither col0 nor col nacc-1), so the code
  republishes col nacc-1 into `_FR13_APC_SSM_LEAF_BY_REQ` (patch:9080) and redirects the snapshot source
  (patch:14644-14646).

**Verified linchpins (read directly in kernel):**
1. `state` register at loop exit == committed leaf (tl.where preserves the last active t=acc_len; kernel:1181)
   — the SAME bytes stored to col nacc-1 at 1191-1198. ⇒ storing `state` to col 0 too is byte-identical.
2. h0 tile is loaded into registers BEFORE any store (kernel:1097-1107, explicit "publish-overwrites-h0-row"
   comment) ⇒ read-col0-then-write-col0 in one launch is safe.

---

## The fix — 3 co-dependent flags (default "0" ⇒ byte-identical); fail-loud if partial

| flag | role |
|---|---|
| `FR13_APC_COMMIT_TO_RUNNING_ROW` | replay kernel also stores committed `state` → col 0 (SSM **and** conv); cache snapshot source → `block_ids[cur_block_idx]` (col 0), map-free; guard-skip leaf publish |
| `FR13_TREE_RUNROW_INIT` | flip both next-step init readers to col 0 (served scan `h0_use_accepted_column=False`; replay `h0_col=0`) |
| `FR13_APC_BURN_NODE_BANK` | after the col-0 write, zero node-bank spec cols `1..num_spec` (col 0 preserved) |

Dep-guard (fail-loud at engine init, next to REPLAY_ROUTE guard patch:817-819): all three form one
lifecycle — enabling any requires all three; all-OFF ⇒ SNAP_FIX path runs verbatim (byte-identical).
Prevents the two proven corruption traps: BURN-without-INIT zeroes the col the scan still reads;
COMMIT-without-retarget leaves the cache reading bias-chosen rows.

### Insertion points (SSM — VERIFIED)
- **Writeback (3a):** constexpr-gated post-loop store of `state` → col 0 in `_tree_gdn_replay_kernel`
  (after kernel:1198) + `_tree_gdn_replay_all_layers_kernel` (after :1532); thread `RUNROW_COMMIT` through
  launches (patch per-layer:10743-10760, all-layers:10680-10699). col0 = SOURCE row ⇒ NOT the retired
  FR13_APC_VERBATIM dead-writer (which wrote the DEST row).
- **Init reroute (3b):** served scan `h0_use_accepted_column=(not RUNROW_INIT)` (patch:5352); replay
  `h0_col = 0 if RUNROW_INIT else max(prev_len-1,0)` (kernel:1095, :1440 all-layers).
- **Cache retarget + SNAP_FIX removal (3c):** in get_temporal_copy_spec override, `src_block_id =
  block_ids[cur_block_idx]` when flag on (replaces redirect patch:14323-14360/14644-14646); guard-skip
  `_fr13_publish_apc_ssm_leaf` (patch:10761-10764).
- **Burn (4):** constexpr `BURN_NODE_BANK` trailing zero-loop over cols `1..num_spec` after the col-0 store.

### Conv half — RESOLVED (wck7xl1oz): CASE B for emulation, cache already col-0
Conv splits: (i) **emulation cross-step carrier** (gather-prior ← per-node write-back) is STALE like SSM —
committed window at the LEAF NODE column, col 0 = root window ⇒ needs COMMIT+INIT+BURN; (ii) **mamba cache**
(`get_conv_copy_spec`) ALREADY reads col-0 whole-row map-free (patch:14065/14089/14101 offset-0) ⇒ NO retarget.
So the COMMIT (col 0 ← committed leaf window) fixes the emulation read AND the conv cache in one write (today
conv cache reads col0=root for nacc≥2 = latent bug + part of carrier B). Reuses the SAME 3 flags, **no Triton,
no new flags** — pure Python index ops in the committer + prepared-rows helper:
- **COMMIT** (`FR13_APC_COMMIT_TO_RUNNING_ROW`): 2nd `index_copy_` after patch:3858 — `_fr10_new_state[leaf]`
  (`_fr13_committed_read_cols[b]` = leaf node id, :2791) → `spec_state_indices[b,0]`.
- **INIT** (`FR13_TREE_RUNROW_INIT`): in `prepare_committed_path_conv_rows` (fr13_tree_conv_fused.py:283-316)
  return `read_bank_rows = spec_state_indices[:,0]` when on; feed to gather at patch:2797; mirror OFF-arm
  gather at kernel:475.
- **BURN** (`FR13_APC_BURN_NODE_BANK`): `conv_state.index_fill_(0, spec_state_indices[b,1:tree_n], 0.0)` after
  COMMIT (page-safe: index_fill_ on the conv as_strided view touches conv rows only).
- **Cache/cleanup**: no edit; CONV_SNAP_FIX/`_FR13_APC_CONV_LEAF_BY_REQ` already default-OFF/inert ⇒ pure dead-code deletion.

**Dep-guard (critical):** the mamba block is ONE physical page holding conv(kv0)+ssm(kv1) as as_strided views;
native restore rewrites col 0 for the WHOLE page. So col 0 must hold the committed frontier for BOTH slices —
partial enable (SSM rerouted, conv not) = mixed-page corruption on restore. Hence ONE flag spans both; the
fail-loud guard adds a third trap: SSM-rerouted-without-conv-rerouted.

**Resolved sub-questions:** width-K window is INTERIOR to one bank row [dim,state_len] (whole-row copy/burn,
no per-tap handling); the `causal_conv1d_update` sliding offset lives ONLY on non-live diagnostic/spine-oracle
arms — the tree emulation reads the window with no offset and the cache copies whole-row offset-0, so col-0 is
a closed offset-free loop.

---

## Invariant (user 2026-07-07, sharpened — the acceptance bar)

The tree must be **behaviorally indistinguishable from native+MTP at the data layer**:
1. **No persistent data beyond native+MTP** — no leaf-map, no SNAP_FIX redirect, no exact_seed staging.
2. **Read/write access behavior identical to native+MTP** — same block-pool/running rows read & written,
   same timing (init reads col0; commit writes col0; cache snapshot reads col0; restore reads col0).
3. **Only the spec-decoding MATH differs** — the branched tree scan. Its within-step branch workspace
   (node-bank cols `1..num_spec`) is the sole delta vs native's narrower linear-MTP draft scratch, and is
   BURNED every step ⇒ **zero lifespan, zero persistent extra data.**

Consequence: if the tree's persistent-data access pattern == native+MTP's, **carrier B is impossible by
construction** (it is a cache-behavior divergence, and native has none). Access-equivalence PROVES it;
the CONC4 give-up test CONFIRMS it.

## Validation (no-setback gates)
1. **Access-equivalence to native+MTP (the PROOF gate)**: instrument the block-pool rows the tree
   reads/writes per step (init-read, commit-write, snapshot-source, restore-read) and assert they equal
   native+MTP's on the same trajectory — same rows, same timing. No tree-extra persistent tensor touched
   on the cache path. If equal ⇒ carrier B cannot exist.
2. **`FR13_APC_CACHEROW_DUMP`**: assert WRITTEN(col0) == SNAPSHOT-SOURCE(col0) == RESTORE(col0) byte-exact
   + graph-safe obs `runrow_commit_events`/`burn_events` prove the path fired in one boot.
3. **NO-CACHE continuous-decode byte-identity**: tree+no-cache, flags OFF vs ON, same seed, temp 0.6,
   CONC=1, GRAPH — streams byte-identical (proves the reroute doesn't regress the 0-give-up path).
4. **Graph-mode CONC1-vs-CONC4 carrier B (the CONFIRM gate)**: live SWE-Verified agentic, flags ON, rows =
   spine5 AND cat8 × graph AND eager; CONC4 give-up rate matches CONC1 + native (the 4/4→0/4 collapse gone).

**Conv is held to the identical bar**: conv init-read / commit-write / snapshot-source / restore-read must
equal native's conv access pattern, no persistent conv-extra (CONV_SNAP_FIX / _FR13_APC_CONV_LEAF_BY_REQ
deleted), conv branch scratch burned. (wck7xl1oz determines whether conv already meets this or needs the reroute.)

Then #14 physically deletes SNAP_FIX (SSM+conv) + leaf-maps + exact_seed/HRS/refold.
