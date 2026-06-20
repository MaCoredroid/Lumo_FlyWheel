# APC SSM align-snapshot carrier — FIXED (2026-06-20)

Branch `fr13-prefix-cache`. The SSM/temporal align-snapshot poison (the marathon carrier) is
**fixed** by a commit-site-equivalent **write-through**, validated lossless in the deployment form.

## The fix (option 3, corrected): FR13_APC_SSM_WRITE_THROUGH
In `collect_mamba_copy_meta` (worker `mamba_utils.py`, injected by
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:_patch_mamba_utils_collect_apc_leaf`), for every
`get_temporal_copy_spec` copy, write the committed-leaf recurrent **value** in-place into the exact
row the stock align snapshot reads:
```
dest = (copy_spec.start_addr - state.data_ptr()) // row_bytes   # the EXACT pointer the apply kernel tl.loads
leaf = _FR13_APC_SSM_LEAF_BY_REQ[req_id]                        # = spec_idx[b][alen-1] (committer write target / decode read)
state[dest].copy_(state[leaf])
```
The apply (`batch_memcpy_kernel`, Triton) reads the VALUE at apply time and runs synchronously after
collect (workflow wd8yxwms3, 2 readers), so the collect-time mutation lands. SUB (option-1 pointer
redirect) is disabled when WRITE_THROUGH=1. Gated, default 0 → byte-identical when off.

## Two bugs the cat9_apc_wt drill exposed (both fixed)
1. **Wrong dest row.** `block_ids[src_block_idx+accept_token_bias]` (=658) ≠ the row `copy_spec`
   actually points to (=716, from `start_addr`). The old WT overwrote a row the snapshot never reads.
   Fix: derive dest from `copy_spec.start_addr` — no formula, the literal kernel-read pointer.
2. **Wrong leaf.** Tap-A `rows[-1]` is systematically `map − 10`; the map (`spec_idx[b][alen-1]`) is
   the committer's real write target / what the decode reads. The old WT (and the old SUB) used the
   +10-off Tap-A leaf → poison persisted. Fix: use the MAP leaf.

## Validation (rigorous)
- **EAGER**: `cat9_apc_wt2` (13033, eager) ran **20/20 coherent** agent messages reasoning correctly
  about the astropy bug — poison gone. (Gave up only on the 25-min eager+tap timeout, a speed
  confound, not garble.)
- **NON-EAGER / CUDA-GRAPH (deployment form)**: the controlled same-boot A/B
  (`scripts/fr13_apc_lossless_ab.sh`: two identical greedy max_tokens=160 requests, req1 cache-MISS
  vs req2 cache-HIT) returns **LOSSLESS-MATCH: C1==C2 byte-identical (385 chars)** with real cache
  reuse (4992 hits / 12964 queries). CUDA-graph capture is clean (`Graph capturing finished`, WT
  fires `did=True` under graph). **So APC + the tree committer is byte-lossless in deployment form.**

## Diag artifact (important)
A non-eager SWE run with **`FR13_APC_SSM_DIAG=1`** (cat9_apc_wt_12907) GARBLED, while the otherwise
identical A/B with **DIAG off** is byte-lossless. The only env difference was the diag flag.
Diagnostics misbehave under CUDA-graph ([[feedback_fr12_subkernel_zero_gate]] regime). **Deployment
must run DIAG OFF** (it is, by default). Confirming end-to-end via cat9_apc_wt_deploy (13236, DIAG OFF).

## Scope (who has the bug)
Stock align formula `state[block_ids[cur+num_accepted-1]]` was designed for native MTP's linear block
layout. **Non-spec decode: no bug. Native MTP spine: no bug.** OUR tree committer writes the leaf to a
NODE BANK (`spec_state_indices`, for branches) → diverges → bug. The write-through re-establishes the
native invariant (leaf back in the block row), so our tree+APC behaves like native MTP+APC.

## Reusable gate
`scripts/fr13_apc_lossless_ab.sh` — boots cat9+APC (non-eager), GATE-A/B/C (health/engaged/hits) +
the LOSSLESS A/B (req-miss vs req-hit byte compare). The fast, controlled APC-lossless instrument
(no SWE-agent noise). Run with `FR13_APC_SSM_SNAPSHOT=1 FR13_APC_SSM_WRITE_THROUGH=1 FR13_APC_CONV_FIX=1`.

## Remaining for the ship gate (surface to user, do not auto-declare)
- e2e coherent SWE convergence in deployment form (cat9_apc_wt_deploy, in flight).
- TTFT cache-hit vs cache-miss win + decode-TPS parity.
- The 4-task coding-quality cache-OFF vs cache-ON.
- Bake: default WRITE_THROUGH+SNAPSHOT on with APC (behavior-preserving when APC off).

Cross-refs: [[why_option1_snapshot_side_failed]], [[apc_ssm_carrier_deep_findings]],
[[sglang_mamba_radix_cache_design]].
