# Deferred MTP KV publication: bounded witness source map

2026-09-29. Source-only implementation guidance for the existing B1 forced cycle. No code, gate, GPU, model, remote operation, or numerical experiment was changed or executed. This concerns publication bytes, not whether the MTP forward computes numerically correct values.

## Exact operation and byte contract

The source-pinned generated Eagle executes its first MTP model forward at lines 821–838, then calls `_fr13_mtp_kv1` at 956–972. `launch_attn_kv_linear_remap_syncfree_fixed1_drafter` in `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:10176–10197` delegates to the existing fixed16 implementation with **one cache and `dst_pi=None`**. It does not use the target verifier's inverse permutation for source slots.

For B1, let `P` be the case's independently fixed prefix length, `T[k]` its current request-owned block table in physical-cache units, `B` the actual MTP cache block size, and `s(p)=T[p//B]*B+p%B`. Require the restored 32-row map to equal `s(P+i)` for every `i=0..31`. With accepted non-root node `a_j=accepted_paths[0,j]`, the expected copy is:

`after[s(P+j+1)] = before[s(P+a_j)]` for `j < accepted_len`.

The root `s(P)` is not a destination. No inverse-PI conversion belongs in this MTP oracle. Cases with zero accepted drafts are meaningful no-change byte controls, not evidence of a nontrivial copy. Nodes, accepted length, neutral path tail and parent chain must match the fixed case.

The shared implementation at 10088–10149 gathers all 16 source rows and all 16 destination-prior rows before writing; active is `(j < accepted_len) and (a_j != j+1)`. It physically writes all 16 destination depths using `where`, including value-identical inactive destinations. Build expected bytes from one immutable BEFORE capture, not sequentially from already-updated expected rows. Include depths 1–16, root, non-destination tree rows and bounded guards. Source/destination overlap is valid; duplicate destination physical addresses or aliasing request blocks are not. Validate every indexed slot, including padded path entries used in the all-16 gather.

## Cache and independent mapping ownership

Runner 6991–7032 obtains the cache through the exact registry singleton `mtp.layers.0.self_attn.attn` (GDN constant at 579), caches it as `runner._fr13_fixed32_mtp_kv_cache`, and places that object in the deferred payload at 7123. Bind all three views: registry layer cache, runner cached view, and actual `payload['mtp_kv']`; also require `runner.drafter is eagle_self`, and bind the same stored runner to the active case/request. The existing target registry helper already distinguishes this singleton from target attention16. The MTP view must not silently substitute or overlap a target cache view; use the existing pointer/storage/shape/stride validation where applicable.

Find the unique group containing the singleton in `runner.kv_cache_config.kv_cache_groups`; require equality to `eagle_self.kv_cache_gid`, and the singleton in the actual draft attention group/layer-name set. Read `runner.input_batch.block_table[gid]` for the current request row, using `num_blocks_per_row` and `get_numpy_array()`. Bind request ID/index, active block count and table bytes to the stored runner. Do not use group 0 by default or select a target attention group merely because its shape matches.

Runner 3444–3459 obtains each common metadata device table from that group's input-batch table. At 3721–3728 it selects the **drafter group's** common metadata. Eagle 7890–7927 derives its `kv_cache_gid` and metadata-builder kernel block size from the same group. Therefore the live `common_attn_metadata.block_table_tensor` is a useful additional identity check: its active request row must agree byte-for-byte with the independent CPU table over all allocated entries needed by the witness. A padded device view need not be the same Python object or have the exact same full shape as the CPU table; compare the bounded active row and record both signatures.

Use actual physical units: require `bt.block_size == mtp_kv.shape[2] == runner._kernel_block_sizes[gid]` and agreement with the draft metadata builder's resolved spec. Record `eagle_self.block_size` and group allocation block size separately and check the actual route's expected agreement. Do not infer physical cache units from `cache_config.block_size=1024`, native reference block size 64, or an FA compute tile. The candidate's pinned tree FA2 route has 1024-page guards, but no independent MTP runtime shape attestation is created by this source review. Derive and validate actual MTP cache dimensions at the hook, including BF16, two planes, declared head geometry, injective strides, storage bounds and indexed CUDA device.

## Restoration and minimal observer seams

Runner 7104–7149 stores the exact payload, accepted path/lens views, request identities, forward/event identity, permutation group/slot map/query starts/spans, and `slot_restore_complete=False`. At 7795–7847 it validates the pending restoration record and applies `sm_span = sm_span[pi]`, clears `_fr13_sr_active`, then marks restoration complete. Eagle's normal `set_inputs_first_pass` returns the same common metadata when no extra input slots are required (6745–6777); `_get_slot_mapping` copies its map to the drafter buffer and pads only its tail (538–554). The actual remap consumes `_slot_mapping_buffer[:slot_mapping_size]`, not the old permuted target map.

Insert a read-only before-hook immediately before Eagle 956 and an after-hook immediately after 972. Keep the original call once, with unchanged kwargs. The before-hook must run **after the real first MTP forward**; O0 or the target-KV before image is too early. The after-hook must precede subsequent MTP work. Stream-order capture/readback is permitted for this untimed observer; do not add another MTP forward or mutate caches, paths, slots or production events.

For the admitted B1 case, require compact_batch=batch_rows=num_reqs=1, spec indices `(0,)`, `batch_indices is None`, exact request IDs, 32-row query span, and unchanged case prefix. Require the actual payload object to be `eagle_self._fr13_fixed32_mtp_kv_payload`, the expected schema/key set, measured event identity, `slot_restore_complete is True`, empty runner restoration stash, and the payload's query-start object to be the common metadata's query starts. Compare both the now-restored payload map and the consumed drafter-buffer tree span to the independent `s(P+i)` sequence. Bind `permutation_group_id` to the identified MTP group for this pinned single-full-attention-group route. Refuse unsupported metadata reshaping/extra-slot modes rather than pretending they retain this 32-row contract.

At the immediate after-hook, the payload is **still live**: production sets `drafter_kv_complete/kv_complete` and clears the payload only at Eagle 1002–1004. Check stable payload/cache/index identities and exact expected bytes there; do not prematurely require completion or clearing. Leave the existing later seal responsible for final event completion. Before remap, the pending event must have target KV complete, drafter/combined completion unset, and the same forward/request/event identity. Preserve these metadata readbacks in the record.

Reuse the target witness's bounded coverage pattern: all allocated blocks intersecting `[P,P+32)`, plus one allocated neighbor block on each side. In those blocks, require every non-destination row unchanged, including root and inactive destination depths. Include block zero as an explicit separate guard only if that is the existing desired null-block coverage; do not claim it or the entire cache was checked otherwise. Preserve raw before/after bytes and independent index/table metadata, compare dtype bytes exactly, and let the offline reducer reconstruct expected rows rather than trusting a boolean. A failed after-check must retain the before evidence and fail the case/process; a failed before-check must not allow the remap to proceed under a falsely valid record.

## Source pins

Generated files are under `identity/generated_source/probe-20260928T035847Z/logs/generated/` in the campaign. Exact SHA-256:

- `eagle.patched.py`: `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62`.
- `gpu_model_runner.patched.py`: `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79`.
- `gdn_linear_attn.patched.py`: `23df7748f02a742753e586487e3e905e0ffb23815a9c5aaf87f5cf2c4a2ff91a`.
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`: `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`.
- `scripts/fr13_patch_fa2_tree_bias.py`: `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2`.
- `tools/q1_candidate_layer_registry_v1.py`: `3c6c4c06cffbc41a7ff7d6755b623cb01450f3fa66040b50e6c044ecdf269783`.
- `tools/q1_target_kv_witness_v1.py`: `311b047bcc1bc59809d89f71baf27dc0f178c065d20d48358e408332a081d0d8` (its current capture enforces 16 caches and its planner uses inverse-PI; neither can be called unchanged as an MTP one-cache oracle).

Any observer implementation still needs its own source binding and bounded CPU controls. This map does not certify implementation or native MTP numerical equivalence.
