# Q1 publication observer seams — bounded source map

The smallest extension is to bracket the existing convolution commit, replay, hidden-state copy/gathers, and deferred single-cache MTP remap. Call each production operation exactly once. Do not substitute a reference implementation or reconstruct its output in place. This is a source map, not execution evidence or gate approval.

`RS`, `Runner`, `Eagle`, and `GDN` below mean the frozen generated files in `identity/generated_source/probe-20260928T035847Z/logs/generated/`. `Kernel` is `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`. Exact paths and hashes are in the adjacent `.sources.json`.

## Minimal seams

| Surface | Existing source seam | Observer obligation |
|---|---|---|
| Convolution-history publication | RS `_fr13_fixed32_device_commit_route`, original `_fixed_conv_commit(...)` at 2151; implementation Kernel `launch_fixed32_conv_commit_to_col0` at 9275 | Snapshot the actual accepted-source stages, path/lens/SSI, and full physical conv banks immediately before; compare expected destinations and untouched storage after the original call. |
| Replay rings and staging | Same RS route, `_fixed_replay(...)` at 2158, then explicit freshness clear at 2203 | Bracket real replay. Preserve full ring/SSI/previous-lens/parameter bytes; separately account for intentional graph scratch writes and the flags update. |
| Target hidden rows entering MTP | Eagle `set_inputs_first_pass` at 6745, copy at 6775 | Compare all `target_hidden_states` rows to the actual first-pass buffer prefix, plus untouched buffer tail. Bind actual selected row separately. |
| MTP hidden-row selection | Eagle existing gathers at 1011 and 1031 | Capture actual first-forward output and selected indices before each existing gather; compare selected output bytes. This is distinct from the target-hidden copy. |
| Deferred MTP KV publication | Eagle post-first-forward block at 840, original `_fr13_mtp_kv1(...)` at 950–972 | Use a separate one-cache, flat-destination witness with a fresh pre-remap snapshot, after the first MTP forward. |
| Final continuation seal | GDN `_fr13_fixed32_drafter_proposal_end` at 6836 and `_fr13_fixed32_complete_pending_event` at 8457; current hooks `on_sealed` at 455 | Require every new witness to belong to the same step/request/case and finish before the existing seal; retain original target/drafter completion and event-counter checks. |

## Convolution: source values, destination, untouched bytes

Kernel `_fr13_fixed32_conv_direct_col0_kernel` (6966–7050) implements, for layer `l`, compact batch row `b`, channel `c`, physical history column `s`:

```
leaf = accepted_paths[b, accepted_lens[b]-1] if accepted_lens[b] > 0 else 0
row = SSI[l,b,0]
bank[l][row,c,s] = source_stagings[l][b*SOURCE_ROWS + state_src[leaf,s],c]
```

When the observed `commit_zero_tail` route is active, columns `s >= commit_live_state_cols` are explicitly zero, rather than copied. The deployed contract checked at Kernel 8090–8110 is BF16, `C=10240`, physical history `L=34`, `SOURCE_ROWS=36`, and three live history columns. The observer must bind the actual state fields rather than assume this route from a name. Direct metadata fusion has its own production kernel at 7153; `launch_fixed32_conv_commit_to_col0` selects it. Both must be witnessed without changing route flags.

The current registry already binds 48 conv banks in **DS** order `[rows,C,L]` to the runner's **SD** view `[rows,L,C]`, including strides and offset (hooks 125–151). Preserve this mapping. Compare every physical destination column, not just the three live taps. All non-destination bank storage and source-stage bytes must remain unchanged; compute the writable union using physical storage addresses/alias classes so an alias of a legitimate destination is not falsely called untouched.

`_FR13_FIXED32_CONV_PREGATHER['state']['source_stagings']` is the accepted-leaf source. Do not confuse it with the separate pregather `staging` buffer populated by `launch_fixed32_conv_col0_pregather` (8932). Preseed source registration is in GDN 14544–14590; persistent source/bank/layout fields are retained at Kernel 8278–8340.

## Replay: distinguish read-only rings from written scratch

The patcher allocates and aliases the stacks at 11348–11498: `k[48,capacity,32,KH,DK]`, `v[48,capacity,32,VH,DV]`, `a/b[48,capacity,32,VH]` in consumed model dtype; optional FP32 `k_norm[...,KH]` and `gate[...,VH,2]`; int32 `prev_lens[48,capacity]`, `spec_idx[48,capacity,32]`, and `flags[48,2]`. GDN 14315–14370 snapshots previous accepted lengths and SSI **at scan time**, before new acceptance lengths replace them. A witness at publication can prove that these sources are preserved and consumed; it cannot by itself prove their earlier production from the scan operands.

Fixed32 goes through Kernel `launch_tree_gdn_replay_all_layers` (16280) → `_fr13_fixed32_committer_replay` (16127) → existing captured `graph.replay()` (16257). Do not instrument only the legacy fallback or expect a Python callback inside capture to run on every graph replay.

For the non-layer-batched graph, `_fr13_fixed32_committer_graph_body` (14551–14703) constructs 16 replay positions: root at position zero, then the first 15 path entries; `valid = position <= accepted_len`. Valid entries copy ring `k/v/a/b`; invalid entries use zero for `k/v/b` and `-1e4` for `a`. Its `ssi` staging repeats running-row SSI column zero. These scratch buffers are intentionally written. Native update is `fused_sigmoid_gating_delta_rule_update(..., inplace_final_state=True)` at 14680. The alternative admitted graph calls `_fr13_fixed32_committer_native_layer_batch` (14358) and its Triton kernel (14171); bind the actual graph choice before asserting scratch coverage.

Compare entire ring/SSI/previous-lens/A_log/dt_bias storage, including inactive capacity, across commit. The route deliberately clears `flags[:,0]` afterward (RS 2203), while `flags[:,1]` should retain its staged-row value. SSM numerical correctness remains the separate numerical comparison; it is not a source-copy equality claim.

## Hidden rows and the deferred MTP work

Runner 8056 passes the padded target hidden prefix into `drafter.propose` at 8065. Eagle 764–795 derives the selected row as `query_start_loc[full_spec_row] + accepted_leaf`, preserving non-spec rows in mixed batches; a zero accepted length selects root. Eagle 6775 copies **all** target hidden rows into the drafter input buffer. Then the first MTP forward occurs at 813–838, and Eagle 1011/1031 gathers its returned hidden rows. Do not compare an MTP output row to the pre-forward target hidden row: they are different boundaries. The first method can be wrapped; the two intermediate gathers need adjacent source-pinned observer seams because the final `propose` return does not expose these buffers.

Runner 7105–7149 retains the deferred payload after target-KV publication. Runner 7795–7850 restores the current target slot map before the first MTP forward. Only then does Eagle 950–972 call Kernel `launch_attn_kv_linear_remap_syncfree_fixed1_drafter` (10176), which shares the fixed16 implementation (10039) with `expected_cache_tensors=1` and `dst_pi=None`.

For each accepted non-root position `j`, the source slot is `slot_mapping[qstart + accepted_paths[b,j]]`; destination is `slot_mapping[qstart + j+1]`. Active changed pairs use source clones, preserving overlaps. Root and every non-destination slot remain unchanged. Cache layout is `[2,nblocks,blocksize,KVheads,head_dim]`. Reuse the existing target witness's physical-slot machinery only with a separately bound drafter cache, flat destination map, and fresh post-forward/pre-remap snapshot. An O0 or target-KV snapshot is too early because the first MTP forward legitimately writes this cache.

Eagle 974–1008 then records 16+1 cache completion and clears the payload. GDN 8457–8490 seals only after both target and drafter KV completion and matching event/request identity; keep that lifecycle unchanged. No extra MTP forward is needed for these observers.

## Integration boundary

Patcher counterparts for the original conv/replay calls are 20215/20222; deferred MTP injection begins at 32320, slot restoration at 38893–38912, and payload construction near 41737. The generated-source hashes and actual installed-source hashes must agree with the separately pinned observer delta before qualification. Snapshots/readbacks belong outside production operations in the untimed qualification observer. This reconnaissance performed only text/byte reads; it did not run scientific code or validate a live layout.
