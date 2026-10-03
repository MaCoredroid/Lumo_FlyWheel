# Native GDN selected-row derivation for common-O0 import

**Use a four-way physical-row equality plus the independent logical-column formula, under the actual aligned native mode.** A positive in-range metadata index alone is insufficient. The current successor must require `mamba_cache_mode == "align"`, target speculation off and MambaSpec speculative blocks zero. No implementation, runtime, GPU or container operation was performed by this reviewer.

## Exact source chain

1. **Prepared sequence extent includes this forward.** Pinned `gpu_model_runner.py:1986–1993` computes positions and GPU `seq_lens = num_computed_tokens + num_scheduled_tokens`. At the root import seam, require one actual request, `ncomp == |P|`, scheduled count one, prepared root/position matching the frozen fixture and GPU sequence length `|P|+1`.
2. **Native state movement occurs before the hook.** `gpu_model_runner.py:3929–3945` calls `preprocess_mamba` only in `align` mode, first applying deferred CPU state corrections. Native `v1/worker/mamba_utils.py:188–205` sets

   `column = ceil((ncomp + n_sched) / spec.block_size) - 1`

   after cancelling the separately added speculative-block count. It stores this **logical block-table column** in `runner.mamba_state_idx[rid]`. If needed, it copies prior state into that column before execution (`:206–219`). `collect_mamba_copy_meta:116–129` resolves physical destinations independently for each group as `request.block_ids[gid][column]`.
3. **The actual group table reaches its own metadata builder.** Runner `:2135–2141` obtains `input_batch.block_table[gid].get_device_tensor(...)`; `:2281–2294` assigns that group table into CommonAttentionMetadata; `:2275–2276` maps the resulting metadata to the group's actual layer names. The adapter must derive the layer's unique group from the live cache configuration/registry and verify it matches the expected 48-layer partition, not assume one common physical row for every GDN group.
4. **The exact image helper makes the same selection.** Newly retained `utils.py:874–892`, SHA `9c105d862fb974f90e9212455cc27d1b033c3483090dd0b30f8296bc4d5e6866`, gathers `clamp((seq_lens - 1) // spec.block_size, min=0) + offsets` in aligned mode, where offsets cover `1 + num_speculative_blocks`. For this positive-length spec-off step, there is one offset, zero. `gdn_attn.py:170–175` invokes this helper; the non-spec branch `:199–207` then selects `block_table_tensor[:,0]`. Full graph handling `:406–418` copies the selected values into stable metadata buffers and fills padded rows with the null block; it does not choose a different real-request index.

The formula therefore agrees exactly across preprocessing and metadata: for positive `seq_len`, `ceil(seq_len/B)-1 == (seq_len-1)//B`.

## Required adapter check at the O0 seam

For each target GDN layer, after verifying its live group, actual request index, native metadata and cache geometry:

```text
require mode == "align", target speculative config is None
require group spec is the expected MambaSpec
require spec.block_size == actual group table block_size == 1024
require spec.num_speculative_blocks == 0
require ncomp == P, n_sched == 1, GPU seq_len == P + 1
column = (P + 1 - 1) // 1024
require runner.mamba_state_idx[rid] == column
require 0 <= column < table.num_blocks_per_row[request_index]
require column < len(request.block_ids[gid])
row = request.block_ids[gid][column]
require row == CPU_group_table[request_index, column]
require row == GPU_group_table[request_index, column]
require row == layer_metadata.non_spec_state_indices_tensor[request_index]
require 0 < row < conv_cache.shape[0] and row < ssm_cache.shape[0]
```

The values must be strict integer metadata, not boolean/coerced alternate IDs. Require the actual request index is the sole B1 index and that no spec sequence masks/spec state branch is present. Verify the real decode/query counts/offsets are consistent with one scheduled token; any graph-padding entries are not import owners. Use the current prepared GPU table, not a copied source record or a speculative scratch table. The importer continues to validate exact live tensor/view/storage identities and global destination disjointness before copying.

For the retained short prefix `P=13487`, the logical column is 13 while physical selected rows for native groups 0/1/2 are 53/54/55. **13 is not a cache row.** At `P=1024`, root processing selects column 1 after native preprocessing copies the prior state out of column 0; using `(P-1)//1024`, the previous selected index, or the source process's row would import into the wrong destination. Do not use `allocated_columns-1` as a substitute for the formula: spare allocated columns are not semantic state ownership.

Only the selected GDN column must be a positive real state row. Do not independently invent an all-positive-history requirement for every earlier GDN table entry; this is the native recurrent checkpoint/allocation table, not the full-attention prefix table. Target-attention hydration retains its separate complete logical-prefix ownership checks.

## All and none are different modes, not fallbacks

The exact helper `utils.py:874–875` returns the input table unchanged in `all` or `none`, so the generic non-spec GDN builder subsequently takes its first column. The runner's `mamba_state_idx` is not maintained by `preprocess_mamba` in those modes; an absent or stale entry cannot be interpreted using aligned semantics. More decisively, the pinned Qwen3.5 model constructor `qwen3_5.py:474–478` rejects `all` outright and requests aligned mode. The current adapter must refuse `all` and `none`; admitting either would be a separately designed configuration, not a missing-value workaround. This check also catches a plausible but incorrect provisional `mode=all` declaration.

## Evidence and bounded controls

Source snapshots and controls are preserved under `p0/monitor/review-response-20260927/native-gdn-row-derivation-review/`. Exact sources:

| Source | SHA-256 |
| --- | --- |
| stock runner | `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0` |
| native worker/mamba_utils.py | `f2b4192646ccc9c4fb2862d464e94550c6ebb9cf63aa778c303dabbc98048f07` |
| native gdn_attn.py | `3c60cde9b0ef4bb6e4c105491afa50d96cad368399d36bfbed81e47a6e1b4b68` |
| native backend utils.py | `9c105d862fb974f90e9212455cc27d1b033c3483090dd0b30f8296bc4d5e6866` |
| pinned Qwen3.5 model | `95013496de9ca5357acd1e72f35347ed8d8d6b2b4af7ea1cb907a1afeebe05e5` |

The last model identity is recorded authoritatively in SOURCE-SNAPSHOT.json; the source path is `identity/native_source/vllm__model_executor__models__qwen3_5.py`.

Four exact `preprocess_mamba` AST controls confirmed ordinary and boundary column/copy behavior with injected copy routines. Fifteen exact-helper AST cases with NumPy-backed stand-ins confirmed four prefix lengths × three groups, unchanged-table behavior for all/none, and the null padded zero-length case. They perform no native tensor/runtime execution. The parent's new helper extraction manifest binds the immutable current image and records the never-started inspection container's removal; that operation was parent-performed.

This is the recommended adapter invariant and source derivation, not acceptance of the yet-unreviewed connected adapter or launch authority.
