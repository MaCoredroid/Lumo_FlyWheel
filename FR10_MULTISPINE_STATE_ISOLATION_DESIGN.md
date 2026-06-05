# FR10 Multi-Spine GDN State Isolation Design

Updated: 2026-06-05

## Decision Context

The no-copy FR10 tree verifier is concluded. The next viable route is copy-recurrent multi-spine: spine A is native MTP-5 by construction, and extra spines are evaluated as isolated candidates. On GB10 the expected GDN linear verify tax is small relative to the per-forward FP8 weight stream: extra recurrent state traffic is tens of MB versus roughly 27 GB of weights per forward. The hard requirement is lossless recurrent-state isolation.

This document is a source-read design only. No server boot or build was performed.

## Source Contracts

- `vllm/v1/spec_decode/eagle.py:1234-1245`: current v1 speculative input preparation keeps `num_reqs=common_attn_metadata.num_reqs` and preserves the same `block_table_tensor`; it rewrites query starts, token indices, and slot mapping only. It is token/logit expansion, not request/state expansion.
- `vllm/v1/worker/gpu/input_batch.py:538-576`: `expand_idx_mapping` expands logits back to the same `req_state_idx`; it does not create a new request id per expanded candidate.
- `vllm/v1/attention/backends/gdn_attn.py:170-175`: GDN builds its Mamba block table through `mamba_get_block_table_tensor(...)`.
- `vllm/v1/attention/backends/gdn_attn.py:255-258` and `276-279`: for spec rows, `spec_state_indices_tensor = block_table_tensor[spec_mask, :num_spec+1]`, i.e. one row of physical state blocks per scheduled request row.
- `vllm/v1/attention/backends/utils.py:854-892`: in Mamba align mode, `mamba_get_block_table_tensor` returns the last `1 + num_speculative_blocks` blocks of each request.
- `vllm/v1/worker/mamba_utils.py:180-205`: running Mamba state is tracked as `mamba_state_idx[req_id] = curr_state_idx`; it is request-id keyed.
- `vllm/v1/worker/mamba_utils.py:222-273`: postprocess copies accepted state from the request's running block to the committed destination block, again per `req_id`.
- `vllm/v1/core/single_type_kv_cache_manager.py:901-927` and `947-1009`: Mamba align mode allocates `1 + num_speculative_blocks` state blocks per request and reuses the speculative tail blocks across steps.
- `vllm/model_executor/layers/mamba/mamba_utils.py:288-367`: Qwen3.5/GDN exposes stock copy descriptors via `MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()`, returning `(get_conv_copy_spec, get_temporal_copy_spec)`.

## Design Goal

For each live request and each candidate spine `s` in `0..N_spines-1`, provide an isolated physical GDN state row set:

```text
request canonical running state
  -> clone into spine_s initial state blocks
  -> run spine_s recurrent update using only spine_s blocks
  -> committer chooses winning spine
  -> copy winner final state back to canonical request state
  -> discard/reuse non-winning spine blocks next step
```

Losslessness condition: spine `0` must be exactly the native MTP-5 linear verify path. It starts from the same pre-spec state, uses its own blocks, and commits through the same accepted-state copy semantics. Extra spines must not write into spine `0` blocks or the request canonical row before winner commit.

## Option A: Real Request-Like Rows

Represent each candidate spine as an actual transient request-like row in the vLLM batch:

- Add synthetic request ids like `<req_id>::spine<s>` to `InputBatch.req_ids`.
- Give each synthetic row its own block table row via `InputBatch.block_table.add_row(...)`.
- Extend `requests` / scheduler-visible state enough for `mamba_utils.preprocess_mamba` and `postprocess_mamba` to see independent `req_id`s.
- Merge the selected winner back into the real request after sampling/commit, then remove synthetic rows.

Pros:

- It matches the stock Mamba isolation model: one `req_id`, one `mamba_state_idx`, one block-table row.
- GDN metadata already works if each spine is a real spec row.
- Stock copy functions and Mamba postprocess can be reused with fewer special cases.

Cons:

- This is invasive. It touches scheduler output semantics, `GPUInputBatch` request accounting, block table condensation/reordering, sampler metadata, logits processors, metrics, request lifecycle, and finished/preempted cleanup.
- It risks confusing public request accounting because transient spines are not real user requests.
- It is not a patcher-only change. It is deeper vLLM scheduler/input-batch surgery.

Scope estimate: high. Expect multi-file vLLM changes plus substantial lifecycle tests. This is the clean architecture if upstreaming, but it is not the fastest GB10 experiment.

## Option B: Custom Per-Spine Physical State Rows

Keep one public vLLM request row, but allocate extra physical Mamba state blocks in that request's speculative tail and expose a custom per-spine `spec_state_indices_tensor` to GDN.

This is the recommended build path for an FR10/FR11 experiment.

### 1. Allocate

Minimal allocation seam:

- Patch `MambaManager` in `vllm/v1/core/single_type_kv_cache_manager.py`.
- For FR multi-spine mode, increase the per-request speculative Mamba tail reservation from:

```text
1 + num_speculative_blocks
```

to:

```text
1 + num_spines * (spine_len + 1)
```

or, if the canonical running block remains outside the per-spine set:

```text
1 + num_spines * spine_state_cols
```

where `spine_state_cols = spine_len + 1` for column-0 initial plus per-position finals.

Concrete seams:

- `single_type_kv_cache_manager.py:901-927`: capacity estimate in `get_num_blocks_to_allocate`.
- `single_type_kv_cache_manager.py:947-1009`: actual block allocation and reuse in `allocate_new_blocks`.
- `gpu_model_runner.py:6439-6450`: `max_num_blocks_per_req` currently adds `kv_cache_spec.num_speculative_blocks`; the runner/input-batch sizing must account for the larger FR tail.
- `worker/block_table.py:68-72`: `BlockTable` row width is fixed at construction, so the larger tail must be reflected before input-batch allocation.

Implementation detail:

- Do not change public `num_speculative_tokens` for the sampler. Add a separate FR runtime value, e.g. `fr10_num_spine_state_blocks`, used only for Mamba block allocation and GDN metadata.
- Keep attention/KV slots flat and public request rows unchanged. Only Mamba/GDN state rows are expanded.

### 2. Clone

At the start of a spec-verify step, clone the canonical pre-spec state into each spine's column-0 block.

Preferred implementation:

- Reuse the stock copy descriptor machinery in `vllm/v1/worker/mamba_utils.py`.
- Add a helper, conceptually:

```python
collect_mamba_copy_meta_for_physical_blocks(
    src_block_idx=canonical_running_idx,
    dest_block_idx=spine_base_col0_idx,
    accept_token_bias=0,
    req_state=req_state,
)
```

- Use `model.get_mamba_state_copy_func()`; for Qwen3.5/GDN this is `(get_conv_copy_spec, get_temporal_copy_spec)`.
- Run one batched memcpy for all layers/states/spines using the existing `MambaCopyBuffers` and `do_mamba_copy_block(...)`.

Why this is safer:

- It avoids hard-coding conv/temporal tensor layouts.
- It preserves both short-conv and temporal SSM state copy semantics.
- It keeps clone cost proportional to state size and graph-compatible once the copy list is static/captured.

Required source adjustment:

- Stock `collect_mamba_copy_meta(...)` takes logical block indices into `req_state.block_ids[mamba_group_id]`. For custom spine blocks, either:
  - append the extra physical blocks into `req_state.block_ids` / block table so logical indices work, or
  - add a physical-block-id variant that writes `state[dest_block_id]` directly.

Minimal recommendation: append the extra blocks into the request's Mamba `req_to_blocks` tail and use logical indices. That keeps free/reuse tied to the request lifecycle.

### 3. Isolated Forward

Build an FR multi-spine GDN metadata view:

```text
spec_state_indices_tensor:
  shape [num_spec_decodes * num_spines, spine_state_cols]
  row (request b, spine s) -> physical blocks reserved for that spine
```

Then run the stock/native GDN spec path as independent linear rows:

- Each spine row has its own `spec_state_indices_tensor[row, :]`.
- Spine A row receives exactly the native MTP-5 token sequence and state clone.
- Spine B/C rows receive their candidate token sequences.
- No row shares state blocks with another row.

Seams:

- `gdn_attn.py:255-279`: today derives one state-index row per request from `block_table_tensor[spec_mask, :num_spec+1]`. Patch/add an FR multi-spine branch that replaces this with the expanded per-spine tensor.
- `gdn_attn.py:51-64`: metadata already carries `spec_query_start_loc`, `spec_state_indices_tensor`, `spec_sequence_masks`, and `num_accepted_tokens`; the shape contract must be extended for `num_spec_decodes * num_spines`.
- Conv path must receive the same expanded row mapping. Existing FR10 work showed conv and temporal state must be kept together; do not isolate only SSM.
- Full-attention/KV positions can remain standard linear per spine. The multi-spine route should avoid the no-copy tree mask entirely; each spine is a linear verify sequence.

Two implementation variants:

- **Patcher/minimal experiment:** inside the GDN attention branch, expand the spec rows and call the existing flat spec conv/scan kernels on the expanded spine rows. This minimizes scheduler changes but requires custom tensor construction for `query_start_loc`, token ordering, and logits gather.
- **Cleaner integration:** add an FR `MultiSpineSpecMetadata` before attention metadata building, so both GDN and full-attn see the same expanded spine rows. This is more work but less fragile.

### 4. Commit

After the committer selects `(winning_spine, accepted_len)`, copy that spine's accepted-final state back to the request canonical running/commit state.

Commit source:

```text
src = spine_state_indices[request, winning_spine, accepted_len - 1]
```

Commit destination:

```text
dest = canonical request running state block, then stock postprocess destination if block-fill occurs
```

Implementation:

- Extend the committer to emit `winning_spine_id` and `accepted_len` per request as CPU/GPU tensors.
- Add an FR Mamba postprocess helper after sampling and before the next decode:
  - collect copy meta from the winning spine source logical/physical block
  - copy into the canonical `mamba_state_idx[req_id]` block
  - let stock `postprocess_mamba` handle block-fill migration, or fold the winner source into the same copy path so stock postprocess does not overwrite it with a linear bias.

Important caveat from FR10:

- Stock `postprocess_mamba` owns the final committed block in align mode. The FR winner-state copy must be the last writer to the canonical row, or stock postprocess must be taught the FR winner source. Silent stock overwrite was a real failure mode in cross-step work.

## Recommended Build Plan

1. Add an FR-only state-tail sizing layer:
   - `fr10_num_spines`
   - `fr10_spine_len`
   - `fr10_spine_state_cols = spine_len + 1`
   - `fr10_extra_mamba_state_blocks = num_spines * spine_state_cols`

2. Patch Mamba allocation in align mode:
   - reserve the larger per-request tail
   - expose tail logical indices for each spine
   - ensure block-table row width and CUDA graph capture sizes are increased consistently

3. Add clone helper:
   - clone canonical pre-spec state into every spine col0 for both conv and temporal state
   - use stock GDN copy funcs
   - add fail-loud accounting: every spec request must clone `num_spines` rows before verify

4. Add isolated GDN metadata branch:
   - construct expanded per-spine `spec_state_indices_tensor`
   - run linear spec GDN for each spine row
   - no tree mask; no shared no-copy tree state

5. Add committer/winner state commit:
   - canonical spine A must equal native MTP-5
   - copy winning spine final state to canonical row
   - make stock Mamba postprocess FR-aware or run FR commit after stock postprocess

6. Gates before speed:
   - Spine A greedy byte/token parity against native MTP-5 on same prompts.
   - Recurrent-state gate: winner committed state equals native serial state for spine A.
   - Multi-spine isolation negative control: mutate spine B state/tokens and prove spine A logits/state do not change.
   - External superset gates: path0/native equality first, then strict-win CI.

## Engineering Scope

Patcher-only feasibility: medium-low.

- A narrow patcher experiment is possible if it only targets Mamba align mode and fixed `num_spines=2`, `spine_len=5`, but it still must patch allocation sizing, GDN metadata, committer outputs, and Mamba postprocess. That crosses several stock seams and will be fragile under scheduler/batch changes.

Deeper vLLM surgery: medium-high but clean.

- Proper support should add an explicit per-request candidate-state tail to the KV/Mamba manager and expose it through attention metadata as a first-class concept.
- This avoids fake request ids but requires real block-pool changes, block-table sizing, graph shape updates, and commit semantics.

Estimated implementation bands:

- Design/test harness and source-gated patch scaffolding: 0.5-1 day.
- Minimal fixed-shape FR patcher prototype: 2-4 days if no scheduler surprises.
- Clean robust integration with dynamic B4, graph capture, preemption/finish cleanup, and gates: 1-2 weeks.

Risk items:

- Align-mode Mamba block reuse across steps (`single_type_kv_cache_manager.py:992-1009`) must not recycle a non-winning spine block before commit.
- `mamba_state_idx[req_id]` remains a single canonical pointer; FR must not confuse canonical running state with candidate-spine state.
- Stock `postprocess_mamba` must not overwrite the FR winner commit.
- Block-table row width must be increased before CUDA graph capture; late dynamic widening will fail.
- Conv and temporal state must be cloned/committed together.

## Losslessness and Cost

Losslessness-by-construction is achievable if and only if spine A is an isolated linear row cloned from the canonical pre-spec state and committed through the same accepted-state rule as native MTP-5. Under those conditions, spine A is native MTP-5, and extra spines are strict candidates rather than perturbations.

GB10 cost remains favorable. Extra spine states are small compared with the model weight stream. For a fixed B4 decode forward, the dominant cost is streaming the FP8 weights; adding a second 5-token spine mostly increases activation/state traffic and a small amount of recurrent compute. The practical blocker is therefore the isolation/commit engineering above, not memory bandwidth economics.
