# Native MTP worker connection and exact planner review

Disposition: **the bounded registration/allocation/profile source connection is coherent; request history is still unconnected and no launch is approved here.** The newly extracted exact native planner resolves the suspected whole-cache overlap concern without weakening the owner. This supplements the earlier owner setup review; it does not change its preserved snapshot.

## Frozen source and CPU checks

| Source | SHA256 |
|---|---|
| `tools/q1_native_mtp_owner_v1.py` including `profile_dummy` | `edf4f60b769f9a92ac138633e620ec554995451414be7cc5192a1ce8ef602690` |
| `tools/q1_patch_native_mtp_worker_v1.py` | `54e92de927008f6bfa559428974c243e2d158a746fe788b88277d6589ca8fd0a` |
| `tools/q1_reference_hooks_v2_2_mtp.py` | `7347dae3add8653b4f1beaf90255a605560876ee5108579de89893e8e37fbea1` |
| Exact-image `v1/core/kv_cache_utils.py` | `4d3475bd1515cf1cbd43199bc3dc2d811482f71f046acf8fc0b678a70405a40e` |
| Exact-image `v1/kv_cache_interface.py` | `0db7ff3efa74e29a53a4a921d4ac4446fef3999f78747186796ee68f58cc8a79` |

Sources and CPU controls are preserved under `p0/monitor/review-response-20260927/native-mtp-worker-connection-review/`. The actual patcher produces seven unique owner anchors and preserves all five reference anchors; the resulting runner compiles (SHA `c6d9122c423721714aa26758dccde929d2041d72fc8f46bbbfbe55158021e1ce`). Wrong input source and repeat patching refuse. Structural checks establish attachment before native draft loading, binding after backend/builders/cache allocation, and owner invalidation before profiling cache destruction. Forty-eight combinations of graph mode, graph-capture state, eager mode, and LoRA specialization compare the actual appended `profile_dummy` body to the native source's boolean/LoRA rules and forwarded arguments; all agree. These controls execute source-extracted bodies with stubs, not vLLM or Torch.

## Exact 65-layer allocation result

The stock runner intentionally supports shared hybrid pools (`gpu_model_runner.py:6496–6502`). The accepted native smoke record `8fb996a798d91aaade5cac94debd7996292f40968ef578ba594102a08bc49ec2` has 48 same-base attention/GDN pairs. That made a whole-tensor disjointness assertion worth checking; it did not prove the new MTP pool would overlap.

Executing the exact extracted planner/interface code with the declared native 48 GDN / 16 attention specs and **MTP appended after those 64 target registrations** gives:

- GDN groups of 16, 16, 16 and one full-attention group of 17.
- Seventeen backing tensors. Tensor index 16 has `shared_by == ['mtp.layers.0.self_attn.attn']`; all sixteen target `shared_by` lists are unchanged from the 64-layer plan.
- The attention and padded GDN allocator pages are 4,194,304 bytes, for 1024 logical tokens. The accepted native FA2 view divides each allocator page into sixteen 64-token kernel blocks. Conv `[3,10240]` BF16 and state `[48,128,128]` FP32 dimensions and 4MiB strides come from that sealed native record; `gdn_attention` comes from the pinned GDN layer source.

This follows the planner's round-robin slicing `layers[i::num_groups]` (`kv_cache_utils.py:1075–1088`), not contiguous slices of 17 GDN layers. The allocation pools zip the resulting groups by layer index (`:1140–1168`), leaving the final MTP slot unshared. A counterfactual MTP-first insertion instead shares its pool with GDN layers 0/1/2, which the owner correctly rejects. Therefore **keep whole-cache MTP/target disjointness** for this exact prospective native layout and record the actual singleton `shared_by`, group census, spec equality, and pointer geometry when connecting allocation. Do not generalize the result to reordered layers or a different MTP cache spec.

The CPU planner run uses a synthetic scalar `num_gpu_blocks_override=23` only to materialize a nonzero structural plan; it allocates no tensors, does not estimate available memory, and changes no runtime config. The real launch must derive its block count from its unchanged resource policy. It also does not establish the newly loaded model's actual specs; those remain runtime attestations against this predicted layout.

## Dispatcher, metadata, and next-slot connection

The bind order is correct: native `initialize_attn_backend` resolves graph mode at 6261; `initialize_kv_cache` builds metadata and binds actual cache tensors at 6764–6782; the inserted owner then initializes its native draft backend and graph dispatcher from that resolved mode. Profiling allocations never become request-ready and cleanup invalidates the generation before clearing registered tensors. Target-only reference selectors use the owner's exact target map and cannot accidentally count the seventeenth attention layer as a target layer.

Two requirements matter for the still-unimplemented history connection:

1. **Treat captured common metadata as untrusted until joined to a real request/step.** Native `_dummy_run` also calls `_build_attention_metadata` with `for_cudagraph_capture=is_graph_capturing` (runner 5431–5440). A noncapturing dummy warmup can therefore reach the new capture call. With the required B1 setting it can leave a structurally valid dummy record. The future history consumer must require the exact active request, case, allocation generation, scheduled step, logical position, and allocated block-table ownership; clear/replace stale pending metadata at the real-step boundary. The current patch intentionally has no consumer, so this is a required connection condition, not an observed false numerical result.
2. **Do not call the full two-step proposer on a block table that has only current target allocation.** Native scheduler lines 215–222 leave `num_lookahead_tokens=0` when the target is spec-off, and lines 463–466 allocate with that value. Eagle's first pass shifts IDs while retaining target positions (657–675), so it uses existing target slots. The second pass increments the position and indexes the block table (545–570). A 64-token kernel boundary within an already allocated 1024-token page is not a missing allocation; the hazard is an entry beyond the request's actually allocated logical-page count. A nonnegative entry in the padded backing table is insufficient evidence of ownership.

The smallest unchanged-scheduler route is to preserve the first-pass MTP hidden state, logits/selected token, position, cache generation, and request/case identity; defer its follow-up until the next real target step has allocated the required slot. Join the fresh allocation to the same request and position, then execute the follow-up using **the saved first-pass MTP hidden state**, not the next target hidden state. Capture the pre-follow-up MTP boundary separately. If its speculative input differs from the teacher-forced continuation, restore/overwrite that speculative row before making it part of persistent teacher-forced MTP history. A request ending before the necessary allocation yields an unexecuted follow-up, never an invented slot. This proposes an implementation seam only; it is not an alternate numerical criterion or authorization to enable target speculative scheduling.

No source implementation, live process, container, GPU, cache, gate, or experiment counter was changed by this review.
