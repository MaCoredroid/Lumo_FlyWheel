# SGLang Qwen MTP draft factory: source conclusion

2026-09-28. Read-only local source/CPU addendum to `worker-partition-factory-bounded-review.md`. No engine/container/remote/model/GPU/workload operations. WP remains closed, workload **0/4**. The accepted vLLM partition repair is unchanged.

**The declared current Qwen EAGLE MTP route selects a hybrid wrapper for the draft, not the plain-MHA branch.** Its own full-attention KV subpool covers one draft layer; the wrapper also references the shared target request/Mamba pool. The current `sglang_worker_metadata_v1.py` safely refuses that draft at `need(not draft, 'unreviewed hybrid draft-pool layout')`, so it needs a narrow applicability repair before installation.

## Closed source chain

The newly copied `workload-plan/inspections/codex-sglang-hybrid-source-20260928T212632Z/hybrid_arch.py` is 3,559 bytes, SHA256 **`48b4bcf47cffb2cd71b835588979edd3b80c6539a3328567f35f36f550a8b7ce`**. Both were checked against its receipt. The receipt records the pinned SGLang image `sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`, created-not-started owned CID `eb1e27ac83c7cb95c9667c2f26d194d065422e5335af5e123e0ccd5d345c9b24`, and its removal. This review read those retained records only.

1. Retained `model_config.py:675–683` rewrites Qwen's draft architecture string to `Qwen3_5ForCausalLMMTP` and sets one next-N layer. It does not replace the config object's Python class there.
2. `hybrid_arch.py:48–60` recognizes the **text-config class**, including `Qwen3_5Config`, rather than checking that architecture string. It has no Qwen draft exclusion. `mambaish_config():116–126` returns that hybrid-GDN config. The nearby Nemotron draft exclusion is specific to Nemotron and does not apply to Qwen.
3. Pinned `kv_cache_configurator.py:233–234,954–976` therefore retains the hybrid-linear choice for both Qwen target and draft. With the declared non-unified/non-page-major BF16 configuration, the wrapper's full-attention backing is ordinary `MHATokenToKVPool`.
4. `_build_hybrid_linear_kv_pool():1390–1434` sets draft full-attention IDs to `[0]` and supplies `req_to_token_pool.mamba_pool` to `HybridLinearKVPool`. `_init_pools():416–438` reuses the target's request pool for this draft; only the Inkling-specific branch clones Mamba state. This is **one draft attention allocation plus a shared target-state reference**, not a new draft recurrent allocation.

These are source-derived expected choices for the declared configuration. Live object classes, flags, dtype/shape and worker identity still have to be observed during the authorized boot.

## Smallest observer treatment

- Accept the source-bound `HybridLinearKVPool` **draft** case and read K/V metadata from its own `full_kv_pool`, with exact draft full-attention mapping `[0]` for this route. Keep the existing BF16/nonquantized/plain-MHA and no-post-capture checks on that backing.
- Preserve the target path's existing observation of real allocated convolution/temporal views. For the draft, **do not enumerate those shared target tensors under `allocated_tensors.convolution/recurrent` or `packed_allocations`**. Explicitly label the Mamba member as a reference to the target's shared pool. Avoid implying the draft owns a separate recurrent state merely because its wrapper exposes that member.
- Bind the shared reference to the admitted target pool for the same boot/worker process. A host object identity paired with PID/boot/target receipt, or an `is` comparison against a supplied weak reference, can establish that relationship without tensor contents, tensor device pointers, CUDA calls, or extended tensor lifetime. Checking only the draft's two aliases against each other does not link them to the recorded target pool.
- If a future source/config selects an independent recurrent draft or a different backing, refuse that unreviewed case. Do not change runtime flags or scientific parameters to make this metadata observer pass.

## CPU evidence

Snapshot: `p0/monitor/review-response-20260927/sglang-factory-conclusion-reviewed-20260928T212755Z/`. Three independent controls execute only AST-extracted functions from the pinned `hybrid_arch.py` against fake config classes and the unchanged fake observer fixture. They show: architecture rewrite preserves Qwen hybrid classification; the architecture string alone does not select hybrid classification; and the current observer refuses the actual hybrid-draft-shaped input. No engine libraries are imported; subprocess/network entry points are blocked.

`test-log-attempt2.txt` retains **3/3 passing controls**. Attempt 1 and its script are preserved: the two predicate controls passed, while the observer control hit a reviewer fixture namespace typo (`S.W` instead of `S.S.W`); only that fixture reference changed before rerun. The refusal is a demonstrated current compatibility limitation, not a launched-engine failure.

Reviewed observer SHA remains `f1f5a327c9c47ff1deaa9f1a71c264e94da1237d26040ec006b9ba35bd3da05c`. The new snapshot binds source copies, receipt, fixtures, both control attempts and this note. The earlier projection source/role/coverage/dispatch obligations remain pending and are not re-audited here.
