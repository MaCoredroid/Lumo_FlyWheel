# G3 — static check of vllm-project/vllm#58021 fix

**Head: `cdcfa2329d7bab4641deb3c78cce92a3469bfe60`**, committed 2026-09-24T19:24Z — *before* dustinCodes84600's 20:45Z comment; nothing pushed since. All citations @cdcfa2329d7.

## VERDICT: YES — follow-up warranted.
QHarshil's "a missing stamp in a real run still aborts at core.py:169" does not hold: that readback checks the engine's own object three lines after the engine stamped it, and observes no worker. With adoption optional at `gpu_worker.py:787`, fail-closed rests on an in-tree accident whose fail-open outcome is #58020's symptom.

## Q1 — What changed (c18f4fd6c9 → cdcfa2329d7)
Rebased onto `e30559b58f`. PR-content delta is **two files**; the other eight (`config/cache.py`, `v1/engine/core.py`, `v1/worker/gpu_worker.py`, …) are byte-identical.
- `vllm/model_executor/layers/mamba/checkpoint.py:81-89` — early `return None` when `cache_config.resolved_hash_block_size is None`, guarding the accessor at `:98`.
- `tests/models/kimi_k3/test_kda_metadata.py:1103-1133` — new `test_checkpoint_builder_is_inert_before_the_engine_resolves_geometry`.
- Commit message dropped its `Co-authored-by: Claude` trailer.

## Q2 — "None only during profiling"? **No — true only by in-tree accident** (defect)
- **(a) Profiling — expected.** `determine_available_memory()` (`core.py:315`) precedes the stamp (`core.py:368-374`); the profiling KV cache is built locally (`gpu/cudagraph_utils.py:986-992`).
- **(b) `core.py:168-169` is not a backstop.** It reads the `scheduler_kv_cache_config` returned at `core.py:405`, which `_initialize_kv_caches` itself stamped at `core.py:371-373` from `resolve_kv_cache_block_sizes` (`kv_cache_utils.py:747`, returns `tuple[int,int]`). It can only raise if the engine failed to stamp itself, and runs once per engine core, never per worker (elastic-EP scale-up launches a fresh core, `core.py:148-152`, which re-stamps).
- **(c) Adoption is optional** — `gpu_worker.py:787-788`. The sole in-tree caller of `Executor.initialize_from_config` (`v1/executor/abstract.py:122`) is `core.py:376`, with stamped configs, so the branch is dead today. Any out-of-tree executor or `worker_cls` building its own `KVCacheConfig` — e.g. via `get_kv_cache_config_from_groups` (`kv_cache_utils.py:1676`), which leaves both fields `None` (`kv_cache_interface.py:1450,1452`) — yields an unstamped worker: **silent `None` at `checkpoint.py:81-89`, checkpoints dropped, no error** — #58020 restored.

## Q3 — dustin's proposal (question)
Unconditional `get_hash_block_size()` at `gpu_worker.py:787` closes (c), free in-tree. The second half does not land: **no profiling state is readable at `MambaPrefillCheckpointBuilder.build()`** (`checkpoint.py:74`). `is_profiling` is only a call parameter (`gpu/model_runner.py:566-571`, `gpu_model_runner.py:6961/7099/7335`; set at `gpu/cudagraph_utils.py:992`) consumed inline (`gpu/model_runner.py:703,748`), never stored nor reaching metadata builders — new plumbing required. `build_for_cudagraph_capture` (`v1/attention/backend.py:685`) marks capture, not profiling — real post-stamp capture uses it too, so it is the wrong discriminator.

## Q4 — Regression test exercises the profiling path? **No** (observation)
`test_kda_metadata.py:1103-1133` builds the builder directly and calls `builder.build(MagicMock(), [0])` after asserting the field's *default* `None` (`:1119`) — no profiling run, no real `CommonAttentionMetadata`, no `split_decodes_and_prefills`, no PIECEWISE capture. It verifies only the "ValueError if the guard is removed" claim, and no test covers `hash_block_size=None` reaching a worker (`test_gpu_worker.py:326,335,345` and `tests/v1/engine/test_kv_cache_geometry_propagation.py` all pass stamped configs).

## Q5 — Regression vs #58020? (observation)
No masked raise: pre-PR the line was `prefix_match_unit or block_size`, which never raised, and the invalid-`prefix_match_unit` `ValueError` still fires in the core (`kv_cache_utils.py:807/835` via `core.py:368`). Residuals: (i) the guard's failure mode *is* #58020's symptom, leaving the consumer no fail-closed signal; (ii) profiling now skips the allocations at `checkpoint.py:113-126`, so `determine_available_memory()` under-measures peak — tiny, but a real divergence.

## DRAFT COMMENT (not posted)

> On "a missing stamp in a real run still aborts at `core.py:169`" — that readback looks like it checks the engine's own object: `_initialize_kv_caches` stamps `scheduler_kv_cache_config` at `core.py:371-373` and returns that same object (`core.py:405`), so `get_hash_block_size()` at `:169` can only fire if the engine failed to stamp itself three lines earlier. It never observes a worker. With adoption optional at `gpu_worker.py:787`, is anything besides "every in-tree `KVCacheConfig` happens to come from `core.py:376`" keeping an unstamped worker out? If not, `checkpoint.py:81` turns that state back into a silently dropped checkpoint — the #58020 symptom.
>
> Separately, `is_profiling` is only a parameter (`gpu/model_runner.py:569`), never persisted, so gating the no-op on real profiling state seems to need new plumbing — worth confirming?
>
> — reviewed with AI assistance (Claude)
