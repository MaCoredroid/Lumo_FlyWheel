# Item G — static review of vllm-project/vllm#58021

"[Core] Make resolved KV-cache geometry authoritative across scheduler and workers" (QHarshil).
Run 2026-09-23. **GitHub read-only; nothing was posted anywhere. No GPU, no code executed, no test run.**

## Named SHAs used for every file:line below

| name | SHA | what it is |
|---|---|---|
| **HEAD** | `c18f4fd6c9f41e5e6c2ce51aed0de50122b5e06a` | PR #58021 head (fetched as `pr-58021`) |
| **BASE** | `c723a831a81cb4ff89ea6d61b2d15109306ed1bc` | merge-base with `origin/main` ("[Fast Start] Cache the MTP draft model … (#57312)") |
| **PRIOR** | `354259683e0a5251c832f464e78a7a58f8945909` | the commit the PR body says was validated (pre-rebase) |
| **E-MAIN** | `382970ee6ca490aeaaaf4e32c53695b581ff61ba` | the main SHA item E and issue #58020 both cite |

Unless a line says otherwise, every `file:line` is **at HEAD**. Diff read as `git diff BASE..HEAD`
(10 files, +500/−8). PR state: OPEN, MERGEABLE, `reviewDecision=REVIEW_REQUIRED`,
**zero human reviews** (the only review row is the bot notice that fork PRs get no automated review).

---

# VERDICT

**No clear defect established, so no merge verdict is offered.** The change is internally consistent:
the resolution site moves earlier but its inputs are not mutated in between (verified below), workers
are stamped before `initialize_from_config`, and the only production consumer of the old
`prefix_match_unit or block_size` fallback is converted. The two findings worth the author's time are
a **possible defect I could not prove reachable** (F1) and a **material accuracy question about the
cited repro geometry** (Q1), plus a direct answer to the author's own fail-closed question (Q2).

Classification key: **F** = would produce wrong behaviour on a concrete path, but reachability of the
configuration is unproven; **Q** = question; **O** = observation.

---

## F1 (possible defect — mechanism concrete, reachability unproven): the resolver's *back-off* value is now fed to the Mamba checkpoint consumer

`resolve_kv_cache_block_sizes` does not always return a prefix-match unit in its second slot. In three
branches it returns the **scheduler alignment** (the LCM) there, deliberately, to make finer hashing
inert:

- `kv_cache_utils.py:769-771` — `len(groups) <= 1` → `(bs, bs)` with `bs = cache_config.block_size * dcp`
- `kv_cache_utils.py:784-786` — prefix caching **and** KV connectors both off → `(scheduler_block_size, scheduler_block_size)`
- `kv_cache_utils.py:791-796` — any Mamba group not in `align` mode → same

The LCM is `>=` every group block size, so in those branches `hash_block_size >= MambaSpec.block_size`.
Before this PR the consumer used the Mamba group's own block size:

```
vllm/model_executor/layers/mamba/checkpoint.py:86 (BASE)
    hash_block_size = self.vllm_config.cache_config.prefix_match_unit or block_size
```
After it, `checkpoint.py:88-89` uses `cache_config.get_resolved_hash_block_size()`, i.e. whatever
`_initialize_kv_caches` stamped (`core.py:368-374`), **including the back-off value**.

A concrete way for the LCM to exceed the Mamba block: `resolve_dcp_kv_block_size`
(`kv_cache_utils.py:682-689`) returns `spec.block_size * dcp_world_size` for all-attention groups and
`spec.block_size` for Mamba (`dcp_world_size_for_kv_cache_spec`, `:709-728`). With
`decode_context_parallel_size > 1` on a hybrid model whose attention and Mamba blocks are equal (see Q1),
`lcm(B*dcp, B) = B*dcp`, while the old consumer used `B`. `checkpoint.py:94-101` then passes the coarser
unit into `compute_mamba_prefill_checkpoints`, moving the checkpoint position and (per the PR body's own
mutation table) risking `is_mamba_prefill_checkpoint_valid` rejecting it and the builder emitting
`checkpoint=None` — the same silent-drop failure mode the PR is fixing, in the opposite direction.

The consumer is **not** gated on prefix caching: `num_prefill_checkpoint_blocks` is set from the KDA
prefill backend alone (`vllm/models/kimi_k3/nvidia/kda.py:741-749`,
`num_prefill_checkpoint_blocks=int(alignment is not None)`), and `KimiK3KDAMetadataBuilder.build` calls
`self.checkpoint_builder.build(...)` whenever `num_prefills > 0`
(`vllm/models/kimi_k3/nvidia/kda_metadata.py:704-709`) with no prefix-caching or cache-mode gate.

**Why this is not filed as a proven defect.** I could not establish statically (a) that
`decode_context_parallel_size > 1` is a supported combination with Kimi-K3 KDA prefill checkpoints, or
(b) whether a dropped checkpoint has any observable effect when prefix caching is off. Both need a run.
A cheap fix if it is real: keep using the group's own block size when the resolver backed off, or have
`_initialize_kv_caches` stamp `None` (leaving the accessor to fail closed) in the back-off branches
rather than stamping an alignment under a name that says "match unit".

## Q1 (highest-value question): the cited "16/1600" geometry is accurate as a *hash-vs-Mamba* description, but is not one the platform produces for a served align-mode hybrid

The citation itself checks out. `tests/v1/core/test_mamba_align_chunk_split.py:33-36` sets
`ATTN_BLOCK_SIZE = 16`, `MAMBA_BLOCK_SIZE = 1600`, and `:120-121` builds the manager with
`scheduler_block_size=MAMBA_BLOCK_SIZE, hash_block_size=ATTN_BLOCK_SIZE`, with
`:141 cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE)`. So the PR body's description —
"the resolver produces a 16-token hash unit while the current Mamba checkpoint consumer falls back to
its local 1600-token block size" — is a **correct** reading of that fixture. (For the record, this is a
*different* divergence from the one #54076 is about; see O7.)

What the fixture does not establish is that real grouping produces it. For every hybrid model
(`platforms/interface.py:630-631`, `if model_config.is_hybrid: cls._align_hybrid_block_size(...)`):

```
vllm/platforms/interface.py:916-922   if cache_config.block_size < attn_block_size:
                                          cache_config.block_size = attn_block_size
vllm/platforms/interface.py:924-925   if cache_config.mamba_cache_mode == "align":
                                          cache_config.mamba_block_size = cache_config.block_size
```
`MambaSpec.block_size` comes from `cache_config.mamba_block_size` and attention specs are built from
`cache_config.block_size`, so in align mode the attention and Mamba groups start out with the **same**
block size, and `unify_kv_cache_spec_page_size` only ever raises a group's block size. The GCD over
prefix-cacheable groups is then the Mamba block, and the old fallback (`prefix_match_unit or block_size`)
already agreed with it — i.e. the bug does not fire. Our item E measured exactly this on 0.28.0:
784/784 with no speculation, 832/832 with a DFlash2 drafter, 1632/1632 with `--block-size 816`
(`docs/vllm-upstream/E_54076_BOOTCHECK_agent.md`, arms 1-3); the same platform code is present at HEAD.

**Question for the author:** which served configuration actually yields
`hash_block_size < MambaSpec.block_size` with `prefix_match_unit=None`? The only in-tree route I found
that lowers a *prefix-cacheable* group's block size below the common one is the hidden-state-cache
re-add, `kv_cache_utils.py:2377-2381` (`group_block_size = math.gcd(...)` then
`new_bs = _largest_divisor_at_most(group_block_size, max_block_size)`; `_largest_divisor_at_most` at `:2265`), on a `HiddenStateCacheSpec` group
(`kv_cache_interface.py:722`, which inherits `prefix_cacheable = True` and so is inside the GCD at
`kv_cache_utils.py:798-802`). Item E booted that configuration and measured a hidden-state group at
block size 200 against 800 for the attention and Mamba groups. Naming a reachable configuration in
#58020 would move this from "latent inconsistency" to "this is dropping checkpoints today", and the
difference matters for how the maintainers will triage it.

Note the fix is harmless in the equal case (`resolved == derived`), so this question is about the
severity claim and the repro, not about the correctness of the change.

## Q2 (answering the author's own "worth a second opinion" question): fail-closed reachability

The author asks whether any path can reach `get_resolved_hash_block_size()` before the engine resolves.
I enumerated the consumers and the pre-stamp window and **found no such path**, with one residual
uncertainty.

**Consumers of the new accessors (whole tree, HEAD):**
- `CacheConfig.get_resolved_hash_block_size()` — production: `checkpoint.py:89` only.
- `KVCacheConfig.get_scheduler_block_size()` / `get_hash_block_size()` — production: `core.py:168-169` only.
- `record_hash_block_size` — production: `core.py:374` (engine core) and `gpu_worker.py:754-755` (worker).

**The pre-stamp window.** Stamping is `core.py:368-374`, immediately before
`self.model_executor.initialize_from_config(kv_cache_configs)` at `:376` and
`compile_or_warm_up_model()` at `:378`. Everything a worker does *before* that is inside
`determine_available_memory()` (`core.py:315`). In that window:
- `Worker.determine_available_memory` → `model_runner.profile_run()` (`gpu_worker.py:578`) →
  `_dummy_run(..., is_profile=True)` (`gpu_model_runner.py:6466-6468`) with `force_attention=False` and
  `cudagraph_runtime_mode` unset, so the `if force_attention or cudagraph_runtime_mode == FULL` guard at
  `gpu_model_runner.py:5971` is not taken and **no attention metadata is built**.
- `Worker.determine_available_memory` → `model_runner.profile_cudagraph_memory()` (`gpu_worker.py:587-591`)
  **does** build a minimal KV cache and metadata builders (`gpu_model_runner.py:6481-6507`,
  `initialize_kv_cache(minimal_config, is_profiling=True)`) and then runs
  `_warmup_and_capture(desc, mode)` with `force_attention = (mode == FULL)` (`:6871`). This is the one
  pre-stamp site that constructs attention metadata. It does **not** reach the checkpoint builder for
  the KDA path, because:
  - `MambaPrefillCheckpointBuilder.build` returns before touching the accessor when
    `num_prefill_checkpoint_blocks == 0` (`checkpoint.py:79-80`);
  - `KimiK3KDAMetadataBuilder` calls it only when `num_prefills > 0` (`kda_metadata.py:704-709`);
  - `KimiK3KDAMetadataBuilder` inherits `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`
    (`vllm/v1/attention/backends/gdn_attn.py:84`; no override in `kda_metadata.py`), so FULL capture
    descriptors are uniform-decode, and for a uniform-decode batch `split_decodes_and_prefills`
    (`vllm/v1/attention/backends/utils.py:811-840`, called at `kda_metadata.py:442-448` with
    `decode_threshold=1, treat_short_extends_as_decodes=False`) yields `num_prefills == 0`.
- Every other `force_attention=True` caller is a warmup module
  (`vllm/model_executor/warmup/{kernel_warmup,replayssm_warmup,flashinfer_sparse_mla_warmup,minimax_m3_msa_warmup}.py`),
  all reached from `compile_or_warm_up_model`, i.e. **after** `core.py:374`.

**Residual uncertainty (could not establish statically).** The step I could not fully close is whether
`cudagraph_dispatcher.get_capture_descs()` can ever hand `profile_cudagraph_memory` a FULL,
non-uniform descriptor for a UNIFORM_BATCH backend (which would make `query_lens[0] > 1` and hence
`num_prefills = num_reqs` via `utils.py:819-821`). Tracing the dispatcher's mode downgrade would settle it;
booting a Kimi-K3 KDA model with `--cudagraph-mode FULL` would settle it faster.

**Other construction paths checked and clear:**
- Every v1 worker inherits the adoption: `CPUWorker(Worker)` (`vllm/v1/worker/cpu_worker.py:32`) and
  `XPUWorker(Worker)` (`vllm/v1/worker/xpu_worker.py:23`) are the only subclasses; `Worker` is the only
  class defining `initialize_from_config` (`gpu_worker.py:740`). See O2 for the out-of-tree caveat.
- Exactly one production caller of `initialize_from_config` (`core.py:376` →
  `Executor.initialize_from_config`, `vllm/v1/executor/abstract.py:122-124` →
  `WorkerWrapperBase.initialize_from_config`, `worker_base.py:327-331`, indexing
  `kv_cache_configs[self.global_rank]` — every element of that list is stamped at `core.py:371-373`).
- Pipeline-parallel and TP stages take their config from the same stamped list.
- Elastic EP scale-up runs a **new** engine-core process (`core.py:148-152`, `_eep_scale_up_before_kv_init`
  at `:2464-2481`) and therefore re-enters `_initialize_kv_caches`; there is no in-tree path that hands a
  worker a `KVCacheConfig` produced by an already-running engine (see O3).
- Speculative-decode drafters get their backends from `initialize_attn_backend`
  (`gpu_model_runner.py:7093`), reached from `initialize_kv_cache`, i.e. after `gpu_worker.py:755`.
- KV connectors: `ensure_kv_transfer_initialized` is at `gpu_worker.py:762`, **after** the record at `:755`.
- The `is_profiling=True` minimal `KVCacheConfig` (`gpu_model_runner.py:6499-6504`) carries
  `hash_block_size=None`, but nothing on the worker side reads the `KVCacheConfig` accessors, so it is inert.
- `KVCacheConfig` is constructed with keyword arguments at all four in-tree sites
  (`kv_cache_utils.py:1695, 1764, 1828`; `vllm/v1/hisparse/layout.py:348`), so inserting the two new
  fields mid-dataclass breaks nothing in-tree.

## O1 — moving `resolve_kv_cache_block_sizes` earlier is behaviour-preserving (verified)

The call moves from `EngineCore.__init__` (BASE `core.py:164-166`) into `_initialize_kv_caches`
(HEAD `core.py:368-370`), i.e. before `initialize_from_config` and `compile_or_warm_up_model` rather than
after. The function is pure — it reads `cache_config.block_size`, `prefix_match_unit`,
`enable_prefix_caching`, `kv_transfer_config`, `decode_context_parallel_size` and the group specs, and
writes nothing (`kv_cache_utils.py:747-839`). None of those are mutated between the two sites: the only
in-tree writers of `cache_config.block_size` / `mamba_block_size` / `enable_prefix_caching` /
`prefix_match_unit` are platform and model config code that runs at config-creation time
(`platforms/interface.py:626,736,904,917,925`, `platforms/cpu.py:254-267,488`, `platforms/xpu.py:417,426`,
`model_executor/models/config.py:177,649,657`, `model_executor/layers/attention/*.py`), plus
`core.py:290` and `core.py:352`, both of which already precede the new site. The group specs are a
`copy.deepcopy` taken at `core.py:339` (`generate_scheduler_kv_cache_config`,
`kv_cache_utils.py:2404-2425`) and nothing in `vllm/v1/worker/` assigns to a spec's `block_size`.
So `EngineCore.__init__` reading back the stamped pair is the same pair the old code computed.

## O2 — the adoption is one class lower than the "sibling property" it models

The PR argues from `kv_cache_layout`. That property is adopted in **two** places: `gpu_worker.py:748-749`
*and* `WorkerBase.set_kv_cache_layout` (`worker_base.py:112-114`), the latter driven by
`model_executor.set_kv_cache_layout(layout.name)` at `core.py:301`, so it reaches any worker that
implements `WorkerBase`. The new hash-unit adoption exists only at `gpu_worker.py:754-755`. Every
in-tree v1 worker subclasses `Worker`, so this is not a defect today; an out-of-tree platform plugin
that implements `WorkerBase` directly would silently skip adoption and then hit the fail-closed raise.
Cheap to make symmetric.

## O3 — the "workers created later" motivation is not exercised by any in-tree path

The PR body and #58020 both motivate the `KVCacheConfig` channel with "workers created after startup
(for example, elastic EP scale-up or restart)". I could find no in-tree path where a worker receives a
`KVCacheConfig` from an engine that has already resolved: `initialize_from_config` has exactly one
production caller (`core.py:376`), and EEP scale-up brings up a new engine core that re-resolves from
its own groups (`core.py:148-152`, `:305-311`, `:2464-2481`). The plumbing is still the right shape —
it is what makes the multiproc worker able to adopt at all — but the stated motivation overstates it.
Relatedly, `record_hash_block_size`'s conflict branch (`kv_cache_utils.py:739-743`) is unreachable in
production today; it is exercised only by the new tests.

## O4 — the front-end process's `CacheConfig` is not stamped

`EngineCoreClient` syncs the engine's resolved geometry back to the front-end process for `block_size`
and `mamba_block_size` (`core_client.py:843-847`) but not for `resolved_hash_block_size`. There is no
front-end consumer today, so this is inert; a future one would hit the fail-closed raise rather than a
wrong value, which is the intended direction.

## O5 — three worker-side call sites still re-derive the geometry locally

After the PR there is **no** remaining `prefix_match_unit or block_size` fallback in `vllm/` — the only
production reader of `prefix_match_unit` is the resolver itself (`kv_cache_utils.py:803`). But three
worker-side components still call `resolve_kv_cache_block_sizes` themselves, against their **own**
`KVCacheConfig` rather than the scheduler's:
`vllm/distributed/aux_output_connector/worker.py:117`,
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:1471`,
`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:151-154`
(plus `offloading/config.py:89`, `mooncake/store/scheduler.py:84` on the scheduler side).
They are not the consumer #58020 is about and the PR does not make them worse, but "authoritative across
scheduler and workers" is broader than what the diff delivers; worth a sentence in the PR body so a
reviewer is not surprised.

## O6 — test notes

- The regression test `test_internal_checkpoint_follows_the_engine_resolved_match_unit`
  (`tests/models/kimi_k3/test_kda_metadata.py:1052-1100`) is the existing
  `test_spec_internal_checkpoint_metadata_targets_replay_boundary[False-16-80]` case (`:347-398`) with the
  unit delivered via `resolved_hash_block_size=16` instead of `prefix_match_unit=16`, same
  `mamba_block_size=64`, same expected `checkpoint_offsets == [80]`. That is the right shape for a
  propagation test and it does assert on `checkpoint_offsets`, not on a config value, as the PR body
  claims. It is CUDA-gated.
- The `_make_builder` shim (`test_kda_metadata.py:125-132`) records
  `resolved or (prefix_match_unit or mamba_block_size)`, which reproduces the old local derivation, so the
  pre-existing parametrisations keep their meaning. Good.
- `tests/v1/engine/test_kv_cache_geometry_propagation.py:38-52` builds both the `FullAttentionSpec` and the
  `MambaSpec` with `block_size=BLOCK_SIZE` (32). The module comment at `:30-32` claims "the group block
  size is far coarser than the unit a worker would pick on its own"; with equal spec block sizes that is
  not obviously what the fixture produces (page unification would raise the *attention* group, not lower
  the hash unit below the Mamba block). The assertions themselves only need the published values to match
  across configs, which they do — the comment overstates the fixture. **Not verified by running it.**
- The two failures the author reports as pre-existing,
  `test_get_kv_cache_config_mamba_hybrid_sharing_pp_group_count_bump` and
  `..._pp_starved_stage`, are at `tests/v1/core/test_kv_cache_utils.py:2830` and `:2864` at **both** BASE
  and HEAD, at identical line numbers, and the PR only appends to that file (`:4386-4426`). Consistent
  with "pre-existing on the branch base, untouched by this change"; the stated cause (world size 2 on a
  1-GPU host) is plausible but not verified.

## O7 — relation to the #54076 / #53798 cluster (scope 3)

- **Does #58021 change what `_mamba_block_aligned_split` reads?** No. `scheduler.py:431` is
  `block_size = self.cache_config.block_size` at HEAD, byte-identical to BASE and to E-MAIN, and
  `cache_config.block_size` is still the `min` over prefix-cacheable groups at `core.py:347-356`,
  untouched by the diff.
- **Does it change what the worker seeds `state_idx` with?** No. #53798/#55507/#55601 all touch
  `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` (and #53798 also `gpu/model_runner.py`,
  `gpu/model_states/interface.py`); #58021 touches none of them.
- **Textual conflict?** None with any of the four. #54076 touches
  `vllm/v1/core/sched/scheduler.py` + `vllm/config/speculative.py`; #53798/#55507/#55601 touch
  `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`. #58021 touches `vllm/config/cache.py`,
  `vllm/model_executor/layers/mamba/checkpoint.py`, `vllm/v1/core/kv_cache_utils.py`,
  `vllm/v1/engine/core.py`, `vllm/v1/kv_cache_interface.py`, `vllm/v1/worker/gpu_worker.py`. Disjoint.
- **Semantic relation.** There are now three distinct numbers in play, and the PR body's "Not a
  duplicate" section is right that they are different consumers: `cache_config.block_size` = min over
  prefix-cacheable groups (`core.py:347-356`), read by `_mamba_block_aligned_split` — that is #54076;
  `scheduler_block_size` = LCM and `hash_block_size` = GCD (`kv_cache_utils.py:776, 804`) — this PR is
  about the GCD; and `MambaSpec.block_size` = `cache_config.mamba_block_size` — what #53798's family
  seeds with. Nothing here is a duplicate of anything there.
- **Would this be the natural carrier for a resolved Mamba state block size?** Observationally, yes: the
  new `KVCacheConfig.scheduler_block_size` / `hash_block_size` fields plus `record_*` are exactly the
  shape #54076 and #53798 would need if the maintainers wanted the Mamba block size published the same
  way instead of each side re-deriving it. Stated as an observation only; **no design proposal is being
  made and none should be posted.**

## O8 — hygiene

- **DCO:** present. The single commit (HEAD) carries `Signed-off-by: QHarshil <harshil_c@hotmail.com>`
  and `Co-authored-by: Claude <noreply@anthropic.com>`.
- **AI disclosure:** present in the PR body ("AI assistance was used for parts of this change…").
- **Validated-commit claim:** the body says it was validated on PRIOR (`354259683e`), which is no longer
  the head. I diffed the two patches' added/removed lines (`PRIOR^..PRIOR` vs `BASE..HEAD`) — **identical
  change content**; only the base moved (a rebase, which also cleared the `mergify` conflict comment of
  2026-09-22T01:06Z). The claim stands.
- **Typing/imports:** `kv_cache_utils.py:16` widens an existing `from vllm.config import VllmConfig` to
  include `CacheConfig` (no new import edge); `gpu_worker.py:82` adds
  `from vllm.v1.core.kv_cache_utils import record_hash_block_size` next to an existing
  `vllm.v1.core.sched.output` import (no new package edge). Both new accessors are annotated `-> int`
  and narrow an `int | None`, which is what mypy wants. Nothing here contradicts the author's
  "pre-commit incl. mypy-3.10 passed"; I did not run it.
- **`compute_hash`:** `resolved_hash_block_size` is added to the *ignored* factors set
  (`vllm/config/cache.py:275-292`), next to `prefix_match_unit`. Correct — it is runtime-derived and does
  not change the compiled graph.
- **`prefix_match_unit` semantics:** preserved. The user's request is never overwritten
  (`core.py:374` writes only `resolved_hash_block_size`), and the resolver still honours it
  (`kv_cache_utils.py:803-804`). The PR's claim here is accurate.
- **Error messages:** both accessors name where resolution happens
  (`cache.py:388-393`, `kv_cache_interface.py:1470-1476, 1479-1485`), and the conflict message names both
  values (`kv_cache_utils.py:740-743`). Good enough to debug from a traceback.

---

# DRAFT review comment (type: COMMENT) — NOT POSTED

> Static review at `c18f4fd6c9`; nothing run.
>
> **Back-off values now reach the new consumer.** `resolve_kv_cache_block_sizes` returns
> `scheduler_block_size` in the `hash_block_size` slot in three cases — ≤1 group
> (`kv_cache_utils.py:769-771`), prefix caching and connectors both off (`:785-786`), non-align Mamba
> group (`:791-796`). That value is the LCM, so it can exceed the Mamba block: with DCP>1,
> `resolve_dcp_kv_block_size` (`:682-689`) scales attention groups only. `checkpoint.py:89` now consumes
> it where the old code used the Mamba group's own `block_size`, and `num_prefill_checkpoint_blocks`
> (`kda.py:747`) isn't gated on prefix caching. Is that combination reachable?
>
> **On the 16/1600 repro.** For a hybrid model `_align_hybrid_block_size` raises
> `cache_config.block_size` to `attn_block_size` and, in align mode, sets
> `mamba_block_size = cache_config.block_size` (`platforms/interface.py:916-925`), so the attention and
> Mamba groups share a block size and the GCD equals the Mamba block. Which served configuration yields
> GCD < `MambaSpec.block_size`? The only route I found that lowers a prefix-cacheable group below it is
> the hidden-state-cache re-add (`kv_cache_utils.py:2377-2381`).
>
> On your fail-closed question I found no pre-stamp path: the only attention metadata built before
> `core.py:368-374` is `profile_cudagraph_memory`, and the KDA builder needs `num_prefills > 0`, which
> uniform-decode capture doesn't produce.
>
> AI assistance was used for this review; I checked every line I cite.

Word count: 191 (≤ 220). No merge verdict, no design proposal, no cluster pitch.

---

# What I could NOT establish statically

1. Whether `decode_context_parallel_size > 1` is a supported configuration together with a Kimi-K3 KDA
   Mamba group carrying prefill checkpoints — the concrete trigger for F1.
2. Whether a dropped `checkpoint` has any observable effect when prefix caching is off (F1's blast radius).
3. Whether `cudagraph_dispatcher.get_capture_descs()` can yield a FULL, non-uniform descriptor for a
   `UNIFORM_BATCH` backend during `profile_cudagraph_memory` — the one remaining crack in the Q2 answer.
4. Whether any real deployment produces `hash_block_size < MambaSpec.block_size` for a Kimi-K3 KDA model
   (Q1). I argued from `platforms/interface.py:916-925` that the align-mode hybrid case does not, and
   pointed at the hidden-state-cache route that does, but neither was booted here.
5. The actual resolved values in `tests/v1/engine/test_kv_cache_geometry_propagation.py` — whether page
   unification leaves the attention group coarser than the Mamba group in that fixture (O6, third bullet).
   One CPU-only pytest run would settle it.
6. The author's CI claims: 12/12 new tests, 167 passed + 2 pre-existing failures, pre-commit/mypy green,
   6/6 mutations caught. I verified the two named failures are pre-existing and untouched, and that the
   validated commit's change content is identical to the head's; I ran nothing.
7. Anything about behaviour under load — this review is entirely about startup-time geometry and which
   number reaches which consumer, exactly as item E was.

---

# Provenance

Read-only throughout. `git fetch origin pull/58021/head:pr-58021` and
`git fetch origin 354259683e…` in `/home/mark/shared/vllm-head`; all reads via `git show <rev>:<path>`.
The clone's checked-out branch (`fix/modelopt-lmhead-quant-gaps` @ `38f7bcef29`) was never changed and no
worktree was created. `gh` used read-only for the PR, its metadata, its commits, issue #58020 and its
(empty) comment list, and the file lists of #54076/#53798/#55507/#55601. Nothing was posted anywhere.
Context inputs: `docs/vllm-upstream/E_54076_BOOTCHECK_agent.md` and
`results/upstream/54076/CORRECTIONS.md`.
