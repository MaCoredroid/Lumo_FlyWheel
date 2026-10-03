# STATIC_FINDINGS — #54076: where `cache_config.block_size` and `MambaSpec.block_size` come from

Two trees were read. **Only the first was executed.**
- **0.28.0 wheel (EXECUTED)** — `/home/mark/shared/exp54928/.venv/lib/python3.12/site-packages/vllm`,
  `vllm.__version__ == "0.28.0"`, aarch64, torch 2.13.
- **origin/main (READ ONLY, NEVER BUILT OR RUN)** — `/home/mark/shared/vllm-head`,
  `git fetch origin main` on 2026-09-21 →
  **`382970ee6ca490aeaaaf4e32c53695b581ff61ba`** ("[Kimi-K3][AMD] Return KDA and MLA projection
  outputs directly (#50592)", authored 2026-09-21 15:44:41 -0500).

---
## (a) Where `cache_config.block_size` is finalized

| | 0.28.0 | origin/main @382970ee6c |
|---|---|---|
| file | `vllm/v1/engine/core.py` | `vllm/v1/engine/core.py` |
| lines | 321-324 | 339-353 |
| code | `vllm_config.cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv_cache_groups)` | `participating = [g.kv_cache_spec.block_size for g in kv_cache_groups if g.kv_cache_spec.prefix_cacheable]` then `min(participating if participating else [...all...])` |

**This is the only difference between the two trees that bears on the question.** main narrows the
`min` to groups whose spec reports `prefix_cacheable`; the in-tree comment names the motivating case
("GLM-5.3-Flash kpool tail, a 1-block/req scratch buffer with block_size=kpool: their small
block_size would otherwise drag the global block_size below the real allocator block size and desync
it from mamba"). 0.28.0 has no `prefix_cacheable` property anywhere in `v1/kv_cache_interface.py`;
its `min` is over every group.

`prefix_cacheable` on main, `vllm/v1/kv_cache_interface.py`:
- base `KVCacheSpec.prefix_cacheable` → `True` (:163)
- overridden to `False` only by `HiSparseHotSpec` (:419), `HiSparseResidentSpec` (:452),
  `CircularBufferSpec` (:900), `KpoolTailSpec` (:1005); `SlidingWindowMLASpec` (:929) returns
  `not self.bounded_replay`.
- **`HiddenStateCacheSpec` (:722) has NO override → `prefix_cacheable is True` → it IS inside main's
  filtered `min`.** This is the load-bearing fact for the answer below.

## (b) The platform normalization jschmied cites

`vllm/platforms/interface.py`, `Platform._align_hybrid_block_size` — 0.28.0 :767, main :764.
Reached from `update_block_size_for_backend` (0.28.0 :609 / main :596) "Phase 2 … if
`model_config.is_hybrid`".

| step | 0.28.0 | main |
|---|---|---|
| mamba page from state shapes | :858-862 | :855-875 |
| floor: `if cache_config.block_size < attn_block_size: cache_config.block_size = attn_block_size` | :909-910, log at :912-914 "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size." | :916-917, log :918-921 |
| align mode: `cache_config.mamba_block_size = cache_config.block_size` | :917-918 | :924-925 |
| pad mamba page to attention page | :931, log :936-940 "Padding mamba page size by %.2f%% …" | :938, log :941-945 |

A `diff` of the two function bodies shows only: a docstring reflow; a `cache_dtype_str=` kwarg on the
MLA spec; a `get_mamba_specs_from_config` branch for Qwen4Exp; and an added indexer block alignment.
**None of these change the floor, the align assignment, or the padding.** jschmied's `:931/:933/:957`
on 0.28.1rc1.dev524 are the same three statements at his tree's line numbers.

Note what this floor does and does not do: it raises `cache_config.block_size` **before** any KV-cache
group exists. It cannot protect the value that `core.py` writes afterwards.

## (c) How a separate drafter's attention KV group gets its block size

The draft model is loaded into the same worker and its attention layers land in the same
`compilation_config.static_forward_context`, so `GPUModelRunner.get_kv_cache_spec`
(0.28.0 `vllm/v1/worker/gpu_model_runner.py`:7899) builds their specs exactly like the target's —
each `Attention.get_kv_cache_spec(vllm_config)` uses `vllm_config.cache_config.block_size`.
Differing per-token page sizes are reconciled afterwards by
`unify_kv_cache_spec_page_size` (0.28.0 `vllm/v1/core/kv_cache_utils.py`:1036-1097): a layer whose
page is **smaller** than the max gets `new_block_size = layer_spec.block_size * ratio` (:1081-1083).
**That path only ever RAISES a block size.** A `MambaSpec` is padded instead (:1070-1076).
The drafter's resulting block size is reported at DEBUG by
`vllm/v1/spec_decode/llm_base_proposer.py`:1801 `"Using block size %d for drafting layers"`.

Consequence: **a separate drafter with its own attention KV-cache group cannot, by itself, pull
`cache_config.block_size` below `MambaSpec.block_size`.** Whatever the drafter's KV dtype or head
geometry, unification moves its block size up, never down. For the specific pair in this rig the
question does not even arise — the per-token pages are equal (see RESULT.md).

## (d) `MambaSpec.block_size`

`vllm/model_executor/layers/mamba/abstract.py` `MambaBase.get_kv_cache_spec`
(0.28.0 :63-79, main :63-81): `block_size=mamba_block_size` where
`mamba_block_size = vllm_config.cache_config.mamba_block_size` (0.28.0 :64, main :66).
In `--mamba-cache-mode align` that is, by (b), exactly the post-floor `cache_config.block_size`.
Identical in both trees.

## (e) `_mamba_block_aligned_split` — what it reads

`vllm/v1/core/sched/scheduler.py`: 0.28.0 `def` at :366, main at :405. The block size it uses:

    block_size = self.cache_config.block_size      # 0.28.0 :392   |   main :431

Byte-identical statement in both trees. `self.cache_config` is `vllm_config.cache_config` — the same
object `core.py` mutated in (a). The divergence is therefore exactly: `core.py`'s `min` writing a
value smaller than the `MambaSpec.block_size` that `mamba_block_size` produced at (b)/(d).

## (f) THE PATH THAT LOWERS A GROUP'S BLOCK SIZE — hidden-state cache layers

`get_kv_cache_groups` pulls `HiddenStateCacheSpec` layers out before page unification and re-adds
them with a **reduced** block size:

    per_token      = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
    max_block_size = max(common_page // per_token, 1)
    new_bs         = _largest_divisor_at_most(group_block_size, max_block_size)
    ...
    aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
    groups.append(KVCacheGroupSpec([name], aligned))

0.28.0 `kv_cache_utils.py` :1783-1821 (log :1812-1818 "Using block size %d for hidden-state cache
layer %s; page alignment wastes %d bytes (%.2f%%) per block").
main `kv_cache_utils.py` :2305-2353 (same log :2344-2350; main adds `_ensure_min_page_size` :2337).
`_largest_divisor_at_most` — 0.28.0 :1725ish / main :2226 — returns a **divisor of** the attention
groups' gcd that is **at most** `common_page // per_token`, i.e. ≤ the attention/mamba block size and
strictly less whenever the hidden state costs more than one attention token's worth of page.

These layers are created by the `extract_hidden_states` speculative method
(`vllm/model_executor/models/extract_hidden_states.py`: `CacheOnlyAttentionLayer.get_kv_cache_spec`
0.28.0 :326-334 / main :315-322; `ExtractHiddenStatesModel.__init__` sets
`num_heads = len(hf_config.eagle_aux_hidden_state_layer_ids)` and `head_size = hidden_size`,
0.28.0 :350-375 / main :339-364). Unchanged between the two trees.

**Therefore, on origin/main @382970ee6c:** a hybrid mamba model in `--mamba-cache-mode align` plus
the `extract_hidden_states` speculative method yields a `HiddenStateCacheSpec` group that is
`prefix_cacheable` (→ inside main's filtered `min`) and whose `block_size` is a proper divisor of the
mamba/attention block size (→ smaller than `MambaSpec.block_size`). The `min` at `core.py:349` then
writes that smaller value into the very `cache_config.block_size` that `scheduler.py:431` reads.
main's `prefix_cacheable` filter closes the kpool-tail route; it does **not** close this one.

## Summary of 0.28.0-vs-main differences that matter
1. main filters the `min` to `prefix_cacheable` groups (core.py:344-348). 0.28.0 does not. This makes
   the divergence **strictly harder** to reach on main — everything reachable on main is reachable on
   0.28.0, but not conversely. Any 0.28.0 observation must be re-checked against that filter, which is
   done above for `HiddenStateCacheSpec` (it passes the filter).
2. Everything else in the chain — the platform floor, the align assignment, the mamba page padding,
   `MambaSpec.block_size = cache_config.mamba_block_size`, `block_size = self.cache_config.block_size`
   inside `_mamba_block_aligned_split`, and the hidden-state block-size reduction — is the same logic
   in both trees.

---
## (g) Does main still use the min-over-groups, or has `resolve_kv_cache_block_sizes` replaced it?
(Added after the coordinator flagged PR vllm-project/vllm#58021, QHarshil, opened 2026-09-21T21:49Z,
"[Core] Make resolved KV-cache geometry authoritative across scheduler and workers". Read read-only,
and only far enough to answer this question; its diff was not reviewed.)

**The min-over-groups logic #54076 describes still exists on main and has NOT been replaced.**
On `origin/main @382970ee6c` the two mechanisms coexist and compute *different* numbers:

| value | where set on main | what it is | who reads it |
|---|---|---|---|
| `cache_config.block_size` | `v1/engine/core.py:349` (inside `_initialize_kv_caches`) | **MIN** over `prefix_cacheable` groups | `Scheduler._mamba_block_aligned_split` via `self.cache_config.block_size` (`v1/core/sched/scheduler.py:431`) |
| `scheduler_block_size` | `resolve_kv_cache_block_sizes` (`v1/core/kv_cache_utils.py:731`), called from `EngineCore.__init__:164` | **LCM** of group block sizes | passed as `Scheduler(block_size=...)` → `self.block_size` |
| `hash_block_size` | same function | `prefix_match_unit` override, else **GCD** over prefix-cacheable groups | `Request.block_hashes` granularity |

So on main the scheduler object carries `self.block_size` (LCM) *and* `self.cache_config.block_size`
(MIN), and `_mamba_block_aligned_split` reads the MIN one. That is precisely the surface #54076 is
about. Note also that on main `resolve_kv_cache_block_sizes` is called in `EngineCore.__init__`
(:164), **after** `_initialize_kv_caches` (:258) returns — not inside it. #58021's subject is that
resolution point (the hash / prefix-match unit and making the resolved geometry authoritative across
scheduler and workers), which is a different quantity from the Mamba *state* block size
(`MambaSpec.block_size` = `cache_config.mamba_block_size`) that #54076 turns on. The two issues touch
the same function neighbourhood; they are not the same quantity.

One caution about #58021's cited "16/1600 hybrid geometry observed on upstream main at 382970ee6" and
`tests/v1/core/test_mamba_align_chunk_split.py`: in that test (main, lines 34-37 and 115-141)
`ATTN_BLOCK_SIZE = 16` is the **hash block size** and `MAMBA_BLOCK_SIZE = 1600`, and the stub sets
`cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE)` — i.e. `cache_config.block_size` is
**1600, equal to** the mamba block. That test therefore documents `hash_block_size (16) < mamba block
(1600)`; it does **not** exhibit `cache_config.block_size < MambaSpec.block_size`. It is not an
instance of the #54076 divergence and should not be cited as one.

Incidental, useful to whoever reruns this on main: `resolve_kv_cache_block_sizes` on main logs
`"kv cache group sizes %s"` and `"kv lcm block sizes %s"` at INFO. **0.28.0 has neither log line**,
which is part of why the instrumentation in DEVIATIONS.md §2 was needed here.
