# RESULT — #54076 boot check (executed on vLLM 0.28.0, GB10, TP=1)

Verdict rule from CARD.md: "reachable" iff, in a configuration that boots,
`cache_config.block_size` (post-`min`, the value `_mamba_block_aligned_split` reads)
is STRICTLY SMALLER than the `MambaSpec` group's `block_size`.

| arm | config | `cache_config.block_size` | `MambaSpec.block_size` | divergent | booted + served |
|---|---|---|---|---|---|
| 1 | target + **DFlash2 drafter** (k=7), prefix caching + `--mamba-cache-mode align` | **832** | **832** | **NO** | yes (365.1 s) |
| 2 | control, no `--speculative-config` | **784** | **784** | **NO** | yes (310.1 s) |
| 3 | arm 1 + explicit `--block-size 816` | **1632** | **1632** | **NO** | yes (365.1 s) |
| 4 | target + **`extract_hidden_states`** speculator (1 aux layer) + ExampleHiddenStatesConnector | **200** | **800** | **YES** | yes (310.1 s) |

All four: `Qwen/Qwen3.8-27B-FP8` @017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, TP=1, `--max-model-len 4096`,
`--max-num-seqs 4`, `--gpu-memory-utilization 0.50`, `--enable-prefix-caching --mamba-cache-mode align`,
`VLLM_LOGGING_LEVEL=DEBUG`. Every number below is quoted from an archived `server.log` with its line number.

---
## Arm 1 — separate DFlash2 drafter: NOT divergent
`runs/arm1_dflash2/server.log`

    1948: INFO [platforms/interface.py:911] Setting attention block size to 832 tokens to ensure that
          attention page size is >= mamba page size.
    1949: INFO [platforms/interface.py:935] Padding mamba page size by 1.71% ...
    3192: INFO [v1/core/kv_cache_utils.py:1869] GPU KV cache size: 35,945 tokens, Maximum concurrency
          for 4,096 tokens per request: 8.78x
    3262: [[E54076]] SCHEDULER_CACHE_CONFIG id(cache_config)=0xffeb25bb3cb0 cache_config.block_size=832
          cache_config.mamba_block_size=832 cache_config.mamba_cache_mode=align
          cache_config.enable_prefix_caching=True cache_config.mamba_page_size_padded=3407872
          cache_config.num_gpu_blocks=1018
    3263: [[E54076]] SCHEDULER_FIELDS scheduler.block_size=832 scheduler.hash_block_size=832
          need_mamba_block_aligned_split=True mamba_partial_cache_hit=False has_mamba_layers=True use_eagle=True
    3264-3278: 15 KV cache groups — 10x MambaSpec, 4x FullAttentionSpec, 1x SlidingWindowSpec —
          EVERY ONE with block_size=832 and page_size_bytes=3407872.
    3279: [[E54076]] VERDICT cache_config.block_size=832 MambaSpec.block_size=832 divergent=False

The drafter's group is idx=14: `spec=SlidingWindowSpec block_size=832 n_layers=5
first_layer=model.layers.64.self_attn.attn` (log line 3278) — the 5 DFlash2 draft layers (the target's
own 64 layers are named `language_model.model.layers.*`), carrying the drafter's `sliding_window`
2048 from its config. It is a genuinely separate attention KV-cache group, and its block size is
**equal to**, not smaller than, the target's.

Why, arithmetically: both models' per-token KV page is the same 4096 B —
target full attention `2 * num_kv_heads(4) * head_dim(256) * 2 B(bf16)`, DFlash2
`2 * num_kv_heads(8) * head_dim(128) * 2 B(bf16)`. Equal per-token pages ⇒ page unification is a
no-op ⇒ identical block sizes (832 * 4096 = 3,407,872 = the `page_size_bytes` printed for every group).
And structurally: `unify_kv_cache_spec_page_size` can only RAISE a group's block size
(STATIC_FINDINGS §c), so no drafter geometry can push the `min` below the mamba block.
**jschmied's "separate drafter with its own attention KV group" is not, by itself, sufficient.**

Booted and served: `/health` 200, `/v1/models` 200, one `/v1/chat/completions` → 200, content `"OK."`
(`runs/arm1_dflash2/smoke_resp.json`, `system_fingerprint: vllm-0.28.0-b8604cd9`). The
`EngineDeadError` at log line 3366 is the SIGINT shutdown path (lines 3344-3358 show
`[shutdown] API server: shutdown triggered` → `force killing remaining process EngineCore`), not a boot failure.

## Arm 2 — control: NOT divergent
`runs/arm2_nospec/server.log`

    1843: INFO [platforms/interface.py:911] Setting attention block size to 784 tokens ...
    1844: INFO [platforms/interface.py:935] Padding mamba page size by 0.13% ...
    2455: INFO [v1/core/kv_cache_utils.py:1869] GPU KV cache size: 160,085 tokens, Maximum concurrency
          for 4,096 tokens per request: 39.08x
    2504: [[E54076]] SCHEDULER_CACHE_CONFIG cache_config.block_size=784 cache_config.mamba_block_size=784
          cache_config.mamba_cache_mode=align cache_config.mamba_page_size_padded=3211264 num_gpu_blocks=469
    2506-2509: 4 groups — 3x MambaSpec, 1x FullAttentionSpec — all block_size=784
    2510: [[E54076]] VERDICT cache_config.block_size=784 MambaSpec.block_size=784 divergent=False

Equal geometry, matching jschmied's own no-speculation result. The absolute value differs from arm 1
(784 vs 832) only because the drafter changes the mamba state shape the platform sizes against
(padded mamba page 3,211,264 B here vs 3,407,872 B with the drafter); the *relation* is the same.

## Arm 4 — `extract_hidden_states`: **DIVERGENT** (this is the answer)
`runs/arm4_extract/server.log`

    1855: INFO [platforms/interface.py:911] Setting attention block size to 800 tokens to ensure that
          attention page size is >= mamba page size.
    1856: INFO [platforms/interface.py:935] Padding mamba page size by 1.52% ...
    2462: INFO [v1/core/kv_cache_utils.py:1812] Using block size 200 for hidden-state cache layer
          cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block
    2481: INFO [v1/core/kv_cache_utils.py:1869] GPU KV cache size: 52,110 tokens, Maximum concurrency
          for 4,096 tokens per request: 12.72x
    2521: [[E54076]] POST_INIT_KV_CACHES id(cache_config)=0xfb3fe21e5400 cache_config.block_size=200
          cache_config.mamba_block_size=800 n_groups=5
    2533: [[E54076]] SCHEDULER_CACHE_CONFIG id(cache_config)=0xfb3fe21e5400 cache_config.block_size=200
          cache_config.mamba_block_size=800 cache_config.mamba_cache_mode=align
          cache_config.enable_prefix_caching=True cache_config.mamba_page_size_padded=3276800
          cache_config.num_gpu_blocks=458
    2534: [[E54076]] SCHEDULER_FIELDS scheduler.block_size=800 scheduler.hash_block_size=200
          need_mamba_block_aligned_split=True mamba_partial_cache_hit=True has_mamba_layers=True use_eagle=False
    2535-2538: 4 groups at block_size=800 (3x MambaSpec of 16 layers each, 1x FullAttentionSpec of 16)
    2539: [[E54076]] KV_GROUP idx=4 spec=HiddenStateCacheSpec block_size=200 page_size_bytes=3276800
          n_layers=1 first_layer=cache_only_layers.64
    2540: [[E54076]] VERDICT cache_config.block_size=200 MambaSpec.block_size=800 divergent=True

The `id(cache_config)` is the same object (`0xfb3fe21e5400`) at `_initialize_kv_caches` exit (line 2521)
and on the scheduler (line 2533) — i.e. the `200` is the value `_mamba_block_aligned_split` reads at
`self.cache_config.block_size` (`v1/core/sched/scheduler.py:392` on 0.28.0, `:431` on main).

Arithmetic, all from the log: the platform floors the attention block to **800** so the attention page
covers the mamba page, and (align mode) sets `mamba_block_size = 800`. The hidden-state layer's
per-token cost is `1 aux layer * hidden_size 5120 * 2 B = 10,240 B`; the common page is
`800 * 4096 = 3,276,800 B`; `3,276,800 // 10,240 = 320`; the largest divisor of 800 that is <= 320 is
**200** — and `3,276,800 - 200*10,240 = 1,228,800 B = 37.50 %` wasted, exactly the number vLLM prints
at line 2462. `min(800, 800, 800, 800, 200) = 200` ⇒ `cache_config.block_size = 200 < 800`.

Booted and served: `/health`, `/v1/models`, one `/v1/chat/completions` → 200 (`smoke_resp.json`), and the
hidden-states connector wrote a file under `runs/ehs_out/`. Same benign SIGINT `EngineDeadError` at shutdown.

## Arm 3 — explicit `--block-size 816`: NOT divergent (and the floor overrides the user)
`runs/arm3_bs816/server.log`

    1948: INFO [platforms/interface.py:911] Setting attention block size to 1632 tokens to ensure that
          attention page size is >= mamba page size.
    1949: INFO [platforms/interface.py:935] Padding mamba page size by 99.51% ...
    3192: INFO [v1/core/kv_cache_utils.py:1869] GPU KV cache size: 20,634 tokens, Maximum concurrency
          for 4,096 tokens per request: 5.04x
    3262: [[E54076]] SCHEDULER_CACHE_CONFIG cache_config.block_size=1632 cache_config.mamba_block_size=1632
          mamba_cache_mode=align mamba_page_size_padded=6684672 num_gpu_blocks=534
    3264-3278: the same 15 groups as arm 1 (10x MambaSpec, 4x FullAttentionSpec, 1x SlidingWindowSpec),
          all at block_size=1632, page_size_bytes=6684672
    3279: [[E54076]] VERDICT cache_config.block_size=1632 MambaSpec.block_size=1632 divergent=False

This is jschmied's floor, live. `--block-size 816` does NOT survive: `_align_hybrid_block_size` takes
`kernel_block_alignment_size = max(min supported kernel block, cache_config.block_size) = 816` and then
`attn_block_size = 816 * cdiv(mamba_page, 816 * 4096) = 816 * 2 = 1632`. The user's 816 is not
"floored back up to 832" — it is used as the ALIGNMENT GRAIN, so the result is the next multiple of
816, i.e. 1632, and the mamba page is then padded by **99.51 %** to match. Correction to CARD.md
prediction P3: P3 said "geometry unchanged from arm 1"; the geometry did change (1632 vs arm 1's 832).
What P3 got right, and what the question turns on, is the RELATION: still `cache_config.block_size ==
MambaSpec.block_size`, still not divergent.

---
## What this does and does not establish
- ESTABLISHED, BY MEASUREMENT, ON 0.28.0: a configuration exists in which
  `cache_config.block_size` (200) < `MambaSpec.block_size` (800) at the moment the scheduler is
  constructed, on a server that boots and answers requests. Arms 1-2 establish that a separate DFlash
  drafter is NOT such a configuration and reproduce jschmied's equal-geometry result.
- NOT ESTABLISHED HERE: anything about what the scheduler then *does* with that geometry at runtime.
  No request was traced through `_mamba_block_aligned_split`; no Mamba state was inspected; #54076's
  correctness claim is neither confirmed nor refuted by this experiment. The startup lines above are
  configuration.
- NOT EXECUTED: `origin/main`. The claim that arm 4's geometry survives on main
  @382970ee6ca490aeaaaf4e32c53695b581ff61ba is a SOURCE claim, argued in STATIC_FINDINGS §a/§f:
  main narrows the `min` to `prefix_cacheable` groups (core.py:344-348), and `HiddenStateCacheSpec`
  (kv_cache_interface.py:722) declares no `prefix_cacheable` override, inheriting `True` from
  `KVCacheSpec` (:163) through `MLAAttentionSpec` -> `FullAttentionSpec` -> `AttentionSpec`, none of
  which override it. So the hidden-state group stays inside main's `min`. Running it is jschmied's step.
