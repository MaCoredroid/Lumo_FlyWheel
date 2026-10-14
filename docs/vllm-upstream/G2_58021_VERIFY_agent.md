# G2 verification — PR #58021 @ c18f4fd6c9f41e5e6c2ce51aed0de50122b5e06a

**VERDICT: CONFIRMED** (one citation wrong; one premise is weaker than reality — the path is reachable under *default* config, not just a contrived one). All cites `@c18f4fd`.

1. **Accessor / consumer — confirmed.** `config/cache.py:379-394` raises when `resolved_hash_block_size` (field `:206`) is None. `mamba/checkpoint.py:87-89` reads it from `self.vllm_config.cache_config`, **not** from a `KVCacheConfig`. It is the only non-test consumer in `vllm/`.

2. **Ordering — confirmed.** `v1/engine/core.py:315` `determine_available_memory()` precedes the only two stamps: `core.py:374` `record_hash_block_size` and `gpu_worker.py:754-755` in `initialize_from_config` (invoked `core.py:376`). Profiling's `runner.initialize_kv_cache(minimal_config, is_profiling=True)` (`gpu/cudagraph_utils.py:988`) never stamps — no other assignment exists. `core.py:292-295` even documents that workers build a KV cache during profiling.

3. **V2 reaches it — confirmed.** `gpu_worker.py:591` → V2's `gpu/model_runner.py:971` → `cudagraph_utils.py:849`; minimal KV cache `:965-989`, `capture_model(profile_only=True)` `:923`. V2 is the **default** runner (`config/vllm.py:696-745`).

4. **PIECEWISE builds real metadata — confirmed.** `cudagraph_utils.py:825-833` passes `for_capture=full_cudagraph`; False for PIECEWISE ⇒ `attn_utils.py:481-500` calls `builder.build()`, not `build_for_cudagraph_capture`.

5. **Query len > 1 — confirmed, and it is the default.** PIECEWISE descs set `num_reqs=None` (`:352-359`); `:650` falls back to `min(num_tokens, max_num_reqs)`; `input_batch.py:143` `base_tokens = num_tokens // num_reqs`. Default `max_num_seqs=128` (`config/scheduler.py:44`) with capture sizes up to 512 ⇒ query_len 2-4. `compilation.py:1507-1510` deliberately declines to cap PIECEWISE sizes at `max_num_seqs`.

6. **Prefill classification — confirmed; the capture guard is bypassed.** `input_batch.py:202` sets `is_prefilling_np` all-False; `kda_metadata.py:437-441` zeroes `no_prior_state`, but `split_decodes_and_prefills` (`v1/attention/backends/utils.py:817-819`) returns `(0, num_reqs, 0, num_tokens)` as soon as `query_lens[0] > decode_threshold=1` — *before* `is_prefill |= is_prefilling` (`:835`). So `num_prefills == num_reqs`.

7. **FlashKDA gating — substance right, citation wrong.** `num_prefill_checkpoint_blocks=int(alignment is not None)` lives at `models/kimi_k3/nvidia/kda.py:744-748`; `kda_checkpoint.py:16-17` is only the `alignment=16` helper.

8. **Reach — confirmed.** `kda_metadata.py:704-709` calls `self.checkpoint_builder.build` (builder always constructed, `:318`); `checkpoint.py:79` passes (blocks==1) ⇒ `:89` raises.

9. **UNIFORM_BATCH — confirmed.** GDN/KDA report `UNIFORM_BATCH` (`gdn_attn.py:84`); FULL decode descs are uniform query_len=1 and skipped when `rounded_num_reqs > max_num_reqs` (`cudagraph_utils.py:311-327`); `compilation.py:1445-1461` gates only `decode_mode()==FULL`.

**Sentences to change:** replace "kda_checkpoint.py:16–17" with "kimi_k3/nvidia/kda.py:744–748"; say the `is_prefilling` guard is *bypassed by an early return*, not merely ineffective; drop "capture size > max_num_seqs (e.g. 32/16)" as exotic — defaults (128 seqs / 512 capture) already satisfy it.

**Residual gap:** static only (no GPU). Unverified at runtime that a Kimi-K3 + `flashkda` profile run actually reaches PIECEWISE capture — the minimal profiling KV cache (`num_gpu_blocks_override = min(max_num_reqs, max_cudagraph_capture_size)`, `cudagraph_utils.py:974-977`) must supply enough Mamba blocks (`kv_cache_interface.py:1063` wants `2 + spec + checkpoint` per request); an earlier failure there would mask the accessor raise.
