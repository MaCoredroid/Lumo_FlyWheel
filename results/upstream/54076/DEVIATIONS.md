# DEVIATIONS — item E / #54076

1. **Arm 4 added (`extract_hidden_states`).** The brief specified three arms (DFlash2 drafter,
   no-spec control, optional explicit `--block-size 816`). The static analysis (STATIC_FINDINGS.md
   §c) shows *before booting* that a separate drafter's attention group can only have its block size
   RAISED by page unification, so arms 1-3 can answer "is a DFlash drafter sufficient?" but cannot
   answer jschmied's actual question ("name a configuration in which the divergence IS reachable").
   §f identifies the one code path that lowers a group's block size — `HiddenStateCacheSpec`, created
   by the `extract_hidden_states` speculative method — and predicts it reaches the divergence on both
   0.28.0 and main. Arm 4 boots that configuration so the answer rests on a measurement rather than
   on reading alone. It uses the same target model, the same rig, the same rules.

2. **Additive logging-only instrumentation used (disclosed in CARD.md, code archived).**
   `exp54076/hook/sitecustomize.py` on `PYTHONPATH`. Stock vLLM 0.28.0 emits no INFO or DEBUG line
   carrying the post-`min` `cache_config.block_size`; the one function that would report it,
   `EngineCore.get_kv_cache_group_metadata` (v1/engine/core.py:419), has no caller and no HTTP route
   in 0.28.0. The hook wraps `Scheduler.__init__` and `EngineCore._initialize_kv_caches`: each wrapper
   calls the original first, unchanged, then only reads attributes and writes to stderr. It assigns
   nothing on any vLLM object and swallows its own exceptions. It prints `id(self.cache_config)` so
   that the object it reports is on the record as the same object `_mamba_block_aligned_split` reads.
   Nothing under `.venv` was modified. The file also replicates the system
   `/usr/lib/python3.12/sitecustomize.py` apport preamble verbatim, since placing our file on
   PYTHONPATH shadows it.

3. **`VLLM_LOGGING_LEVEL=DEBUG` on every arm** (the CARD says so). Needed for
   `llm_base_proposer.py:1801` "Using block size %d for drafting layers", which is DEBUG-only. Side
   effect: per-weight load logging makes each boot noticeably slower than the rig's previous INFO-level
   runs. No behavioural effect.

4. **Citation correction (CARD.md).** The CARD cites the 0.28.0 platform floor as
   `interface.py:908-914`. The exact lines are `:909` (`if cache_config.block_size < attn_block_size`),
   `:910` (the assignment) and `:912-914` (the log call). STATIC_FINDINGS.md carries the corrected
   numbers. The CARD is left as written.

5. **Prefix caching is ON in all arms** (`--enable-prefix-caching --mamba-cache-mode align`), unlike
   `exp54928/runner.py`, which passes `--no-enable-prefix-caching`. Required: `mamba_cache_mode`
   defaults to `"none"` (config/cache.py:137) and the align-mode assignment
   `cache_config.mamba_block_size = cache_config.block_size` (interface.py:918) is what makes
   `MambaSpec.block_size` comparable to `cache_config.block_size` at all.

6. **CARD prediction P3 partially wrong — recorded, not edited.** P3 predicted that an explicit
   `--block-size 816` would be raised "back to the hybrid-required value" with "geometry unchanged
   from Arm 1". Arm 3 measured `cache_config.block_size = 1632`, not 832: the user's value is
   consumed as the kernel *alignment grain*, so the floor lands on the next multiple of 816. The
   relation P3 was about (`cache_config.block_size == MambaSpec.block_size`, not divergent) held.
   RESULT.md §"Arm 3" states the correction in place.

7. **STATIC_FINDINGS.md §g added after the arms were launched**, in response to the coordinator's
   mid-task note about PR vllm-project/vllm#58021. It is a static reading only; it did not change the
   GPU plan, and no arm was re-run because of it.

8. **The "Using block size %d for drafting layers" DEBUG line (llm_base_proposer.py:1801) never fired**
   in arms 1 and 3 — the DFlash proposer does not go through that code path. The drafter's block size
   is instead taken from the scheduler's own `kv_cache_config.kv_cache_groups` (group idx=14,
   `SlidingWindowSpec`, 5 layers, `model.layers.64.self_attn.attn`), which is stronger evidence anyway.
