DRAFT — not posted. Target: vllm-project/vllm#57605 (jsolman).

---

Ran the align-mode allocation paths CPU-only at `ef97fad` and at merge-base
`63d9ad0`.

At merge-base `tests/v1/core/test_single_type_kv_cache_manager.py` is 34/34
green; at this head 18 fail, plus 11 in `test_mamba_align_chunk_split.py` and
`prefix_cache/test_partial_prefix_cache_hits.py` — every one at
`single_type_kv_cache_manager.py:1932`. Twelve need no lookahead at all: the
rewritten bound drops the `checkpoint_block` term the merge-base bound carried
(1882-1885), and a chunk exporting an internal checkpoint needs exactly that
extra block.

Two questions rather than claims:

- At a page-aligned main end the estimate returns 1 while `allocate_new_blocks`
  takes 2 — `num_required_blocks` already carries the reserved page and 1938
  adds another. Squeezing the pool to the estimate gives
  `ValueError: Cannot get 2 free blocks from the pool`. Is
  `physical_block_cap += 1` (1840) intended to bind? The `min` never clamps
  there.
- With `num_speculative_blocks > 0` that next page column already holds a real
  speculative block at merge-base. Is the corruption about the block's contents
  rather than its allocation?

Model-free repro: {{BRANCH_LINK}}

Drafted with AI assistance; happy to share the CPU-only logs.
