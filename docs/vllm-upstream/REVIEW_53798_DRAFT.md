# Review of #53798 — "[Bugfix] Seed align-mode Mamba state_idx in Mamba blocks" — draft v1

> Plan v3 item P3. Reviewed head `af5357c2b90b37bd2033578bbc97d0ddfa6cc69f`.
> Local test run and merge check: see "Verification". Posting form: one
> GitHub review using Comment. Mark posts after Codex + Claude GO and Mark's
> review. (v2: Codex round-2 replacements applied verbatim.)

---

Reviewed `af5357c2b` locally; traced the invariant through the production
call paths rather than the description.

**Invariant.** `MambaHybridModelState._mamba_state_idx_gpu` indexes the
align block table in `MambaSpec.block_size` units. `preprocess_state`
advances it using `MAMBA_BLOCK_SIZE=mamba_spec.block_size`, and the pre-copy
kernel uses it as a block-table column. On main at
`8359e15aae32dee9dc1f259a9b2574fb72b5507e`, `add_request` instead seeds it
with `(num_computed_tokens - 1) // cache_config.block_size`. When these
block sizes differ, that seed can select the wrong column or exceed the
table. V1 already uses `mamba_spec.block_size` for `prev_state_idx`; this
PR makes MRV2 use the same unit.

**Binding order.** `GPUModelRunner.initialize_kv_cache` calls
`set_kv_cache_config` after `init_attn_backend` and before
`get_additional_cg_support`; requests are added after KV initialization.
All three production callers of `_get_mamba_group_info()` (`add_request`,
`preprocess_state`, `prepare_attn`) are guarded by align mode, matching the
early return in `set_kv_cache_config` for non-align modes. These paths
therefore do not reach the new bound-spec assertion in `none`/`all` modes.
The existing assertion that Mamba groups share cache scheduling parameters
now runs during KV initialization rather than on the first batch.

**Pre-existing semantics preserved.** A fresh request (`num_computed_tokens
== 0`) still seeds `-1`; that is unchanged from `main` and is the value the
pre-copy kernel fast-exits on (`src_col < 0`). `_ensure_align_ctx` still
resolves copy funcs from `kv_cache_config` independently; unaffected.

**Test.** `test_add_request_seeds_state_idx_in_mamba_blocks` constructs the
state with `object.__new__`, binds an 880-token Mamba block beside a
16-token scheduler block, and checks `(107_360 - 1) // 880 == 121`. The old
denominator would produce 6709. This CPU test covers the seed calculation;
it does not execute the pre-copy kernel or reproduce the illegal memory
access. The two `test_mamba_utils.py` edits exercise the new binding step
before the no-argument lookup.

**Verification (local).**
- `git merge-tree --write-tree 8359e15aae32dee9dc1f259a9b2574fb72b5507e af5357c2b90b37bd2033578bbc97d0ddfa6cc69f`
  completed without conflicts. That main snapshot has 52 commits absent
  from the PR head. The test result below is for the PR head, not the
  merged tree.
- `pytest tests/v1/worker/test_mamba_hybrid_model_state.py tests/v1/worker/test_mamba_utils.py tests/v1/worker/test_gpu_model_runner_v2.py`
  at `af5357c2b` on a GB10 (sm_121, CUDA tests included): 56 passed in 9.2 s.

**Scope note, not a blocker.** This fixes the worker-side seed. The
scheduler-side chunk grid for unequal geometries is #54076's concern; the
two are independent layers and this PR does not depend on it.
