# Review of #53798 — "[Bugfix] Seed align-mode Mamba state_idx in Mamba blocks" — draft v1

> Plan v3 item P3. Reviewed head `af5357c2b` (ptorsten/vllm, merge of main
> 2026-09-09; 52 commits behind origin/main at review time). Local test run
> and merge-forward check: see "Verification" (filled from the local run).
> Posting form: one GitHub review (Comment, not Approve — we are not
> maintainers; not "request changes" — nothing blocks). Mark posts after
> Codex + Claude GO and Mark's review.

---

Reviewed `af5357c2b` locally; traced the invariant through the production
call paths rather than the description.

**The invariant, as I read it.** `MambaHybridModelState._mamba_state_idx_gpu`
is consumed in Mamba-block units: `preprocess_state` passes it to the
state-advance/pre-copy kernel with `MAMBA_BLOCK_SIZE=mamba_spec.block_size`,
and the align block table it indexes is laid out in `MambaSpec.block_size`
blocks. On `main`, `add_request` seeds it with
`(num_computed_tokens - 1) // cache_config.block_size` — scheduler units.
The two coincide only until `unify_kv_cache_spec_page_size` scales the Mamba
block past the scheduler block, at which point a resumed request's seed
lands in the wrong column or past the table. The V1 runner already seeds in
Mamba units (`mamba_utils.py`: `block_size = mamba_spec.block_size` →
`prev_state_idx = (num_computed_tokens - 1) // block_size`), so this PR
brings MRV2 in line with V1. The fix is the right unit.

**Binding order.** `set_kv_cache_config` is called from
`GPUModelRunner.initialize_kv_cache` right after `self.kv_cache_config` is
bound and before `get_additional_cg_support`; `add_request` runs from the
input-batch update on real batches, after KV init. `_get_mamba_group_info()`
now asserts the spec is bound — I checked all three callers
(`add_request`, `preprocess_state`, `prepare_attn`) and each is behind
`if self._align_mode`, matching the early return in `set_kv_cache_config`
for non-align modes, so the assert cannot fire in `none`/`all` modes.
Moving the "all mamba groups share cache scheduling parameters" assert from
first-batch to KV-init is a small improvement: it fails at startup instead
of on the first real batch.

**Pre-existing semantics preserved.** A fresh request (`num_computed_tokens
== 0`) still seeds `-1`; that is unchanged from `main` and is the value the
pre-copy kernel fast-exits on (`src_col < 0`). `_ensure_align_ctx` still
resolves copy funcs from `kv_cache_config` independently; unaffected.

**Test.** `test_add_request_seeds_state_idx_in_mamba_blocks` builds the
state via `object.__new__` with an 880-token Mamba block beside a 16-token
scheduler block and checks `(107_360 - 1) // 880 == 121`. It guards exactly
the crash class and would fail on `main` (which would seed 6709). The two
`test_mamba_utils.py` edits follow the signature change.

**Verification (local).**
- Merge-forward against `origin/main` (2026-09-10, 52 commits ahead of the
  PR's merge base): `git merge-tree` clean, no conflicts.
- `pytest tests/v1/worker/test_mamba_hybrid_model_state.py tests/v1/worker/test_mamba_utils.py tests/v1/worker/test_gpu_model_runner_v2.py`
  at `af5357c2b` on a GB10 (sm_121, CUDA tests included): 56 passed in 9.2 s.

**Scope note, not a blocker.** This fixes the worker-side seed. The
scheduler-side chunk grid for unequal geometries is #54076's concern; the
two are independent layers and this PR does not depend on it.

Reviewed as a community reviewer; no approval authority implied.
