# P8 — Is there a state-level test gap on the align-mode Mamba resume path under heterogeneous block geometry?

Pinned SHA: `8359e15aae32dee9dc1f259a9b2574fb72b5507e` (== `origin/main` HEAD, 2026-09-10). Read-only.

**Verdict: the gap is REAL.** Nothing upstream — merged or open — exercises the align-mode
prefix-cache resume path with `cache_config.block_size != MambaSpec.block_size` while asserting
anything about **state tensor contents**. Every heterogeneous-geometry test is pure-int; every
state-tensor test is homogeneous.

## 1. The geometry actually diverges at runtime (not hypothetical)

Two independent sites make the scheduler/attention block unit differ from the Mamba state grid:

- `vllm/v1/engine/core.py:345-349` — after KV-cache resolution, `cache_config.block_size` is
  **overwritten** with `min(block_size of prefix-cacheable groups)`. A prefix-cacheable drafter
  attention group at 816 drags it below a Mamba group at 1648. The comment at `:336-339` shows this
  was already patched once (to exclude non-prefix-cacheable groups) — a *prefix-cacheable* small
  group still reaches it.
- `vllm/v1/core/kv_cache_utils.py:1372-1387` (`unify_kv_cache_spec_page_size`) — a `MambaSpec` keeps
  its `block_size` and only grows `page_size_padded` (`:1379`), while non-Mamba specs get
  `block_size * ratio` (`:1387`). The comment at `:1376-1378` names the draft model explicitly.

`vllm/platforms/interface.py:939-940` (`mamba_block_size = cache_config.block_size` in align mode)
runs *before* both, so it does not hold afterwards. This is why #53142 reproduces for some
reporters and not others (uniproc vs multiproc config sharing), and why `bestxrr` saw equal sizes.

## 2. Where the invariant is assumed vs. enforced

| Stage | File:line @ SHA | Block unit assumed |
|---|---|---|
| Hash publication | `kv_cache_utils.py:705-793` | `hash_block_size` = GCD; `scheduler_block_size` = LCM |
| Cache hit | `single_type_kv_cache_manager.py:1476-1509` | maps hash units → `kv_cache_spec.block_size` |
| Alloc / checkpoint | `single_type_kv_cache_manager.py:1710-1745` | `self.block_size` (= `MambaSpec.block_size`) |
| **Worker seed (MRV2)** | **`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:121-122`** | **`self.cache_config.block_size`** ← the only Mamba quantity in `v1/worker/` read from `cache_config.block_size` |
| State advance | `mamba_hybrid.py:212-223` → `mamba_utils.py:545` | `MAMBA_BLOCK_SIZE = mamba_spec.block_size` |

The seed and the advance use **different divisors of the same quantity**. They coincide iff the
geometry is homogeneous. The seed is not a cosmetic index: `mamba_utils.py:534-538` stores it as
`src_col`, i.e. the column the pre-copy **reads the resumed state from**. Wrong seed ⇒ either
out-of-range read (#53142 IMA) or a silently wrong state restored.

## 3. Test inventory — every half is covered, the composition is not

- **Copy exactness given columns**: `tests/kernels/mamba/test_precopy_mamba_align.py:224-225`
  (`rtol=0, atol=0`). But `src_col`/`dst_col` are *inputs*, and the harness has **no attention block
  size at all** — it cannot see a seed bug.
- **Seed arithmetic**: `test_precopy_mamba_align.py:392-418` (`np.testing.assert_array_equal` on
  `state_idx`/`src_col`, uniform `block_size=4`); #53798's new
  `test_add_request_seeds_state_idx_in_mamba_blocks` (heterogeneous 16 vs 880) asserts exactly one
  int. No tensors.
- **Heterogeneous geometry**: only `tests/v1/core/test_mamba_align_chunk_split.py`
  (`ATTN_BLOCK_SIZE=16` vs `MAMBA_BLOCK_SIZE=1600/1536/12288`) and some
  `test_partial_prefix_cache_hits.py` cases (2/4, 16/32). All scheduler-level ints; `:276-279`
  compares `block_hash_num_tokens` against a dict the test builds itself, never a state tensor.
- **State-tensor oracles**: all homogeneous. `test_hybrid.py::test_same_mamba_output_apc_on_vs_off`
  = `block_size=16`/`mamba_block_size=16`, output-level. `tests/v1/e2e/test_replayssm_decode.py`
  = output-level, no `block_size` passed. `tests/v1/e2e/general/test_mamba_prefix_cache.py`
  = `BLOCK_SIZE=560` for everything; its MRV1 oracle (`:478-503`) is `torch.allclose(1e-2)` **with a
  `<1%-of-elements-differ` escape hatch** that would not flag a wrong-block restore.

**Nothing composes seed → src_col → bytes under unequal geometry.**

## 4. The single smallest missing invariant

> **(R) Resume-restore fidelity in Mamba block units.** In `mamba_cache_mode="align"` with
> `cache_config.block_size != MambaSpec.block_size`, for a request admitted with
> `num_computed_tokens = n > 0`, the first `preprocess_state` must pre-copy from block-table column
> `cdiv(n, MambaSpec.block_size) - 1`, and the destination running column must afterwards be
> **bit-identical** (`rtol=0, atol=0`) to that source block's pre-step contents.

This is the *restore* class (copy/index correctness → zero-tolerance oracle), **not** recomputation
parity (numerics → tolerance oracle). Conflating the two is why the existing e2e oracles are loose
enough to miss it.

Why an int assertion is insufficient: three open PRs fix the seed with three **different divisor
sources** — #53798 `mamba_spec.block_size`, #55688 `cache_config.mamba_block_size`, #53803 a
`_seed_state_block_idx` helper. Each asserts against its own `SimpleNamespace` stub, so each passes
its own test while only the spec-derived one is reliable in the field (`cache_config.mamba_block_size`
is overwritten at `interface.py:940`). A state-level test pins semantics rather than the expression.
`fabiopili` on #53142 warns that landing the seed fix alone **trades the crash for silent wrong-state
resumes** — precisely what no current oracle detects.

## 5. Where a regression would live

**`/home/mark/shared/vllm-head/tests/v1/worker/test_mamba_hybrid_model_state.py` — 138 lines**, the
smallest align-mode worker test file. It already (a) uses `object.__new__(MambaHybridModelState)` +
`SimpleNamespace`, (b) sets `_align_mode=True` and pokes `_mamba_state_idx_gpu` (`:110-138`),
(c) contains CUDA-gated tests (`:61`, `:109`), and (d) is the file **both** #53798 and #55688 extend
— so the regression lands beside the int tests and upgrades them.

**GPU-required but model-free** — unavoidable and deliberate. The invariant is about which bytes the
fused Triton kernels move; a CPU reimplementation would re-assert the formula, which is exactly the
failure mode of the existing int tests. Cost is a 3-layer fake state pool, ~1 s, no weights, no new
CI tier. (`tests/kernels/mamba/utils.py` plus `_build_state`/`_build_meta` from
`test_precopy_mamba_align.py:67-124` provide the harness; they would move to the shared util.)

Sketch (~18 lines):

```python
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_resume_restores_committed_state_under_unequal_block_sizes():
    MAMBA_BS, ATTN_BS, n = 1648, 816, 1648 * 3      # #54076 repro geometry
    st = object.__new__(MambaHybridModelState)
    st._align_mode, st.max_num_reqs, st.device = True, 2, torch.device("cuda")
    st.cache_config = SimpleNamespace(block_size=ATTN_BS, mamba_cache_mode="align")
    st.rope_state = st.prompt_embeds_state = st.recoverssm = None
    _install_fake_align_ctx(st, mamba_block_size=MAMBA_BS, num_cols=4)  # fake state pool
    st.add_request(0, SimpleNamespace(num_computed_tokens=n, mm_features=[]))

    src_col = n // MAMBA_BS - 1                       # == 2, NOT (n-1)//ATTN_BS == 6
    assert int(st._mamba_state_idx_gpu[0]) == src_col
    committed = [s[bt[0, src_col]].clone() for s in st._fake_state]
    st.preprocess_state(input_batch, block_tables, kv_cache_config,
                        num_computed_tokens=torch.tensor([n], device="cuda"))
    dst = int(st._mamba_state_idx_gpu[0])
    for s, ref in zip(st._fake_state, committed):     # restore must be byte-exact
        torch.testing.assert_close(s[bt[0, dst]], ref, rtol=0, atol=0)
```

At the pinned SHA this fails on the `src_col` assert (6 vs 2) and, with the seed patched but the
copy mis-wired, on the zero-tolerance compare.

**Effort: ~0.5–1 day.** Helpers exist; the work is standing up `MambaSpecDecodeGPUContext` over a
fake state pool without a model (`_ensure_align_ctx` needs `model.get_mamba_state_copy_funcs` and
`validate_mamba_state_copy_funcs` mocked — `test_precopy_mamba_align.py:246-288` already does this).

## 6. Do the in-flight PRs close it?

**No.** #54076 (open) — scheduler ints only, 816/1648, never runs the worker or any kernel.
#53798 (open) — one int assert (`_mamba_state_idx_gpu[1] == 121`), no tensors. #55688 (open) —
does *not* create `test_mamba_prefix_cache.py` (it pre-exists at 1227 lines); its change there is
env-var scoping hygiene with zero new assertions, and its geometry is homogeneous 560/560. #53803's
`test_mamba_align_resume_seed.py` is `cpu_test`, pure int. Invariant (R) remains unasserted after
all four merge.
