# P8 implementation report — align-mode resume restore fidelity

**Status: DELIVERED** (not blocked). Test-only, committed on a local branch, nothing pushed,
no GitHub writes (only `gh pr view` / `gh pr diff` reads).

- Repo: `/home/mark/shared/vllm-head`
- Branch: `p8-restore-fidelity`, based on the pinned head of PR #53798
  (`af5357c2b90b37bd2033578bbc97d0ddfa6cc69f`)
- Commit: `ca1d410ae7e74fcfd718e0421bcc0f03bf326a10` (DCO signed off, Fable co-author trailer)
- File touched: `tests/v1/worker/test_mamba_hybrid_model_state.py` (+358 / -1). No production
  code was modified in the committed diff.
- Repo left on `fix/modelopt-lmhead-quant-gaps` with a clean working tree.
- Total exclusive GPU wall time consumed: **~50 s** across four `flock`-serialized runs
  (budget was 60 min). Every GPU command was wrapped in
  `flock /home/mark/shared/exp54928/gpu.lock ...`; nothing else under `exp54928` was touched.

## What was built

`tests/v1/worker/test_mamba_hybrid_model_state.py` gains, next to the existing integer seed
test, a model-free but fully real admission → preprocess → copy harness:

- `_make_state_pool` builds a conv (bf16) and an SSM (fp32) pool per layer as an `as_strided`
  view over storage with **padded pages** (mirrors `MambaSpec.page_size_padded`, which is what
  `unify_kv_cache_spec_page_size` actually produces). Block `b` carries the distinct finite
  marker `b + 1` plus a quarter-step ramp (exact in bf16 and fp32); page padding carries the
  negated marker. Distinctness and finiteness are **asserted**, not assumed.
- `_run_align_resume_scenario` drives the production path end to end:
  `state.set_kv_cache_config(kv_cache_config)` → `state.add_request(...)` →
  `state.preprocess_state(...)`. It resolves the real metadata
  (`get_mamba_groups`, `validate_mamba_state_copy_funcs`, `MambaSpecDecodeGPUContext.create`,
  `initialize_from_forward_context` / `_populate_metadata`) and launches both real Triton
  kernels (`preprocess_mamba_align_fused_kernel`, `precopy_mamba_align_fused_kernel`).
  Nothing about the copy is mocked and no expected answer is substituted for production
  metadata selection; the only fake objects are the model's
  `get_mamba_state_copy_funcs` (returning the real `get_conv_copy_spec` /
  `get_temporal_copy_spec`) and the `static_forward_context` layer holders.
- Geometry, exactly as specified: `M = 1648`, `cache_config.block_size = 816`,
  `n = 3 * M = 4944`, one scheduled token, stale acceptance count 5 in the slot (so admission's
  reset to 1 is load-bearing — otherwise the conv window would be shifted by 4),
  8 block-table columns (so the pre-fix global-unit column **6** is real storage and the wrong
  restore is silent rather than an IMA), 33 physical blocks, a non-identity block table
  (`bt[r] = reverse(arange(1 + 8r, 9 + 8r))`, so column 2 → block 6, column 3 → block 5,
  column 6 → block 2), and request slot **1** with batch row 0.
- `_assert_restore_is_bit_identical` compares **byte views** (`.view(torch.uint8)`) of the
  logical block contents, page padding excluded, against the pre-step image with exactly the
  destination block's bytes replaced by the source block's. Padding is compared separately.
  `rtol=atol=0` is deliberately *not* used (it is numerical equality, and would also reject an
  exact restore of a NaN-bearing state). Source block, every unrelated block and all padding
  must be untouched.
- Expected columns are derived independently from the invariant
  (`src = n // M - 1`, `dst = cdiv(n + 1, M) - 1`), not read back from the buffers under test,
  and the test asserts they are 2 and 3.
- **Order matters**: the content oracle runs *before* the index assertions, so an index-only
  failure cannot mask the composed state-level failure (Codex's point 7).

Tests added (4 cases, all CUDA-gated):

| Test | Ids |
|---|---|
| `test_align_resume_restores_committed_state_bitwise` | `unequal_geometry` (1648/816), `equal_geometry_control` (1648/1648) |
| `test_align_resume_restore_oracle_rejects_broken_precopy` | `suppressed`, `misdirected` |

The negative control patches `MambaSpecDecodeGPUContext.run_fused_precopy` either to a no-op
(suppression) or to a wrapper that rewrites `src_col` to the global-unit column 6 and then
calls the **real** method (misdirection), and requires the oracle to raise.

## Validation

### (1) New tests pass at the #53798 head

```
$ cd /home/mark/shared/vllm-head && time flock /home/mark/shared/exp54928/gpu.lock \
    .venv/bin/python -m pytest tests/v1/worker/test_mamba_hybrid_model_state.py -v
tests/v1/worker/test_mamba_hybrid_model_state.py::test_prepare_attn_forwards_positions PASSED [ 10%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_add_request_seeds_state_idx_in_mamba_blocks PASSED [ 20%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_align_resume_restores_committed_state_bitwise[unequal_geometry] PASSED [ 30%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_align_resume_restores_committed_state_bitwise[equal_geometry_control] PASSED [ 40%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_align_resume_restore_oracle_rejects_broken_precopy[suppressed] PASSED [ 50%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_align_resume_restore_oracle_rejects_broken_precopy[misdirected] PASSED [ 60%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_postprocess_state_scalar_with_int32_mapping[0-1] PASSED [ 70%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_postprocess_state_scalar_with_int32_mapping[3-3] PASSED [ 80%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_recoverssm_commits_accepted_window_after_v2_sampling PASSED [ 90%]
tests/v1/worker/test_mamba_hybrid_model_state.py::test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias PASSED [100%]
======================= 10 passed, 14 warnings in 2.23s ========================

real    0m7.037s
```

This is also validation item (4): the rest of the file (6 pre-existing tests) still passes.

### (2) Old-predicate failure

Temporary, uncommitted substitution of the pre-#53798 divisor in
`vllm/v1/worker/gpu/model_states/mamba_hybrid.py::add_request`
(`// mamba_spec.block_size` → `// self.cache_config.block_size`, i.e. exactly what
`git show origin/main:...` has at main SHA `8359e15aae32dee9dc1f259a9b2574fb72b5507e`):

```
$ time flock /home/mark/shared/exp54928/gpu.lock .venv/bin/python -m pytest \
    "tests/v1/worker/test_mamba_hybrid_model_state.py::test_align_resume_restores_committed_state_bitwise" -q
...
E   AssertionError: state pool 0: expected block 6 (column 2) restored into block 5 (column 3)
    with every other block untouched; blocks differing from that image: [5]

FAILED tests/v1/worker/test_mamba_hybrid_model_state.py::test_align_resume_restores_committed_state_bitwise[unequal_geometry]
1 failed, 1 passed, 14 warnings in 1.20s

real    0m5.999s
```

The repr of the scenario in the failure output shows `seeded_col=6, src_col=2, dst_col=3` —
i.e. the failure is the **state-content** failure caused by the wrong seed, not a bare integer
assertion, and the destination block 5 received block 2's bytes instead of block 6's. The
`equal_geometry_control` case passed under the same substitution, confirming the failure is
geometry-specific and not an artifact of the harness. The production file was then restored
with `git checkout --` and re-verified (diff clean; `add_request` shows `mamba_spec.block_size`).

### (3) Negative control

Covered by the `suppressed` / `misdirected` parametrizations above, which pass only because the
oracle raises `AssertionError` matching `"blocks differing from that image"`. The `misdirected`
case runs the real copy kernel from column 6 — the exact failure the old seed produces.

### (4) Neighbouring suites unaffected (extra)

```
$ time flock /home/mark/shared/exp54928/gpu.lock .venv/bin/python -m pytest \
    tests/v1/worker/test_mamba_utils.py tests/kernels/mamba/test_precopy_mamba_align.py -q
118 passed, 14 warnings in 18.19s

real    0m22.953s
```

### (5) Lint

```
$ .venv/bin/pre-commit run ruff-check  --files tests/v1/worker/test_mamba_hybrid_model_state.py
ruff check...............................................................Passed
$ .venv/bin/pre-commit run ruff-format --files tests/v1/worker/test_mamba_hybrid_model_state.py
ruff format..............................................................Passed
```

(`ruff-format` reformatted the file once on first run; the committed file is the formatted one
and both hooks pass cleanly against it.)

## Runtimes

| Step | Wall |
|---|---|
| Full target file, 10 tests (PR head) | 7.0 s (2.2 s in pytest) |
| Two geometry cases under the old predicate | 6.0 s (1.2 s in pytest) |
| First full-file run (Triton warm-up included) | 13.7 s (3.5 s in pytest) |
| `test_mamba_utils.py` + `test_precopy_mamba_align.py`, 118 tests | 23.0 s |
| **Total exclusive GPU time** | **~50 s** |

The four new cases add roughly **1 s** to the file. No model weights, no server, no new CI tier.

## Scope limit (must be carried into any upstream framing)

This tests **worker restore fidelity only**: that the bytes committed at the source column are
the bytes that reappear at the destination column, in Mamba block units. It cannot detect a
snapshot that was already taken at the wrong token position — a correctly-copied but
semantically stale state passes. The separate scheduler-publication defect reported by
`fabiopili` on #53142 is therefore **not** covered, and this test must not be described as
closing it. Steps that perform no copy, and nonzero acceptance offsets, are separate contracts
and are not asserted here either (the kernel-level test
`tests/kernels/mamba/test_precopy_mamba_align.py` already covers the bias cases given fixed
columns).

## Would #55688 invalidate the fixture?

**Yes, mechanically — but not conceptually.** #55688 (`[Mamba] Unify FI ReplaySSM
STP/MTP/Prefix Caching Lifecycle`, head `89f5ad734bf2704d9087e5bc77d338c54b87ed0d`, OPEN) is
branched from main, not from #53798, and rewrites both files
(`vllm/v1/worker/gpu/model_states/mamba_hybrid.py` +207/-40,
`vllm/v1/worker/mamba_utils.py` +214/-89,
`tests/v1/worker/test_mamba_hybrid_model_state.py` +176/-16).

What breaks in the harness:

- There is **no `set_kv_cache_config` hook**; `_get_mamba_group_info(kv_cache_config)` keeps its
  old lazy signature, so the scenario's `state.set_kv_cache_config(...)` call disappears.
- The align gate becomes `self._needs_prefix_state_migration` (plus
  `self._use_flashinfer_replayssm`, `self._mamba_prev_last_scheduled_idx_gpu`,
  `self._is_prefilling_gpu`), so setting `_align_mode = True` alone no longer arms
  `preprocess_state`; the fixture would `AttributeError`.
- `_ensure_align_ctx` is renamed `_ensure_mamba_postprocess_ctx` (the fixture does not call it
  directly, so this is only a comment update), `preprocess_mamba_align_fused_kernel` gains a
  `PRESERVE_ACCEPTED` constexpr, and `MambaSpecDecodeGPUContext` gains `replayssm` /
  `has_flashinfer_replayssm` and renames `num_accepted_tokens_out` →
  `num_accepted_tokens_snapshot`.
- Its test-file delta rewrites `test_prepare_attn_forwards_positions` and adds
  `test_add_request_seeds_state_with_scoped_block_size`, so this patch would conflict textually
  with it as well as with #53798's test.

What survives: the invariant, the geometry, the padded state pool, the byte oracle and both
controls are all independent of those renames. Re-arming the fixture is roughly four attribute
assignments (`_needs_prefix_state_migration = True`, `_use_flashinfer_replayssm = False`,
`_mamba_prev_last_scheduled_idx_gpu = None`) plus dropping the `set_kv_cache_config` call.

Worth flagging: at #55688's head the **ordinary align path still divides by
`cache_config.block_size`** — its new `mamba_block_size` scoping applies only when
`self._use_flashinfer_replayssm` is true. So the `unequal_geometry` case would still FAIL on
#55688 as written, i.e. #55688 does not close this invariant; only #53798's spec-derived divisor
does. This matches Codex's review point 5.

## Exact diff

```
commit ca1d410ae7e74fcfd718e0421bcc0f03bf326a10
Author: mark ma <coredroid0401@gmail.com>
Date:   Fri Sep 11 23:29:31 2026 +0000

    [Test] Cover align-mode resume restore fidelity under unequal block sizes
    
    Page unification can leave MambaSpec.block_size above
    cache_config.block_size, so seeding the align running-state column in the
    global unit names a different column of the same Mamba-unit block table.
    The existing coverage splits along that seam: the pre-copy kernel test
    fixes src/dst columns as inputs, the seed tests assert a single integer,
    and every state-content oracle runs at homogeneous geometry.
    
    Compose the two halves: admit one request at n = 3 * M, schedule one
    token, and run the real set_kv_cache_config / add_request /
    preprocess_state path over a model-free padded state pool, then check
    that column 2 is restored into column 3 byte for byte with every other
    block and all page padding untouched. Byte views rather than
    rtol=atol=0, which is numerical rather than bitwise equality.
    
    An equal-geometry case (M == cache_config.block_size) is the control
    that isolates the unequal one, and a negative control drops and then
    misdirects the pre-copy to show the oracle rejects both.
    
    This covers worker restore fidelity only: a snapshot that was already
    taken at the wrong token position would be restored faithfully and pass.
    
    Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>
    Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01VLD8eK1yhEDSqocmQkR7ty
    Signed-off-by: mark ma <coredroid0401@gmail.com>

diff --git a/tests/v1/worker/test_mamba_hybrid_model_state.py b/tests/v1/worker/test_mamba_hybrid_model_state.py
index 81fd793ee..5ba443b87 100644
--- a/tests/v1/worker/test_mamba_hybrid_model_state.py
+++ b/tests/v1/worker/test_mamba_hybrid_model_state.py
@@ -1,22 +1,35 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
+import math
+from collections.abc import Callable
+from contextlib import nullcontext
+from dataclasses import dataclass
 from types import SimpleNamespace
-from unittest.mock import Mock
+from unittest.mock import Mock, patch
 
 import pytest
 import torch
 
 from vllm.config.compilation import CUDAGraphMode
+from vllm.model_executor.layers.mamba.mamba_utils import (
+    MambaStateCopyFuncsByType,
+    get_conv_copy_spec,
+    get_temporal_copy_spec,
+    is_conv_state_dim_first,
+)
 from vllm.platforms import current_platform
+from vllm.utils.math_utils import cdiv
 from vllm.v1.attention.backends.recoverssm_metadata import (
     RecoverSSMMetadata,
     RecoverSSMPostprocessMetadata,
 )
+from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
 from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
 from vllm.v1.worker.gpu.model_states import mamba_hybrid
 from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
 from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
+from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
 
 
 def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
@@ -87,6 +100,350 @@ def test_add_request_seeds_state_idx_in_mamba_blocks() -> None:
     assert state._mamba_state_idx_gpu[1] == 121
 
 
+# ---------------------------------------------------------------------------
+# Align-mode resume: admission -> preprocess -> restored state bytes
+# ---------------------------------------------------------------------------
+
+# Page unification keeps a MambaSpec at its own block size while the engine
+# drops cache_config.block_size to the smallest prefix-cacheable group, so the
+# two divisors disagree and a seed taken in global units names a different
+# column of the same (Mamba-unit) align block table.
+_MAMBA_BLOCK_SIZE = 1648
+_GLOBAL_BLOCK_SIZE = 816
+_NUM_COMPUTED = 3 * _MAMBA_BLOCK_SIZE
+# Column a global-unit seed picks here. It is backed by real storage, so the
+# wrong restore is silent rather than an illegal access.
+_GLOBAL_UNIT_COL = (_NUM_COMPUTED - 1) // _GLOBAL_BLOCK_SIZE
+
+_MAX_NUM_REQS = 4
+_REQ_SLOT = 1  # nonzero request slot; batch row 0 maps onto it
+_NUM_COLS = 8  # block-table columns, past the global-unit column
+_NUM_LAYERS = 2
+_CONV_WIDTH = 4
+_CONV_DIM = 16
+_SSM_SHAPE = (2, 8)
+_PAGE_PAD_ELEMS = 4  # MambaSpec.page_size_padded slack, in elements
+
+_ALIGN_COPY_FUNCS: MambaStateCopyFuncsByType = {
+    MambaAttentionBackendEnum.MAMBA2: (get_conv_copy_spec, get_temporal_copy_spec),
+}
+_REAL_RUN_FUSED_PRECOPY = MambaSpecDecodeGPUContext.run_fused_precopy
+
+
+@dataclass
+class _StatePool:
+    """One (layer, state-type) state pool laid out as padded pages."""
+
+    view: torch.Tensor  # [num_blocks, *block_shape] view over `storage`
+    storage: torch.Tensor  # flat backing storage, padding included
+    page_elems: int
+    block_elems: int
+
+    def logical_bytes(self) -> torch.Tensor:
+        """[num_blocks, block_bytes] byte view, page padding excluded."""
+        return self.view.reshape(self.view.shape[0], -1).contiguous().view(torch.uint8)
+
+    def padding_bytes(self) -> torch.Tensor:
+        """[num_blocks, pad_bytes] byte view of the page padding only."""
+        return (
+            self.storage.view(-1, self.page_elems)[:, self.block_elems :]
+            .contiguous()
+            .view(torch.uint8)
+        )
+
+
+@dataclass
+class _AlignResumeScenario:
+    state: MambaHybridModelState
+    pools: list[_StatePool]
+    pre_logical: list[torch.Tensor]
+    pre_padding: list[torch.Tensor]
+    block_table: torch.Tensor
+    seeded_col: int
+    src_col: int
+    dst_col: int
+
+
+def _make_state_pool(
+    num_blocks: int,
+    block_shape: tuple[int, ...],
+    dtype: torch.dtype,
+    device: torch.device,
+) -> _StatePool:
+    """A state pool whose pages carry MambaSpec-style padding.
+
+    Each block is contiguous but consecutive blocks sit a padded page apart, as
+    ``unify_kv_cache_spec_page_size`` leaves them. Block ``b`` gets the distinct
+    finite marker ``b + 1`` plus a quarter-step ramp (exact in bf16 and fp32);
+    the padding gets the negated marker, so a copy sized by the page stride
+    rather than the block contents is visible too.
+    """
+    block_elems = math.prod(block_shape)
+    page_elems = block_elems + _PAGE_PAD_ELEMS
+    ramp = (torch.arange(page_elems, device=device) % 4).to(dtype) * 0.25
+    markers = torch.arange(1, num_blocks + 1, device=device).to(dtype)[:, None]
+    pages = markers + ramp[None, :]
+    pages[:, block_elems:] *= -1
+    storage = pages.reshape(-1)
+    inner_strides: list[int] = []
+    acc = 1
+    for dim in reversed(block_shape):
+        inner_strides.append(acc)
+        acc *= dim
+    inner_strides.reverse()
+    view = torch.as_strided(
+        storage, (num_blocks, *block_shape), (page_elems, *inner_strides)
+    )
+    return _StatePool(
+        view=view, storage=storage, page_elems=page_elems, block_elems=block_elems
+    )
+
+
+def _run_align_resume_scenario(
+    *,
+    mamba_block_size: int,
+    global_block_size: int,
+    precopy_hook: Callable[..., None] | None = None,
+) -> _AlignResumeScenario:
+    """Admit one request at ``3 * mamba_block_size`` computed tokens, schedule a
+    single token, and run the real align preprocess + pre-copy over a model-free
+    state pool.
+
+    Nothing about the copy is stubbed: ``set_kv_cache_config`` /
+    ``add_request`` / ``preprocess_state`` run as in production, resolving the
+    metadata and launching both fused kernels. ``precopy_hook`` exists only for
+    the negative control, which breaks the copy on purpose.
+    """
+    device = torch.device("cuda")
+    num_computed = 3 * mamba_block_size
+    num_blocks = _MAX_NUM_REQS * _NUM_COLS + 1
+    conv_shape = (
+        (_CONV_DIM, _CONV_WIDTH)
+        if is_conv_state_dim_first()
+        else (_CONV_WIDTH, _CONV_DIM)
+    )
+
+    layer_names = [f"mamba.{i}" for i in range(_NUM_LAYERS)]
+    pools: list[_StatePool] = []
+    forward_context: dict[str, SimpleNamespace] = {}
+    for layer_name in layer_names:
+        conv = _make_state_pool(num_blocks, conv_shape, torch.bfloat16, device)
+        ssm = _make_state_pool(num_blocks, _SSM_SHAPE, torch.float32, device)
+        pools += [conv, ssm]
+        forward_context[layer_name] = SimpleNamespace(kv_cache=[conv.view, ssm.view])
+
+    for pool in pools:
+        assert torch.isfinite(pool.view).all(), "block markers must be finite"
+        blocks = pool.logical_bytes()
+        assert blocks.unique(dim=0).shape[0] == num_blocks, (
+            "block markers must be distinct, else a misdirected copy is invisible"
+        )
+
+    mamba_spec = MambaSpec(
+        shapes=(conv_shape, _SSM_SHAPE),
+        dtypes=(torch.bfloat16, torch.float32),
+        block_size=mamba_block_size,
+        mamba_type=MambaAttentionBackendEnum.MAMBA2,
+        mamba_cache_mode="align",
+    )
+    kv_cache_config = KVCacheConfig(
+        num_blocks=num_blocks,
+        kv_cache_tensors=[],
+        kv_cache_groups=[KVCacheGroupSpec(layer_names, mamba_spec)],
+    )
+
+    # Non-identity physical mapping: columns run backwards within each row, so
+    # reading the right column of the wrong row (or a column as if it were a
+    # physical id) lands on a different block.
+    block_table = torch.empty(
+        (_MAX_NUM_REQS, _NUM_COLS), dtype=torch.int32, device=device
+    )
+    for row in range(_MAX_NUM_REQS):
+        block_table[row] = torch.arange(
+            1 + row * _NUM_COLS,
+            1 + (row + 1) * _NUM_COLS,
+            dtype=torch.int32,
+            device=device,
+        ).flip(0)
+
+    state = object.__new__(MambaHybridModelState)
+    state._align_mode = True
+    state.max_num_reqs = _MAX_NUM_REQS
+    state.device = device
+    state.cache_config = SimpleNamespace(
+        block_size=global_block_size, mamba_cache_mode="align"
+    )
+    state.vllm_config = SimpleNamespace(
+        compilation_config=SimpleNamespace(static_forward_context=forward_context)
+    )
+    state.model = SimpleNamespace(
+        get_mamba_state_copy_funcs=lambda _types: _ALIGN_COPY_FUNCS
+    )
+    state.rope_state = None
+    state.prompt_embeds_state = None
+    state.recoverssm = None
+    # A stale acceptance count left by the slot's previous occupant: admission
+    # must reset it to 1 so the pre-copy runs with the neutral token bias.
+    state.num_accepted_tokens_gpu = torch.full(
+        (_MAX_NUM_REQS,), 5, dtype=torch.int32, device=device
+    )
+    state._mamba_state_idx_gpu = torch.zeros(
+        _MAX_NUM_REQS, dtype=torch.int32, device=device
+    )
+    state._mamba_src_col_gpu = torch.full(
+        (_MAX_NUM_REQS,), -1, dtype=torch.int32, device=device
+    )
+    state._mamba_src_off_gpu = torch.zeros(
+        _MAX_NUM_REQS, dtype=torch.int32, device=device
+    )
+    state._mamba_ctx = None
+    state._mamba_group_ids = []
+    state._mamba_spec = None
+    state._mamba_state_copy_funcs = None
+
+    state.set_kv_cache_config(kv_cache_config)
+    state.add_request(
+        _REQ_SLOT, SimpleNamespace(num_computed_tokens=num_computed, mm_features=[])
+    )
+    seeded_col = int(state._mamba_state_idx_gpu[_REQ_SLOT])
+
+    input_batch = SimpleNamespace(
+        num_reqs=1,
+        idx_mapping=torch.tensor([_REQ_SLOT], dtype=torch.int64, device=device),
+        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
+    )
+    num_computed_tokens = torch.zeros(_MAX_NUM_REQS, dtype=torch.int32, device=device)
+    num_computed_tokens[_REQ_SLOT] = num_computed
+
+    pre_logical = [pool.logical_bytes().clone() for pool in pools]
+    pre_padding = [pool.padding_bytes().clone() for pool in pools]
+
+    patch_precopy = (
+        nullcontext()
+        if precopy_hook is None
+        else patch.object(MambaSpecDecodeGPUContext, "run_fused_precopy", precopy_hook)
+    )
+    with patch_precopy:
+        state.preprocess_state(
+            input_batch, (block_table,), kv_cache_config, num_computed_tokens
+        )
+    torch.cuda.synchronize()
+
+    return _AlignResumeScenario(
+        state=state,
+        pools=pools,
+        pre_logical=pre_logical,
+        pre_padding=pre_padding,
+        block_table=block_table,
+        seeded_col=seeded_col,
+        # Expected columns, derived from the invariant rather than read back
+        # from the buffers under test.
+        src_col=num_computed // mamba_block_size - 1,
+        dst_col=cdiv(num_computed + 1, mamba_block_size) - 1,
+    )
+
+
+def _assert_restore_is_bit_identical(scenario: _AlignResumeScenario) -> None:
+    """Each pool must equal its pre-step image with exactly the destination
+    block's logical bytes replaced by the source block's.
+
+    Byte views, not ``rtol=atol=0``: numerical equality accepts a different bit
+    pattern for the same value and rejects an exact restore of a NaN-bearing
+    state. Page padding is excluded from the content compare and checked
+    separately, since the copy is sized by the block contents, not the page.
+    """
+    src_blk = int(scenario.block_table[0, scenario.src_col])
+    dst_blk = int(scenario.block_table[0, scenario.dst_col])
+    assert src_blk != dst_blk
+    for idx, pool in enumerate(scenario.pools):
+        pre = scenario.pre_logical[idx]
+        expected = pre.clone()
+        expected[dst_blk] = pre[src_blk]
+        got = pool.logical_bytes()
+        assert torch.equal(got, expected), (
+            f"state pool {idx}: expected block {src_blk} (column "
+            f"{scenario.src_col}) restored into block {dst_blk} (column "
+            f"{scenario.dst_col}) with every other block untouched; blocks "
+            f"differing from that image: "
+            f"{(got != expected).any(dim=1).nonzero().flatten().tolist()}"
+        )
+        assert torch.equal(pool.padding_bytes(), scenario.pre_padding[idx]), (
+            f"state pool {idx}: the pre-copy wrote into page padding"
+        )
+
+
+def _suppressed_precopy(self, *args, **kwargs) -> None:
+    """Negative control: drop the pre-copy entirely."""
+    return None
+
+
+def _misdirected_precopy(
+    self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
+) -> None:
+    """Negative control: run the real pre-copy from the global-unit column."""
+    src_col_gpu[_REQ_SLOT] = _GLOBAL_UNIT_COL
+    return _REAL_RUN_FUSED_PRECOPY(
+        self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
+    )
+
+
+@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
+@pytest.mark.parametrize(
+    ("mamba_block_size", "global_block_size"),
+    [
+        (_MAMBA_BLOCK_SIZE, _GLOBAL_BLOCK_SIZE),
+        (_MAMBA_BLOCK_SIZE, _MAMBA_BLOCK_SIZE),
+    ],
+    ids=["unequal_geometry", "equal_geometry_control"],
+)
+def test_align_resume_restores_committed_state_bitwise(
+    mamba_block_size: int, global_block_size: int
+) -> None:
+    """A request admitted at ``3 * M`` computed tokens and then given one token
+    must restore column 2 of its block table into column 3, byte for byte.
+
+    The parametrizations differ only in ``cache_config.block_size``. When it
+    equals ``MambaSpec.block_size`` both divisors agree and any seed lands on
+    the same column, so the equal-geometry case is the control that isolates
+    the unequal one.
+    """
+    scenario = _run_align_resume_scenario(
+        mamba_block_size=mamba_block_size, global_block_size=global_block_size
+    )
+
+    # Contents first: the state bytes are the invariant, and an index-only
+    # failure would stop the test before it ever checked them.
+    _assert_restore_is_bit_identical(scenario)
+
+    assert (scenario.src_col, scenario.dst_col) == (2, 3)
+    assert scenario.seeded_col == scenario.src_col
+    assert int(scenario.state._mamba_src_col_gpu[_REQ_SLOT]) == scenario.src_col
+    assert int(scenario.state._mamba_state_idx_gpu[_REQ_SLOT]) == scenario.dst_col
+    assert int(scenario.state._mamba_src_off_gpu[_REQ_SLOT]) == 0
+    assert int(scenario.state.num_accepted_tokens_gpu[_REQ_SLOT]) == 1
+
+
+@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
+@pytest.mark.parametrize(
+    "precopy_hook",
+    [_suppressed_precopy, _misdirected_precopy],
+    ids=["suppressed", "misdirected"],
+)
+def test_align_resume_restore_oracle_rejects_broken_precopy(
+    precopy_hook: Callable[..., None],
+) -> None:
+    """Negative control: the byte oracle must reject a pre-copy that is dropped
+    or aimed at the column a global-unit seed would pick."""
+    scenario = _run_align_resume_scenario(
+        mamba_block_size=_MAMBA_BLOCK_SIZE,
+        global_block_size=_GLOBAL_BLOCK_SIZE,
+        precopy_hook=precopy_hook,
+    )
+
+    with pytest.raises(AssertionError, match="blocks differing from that image"):
+        _assert_restore_is_bit_identical(scenario)
+
+
 @pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
 @pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
 def test_postprocess_state_scalar_with_int32_mapping(
```
