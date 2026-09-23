# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution coverage for request-slot indexing in
``MambaSpecDecodeGPUContext.compute_aligned_state_indices``.

Model-free GPU tests for the second half of the mamba spec-decode block-table
fix: once the context is bound to the SOURCE per-request-slot block tables
(``BlockTables.block_tables[i].gpu``) instead of the per-step gathered views,
the aligned-index kernel must resolve its table row through
``input_batch.idx_mapping`` (batch row -> request-state slot) while keeping
``seq_lens`` and its output in batch order.

The block ids below are unique per (group, request slot, column), so reading
any wrong table row shows up in the output.  Every case is a pure index
comparison against a CPU oracle -- no model, no cache allocation.

The tests require CUDA because the unit under test is a Triton kernel.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncsByType,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    _reinterpret_u64_as_i64 as _as_i64,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

MAMBA_BLOCK_SIZE = 16
NUM_SPECULATIVE_BLOCKS = 2
NUM_STATE_SLOTS = 1 + NUM_SPECULATIVE_BLOCKS
MAX_NUM_REQS = 8
MAX_BLOCKS_PER_REQ = 12
NUM_GROUPS = 2
CONV_WIDTH = 4
CONV_INNER_DIM = 8
TEMPORAL_DIM = 16
# Pre-fill for the persistent output buffer: any row the kernel does not write
# keeps this value, which no real block id can take.
UNWRITTEN = -7

_COPY_FUNCS: MambaStateCopyFuncsByType = {
    MambaAttentionBackendEnum.MAMBA2: (get_conv_copy_spec, get_temporal_copy_spec),
}

# Pre-#55506 the method took no mapping.  Keeping the two-line fallback lets the
# exact same expectations run against a pre-fix build as a negative control;
# upstream can drop it once the fix has landed.
_TAKES_IDX_MAPPING = (
    "idx_mapping"
    in inspect.signature(
        MambaSpecDecodeGPUContext.compute_aligned_state_indices
    ).parameters
)


def _compute(ctx, seq_lens, num_reqs, idx_mapping):
    if _TAKES_IDX_MAPPING:
        return ctx.compute_aligned_state_indices(seq_lens, num_reqs, idx_mapping)
    return ctx.compute_aligned_state_indices(seq_lens, num_reqs)


def _block_id(group: int, slot: int, col: int) -> int:
    """A unique, non-zero block id per (group, request slot, column)."""
    return 1 + group * 10_000 + slot * 100 + col


def _first_state_slot(seq_len: int) -> int:
    return max((int(seq_len) - 1) // MAMBA_BLOCK_SIZE, 0)


def _make_kv_cache_config() -> KVCacheConfig:
    spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((CONV_WIDTH, CONV_INNER_DIM), (TEMPORAL_DIM,)),
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        mamba_cache_mode="align",
        num_speculative_blocks=NUM_SPECULATIVE_BLOCKS,
    )
    groups = [
        KVCacheGroupSpec(layer_names=[f"layer_{g}"], kv_cache_spec=spec)
        for g in range(NUM_GROUPS)
    ]
    return KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups)


class _MockBuffer:
    def __init__(self, size: int, dtype: torch.dtype, device: torch.device):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.gpu = torch.zeros(size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


class _MockAttention:
    def __init__(self, kv_cache: list[torch.Tensor]):
        self.kv_cache = kv_cache


def _make_source_block_tables(device: torch.device) -> list[torch.Tensor]:
    """Persistent per-request-slot tables: row == request-state slot."""
    tables = []
    for g in range(NUM_GROUPS):
        table = torch.empty(
            (MAX_NUM_REQS, MAX_BLOCKS_PER_REQ), dtype=torch.int32, device=device
        )
        for slot in range(MAX_NUM_REQS):
            for col in range(MAX_BLOCKS_PER_REQ):
                table[slot, col] = _block_id(g, slot, col)
        tables.append(table)
    return tables


def _make_forward_context(device: torch.device) -> dict[str, _MockAttention]:
    return {
        f"layer_{g}": _MockAttention(
            [
                torch.zeros(
                    (64, CONV_WIDTH, CONV_INNER_DIM),
                    dtype=torch.float16,
                    device=device,
                ),
                torch.zeros((64, TEMPORAL_DIM), dtype=torch.float16, device=device),
            ]
        )
        for g in range(NUM_GROUPS)
    }


def _make_ctx(
    block_tables: list[torch.Tensor], device: torch.device
) -> MambaSpecDecodeGPUContext:
    """Build a context through the real ``create`` +
    ``initialize_from_forward_context`` path, so the block-table pointer capture
    under test is the production one."""
    kv_cache_config = _make_kv_cache_config()
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=MAX_NUM_REQS,
        kv_cache_config=kv_cache_config,
        copy_funcs=_COPY_FUNCS,
        device=device,
        make_buffer=lambda n, dtype: _MockBuffer(n, dtype, device),
    )
    forward_context = _make_forward_context(device)
    ctx.initialize_from_forward_context(
        kv_cache_config, forward_context, _COPY_FUNCS, block_tables
    )
    assert ctx.aligned_state_indices is not None
    assert ctx.aligned_state_indices.shape == (
        NUM_GROUPS,
        MAX_NUM_REQS,
        NUM_STATE_SLOTS,
    )
    return ctx


def _oracle(
    seq_lens: list[int], idx_mapping: list[int], num_reqs: int
) -> torch.Tensor:
    """expected[g, batch_row, s] == source_table[g][slot, first + s]."""
    out = torch.full(
        (NUM_GROUPS, num_reqs, NUM_STATE_SLOTS), UNWRITTEN, dtype=torch.int32
    )
    for g in range(NUM_GROUPS):
        for row in range(num_reqs):
            slot = idx_mapping[row]
            if slot < 0:
                continue  # sentinel row: kernel masks it off
            first = _first_state_slot(seq_lens[row])
            for s in range(NUM_STATE_SLOTS):
                out[g, row, s] = _block_id(g, slot, first + s)
    return out


def _reset_output(ctx: MambaSpecDecodeGPUContext) -> None:
    assert ctx.aligned_state_indices is not None
    ctx.aligned_state_indices.fill_(UNWRITTEN)


def _run(
    ctx: MambaSpecDecodeGPUContext,
    seq_lens: list[int],
    idx_mapping: list[int],
    num_reqs: int,
    device: torch.device,
) -> torch.Tensor:
    seq_lens_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device)
    seq_lens_gpu[: len(seq_lens)] = torch.tensor(seq_lens, dtype=torch.int32)
    mapping_gpu = torch.tensor(idx_mapping, dtype=torch.int64, device=device)
    _reset_output(ctx)
    out = _compute(ctx, seq_lens_gpu[:num_reqs], num_reqs, mapping_gpu)
    return out.clone()


# ---------------------------------------------------------------------------
# C1: control -- batch order == request-slot order.
# ---------------------------------------------------------------------------
def test_identity_mapping_matches_unmapped():
    """With ``idx_mapping == arange(num_reqs)`` the mapped launch must agree
    with the unmapped one.  Passes both before and after the fix; it is the
    control that shows the permuted case below is not just "any change"."""
    device = torch.device("cuda")
    tables = _make_source_block_tables(device)
    ctx = _make_ctx(tables, device)
    seq_lens = [17, 33, 5, 64, 100]
    num_reqs = len(seq_lens)
    identity = list(range(num_reqs))

    mapped = _run(ctx, seq_lens, identity, num_reqs, device)

    seq_lens_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device)
    seq_lens_gpu[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
    _reset_output(ctx)
    unmapped = ctx.compute_aligned_state_indices(
        seq_lens_gpu[:num_reqs], num_reqs
    ).clone()

    assert torch.equal(mapped, unmapped)
    assert torch.equal(mapped.cpu(), _oracle(seq_lens, identity, num_reqs))


# ---------------------------------------------------------------------------
# C2: the actual regression -- permuted batch order.
# ---------------------------------------------------------------------------
def test_permuted_mapping_reads_source_slot_rows():
    """Batch row order != request-slot order.

    The context holds the SOURCE per-request-slot tables, so batch row ``r``
    must read table row ``idx_mapping[r]``.  Indexing by ``r`` (the pre-fix
    expression) reads another request's block ids.
    """
    device = torch.device("cuda")
    tables = _make_source_block_tables(device)
    ctx = _make_ctx(tables, device)
    # batch row: 0    1    2    3    4
    # req slot:  5    0    7    2    1
    idx_mapping = [5, 0, 7, 2, 1]
    seq_lens = [17, 1, 48, 33, 80]
    num_reqs = len(idx_mapping)

    out = _run(ctx, seq_lens, idx_mapping, num_reqs, device)

    assert torch.equal(out.cpu(), _oracle(seq_lens, idx_mapping, num_reqs))


def test_permuted_mapping_differs_from_batch_row_indexing():
    """Guard on the test itself: for this input the two indexings really do
    disagree, so ``test_permuted_mapping_reads_source_slot_rows`` discriminates."""
    idx_mapping = [5, 0, 7, 2, 1]
    seq_lens = [17, 1, 48, 33, 80]
    num_reqs = len(idx_mapping)
    by_slot = _oracle(seq_lens, idx_mapping, num_reqs)
    by_batch_row = _oracle(seq_lens, list(range(num_reqs)), num_reqs)
    assert not torch.equal(by_slot, by_batch_row)


# ---------------------------------------------------------------------------
# C3: the -1 sentinel.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _TAKES_IDX_MAPPING, reason="requires the mapped signature")
def test_sentinel_rows_are_masked_off():
    """A ``-1`` mapping entry must not form a negative table row.  The kernel
    masks such rows off entirely, so the persistent output buffer KEEPS ITS
    PREVIOUS CONTENTS for them -- asserted here so the behaviour is pinned."""
    device = torch.device("cuda")
    tables = _make_source_block_tables(device)
    ctx = _make_ctx(tables, device)
    idx_mapping = [3, -1, 6, -1, 0]
    seq_lens = [17, 999, 48, 999, 33]
    num_reqs = len(idx_mapping)

    out = _run(ctx, seq_lens, idx_mapping, num_reqs, device)

    expected = _oracle(seq_lens, idx_mapping, num_reqs)
    assert torch.equal(out.cpu(), expected)
    # Rows 1 and 3 were left at the pre-fill value, i.e. never written.
    assert (out[:, 1, :] == UNWRITTEN).all()
    assert (out[:, 3, :] == UNWRITTEN).all()


# ---------------------------------------------------------------------------
# C4 / C5: CUDA graph capture and replay.
# ---------------------------------------------------------------------------
def _capture(ctx, seq_lens_gpu, num_reqs, mapping_gpu):
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            _compute(ctx, seq_lens_gpu[:num_reqs], num_reqs, mapping_gpu)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _compute(ctx, seq_lens_gpu[:num_reqs], num_reqs, mapping_gpu)
    return graph


def test_cuda_graph_replay_matches_eager():
    """Capture the launch, then replay it after updating the mapping and the
    sequence lengths IN PLACE (the block-table contents are left unchanged).
    The replay must be bytewise identical to an eager launch over the same
    buffers.

    This shows the kernel's addressing depends only on its pointer arguments
    and on the runtime ``num_requests`` scalar.  It does NOT show that the
    production path is capture-safe: in the V2 runner this kernel is launched
    from ``prepare_attn``, outside the captured region.
    """
    device = torch.device("cuda")
    tables = _make_source_block_tables(device)
    ctx = _make_ctx(tables, device)

    num_reqs = 5
    seq_lens = [17, 1, 48, 33, 80]
    idx_mapping = [5, 0, 7, 2, 1]
    seq_lens_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device)
    seq_lens_gpu[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
    mapping_gpu = torch.tensor(idx_mapping, dtype=torch.int64, device=device)

    graph = _capture(ctx, seq_lens_gpu, num_reqs, mapping_gpu)

    # A different step: new permutation, new lengths; tables unchanged.
    new_mapping = [1, 6, 2, 7, 4]
    new_seq_lens = [64, 20, 96, 5, 49]
    mapping_gpu.copy_(torch.tensor(new_mapping, dtype=torch.int64))
    seq_lens_gpu[:num_reqs] = torch.tensor(new_seq_lens, dtype=torch.int32)

    _reset_output(ctx)
    graph.replay()
    torch.cuda.synchronize()
    replayed = ctx.aligned_state_indices[:, :num_reqs].clone()

    _reset_output(ctx)
    eager = _compute(ctx, seq_lens_gpu[:num_reqs], num_reqs, mapping_gpu).clone()

    assert torch.equal(replayed, eager)
    assert torch.equal(replayed.cpu(), _oracle(new_seq_lens, new_mapping, num_reqs))


@pytest.mark.skipif(not _TAKES_IDX_MAPPING, reason="requires the mapped signature")
def test_cuda_graph_replay_ignores_a_reallocated_mapping():
    """A captured launch bakes the mapping tensor's ADDRESS.

    ``GPUModelRunner.prepare_inputs`` builds ``idx_mapping`` with
    ``async_tensor_h2d`` -- a fresh allocation every step -- so this kernel must
    keep being launched eagerly (as it is today, from ``prepare_attn``).  This
    test pins that precondition rather than any behaviour of the fix.
    """
    device = torch.device("cuda")
    tables = _make_source_block_tables(device)
    ctx = _make_ctx(tables, device)

    num_reqs = 4
    seq_lens = [17, 33, 48, 64]
    captured_mapping = [4, 1, 6, 3]
    seq_lens_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device)
    seq_lens_gpu[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
    mapping_gpu = torch.tensor(captured_mapping, dtype=torch.int64, device=device)

    graph = _capture(ctx, seq_lens_gpu, num_reqs, mapping_gpu)

    # Keep the captured tensor alive so the new one lands elsewhere.
    fresh_mapping = torch.tensor([0, 7, 2, 5], dtype=torch.int64, device=device)
    if fresh_mapping.data_ptr() == mapping_gpu.data_ptr():
        pytest.skip("allocator reused the address; cannot demonstrate")

    _reset_output(ctx)
    graph.replay()
    torch.cuda.synchronize()
    replayed = ctx.aligned_state_indices[:, :num_reqs].clone()

    assert torch.equal(replayed.cpu(), _oracle(seq_lens, captured_mapping, num_reqs))
    assert not torch.equal(
        replayed.cpu(), _oracle(seq_lens, fresh_mapping.tolist(), num_reqs)
    )


# ---------------------------------------------------------------------------
# C6: the call the runner actually makes under FULL cudagraphs.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _TAKES_IDX_MAPPING, reason="requires the mapped signature")
def test_padded_rows_do_not_read_past_idx_mapping():
    """``MambaHybridModelState.prepare_attn`` passes
    ``num_reqs = input_batch.num_reqs_after_padding`` under
    ``CUDAGraphMode.FULL`` but ``idx_mapping`` has only ``num_reqs`` entries
    (``GPUModelRunner.prepare_inputs``).  The kernel's ``idx_mapping`` load is
    masked with ``rows < num_requests``, i.e. with the PADDED count, so the
    cudagraph-padding rows read past the end of the tensor.

    Here the mapping tensor is a prefix view of a longer allocation whose tail
    holds a known slot id, so the out-of-bounds read is observable instead of
    merely undefined.  Padded rows must not depend on that tail.
    """
    device = torch.device("cuda")
    tables = _make_source_block_tables(device)
    ctx = _make_ctx(tables, device)

    num_reqs = 3
    num_reqs_padded = 6
    real_mapping = [2, 0, 5]
    poison_slot = 7
    backing = torch.full(
        (num_reqs_padded,), poison_slot, dtype=torch.int64, device=device
    )
    backing[:num_reqs] = torch.tensor(real_mapping, dtype=torch.int64)
    idx_mapping = backing[:num_reqs]
    assert idx_mapping.numel() == num_reqs

    seq_lens_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device)
    seq_lens_gpu[:num_reqs] = torch.tensor([17, 33, 48], dtype=torch.int32)

    _reset_output(ctx)
    out = _compute(
        ctx, seq_lens_gpu[:num_reqs_padded], num_reqs_padded, idx_mapping
    ).clone()

    padded = out[:, num_reqs:num_reqs_padded, :].cpu()
    leaked = torch.tensor(
        [
            [_block_id(g, poison_slot, s) for s in range(NUM_STATE_SLOTS)]
            for g in range(NUM_GROUPS)
        ],
        dtype=torch.int32,
    )
    assert not (padded == leaked[:, None, :]).all(), (
        "cudagraph-padding rows took their table row from memory past the end "
        f"of idx_mapping (slot {poison_slot} leaked into rows "
        f"{num_reqs}..{num_reqs_padded})"
    )


# ---------------------------------------------------------------------------
# C7: which block tables the context binds to.
# ---------------------------------------------------------------------------
def test_context_binding_is_first_writer_wins():
    """``initialize_from_forward_context`` is idempotent by design, so the
    FIRST caller decides which tables every later launch reads.

    ``MambaHybridModelState._ensure_align_ctx`` is reached from two places with
    two different table sets: ``preprocess_state`` passes the source
    per-request-slot tables, ``prepare_attn`` passes the per-step gathered
    views.  On real batches ``preprocess_state`` runs first, but dummy runs
    (cudagraph capture, DP padding) skip it.
    """
    device = torch.device("cuda")
    source_tables = _make_source_block_tables(device)
    gathered_tables = [torch.zeros_like(t) for t in source_tables]

    ctx = _make_ctx(gathered_tables, device)  # first writer: the gathered views
    first_ptrs = ctx.block_table_ptrs.clone()

    kv_cache_config = _make_kv_cache_config()
    forward_context = _make_forward_context(device)
    ctx.initialize_from_forward_context(
        kv_cache_config, forward_context, _COPY_FUNCS, source_tables
    )

    assert torch.equal(ctx.block_table_ptrs, first_ptrs), (
        "second initialize_from_forward_context rebound the tables"
    )

    # And the launch reads the first-bound tables, not the source ones.
    idx_mapping = [5, 0, 7, 2, 1]
    seq_lens = [17, 1, 48, 33, 80]
    num_reqs = len(idx_mapping)
    out = _run(ctx, seq_lens, idx_mapping, num_reqs, device).cpu()
    assert (out == 0).all(), "expected the zeroed first-bound tables"
    assert not torch.equal(out, _oracle(seq_lens, idx_mapping, num_reqs))


def test_ensure_align_ctx_keeps_the_capture_time_tables():
    """Drive ``MambaHybridModelState._ensure_align_ctx`` in the order the engine
    uses it when the model has an aligned-index builder (Kimi K3 / KDA).

    Order in the engine:

    1. ``capture_model`` -> ``cudagraph_utils`` builds capture metadata with
       ``block_tables.get_dummy_block_tables(num_reqs)`` -- a slice of the
       GATHERED ``BlockTables.input_block_tables`` -- and calls
       ``model_state.prepare_attn`` (cudagraph_utils.py:785, :824).  For an
       align-mode model with an aligned-index builder that reaches
       ``_ensure_align_ctx`` and binds the context.
    2. Only later, on the first real batch, does ``preprocess_state`` offer the
       SOURCE per-request-slot tables
       (``tuple(bt.gpu for bt in self.block_tables.block_tables)``,
       model_runner.py:1723) -- but ``initialize_from_forward_context`` is
       idempotent, so that binding is dropped.

    Dummy runs skip ``preprocess_state`` entirely (it sits under
    ``if not dummy_run:``), so nothing restores the intended binding.
    """
    device = torch.device("cuda")
    source_tables = _make_source_block_tables(device)
    gathered_tables = [torch.zeros_like(t) for t in source_tables]

    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state._mamba_ctx = None
    state._mamba_state_copy_funcs = _COPY_FUNCS
    state.max_num_reqs = MAX_NUM_REQS
    state.device = device
    state.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context=_make_forward_context(device)
        )
    )

    kv_cache_config = _make_kv_cache_config()
    group_ids = list(range(NUM_GROUPS))

    # 1. capture time
    ctx = state._ensure_align_ctx(kv_cache_config, group_ids, tuple(gathered_tables))
    capture_ptrs = ctx.block_table_ptrs.clone()
    # 2. first real batch
    ctx2 = state._ensure_align_ctx(kv_cache_config, group_ids, tuple(source_tables))

    assert ctx2 is ctx
    assert torch.equal(ctx.block_table_ptrs, capture_ptrs)
    assert ctx.block_table_ptrs.tolist() != [
        _as_i64(t.data_ptr()) for t in source_tables
    ], "context bound the source per-request-slot tables"
