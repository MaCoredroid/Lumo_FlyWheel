from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


NODE_FAMILIES = (2, 3, 6, 8, 14)
QK_HEADS = 16
V_HEADS = 48
# Legacy equal-head synthetic default used by the Phase 2 microbench.
H = V_HEADS
K = 128
V = 128
BV = 16


@dataclass(frozen=True)
class Tree:
    parent: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.parent)

    def ancestors(self, node: int) -> tuple[int, ...]:
        out = []
        cur = self.parent[node]
        while cur >= 0:
            out.append(cur)
            cur = self.parent[cur]
        return tuple(reversed(out))

    def path(self, node: int) -> tuple[int, ...]:
        return (*self.ancestors(node), node)

    def is_single_spine(self) -> bool:
        return self.parent == tuple([-1, *range(0, self.n - 1)])

    def masks(self, device: torch.device, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
        visible = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
        for i in range(self.n):
            visible[i, i] = 1
            for j in self.ancestors(i):
                strict[i, j] = 1
                visible[i, j] = 1
        return strict, visible


def make_tree(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    if n == 2:
        return Tree((-1, 0))
    if n == 3:
        return Tree((-1, 0, 0))
    if n == 6:
        return Tree((-1, 0, 1, 2, 2, 1))
    if n == 8:
        return Tree((-1, 0, 1, 2, 3, 3, 2, 1))
    return Tree((-1, 0, 1, 2, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0))


def make_spine_tree(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    return Tree(tuple([-1, *range(0, n - 1)]))


def padded_nodes(n: int) -> int:
    n_pad = 1 << (n - 1).bit_length()
    if n_pad > 16:
        raise ValueError(f"FR10 tree kernel only warms padded node blocks up to 16, got {n}")
    return n_pad


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


@triton.jit
def _linear_remap_rows_kernel(
    state,
    spec_state_indices,
    accepted_paths,
    num_accepted_tokens,
    B: tl.constexpr,
    PATH_COLS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    ROW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_blk = tl.program_id(2)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    accepted_len = tl.load(num_accepted_tokens + pid_b)
    valid_path = (pid_b < B) & (pid_k < PATH_COLS) & (pid_k < SPEC_COLS) & (pid_k < accepted_len)
    src_col = tl.load(
        accepted_paths + pid_b * PATH_COLS + pid_k,
        mask=valid_path,
        other=0,
    )
    src_col = tl.maximum(0, tl.minimum(src_col, SPEC_COLS - 1))
    src_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + src_col,
        mask=valid_path,
        other=0,
    )
    dst_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + pid_k,
        mask=valid_path,
        other=0,
    )
    mask = valid_path & (offs < ROW_ELEMS)
    vals = tl.load(state + src_bank * ROW_ELEMS + offs, mask=mask)
    tl.store(state + dst_bank * ROW_ELEMS + offs, vals, mask=mask)


@triton.jit
def _linear_remap_rows_gather_kernel(
    state,
    spec_state_indices,
    accepted_paths,
    num_accepted_tokens,
    B: tl.constexpr,
    PATH_COLS: tl.constexpr,
    PATH_POW2: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    ROW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # FR13_TREE_REMAP_SEQ: race-free remap. The legacy kernel parallelizes over
    # path columns, but accepted spine paths overlap their destinations (src
    # cols [1..L] -> dst cols [0..L-1]) so the program writing column k races
    # the program reading column k as its source (nondeterministic winner and
    # corrupted state for every accepted_len >= 2). Here a single program owns
    # ALL path columns for one (batch, element-block) slice: every source row
    # slice is loaded into registers before any destination row slice is
    # stored, which makes the in-place overlapping permutation exact.
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, PATH_POW2)
    accepted_len = tl.load(num_accepted_tokens + pid_b)
    valid_path = (
        (pid_b < B) & (ks < PATH_COLS) & (ks < SPEC_COLS) & (ks < accepted_len)
    )
    src_col = tl.load(
        accepted_paths + pid_b * PATH_COLS + ks,
        mask=valid_path,
        other=0,
    )
    src_col = tl.maximum(0, tl.minimum(src_col, SPEC_COLS - 1))
    src_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + src_col,
        mask=valid_path,
        other=0,
    ).to(tl.int64)
    dst_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + ks,
        mask=valid_path,
        other=0,
    ).to(tl.int64)
    mask = valid_path[:, None] & (offs[None, :] < ROW_ELEMS)
    vals = tl.load(
        state + src_bank[:, None] * ROW_ELEMS + offs[None, :], mask=mask
    )
    tl.store(
        state + dst_bank[:, None] * ROW_ELEMS + offs[None, :], vals, mask=mask
    )


def _remap_state_rows(
    state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    *,
    num_spec_decodes: int,
    max_path_len: int,
    block: int = 256,
) -> None:
    if num_spec_decodes <= 0 or max_path_len <= 0:
        return
    if state.ndim < 2:
        raise ValueError(f"state bank must have row dimension plus payload, got {tuple(state.shape)}")
    if spec_state_indices.ndim != 2:
        raise ValueError(f"spec_state_indices must be 2D, got {tuple(spec_state_indices.shape)}")
    if accepted_paths.ndim != 2:
        raise ValueError(f"accepted_paths must be 2D, got {tuple(accepted_paths.shape)}")
    if accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            "accepted_paths batch rows must cover num_spec_decodes="
            f"{num_spec_decodes}, got {accepted_paths.shape[0]}"
        )
    if num_accepted_tokens.numel() < num_spec_decodes:
        raise ValueError(
            "num_accepted_tokens must cover num_spec_decodes="
            f"{num_spec_decodes}, got {num_accepted_tokens.numel()}"
        )
    row_elems = state.stride(0)
    path_cols = min(int(accepted_paths.shape[1]), int(max_path_len))
    spec_cols = int(spec_state_indices.shape[1])
    if path_cols <= 0 or spec_cols <= 0:
        return
    if True:  # FR13_TREE_REMAP_SEQ baked ON (gather-then-scatter remap)
        # Race-free gather-then-scatter remap (see kernel docstring). Default
        # ON: it computes the intended permutation exactly; the legacy racy
        # A/B kernel path is now dead (flag baked to constant True).
        gather_block = min(block, 128)
        path_pow2 = max(1, triton.next_power_of_2(path_cols))
        grid = (int(num_spec_decodes), triton.cdiv(row_elems, gather_block))
        _linear_remap_rows_gather_kernel[grid](
            state,
            spec_state_indices,
            accepted_paths,
            num_accepted_tokens,
            B=int(num_spec_decodes),
            PATH_COLS=path_cols,
            PATH_POW2=path_pow2,
            SPEC_COLS=spec_cols,
            ROW_ELEMS=row_elems,
            BLOCK=gather_block,
        )
        return
    grid = (int(num_spec_decodes), path_cols, triton.cdiv(row_elems, block))
    _linear_remap_rows_kernel[grid](
        state,
        spec_state_indices,
        accepted_paths,
        num_accepted_tokens,
        B=int(num_spec_decodes),
        PATH_COLS=path_cols,
        SPEC_COLS=spec_cols,
        ROW_ELEMS=row_elems,
        BLOCK=block,
    )


def launch_tree_state_linear_remap(
    *,
    ssm_state: torch.Tensor | None,
    conv_state: torch.Tensor | None,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    max_path_len: int,
) -> None:
    """Materialize accepted tree-path rows into stock linear state columns.

    The committer publishes accepted_paths as tree node columns. vLLM's GDN and
    causal-conv consumers read recurrent state by linear accepted-token position,
    so column k must contain the state for accepted_paths[b, k].
    """
    if ssm_state is not None:
        _remap_state_rows(
            ssm_state,
            spec_state_indices,
            accepted_paths,
            num_accepted_tokens,
            num_spec_decodes=num_spec_decodes,
            max_path_len=max_path_len,
        )
    if conv_state is not None:
        _remap_state_rows(
            conv_state,
            spec_state_indices,
            accepted_paths,
            num_accepted_tokens,
            num_spec_decodes=num_spec_decodes,
            max_path_len=max_path_len,
        )


def gather_committed_path_conv_prior(
    *,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    num_spec_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the prior conv window from the COMMITTED path's accepted leaf.

    FR13_CONV_COMMITTED_PATH: the legacy prior-window read gathers
    ``spec_state_indices[:, accepted_len - 1]`` AFTER
    :func:`launch_tree_state_linear_remap` has permuted the bank in place —
    linear-column arithmetic that is only spine-valid by construction and
    depends on the remap having executed exactly. This helper instead reads
    the accepted path's LEAF NODE column (``accepted_paths[b, len - 1]``,
    node-indexed pre-remap layout). The per-node tree-conv write-back stores
    each node's window as the last (width - 1) taps of
    ``(prior ++ that node's root-path tokens)``, so the accepted leaf's bank
    row IS the committed token path's window by construction — valid for
    BRANCH winners ([0,2], [0,1,4]); for spine winners it is byte-identical
    to the legacy post-remap linear read (the leaf's source row is never a
    remap destination), which is the semantics-preserving license.

    Must be called BEFORE ``launch_tree_state_linear_remap`` mutates the
    bank. ``accepted_len == 0`` (no draft accepted) reads node column 0 (the
    committed root token's window), matching legacy. All ops are tensor ops
    with no host sync, so the gather is CUDA-graph safe.

    Returns ``(read_node_cols [B,1], bank_rows [B,1], prior_state_bank)``.
    """
    b = int(num_spec_decodes)
    spec_cols = int(spec_state_indices.size(-1))
    device = spec_state_indices.device
    if accepted_paths is None or num_accepted_tokens is None:
        read_node_cols = torch.zeros((b, 1), dtype=torch.long, device=device)
    else:
        lens = num_accepted_tokens[:b].to(torch.long).view(-1, 1)
        path_cols = torch.clamp(
            lens - 1, min=0, max=int(accepted_paths.size(-1)) - 1
        )
        read_node_cols = accepted_paths[:b].to(torch.long).gather(1, path_cols)
        # len == 0 commits no draft node: the prior window is node 0's (the
        # committed root token's). The committer zero-fills path rows, but
        # enforce explicitly so a stale buffer cannot redirect the read.
        read_node_cols = torch.where(
            lens > 0, read_node_cols, torch.zeros_like(read_node_cols)
        )
        read_node_cols = torch.clamp(read_node_cols, min=0, max=spec_cols - 1)
    bank_rows = spec_state_indices[:b].to(torch.long).gather(1, read_node_cols)
    prior_state_bank = torch.index_select(
        conv_state, 0, bank_rows.reshape(-1)
    )
    return read_node_cols, bank_rows, prior_state_bank


@triton.jit
def _gdn_node_step(
    state_i,
    b_q,
    b_k,
    b_v,
    b_b,
    b_g,
    b_raw_a,
    b_raw_b,
    b_a_log,
    b_dt_bias,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
):
    # FR13_REPLAY_ROUTE shared per-node update body. This is the SINGLE
    # source of the GDN rank-1 node update used by BOTH the tree scan kernel
    # (_tree_gdn_kernel) and the accepted-path replay kernel
    # (_tree_gdn_replay_kernel). Replay bit-exactness is by re-execution of
    # the identical fp32 instruction sequence on bit-identical inputs, so the
    # two kernels MUST inline this one body with identical constexprs
    # (DIM_K/BLOCK_V via operand shapes, OUTPUT_SCALE, USE_QK_L2NORM_IN_KERNEL,
    # RAW_GATING) and identical num_warps=8. Codegen identity across the two
    # compilations (FMA contraction/scheduling per unrolled instance) is NOT
    # spec-guaranteed: it is gated by the one-time byte A/B on captured
    # payloads (GPU-gated obligation; see FR13_REPLAY_ROUTE_BUILD.md).
    b_beta = b_b
    if RAW_GATING:
        x = b_raw_a + b_dt_bias
        softplus_x = tl.where(
            x <= 20.0,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        b_g = -tl.exp(b_a_log) * softplus_x
        b_beta = tl.sigmoid(b_raw_b.to(tl.float32))
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * OUTPUT_SCALE
    state_i *= tl.exp(b_g)
    b_v -= tl.sum(state_i * b_k[None, :], axis=1)
    b_v *= b_beta
    state_i += b_v[:, None] * b_k[None, :]
    out_i = tl.sum(state_i * b_q[None, :], axis=1)
    return state_i, out_i


@triton.jit
def _tree_gdn_kernel(
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    A_log,
    dt_bias,
    h0,
    h0_indices,
    h0_num_accepted_tokens,
    invocation_counter,
    strict_mask,
    visible_mask,
    out,
    state,
    N_ACTUAL: tl.constexpr,
    N_PAD: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    H0_IS_BANK: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    STORE_NODE_STATES: tl.constexpr,
):
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_vh == 0) & (pid_v == 0),
        )

    h0_base = h0
    if H0_IS_BANK:
        h0_column = 0
        if H0_USE_ACCEPTED_COLUMN:
            h0_column = tl.maximum(
                tl.load(h0_num_accepted_tokens + H0_BATCH_INDEX).to(tl.int64) - 1,
                0,
            )
        h0_index = tl.load(h0_indices + H0_INDEX_ROW + h0_column)
        h0_base = h0 + h0_index * H0_BANK_STRIDE
    b_h0 = tl.load(
        h0_base + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Sequential rank-1 tree scan. Each row caches the post-token state for one
    # tree node, so children start from their parent's fp32 checkpoint without
    # reloading h0 or replaying ancestors from HBM.
    h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K), dtype=tl.float32)
    for i in tl.static_range(0, N_PAD):
        state_i = b_h0
        for j in tl.static_range(0, i):
            ancestor = (tl.load(strict_mask + i * N_PAD + j) != 0) & (j < N_ACTUAL)
            h_j = tl.sum(
                tl.where((offs_n == j)[:, None, None], h_cache, 0.0),
                axis=0,
            )
            state_i = tl.where(ancestor, h_j, state_i)

        b_q = tl.load(
            q + (i * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_k = tl.load(
            k + (i * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            mask=(i < N_ACTUAL) & v_mask,
            other=0.0,
        ).to(tl.float32)
        b_b = tl.load(
            beta + i * NUM_VH + pid_vh,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            g + i * NUM_VH + pid_vh,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_raw_a = b_g
        b_raw_b = b_b
        b_a_log = b_g
        b_dt_bias = b_b
        if RAW_GATING:
            b_raw_b = tl.load(
                raw_b + i * NUM_VH + pid_vh,
                mask=i < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            b_raw_a = tl.load(
                raw_a + i * NUM_VH + pid_vh,
                mask=i < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
            b_a_log = tl.load(A_log + pid_vh).to(tl.float32)

        state_i, out_i = _gdn_node_step(
            state_i,
            b_q,
            b_k,
            b_v,
            b_b,
            b_g,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
        )
        h_cache = tl.where((offs_n == i)[:, None, None], state_i[None, :, :], h_cache)
        tl.store(
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=(i < N_ACTUAL) & v_mask,
        )
        if STORE_NODE_STATES:
            # FR13_REPLAY_ROUTE: this per-node HBM export is PURE EXPORT.
            # Children resume from the h_cache registers above; nothing
            # in-kernel reads this store, so skipping it cannot perturb the
            # scan. Under the replay route the accepted path is re-executed
            # from the activation ring instead (see _tree_gdn_replay_kernel).
            tl.store(
                state + ((i * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
                state_i,
                mask=(i < N_ACTUAL) & v_mask[:, None],
            )

@triton.jit
def _tree_gdn_replay_kernel(
    k_ring,
    v_ring,
    a_ring,
    b_ring,
    A_log,
    dt_bias,
    state_bank,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    prev_lens,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    N_PAD: tl.constexpr,
    PATH_COLS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    BANK_STRIDE: tl.constexpr,
    RING_B_STRIDE_K: tl.constexpr,
    RING_N_STRIDE_K: tl.constexpr,
    RING_B_STRIDE_V: tl.constexpr,
    RING_N_STRIDE_V: tl.constexpr,
    RING_B_STRIDE_AB: tl.constexpr,
    RING_N_STRIDE_AB: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
):
    # FR13_REPLAY_ROUTE accepted-path chain replay (sibling of the scan).
    #
    # The scan no longer exports per-node states to HBM
    # (STORE_NODE_STATES=False); instead this kernel re-executes the
    # committed accepted path from the activation ring (k pre-l2norm, v,
    # raw_a, raw_b at consumed precision) on the IDENTICAL shared
    # _gdn_node_step body, in the NATIVE gate-folding basis (no rescaled-exp
    # reconstruction), and publishes the post-step states directly to the
    # bank's LINEAR columns (column t = t-th accepted token), which removes
    # the ssm half of the next-step remap under the flag.
    #
    # No h_cache: one (BLOCK_V, DIM_K) register tile per program, so the
    # replay is spill-free at any tree size.
    pid_b = tl.program_id(0)
    pid_vh = tl.program_id(1)
    pid_v = tl.program_id(2)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    # h0 = same column convention as the scan's in-kernel h0 gather:
    # column clamp(prev_accepted_len - 1, 0). prev_lens is the SCAN-TIME
    # snapshot of the accepted-lens buffer (the committer refills the live
    # buffer with the NEW lens before this kernel launches).
    prev_len = tl.load(prev_lens + pid_b).to(tl.int64)
    h0_col = tl.maximum(prev_len - 1, 0)
    h0_row = tl.load(spec_state_indices + pid_b * SPEC_COLS + h0_col).to(tl.int64)
    # Read the whole h0 tile into registers BEFORE any store: a later
    # publish in this same program may target the h0 bank row itself
    # (publish-overwrites-h0-row case).
    state = tl.load(
        state_bank
        + h0_row * BANK_STRIDE
        + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
        + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc_len = tl.load(accepted_lens + pid_b)
    # q is not stored: the q-side ops never touch state, out was already
    # emitted by the scan. A zero q keeps the shared body's signature and
    # constexprs identical; out_i is discarded.
    b_q = tl.zeros((DIM_K,), dtype=tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
    for t in tl.static_range(0, PATH_COLS + 1):
        if t == 0:
            # Root (gdn node 0) replays unconditionally: row 0 must be
            # refreshed even on a ZERO-ACCEPT event (the next h0 read clamps
            # accepted_len-1 to column 0), and the scan applies NO handoff
            # normalization to h0 before the root update.
            active = acc_len >= 0
            node = 0
        else:
            active = (t - 1) < acc_len
            node = tl.load(
                accepted_paths + pid_b * PATH_COLS + (t - 1),
                mask=active,
                other=0,
            ).to(tl.int64)
            node = tl.maximum(node, 0)
            node = tl.minimum(node, N_PAD - 1)
            # Parent-handoff normalization: the scan reads the parent state
            # through tl.sum(tl.where(offs_n == j, h_cache, 0.0), axis=0),
            # which flips -0.0 to +0.0 exactly once per edge. `+ 0.0`
            # reproduces that bit behavior; the root above gets none.
            state = state + 0.0
        b_k = tl.load(
            k_ring
            + pid_b * RING_B_STRIDE_K
            + node * RING_N_STRIDE_K
            + pid_kh * DIM_K
            + offs_k,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v_ring
            + pid_b * RING_B_STRIDE_V
            + node * RING_N_STRIDE_V
            + pid_vh * DIM_V
            + offs_v,
            mask=active & v_mask,
            other=0.0,
        ).to(tl.float32)
        b_raw_b = tl.load(
            b_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_raw_a = tl.load(
            a_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        new_state, out_i = _gdn_node_step(
            state,
            b_q,
            b_k,
            b_v,
            0.0,
            0.0,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
        )
        state = tl.where(active, new_state, state)
        if t == 0:
            dst_col = 0
        else:
            dst_col = t - 1
        dst_row = tl.load(
            spec_state_indices + pid_b * SPEC_COLS + dst_col,
            mask=active,
            other=0,
        ).to(tl.int64)
        tl.store(
            state_bank
            + dst_row * BANK_STRIDE
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state,
            mask=active & v_mask[:, None],
        )


def launch_tree_gdn_replay(
    *,
    state_bank: torch.Tensor,
    spec_state_indices: torch.Tensor,
    prev_lens: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    a_ring: torch.Tensor,
    b_ring: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_spec_decodes: int,
    output_scale: float,
    use_qk_l2norm_in_kernel: bool = True,
) -> None:
    """Launch the FR13 accepted-path replay (the durable-state publish).

    Replaces the legacy all-rows publish + next-step ssm remap under
    FR13_REPLAY_ROUTE: replays root + accepted path from the activation ring
    and writes post-step states to bank LINEAR columns (and always column 0,
    covering the zero-accept path). All inputs must be persistent
    preallocated buffers (graph-stable addresses); per-step pinned scratch is
    the gate-4 failure mode and is banned here.
    """
    if num_spec_decodes <= 0:
        return
    if state_bank.ndim != 4:
        raise ValueError(
            f"state bank must be (rows, num_vh, dim_v, dim_k), got {tuple(state_bank.shape)}"
        )
    bank_rows, num_vh, dim_v, dim_k = state_bank.shape
    if state_bank.dtype != torch.float32:
        raise ValueError(
            f"FR13 replay requires an fp32 GDN state bank, got {state_bank.dtype}"
        )
    if (
        state_bank.stride(3) != 1
        or state_bank.stride(2) != dim_k
        or state_bank.stride(1) != dim_v * dim_k
    ):
        raise ValueError("state bank payload must be row-contiguous")
    if k_ring.ndim != 4 or v_ring.ndim != 4 or a_ring.ndim != 3 or b_ring.ndim != 3:
        raise ValueError(
            "activation ring shapes must be k(B,N,KH,DK)/v(B,N,VH,DV)/a,b(B,N,VH), got "
            f"k={tuple(k_ring.shape)} v={tuple(v_ring.shape)} "
            f"a={tuple(a_ring.shape)} b={tuple(b_ring.shape)}"
        )
    ring_bs, n_pad, num_kh, ring_dim_k = k_ring.shape
    if n_pad > 16 or n_pad & (n_pad - 1):
        raise ValueError(f"ring n_pad must be a power of two <=16, got {n_pad}")
    if ring_dim_k != dim_k:
        raise ValueError(f"ring k dim {ring_dim_k} != bank dim_k {dim_k}")
    if v_ring.shape != (ring_bs, n_pad, num_vh, dim_v):
        raise ValueError(
            f"v ring shape {tuple(v_ring.shape)} != {(ring_bs, n_pad, num_vh, dim_v)}"
        )
    if a_ring.shape != (ring_bs, n_pad, num_vh) or b_ring.shape != (ring_bs, n_pad, num_vh):
        raise ValueError(
            f"a/b ring shapes must be {(ring_bs, n_pad, num_vh)}, got "
            f"{tuple(a_ring.shape)}/{tuple(b_ring.shape)}"
        )
    if not (
        k_ring.is_contiguous()
        and v_ring.is_contiguous()
        and a_ring.is_contiguous()
        and b_ring.is_contiguous()
    ):
        raise ValueError("activation rings must be contiguous")
    if num_vh % num_kh != 0:
        raise ValueError(f"value heads must be a multiple of k heads, got {num_vh}/{num_kh}")
    if ring_bs < num_spec_decodes:
        raise ValueError(
            f"ring batch {ring_bs} < num_spec_decodes {num_spec_decodes}"
        )
    if spec_state_indices.ndim != 2 or spec_state_indices.shape[0] < num_spec_decodes:
        raise ValueError(
            f"spec_state_indices must be 2D covering {num_spec_decodes} rows, "
            f"got {tuple(spec_state_indices.shape)}"
        )
    spec_cols = int(spec_state_indices.shape[1])
    if accepted_paths.ndim != 2 or accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            f"accepted_paths must be 2D covering {num_spec_decodes} rows, "
            f"got {tuple(accepted_paths.shape)}"
        )
    path_cols = int(accepted_paths.shape[1])
    if path_cols > spec_cols:
        raise ValueError(
            f"path cols {path_cols} exceed spec cols {spec_cols}; linear publish "
            "columns must be valid spec columns"
        )
    if prev_lens.numel() < num_spec_decodes or accepted_lens.numel() < num_spec_decodes:
        raise ValueError(
            "prev_lens/accepted_lens must cover num_spec_decodes="
            f"{num_spec_decodes}, got {prev_lens.numel()}/{accepted_lens.numel()}"
        )
    if A_log.numel() < num_vh or dt_bias.numel() < num_vh:
        raise ValueError(
            f"A_log/dt_bias must cover {num_vh} value heads, got "
            f"{A_log.numel()}/{dt_bias.numel()}"
        )
    grid = (int(num_spec_decodes), num_vh, triton.cdiv(dim_v, BV))
    _tree_gdn_replay_kernel[grid](
        k_ring,
        v_ring,
        a_ring,
        b_ring,
        A_log,
        dt_bias,
        state_bank,
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        prev_lens,
        NUM_KH=num_kh,
        NUM_VH=num_vh,
        DIM_K=dim_k,
        DIM_V=dim_v,
        BLOCK_V=BV,
        N_PAD=n_pad,
        PATH_COLS=path_cols,
        SPEC_COLS=spec_cols,
        BANK_STRIDE=state_bank.stride(0),
        RING_B_STRIDE_K=k_ring.stride(0),
        RING_N_STRIDE_K=k_ring.stride(1),
        RING_B_STRIDE_V=v_ring.stride(0),
        RING_N_STRIDE_V=v_ring.stride(1),
        RING_B_STRIDE_AB=a_ring.stride(0),
        RING_N_STRIDE_AB=a_ring.stride(1),
        OUTPUT_SCALE=output_scale,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        RAW_GATING=True,
        num_warps=8,
    )


@triton.jit
def _tree_gdn_replay_all_layers_kernel(
    k_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    bank_anchor,
    bank_off16,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    prev_lens,
    NUM_SPEC: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    N_PAD: tl.constexpr,
    PATH_COLS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    BANK_STRIDE: tl.constexpr,
    RING_L_STRIDE_K: tl.constexpr,
    RING_B_STRIDE_K: tl.constexpr,
    RING_N_STRIDE_K: tl.constexpr,
    RING_L_STRIDE_V: tl.constexpr,
    RING_B_STRIDE_V: tl.constexpr,
    RING_N_STRIDE_V: tl.constexpr,
    RING_L_STRIDE_AB: tl.constexpr,
    RING_B_STRIDE_AB: tl.constexpr,
    RING_N_STRIDE_AB: tl.constexpr,
    SPEC_L_STRIDE: tl.constexpr,
    PREV_L_STRIDE: tl.constexpr,
    GATE_L_STRIDE: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
):
    # FR13_EAGER_PACK (FIX-2 item 2b): all-layer batched sibling of
    # _tree_gdn_replay_kernel. ONE launch covers every GDN layer; pid0 packs
    # (layer, spec) as layer * NUM_SPEC + spec. Each program's instruction
    # sequence is source-identical to the single-layer kernel (same inlined
    # _gdn_node_step, same constexprs, same num_warps=8); only the base
    # addresses gain a per-layer offset (stacked rings / gates / snapshots,
    # plus an int64 bank OFFSET table relative to the layer-0 bank anchor
    # because each layer's ssm bank is a distinct KV-pool tensor -- see the
    # state_bank addptr note below for why offsets, not raw pointers).
    # Layers are independent: a program reads only
    # its own layer's ring/spec/prev rows and writes only its own layer's
    # bank rows (accepted_paths/lens are shared READ-ONLY), so inter-program
    # concurrency reorders nothing within any program's sequential replay
    # (playbook class 3: no overlapping writes across programs).
    # Class-10 caveat: codegen identity vs the legacy per-layer launch loop
    # is NOT assumed from source identity; it is gated by the int-view byte
    # A/B of bank bytes (never atol) before any live boot.
    pid_lb = tl.program_id(0)
    pid_l = pid_lb // NUM_SPEC
    pid_b = pid_lb % NUM_SPEC
    pid_vh = tl.program_id(1)
    pid_v = tl.program_id(2)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    # Byte-A/B fix (class 10, GPU gate 2026-06-12): the original
    # `tl.load(bank_ptrs + pid_l).to(tl.pointer_type(tl.float32))` form loses
    # ALL alignment info -- this Triton's AxisInfo does not propagate
    # divisibility through tt.int_to_ptr (verified by container microbench;
    # tl.multiple_of/shift hints on the loaded integer do NOT survive the
    # cast either). The whole kernel then compiles with a scalarized layout
    # (sizePerThread=[1,1], st.global.b32) instead of the legacy kernel's
    # vectorized layout (sizePerThread=[1,4], st.global.v4.b32), which
    # reshapes every tl.sum reduction tree in _gdn_node_step and changes
    # fp32 rounding (~1-2 ULP on published rows; measured, both arms
    # deterministic). Fix: address banks as ANCHOR + ELEMENT OFFSET through
    # tt.addptr, whose AxisInfo math is exact: bank_anchor is the layer-0
    # bank ARG (divisibility 16 from arg specialization) and bank_off16
    # holds (data_ptr - anchor_ptr) // 16 per layer, so `off * 4` fp32
    # elements is structurally 16-byte divisible. Host-side data_ptr()%16
    # fail-loud checks in build_replay_bank_pointer_table keep this exact.
    state_bank = bank_anchor + tl.load(bank_off16 + pid_l) * 4
    k_ring = k_rings + pid_l * RING_L_STRIDE_K
    v_ring = v_rings + pid_l * RING_L_STRIDE_V
    a_ring = a_rings + pid_l * RING_L_STRIDE_AB
    b_ring = b_rings + pid_l * RING_L_STRIDE_AB
    A_log = A_logs + pid_l * GATE_L_STRIDE
    dt_bias = dt_biases + pid_l * GATE_L_STRIDE
    spec_layer = spec_state_indices + pid_l * SPEC_L_STRIDE
    prev_layer = prev_lens + pid_l * PREV_L_STRIDE

    # h0 = same column convention as the scan's in-kernel h0 gather:
    # column clamp(prev_accepted_len - 1, 0). prev_lens is the SCAN-TIME
    # snapshot of the accepted-lens buffer (the committer refills the live
    # buffer with the NEW lens before this kernel launches).
    prev_len = tl.load(prev_layer + pid_b).to(tl.int64)
    h0_col = tl.maximum(prev_len - 1, 0)
    h0_row = tl.load(spec_layer + pid_b * SPEC_COLS + h0_col).to(tl.int64)
    # Read the whole h0 tile into registers BEFORE any store: a later
    # publish in this same program may target the h0 bank row itself
    # (publish-overwrites-h0-row case).
    state = tl.load(
        state_bank
        + h0_row * BANK_STRIDE
        + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
        + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc_len = tl.load(accepted_lens + pid_b)
    # q is not stored: the q-side ops never touch state, out was already
    # emitted by the scan. A zero q keeps the shared body's signature and
    # constexprs identical; out_i is discarded.
    b_q = tl.zeros((DIM_K,), dtype=tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
    for t in tl.static_range(0, PATH_COLS + 1):
        if t == 0:
            # Root (gdn node 0) replays unconditionally: row 0 must be
            # refreshed even on a ZERO-ACCEPT event (the next h0 read clamps
            # accepted_len-1 to column 0), and the scan applies NO handoff
            # normalization to h0 before the root update.
            active = acc_len >= 0
            node = 0
        else:
            active = (t - 1) < acc_len
            node = tl.load(
                accepted_paths + pid_b * PATH_COLS + (t - 1),
                mask=active,
                other=0,
            ).to(tl.int64)
            node = tl.maximum(node, 0)
            node = tl.minimum(node, N_PAD - 1)
            # Parent-handoff normalization: the scan reads the parent state
            # through tl.sum(tl.where(offs_n == j, h_cache, 0.0), axis=0),
            # which flips -0.0 to +0.0 exactly once per edge. `+ 0.0`
            # reproduces that bit behavior; the root above gets none.
            state = state + 0.0
        b_k = tl.load(
            k_ring
            + pid_b * RING_B_STRIDE_K
            + node * RING_N_STRIDE_K
            + pid_kh * DIM_K
            + offs_k,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v_ring
            + pid_b * RING_B_STRIDE_V
            + node * RING_N_STRIDE_V
            + pid_vh * DIM_V
            + offs_v,
            mask=active & v_mask,
            other=0.0,
        ).to(tl.float32)
        b_raw_b = tl.load(
            b_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_raw_a = tl.load(
            a_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        new_state, out_i = _gdn_node_step(
            state,
            b_q,
            b_k,
            b_v,
            0.0,
            0.0,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
        )
        state = tl.where(active, new_state, state)
        if t == 0:
            dst_col = 0
        else:
            dst_col = t - 1
        dst_row = tl.load(
            spec_layer + pid_b * SPEC_COLS + dst_col,
            mask=active,
            other=0,
        ).to(tl.int64)
        tl.store(
            state_bank
            + dst_row * BANK_STRIDE
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state,
            mask=active & v_mask[:, None],
        )


def build_replay_bank_pointer_table(
    banks: list[torch.Tensor],
) -> tuple[list[int], tuple[int, int, int, int], int]:
    """Validate per-layer GDN state banks for the batched all-layer replay.

    FR13_EAGER_PACK (FIX-2 item 2b): each layer's ssm bank is a distinct
    KV-pool tensor, so the batched kernel addresses them as the layer-0
    bank ANCHOR plus an int64 device OFFSET table ((ptr - ptr0) // 16,
    derived from this host pointer list; offsets-not-pointers is the
    byte-A/B alignment fix, see the kernel). This helper validates every
    bank exactly like
    launch_tree_gdn_replay (fp32, 4D, row-contiguous payload) plus the
    stacking preconditions (identical shape and stride across layers) and
    returns (host pointer list, bank shape, bank row stride). FAIL-LOUD on
    any precondition miss -- no silent per-layer fallback (playbook class 9).
    The caller must re-assert the host pointer list against the live banks'
    data_ptr() on every commit (cheap Python int compares) before launching.
    """
    if not banks:
        raise ValueError("FR13_EAGER_PACK bank table requires at least one bank")
    shape0 = tuple(banks[0].shape)
    stride0 = banks[0].stride(0)
    ptrs: list[int] = []
    for i, bank in enumerate(banks):
        if bank.ndim != 4:
            raise ValueError(
                f"bank[{i}] must be (rows, num_vh, dim_v, dim_k), got {tuple(bank.shape)}"
            )
        if bank.dtype != torch.float32:
            raise ValueError(
                f"FR13 replay requires fp32 GDN state banks, bank[{i}] is {bank.dtype}"
            )
        rows_i, num_vh_i, dim_v_i, dim_k_i = bank.shape
        if (
            bank.stride(3) != 1
            or bank.stride(2) != dim_k_i
            or bank.stride(1) != dim_v_i * dim_k_i
        ):
            raise ValueError(f"bank[{i}] payload must be row-contiguous")
        if tuple(bank.shape) != shape0 or bank.stride(0) != stride0:
            raise ValueError(
                "FR13_EAGER_PACK stacking precondition failed: bank["
                f"{i}] shape/stride {tuple(bank.shape)}/{bank.stride(0)} != "
                f"bank[0] {shape0}/{stride0}"
            )
        ptr_i = int(bank.data_ptr())
        if ptr_i % 16 != 0:
            # The batched kernel asserts tl.multiple_of(bank_ptr, 16) -- the
            # divisibility a kernel pointer ARG gets from Triton arg
            # specialization. An unaligned bank would make that hint UNSOUND
            # (silent wrong codegen), so fail loud here instead (class 9).
            raise ValueError(
                f"bank[{i}] data_ptr {ptr_i:#x} is not 16-byte aligned; the "
                "batched replay kernel's tl.multiple_of(16) hint would be "
                "unsound"
            )
        ptrs.append(ptr_i)
    return ptrs, (int(shape0[0]), int(shape0[1]), int(shape0[2]), int(shape0[3])), int(stride0)


def launch_tree_gdn_replay_all_layers(
    *,
    bank_anchor: torch.Tensor,
    bank_off16: torch.Tensor,
    bank_shape: tuple[int, int, int, int],
    bank_stride: int,
    spec_state_indices: torch.Tensor,
    prev_lens: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    k_rings: torch.Tensor,
    v_rings: torch.Tensor,
    a_rings: torch.Tensor,
    b_rings: torch.Tensor,
    A_logs: torch.Tensor,
    dt_biases: torch.Tensor,
    num_layers: int,
    num_spec_decodes: int,
    output_scale: float,
    use_qk_l2norm_in_kernel: bool = True,
) -> None:
    """Launch the FR13_EAGER_PACK batched all-layer accepted-path replay.

    Semantics-preserving sibling of launch_tree_gdn_replay: one launch
    replaces the legacy per-layer loop (48 launches + 48 flag clears). Every
    input must be a persistent preallocated stacked buffer (graph-stable
    addresses) allocated at GDN metadata-builder init; per-step scratch is
    the gate-4 failure mode and is banned here. bank_anchor is the LAYER-0
    bank tensor (pointer arg = alignment anchor; byte-A/B fix, see the
    kernel) and bank_off16 is the int64 device table of
    (bank[i].data_ptr() - bank[0].data_ptr()) // 16 derived from
    build_replay_bank_pointer_table's pointer list (the caller re-asserts
    the live banks' data_ptr() each commit, which also pins the anchor).
    """
    if num_spec_decodes <= 0:
        return
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if bank_off16.dtype != torch.int64 or bank_off16.numel() < num_layers:
        raise ValueError(
            f"bank_off16 must be int64 covering {num_layers} layers, got "
            f"{bank_off16.dtype} numel={bank_off16.numel()}"
        )
    if bank_anchor.dtype != torch.float32 or bank_anchor.data_ptr() % 16 != 0:
        raise ValueError(
            "bank_anchor must be the fp32 layer-0 GDN state bank with a "
            f"16-byte-aligned data_ptr, got {bank_anchor.dtype} "
            f"ptr={int(bank_anchor.data_ptr()):#x}"
        )
    bank_rows, num_vh, dim_v, dim_k = (int(x) for x in bank_shape)
    if k_rings.ndim != 5 or v_rings.ndim != 5 or a_rings.ndim != 4 or b_rings.ndim != 4:
        raise ValueError(
            "stacked ring shapes must be k(L,B,N,KH,DK)/v(L,B,N,VH,DV)/a,b(L,B,N,VH), got "
            f"k={tuple(k_rings.shape)} v={tuple(v_rings.shape)} "
            f"a={tuple(a_rings.shape)} b={tuple(b_rings.shape)}"
        )
    ring_layers, ring_bs, n_pad, num_kh, ring_dim_k = k_rings.shape
    if ring_layers < num_layers:
        raise ValueError(f"ring layers {ring_layers} < num_layers {num_layers}")
    if n_pad > 16 or n_pad & (n_pad - 1):
        raise ValueError(f"ring n_pad must be a power of two <=16, got {n_pad}")
    if ring_dim_k != dim_k:
        raise ValueError(f"ring k dim {ring_dim_k} != bank dim_k {dim_k}")
    if v_rings.shape != (ring_layers, ring_bs, n_pad, num_vh, dim_v):
        raise ValueError(
            f"v rings shape {tuple(v_rings.shape)} != "
            f"{(ring_layers, ring_bs, n_pad, num_vh, dim_v)}"
        )
    if a_rings.shape != (ring_layers, ring_bs, n_pad, num_vh) or b_rings.shape != (
        ring_layers,
        ring_bs,
        n_pad,
        num_vh,
    ):
        raise ValueError(
            f"a/b rings must be {(ring_layers, ring_bs, n_pad, num_vh)}, got "
            f"{tuple(a_rings.shape)}/{tuple(b_rings.shape)}"
        )
    if not (
        k_rings.is_contiguous()
        and v_rings.is_contiguous()
        and a_rings.is_contiguous()
        and b_rings.is_contiguous()
    ):
        raise ValueError("stacked activation rings must be contiguous")
    if num_vh % num_kh != 0:
        raise ValueError(f"value heads must be a multiple of k heads, got {num_vh}/{num_kh}")
    if ring_bs < num_spec_decodes:
        raise ValueError(f"ring batch {ring_bs} < num_spec_decodes {num_spec_decodes}")
    if spec_state_indices.ndim != 3 or spec_state_indices.shape[0] < num_layers or (
        spec_state_indices.shape[1] < num_spec_decodes
    ):
        raise ValueError(
            "stacked spec_state_indices must be (L, B, SPEC_COLS) covering "
            f"{num_layers}x{num_spec_decodes}, got {tuple(spec_state_indices.shape)}"
        )
    spec_cols = int(spec_state_indices.shape[2])
    if accepted_paths.ndim != 2 or accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            f"accepted_paths must be 2D covering {num_spec_decodes} rows, "
            f"got {tuple(accepted_paths.shape)}"
        )
    path_cols = int(accepted_paths.shape[1])
    if path_cols > spec_cols:
        raise ValueError(
            f"path cols {path_cols} exceed spec cols {spec_cols}; linear publish "
            "columns must be valid spec columns"
        )
    if prev_lens.ndim != 2 or prev_lens.shape[0] < num_layers or (
        prev_lens.shape[1] < num_spec_decodes
    ):
        raise ValueError(
            "stacked prev_lens must be (L, B) covering "
            f"{num_layers}x{num_spec_decodes}, got {tuple(prev_lens.shape)}"
        )
    if accepted_lens.numel() < num_spec_decodes:
        raise ValueError(
            f"accepted_lens must cover num_spec_decodes={num_spec_decodes}, "
            f"got {accepted_lens.numel()}"
        )
    if A_logs.ndim != 2 or A_logs.shape[0] < num_layers or A_logs.shape[1] < num_vh:
        raise ValueError(
            f"stacked A_logs must be (L, VH) covering {num_layers}x{num_vh}, "
            f"got {tuple(A_logs.shape)}"
        )
    if dt_biases.shape != A_logs.shape:
        raise ValueError(
            f"stacked dt_biases shape {tuple(dt_biases.shape)} != A_logs "
            f"{tuple(A_logs.shape)}"
        )
    if not (A_logs.is_contiguous() and dt_biases.is_contiguous()):
        raise ValueError("stacked A_logs/dt_biases must be contiguous")
    if not (spec_state_indices.is_contiguous() and prev_lens.is_contiguous()):
        raise ValueError("stacked spec_state_indices/prev_lens must be contiguous")
    grid = (int(num_layers) * int(num_spec_decodes), num_vh, triton.cdiv(dim_v, BV))
    _tree_gdn_replay_all_layers_kernel[grid](
        k_rings,
        v_rings,
        a_rings,
        b_rings,
        A_logs,
        dt_biases,
        bank_anchor,
        bank_off16,
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        prev_lens,
        NUM_SPEC=int(num_spec_decodes),
        NUM_KH=num_kh,
        NUM_VH=num_vh,
        DIM_K=dim_k,
        DIM_V=dim_v,
        BLOCK_V=BV,
        N_PAD=n_pad,
        PATH_COLS=path_cols,
        SPEC_COLS=spec_cols,
        BANK_STRIDE=int(bank_stride),
        RING_L_STRIDE_K=k_rings.stride(0),
        RING_B_STRIDE_K=k_rings.stride(1),
        RING_N_STRIDE_K=k_rings.stride(2),
        RING_L_STRIDE_V=v_rings.stride(0),
        RING_B_STRIDE_V=v_rings.stride(1),
        RING_N_STRIDE_V=v_rings.stride(2),
        RING_L_STRIDE_AB=a_rings.stride(0),
        RING_B_STRIDE_AB=a_rings.stride(1),
        RING_N_STRIDE_AB=a_rings.stride(2),
        SPEC_L_STRIDE=spec_state_indices.stride(0),
        PREV_L_STRIDE=prev_lens.stride(0),
        GATE_L_STRIDE=A_logs.stride(0),
        OUTPUT_SCALE=output_scale,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        RAW_GATING=True,
        num_warps=8,
    )


def launch_tree_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    tree: Tree,
    *,
    strict_mask: torch.Tensor | None = None,
    visible_mask: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    output_scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    invocation_counter: torch.Tensor | None = None,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    store_node_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the FR10 dense tree verifier.

    For CUDA graph capture, pass preallocated masks, output, and state buffers.
    The allocation path is only for probes and offline validation.
    """
    n = tree.n
    n_pad = padded_nodes(n)
    if strict_mask is None or visible_mask is None:
        strict_mask, visible_mask = tree.masks(q.device, n_pad)
    return launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        h0=h0,
        n_actual=n,
        n_pad=n_pad,
        strict_mask=strict_mask,
        visible_mask=visible_mask,
        out=out,
        state=state,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        invocation_counter=invocation_counter,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=A_log,
        dt_bias=dt_bias,
        store_node_states=store_node_states,
    )


def launch_tree_gdn_prepared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    *,
    n_actual: int,
    n_pad: int,
    strict_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    out: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    output_scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    h0_indices: torch.Tensor | None = None,
    h0_num_accepted_tokens: torch.Tensor | None = None,
    h0_is_bank: bool = False,
    h0_index_row: int = 0,
    h0_batch_index: int = 0,
    h0_use_accepted_column: bool = False,
    invocation_counter: torch.Tensor | None = None,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    store_node_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch with precomputed graph-safe tree descriptors.

    store_node_states=False (FR13_REPLAY_ROUTE) compiles the scan with the
    per-node HBM state export elided and skips the scratch state allocation;
    the durable accepted states are produced by launch_tree_gdn_replay at the
    committer instead. Returns (out, None) in that mode.
    """
    if n_actual <= 0 or n_actual > n_pad:
        raise ValueError(f"invalid tree node counts n_actual={n_actual}, n_pad={n_pad}")
    if n_pad > 16 or n_pad & (n_pad - 1):
        raise ValueError(f"n_pad must be a power of two <=16, got {n_pad}")
    if q.shape[0] < n_actual:
        raise ValueError(f"q has {q.shape[0]} rows but n_actual={n_actual}")
    if k.shape[0] < n_actual:
        raise ValueError(f"k has {k.shape[0]} rows but n_actual={n_actual}")
    if v.shape[0] < n_actual:
        raise ValueError(f"v has {v.shape[0]} rows but n_actual={n_actual}")
    if g.shape[0] < n_actual or beta.shape[0] < n_actual:
        raise ValueError(
            f"g/beta rows must cover n_actual={n_actual}, got {g.shape[0]}/{beta.shape[0]}"
        )
    if strict_mask.shape[0] < n_pad or strict_mask.shape[1] < n_pad:
        raise ValueError(f"strict_mask must cover {n_pad}x{n_pad}, got {tuple(strict_mask.shape)}")
    if visible_mask.shape[0] < n_pad or visible_mask.shape[1] < n_pad:
        raise ValueError(f"visible_mask must cover {n_pad}x{n_pad}, got {tuple(visible_mask.shape)}")
    num_kh = q.shape[1]
    num_vh = v.shape[1]
    dim_k = q.shape[2]
    dim_v = v.shape[2]
    if k.shape[1] != num_kh or k.shape[2] != dim_k:
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)}")
    if g.shape[1] != num_vh or beta.shape[1] != num_vh:
        raise ValueError(f"g/beta must use value-head count {num_vh}")
    if h0_is_bank:
        if h0.ndim != 4 or h0.shape[1:] != (num_vh, dim_v, dim_k):
            raise ValueError(
                f"h0 bank shape must be (*, {num_vh}, {dim_v}, {dim_k}), got {tuple(h0.shape)}"
            )
        if h0_indices is None:
            raise ValueError("h0_indices is required when h0_is_bank=True")
        if h0_use_accepted_column and h0_num_accepted_tokens is None:
            raise ValueError(
                "h0_num_accepted_tokens is required when h0_use_accepted_column=True"
            )
        if h0_index_row < 0 or h0_index_row >= h0_indices.numel():
            raise ValueError(
                f"h0_index_row {h0_index_row} outside h0_indices numel {h0_indices.numel()}"
            )
        if h0_use_accepted_column:
            if h0_batch_index < 0 or h0_batch_index >= h0_num_accepted_tokens.numel():
                raise ValueError(
                    "h0_batch_index "
                    f"{h0_batch_index} outside num_accepted_tokens numel "
                    f"{h0_num_accepted_tokens.numel()}"
                )
        if h0_indices.is_cuda:
            # Avoid GPU->CPU sync during capture. This range check is for eager
            # launches and debug repros; graph-captured serving relies on the
            # row-count guard above and prevalidated metadata.
            pass
        else:
            idx = int(h0_indices.reshape(-1)[h0_index_row].item())
            if idx < 0 or idx >= h0.shape[0]:
                raise ValueError(f"h0 bank index {idx} outside bank rows {h0.shape[0]}")
        h0_bank_stride = h0.stride(0)
    elif h0.shape != (num_vh, dim_v, dim_k):
        raise ValueError(f"h0 shape must be {(num_vh, dim_v, dim_k)}, got {tuple(h0.shape)}")
    else:
        h0_bank_stride = 0
    if h0_indices is None:
        h0_indices = strict_mask
    if h0_num_accepted_tokens is None:
        h0_num_accepted_tokens = strict_mask
    count_invocation = invocation_counter is not None
    if invocation_counter is None:
        invocation_counter = strict_mask
    raw_gating = (
        raw_a is not None
        or raw_b is not None
        or A_log is not None
        or dt_bias is not None
    )
    if raw_gating:
        if raw_a is None or raw_b is None or A_log is None or dt_bias is None:
            raise ValueError("raw_a, raw_b, A_log, and dt_bias must be provided together")
        if raw_a.shape[0] < n_actual or raw_a.shape[1] != num_vh:
            raise ValueError(
                f"raw_a must cover ({n_actual}, {num_vh}), got {tuple(raw_a.shape)}"
            )
        if raw_b.shape[0] < n_actual or raw_b.shape[1] != num_vh:
            raise ValueError(
                f"raw_b must cover ({n_actual}, {num_vh}), got {tuple(raw_b.shape)}"
            )
        if A_log.numel() < num_vh or dt_bias.numel() < num_vh:
            raise ValueError(
                f"A_log/dt_bias must cover {num_vh} value heads, got {A_log.numel()}/{dt_bias.numel()}"
            )
    else:
        raw_a = g
        raw_b = beta
        A_log = g
        dt_bias = beta
    if num_vh % num_kh != 0:
        raise ValueError(f"value heads must be a multiple of q/k heads, got {num_vh}/{num_kh}")
    if out is None:
        out = torch.empty((n_pad, num_vh, dim_v), device=q.device, dtype=q.dtype)
    elif out.shape[0] < n_actual or out.shape[1:] != (num_vh, dim_v):
        raise ValueError(
            f"out must be at least ({n_actual}, {num_vh}, {dim_v}), got {tuple(out.shape)}"
        )
    if not store_node_states:
        # FR13_REPLAY_ROUTE: the per-node state export is compiled out, so do
        # NOT allocate the 201.3MB/layer scratch (the capture-blocking
        # alloc). A caller passing a state buffer while disabling the store
        # is a wiring bug -- fail loud.
        if state is not None:
            raise ValueError(
                "state buffer provided but store_node_states=False; the FR13 "
                "replay route must not stage per-node states"
            )
        state = strict_mask  # dummy pointer; no store reaches it
    elif state is None:
        state = torch.empty((n_pad, num_vh, dim_v, dim_k), device=q.device, dtype=torch.float32)
    elif state.shape[0] < n_actual or state.shape[1:] != (num_vh, dim_v, dim_k):
        raise ValueError(
            "state must be at least "
            f"({n_actual}, {num_vh}, {dim_v}, {dim_k}), got {tuple(state.shape)}"
        )
    grid = (num_vh, triton.cdiv(dim_v, BV))
    _tree_gdn_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        raw_a,
        raw_b,
        A_log,
        dt_bias,
        h0,
        h0_indices,
        h0_num_accepted_tokens,
        invocation_counter,
        strict_mask,
        visible_mask,
        out,
        state,
        N_ACTUAL=n_actual,
        N_PAD=n_pad,
        NUM_KH=num_kh,
        NUM_VH=num_vh,
        DIM_K=dim_k,
        DIM_V=dim_v,
        BLOCK_V=BV,
        OUTPUT_SCALE=output_scale,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        H0_IS_BANK=h0_is_bank,
        H0_INDEX_ROW=h0_index_row,
        H0_BATCH_INDEX=h0_batch_index,
        H0_BANK_STRIDE=h0_bank_stride,
        H0_USE_ACCEPTED_COLUMN=h0_use_accepted_column,
        RAW_GATING=raw_gating,
        COUNT_INVOCATION=count_invocation,
        STORE_NODE_STATES=store_node_states,
        num_warps=8,
    )
    if not store_node_states:
        return out, None
    return out, state
