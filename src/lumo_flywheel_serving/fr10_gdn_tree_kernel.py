from __future__ import annotations

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
    FLA_BF16_BOUNDARIES: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
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

    # FR12 losslessness gate: replay each node's ancestor path with the same
    # recurrent update order used by vLLM decode. This keeps spine results
    # independent of sibling rows and avoids the triangular-solve op-order gap.
    for i in tl.static_range(0, N_PAD):
        q_i = tl.load(
            q + (i * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            q_i = q_i * tl.rsqrt(tl.sum(q_i * q_i) + 1e-6)
        q_i *= OUTPUT_SCALE
        state_i = b_h0
        for j in tl.range(0, i + 1):
            vis = tl.load(visible_mask + i * N_PAD + j) != 0
            k_j = tl.load(
                k + (j * NUM_KH + pid_kh) * DIM_K + offs_k,
                mask=j < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                k_j = k_j * tl.rsqrt(tl.sum(k_j * k_j) + 1e-6)
            v_j = tl.load(
                v + (j * NUM_VH + pid_vh) * DIM_V + offs_v,
                mask=(j < N_ACTUAL) & v_mask,
                other=0.0,
            ).to(tl.float32)
            g_j = tl.load(
                g + j * NUM_VH + pid_vh,
                mask=j < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            beta_j = tl.load(
                beta + j * NUM_VH + pid_vh,
                mask=j < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            if RAW_GATING:
                x_j = tl.load(
                    raw_a + j * NUM_VH + pid_vh,
                    mask=j < N_ACTUAL,
                    other=0.0,
                ).to(tl.float32) + tl.load(dt_bias + pid_vh).to(tl.float32)
                softplus_j = tl.where(
                    x_j <= 20.0,
                    tl.log(1.0 + tl.exp(x_j)),
                    x_j,
                )
                g_j = -tl.exp(tl.load(A_log + pid_vh).to(tl.float32)) * softplus_j
                beta_j = tl.sigmoid(
                    tl.load(
                        raw_b + j * NUM_VH + pid_vh,
                        mask=j < N_ACTUAL,
                        other=0.0,
                    ).to(tl.float32)
                )
            decayed = state_i * tl.exp(g_j)
            delta_v = v_j - tl.sum(decayed * k_j[None, :], axis=1)
            delta_v *= beta_j
            updated = decayed + delta_v[:, None] * k_j[None, :]
            state_i = tl.where(vis, updated, state_i)
        state_store_i = state_i
        if i > 0:
            state_store_i = b_h0
            for j in tl.range(1, i + 1):
                vis = tl.load(visible_mask + i * N_PAD + j) != 0
                k_j = tl.load(
                    k + (j * NUM_KH + pid_kh) * DIM_K + offs_k,
                    mask=j < N_ACTUAL,
                    other=0.0,
                ).to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    k_j = k_j * tl.rsqrt(tl.sum(k_j * k_j) + 1e-6)
                v_j = tl.load(
                    v + (j * NUM_VH + pid_vh) * DIM_V + offs_v,
                    mask=(j < N_ACTUAL) & v_mask,
                    other=0.0,
                ).to(tl.float32)
                g_j = tl.load(
                    g + j * NUM_VH + pid_vh,
                    mask=j < N_ACTUAL,
                    other=0.0,
                ).to(tl.float32)
                beta_j = tl.load(
                    beta + j * NUM_VH + pid_vh,
                    mask=j < N_ACTUAL,
                    other=0.0,
                ).to(tl.float32)
                if RAW_GATING:
                    x_j = tl.load(
                        raw_a + j * NUM_VH + pid_vh,
                        mask=j < N_ACTUAL,
                        other=0.0,
                    ).to(tl.float32) + tl.load(dt_bias + pid_vh).to(tl.float32)
                    softplus_j = tl.where(
                        x_j <= 20.0,
                        tl.log(1.0 + tl.exp(x_j)),
                        x_j,
                    )
                    g_j = -tl.exp(tl.load(A_log + pid_vh).to(tl.float32)) * softplus_j
                    beta_j = tl.sigmoid(
                        tl.load(
                            raw_b + j * NUM_VH + pid_vh,
                            mask=j < N_ACTUAL,
                            other=0.0,
                        ).to(tl.float32)
                    )
                decayed = state_store_i * tl.exp(g_j)
                delta_v = v_j - tl.sum(decayed * k_j[None, :], axis=1)
                delta_v *= beta_j
                updated = decayed + delta_v[:, None] * k_j[None, :]
                state_store_i = tl.where(vis, updated, state_store_i)
        out_i = tl.sum(state_i * q_i[None, :], axis=1)
        tl.store(
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=v_mask & (i < N_ACTUAL),
        )
        tl.store(
            state + ((i * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
            state_store_i,
            mask=v_mask[:, None] & (i < N_ACTUAL),
        )


@triton.jit
def _tree_gdn_wy_kernel(
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
    FLA_BF16_BOUNDARIES: tl.constexpr,
    FLA_BF16_OUTPUT_SPLIT: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
):
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    n_mask = offs_n < N_ACTUAL
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

    b_g = tl.load(g + offs_n * NUM_VH + pid_vh, mask=n_mask, other=0.0).to(tl.float32)
    b_beta = tl.load(
        beta + offs_n * NUM_VH + pid_vh,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    if RAW_GATING:
        b_a = tl.load(
            raw_a + offs_n * NUM_VH + pid_vh,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        b_b = tl.load(
            raw_b + offs_n * NUM_VH + pid_vh,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        x = b_a + tl.load(dt_bias + pid_vh).to(tl.float32)
        softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        b_g = -tl.exp(tl.load(A_log + pid_vh).to(tl.float32)) * softplus
        b_beta = tl.sigmoid(b_b)
        b_g = tl.where(n_mask, b_g, 0.0)
        b_beta = tl.where(n_mask, b_beta, 0.0)

    b_q = tl.load(
        q + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    b_k = tl.load(
        k + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=1)[:, None] + 1e-6)
        b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=1)[:, None] + 1e-6)
        b_k_state = b_k
        if FLA_BF16_BOUNDARIES:
            b_q = b_q.to(tl.bfloat16).to(tl.float32)
            b_k = b_k.to(tl.bfloat16).to(tl.float32)
    else:
        b_k_state = b_k
    b_v = tl.load(
        v + (offs_n[:, None] * NUM_VH + pid_vh) * DIM_V + offs_v[None, :],
        mask=n_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    m_strict = tl.load(strict_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    m_visible = tl.load(visible_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    m_strict = m_strict & n_mask[:, None] & n_mask[None, :]
    m_visible = m_visible & n_mask[:, None] & n_mask[None, :]

    cum_g = tl.sum(tl.where(m_visible, b_g[None, :], 0.0), axis=1)
    decay = tl.exp(cum_g[:, None] - cum_g[None, :])
    if FLA_BF16_BOUNDARIES:
        b_kb = (b_k * b_beta[:, None]).to(tl.bfloat16)
        kk = tl.dot(b_kb, tl.trans(b_k).to(tl.bfloat16))
        system = tl.where(m_strict, kk * decay, 0.0)
        kk_state = tl.dot(b_k_state, tl.trans(b_k_state), input_precision="ieee")
        system_state = tl.where(
            m_strict,
            kk_state * b_beta[:, None] * decay,
            0.0,
        )
    else:
        kk = tl.dot(b_k, tl.trans(b_k), input_precision="ieee")
        system = tl.where(m_strict, kk * b_beta[:, None] * decay, 0.0)
        system_state = system

    solved_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    solved_k = tl.zeros((N_PAD, DIM_K), dtype=tl.float32)
    solved_state_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    solved_state_k = tl.zeros((N_PAD, DIM_K), dtype=tl.float32)
    trans_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    trans_state_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)

    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        coeff = tl.sum(tl.where(row_i[:, None], system, 0.0), axis=0)
        coeff_state = tl.sum(tl.where(row_i[:, None], system_state, 0.0), axis=0)
        beta_i = tl.sum(tl.where(row_i, b_beta, 0.0), axis=0)
        cumg_i = tl.sum(tl.where(row_i, cum_g, 0.0), axis=0)
        v_i = tl.sum(tl.where(row_i[:, None], b_v, 0.0), axis=0)
        k_i = tl.sum(tl.where(row_i[:, None], b_k, 0.0), axis=0)
        k_state_i = tl.sum(tl.where(row_i[:, None], b_k_state, 0.0), axis=0)
        y_i = beta_i * v_i
        sk_i = beta_i * k_i * tl.exp(cumg_i)
        y_state_i = y_i
        sk_state_i = beta_i * k_state_i * tl.exp(cumg_i)
        if FLA_BF16_BOUNDARIES:
            y_i = y_i.to(tl.bfloat16).to(tl.float32)
            sk_i = sk_i.to(tl.bfloat16).to(tl.float32)
        for j in tl.static_range(0, i):
            row_j = offs_n == j
            coeff_j = tl.sum(tl.where(row_j, coeff, 0.0), axis=0)
            coeff_state_j = tl.sum(tl.where(row_j, coeff_state, 0.0), axis=0)
            solved_v_j = tl.sum(tl.where(row_j[:, None], solved_v, 0.0), axis=0)
            solved_k_j = tl.sum(tl.where(row_j[:, None], solved_k, 0.0), axis=0)
            solved_state_v_j = tl.sum(
                tl.where(row_j[:, None], solved_state_v, 0.0),
                axis=0,
            )
            solved_state_k_j = tl.sum(
                tl.where(row_j[:, None], solved_state_k, 0.0),
                axis=0,
            )
            y_i -= coeff_j * solved_v_j
            sk_i -= coeff_j * solved_k_j
            y_state_i -= coeff_state_j * solved_state_v_j
            sk_state_i -= coeff_state_j * solved_state_k_j
        y_store_i = y_i
        sk_store_i = sk_i
        if FLA_BF16_BOUNDARIES:
            y_store_i = y_store_i.to(tl.bfloat16).to(tl.float32)
            sk_store_i = sk_store_i.to(tl.bfloat16).to(tl.float32)
        solved_v = tl.where((offs_n == i)[:, None], y_store_i[None, :], solved_v)
        solved_k = tl.where((offs_n == i)[:, None], sk_store_i[None, :], solved_k)
        solved_state_v = tl.where(
            (offs_n == i)[:, None],
            y_state_i[None, :],
            solved_state_v,
        )
        solved_state_k = tl.where(
            (offs_n == i)[:, None],
            sk_state_i[None, :],
            solved_state_k,
        )
        incoming_i = tl.sum(b_h0 * sk_i[None, :], axis=1)
        tv_i = y_i - incoming_i
        tv_state_i = y_state_i - tl.sum(b_h0 * sk_state_i[None, :], axis=1)
        if FLA_BF16_BOUNDARIES:
            tv_i = tv_i.to(tl.bfloat16).to(tl.float32)
        trans_v = tl.where((offs_n == i)[:, None], tv_i[None, :], trans_v)
        trans_state_v = tl.where(
            (offs_n == i)[:, None],
            tv_state_i[None, :],
            trans_state_v,
        )

    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        cumg_i = tl.sum(tl.where(row_i, cum_g, 0.0), axis=0)
        q_i = tl.sum(tl.where(row_i[:, None], b_q, 0.0), axis=0) * OUTPUT_SCALE
        state_i = b_h0 * tl.exp(cumg_i)
        state_inter_i = state_i
        out_intra_i = tl.zeros((BLOCK_V,), dtype=tl.float32)
        state_store_i = tl.zeros((BLOCK_V, DIM_K), dtype=tl.float32)
        state_store_i += b_h0
        for j in tl.static_range(0, N_PAD):
            vis = tl.load(visible_mask + i * N_PAD + j) != 0
            row_j = offs_n == j
            trans_j = tl.sum(tl.where(row_j[:, None], trans_v, 0.0), axis=0)
            k_j = tl.sum(tl.where(row_j[:, None], b_k, 0.0), axis=0)
            cumg_j = tl.sum(tl.where(row_j, cum_g, 0.0), axis=0)
            decay_ij = tl.exp(cumg_i - cumg_j)
            state_update_ij = trans_j[:, None] * k_j[None, :] * decay_ij
            k_store_j = tl.load(
                k + (j * NUM_KH + pid_kh) * DIM_K + offs_k,
                mask=j < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                k_store_j *= tl.rsqrt(tl.sum(k_store_j * k_store_j) + 1e-6)
            v_store_j = tl.load(
                v + (j * NUM_VH + pid_vh) * DIM_V + offs_v,
                mask=(j < N_ACTUAL) & v_mask,
                other=0.0,
            ).to(tl.float32)
            g_store_j = tl.load(
                g + j * NUM_VH + pid_vh,
                mask=j < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            beta_store_j = tl.load(
                beta + j * NUM_VH + pid_vh,
                mask=j < N_ACTUAL,
                other=0.0,
            ).to(tl.float32)
            if RAW_GATING:
                x_store_j = tl.load(
                    raw_a + j * NUM_VH + pid_vh,
                    mask=j < N_ACTUAL,
                    other=0.0,
                ).to(tl.float32) + tl.load(dt_bias + pid_vh).to(tl.float32)
                softplus_store_j = tl.where(
                    x_store_j <= 20.0,
                    (1.0 / 1.0) * tl.log(1.0 + tl.exp(1.0 * x_store_j)),
                    x_store_j,
                )
                g_store_j = -tl.exp(tl.load(A_log + pid_vh).to(tl.float32)) * softplus_store_j
                beta_store_j = tl.sigmoid(
                    tl.load(
                        raw_b + j * NUM_VH + pid_vh,
                        mask=j < N_ACTUAL,
                        other=0.0,
                    ).to(tl.float32)
                )
            state_store_update_j = state_store_i
            state_store_update_j *= tl.exp(g_store_j)
            delta_store_j = v_store_j
            delta_store_j -= tl.sum(state_store_update_j * k_store_j[None, :], axis=1)
            delta_store_j *= beta_store_j
            state_store_update_j += delta_store_j[:, None] * k_store_j[None, :]
            if FLA_BF16_OUTPUT_SPLIT:
                a_ij = tl.sum(q_i * k_j) * decay_ij
                a_ij = tl.where(
                    vis & (i < N_ACTUAL) & (j < N_ACTUAL),
                    a_ij,
                    0.0,
                )
                a_ij = a_ij.to(tl.bfloat16).to(tl.float32)
                out_intra_i += a_ij * trans_j
            state_i += tl.where(
                vis & (i < N_ACTUAL) & (j < N_ACTUAL),
                state_update_ij,
                0.0,
            )
            state_store_i = tl.where(
                vis & (i < N_ACTUAL) & (j < N_ACTUAL),
                state_store_update_j,
                state_store_i,
            )
        if FLA_BF16_OUTPUT_SPLIT:
            out_inter_i = tl.sum(state_inter_i * q_i[None, :], axis=1)
            out_i = out_inter_i + out_intra_i
        else:
            out_i = tl.sum(state_i * q_i[None, :], axis=1)
        tl.store(
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=v_mask & (i < N_ACTUAL),
        )
        tl.store(
            state + ((i * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
            state_store_i,
            mask=v_mask[:, None] & (i < N_ACTUAL),
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
    fla_bf16_boundaries: bool = False,
    fla_bf16_output_split: bool = False,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_wy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        fla_bf16_boundaries=fla_bf16_boundaries,
        fla_bf16_output_split=fla_bf16_output_split,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=A_log,
        dt_bias=dt_bias,
        use_wy=use_wy,
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
    fla_bf16_boundaries: bool = False,
    fla_bf16_output_split: bool = False,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_wy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch with precomputed graph-safe tree descriptors."""
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
    if state is None:
        state = torch.empty((n_pad, num_vh, dim_v, dim_k), device=q.device, dtype=torch.float32)
    elif state.shape[0] < n_actual or state.shape[1:] != (num_vh, dim_v, dim_k):
        raise ValueError(
            "state must be at least "
            f"({n_actual}, {num_vh}, {dim_v}, {dim_k}), got {tuple(state.shape)}"
        )
    grid = (num_vh, triton.cdiv(dim_v, BV))
    if use_wy:
        _tree_gdn_wy_kernel[grid](
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
            FLA_BF16_BOUNDARIES=bool(fla_bf16_boundaries),
            FLA_BF16_OUTPUT_SPLIT=bool(fla_bf16_output_split),
            RAW_GATING=raw_gating,
            COUNT_INVOCATION=count_invocation,
        )
        return out, state
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
        FLA_BF16_BOUNDARIES=bool(fla_bf16_boundaries),
        RAW_GATING=raw_gating,
        COUNT_INVOCATION=count_invocation,
    )
    return out, state
