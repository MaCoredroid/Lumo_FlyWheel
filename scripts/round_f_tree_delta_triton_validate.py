#!/usr/bin/env python3
"""Offline validation for the Round-F fused tree-delta Triton kernel."""

import sys

import torch

if not torch.cuda.is_available():
    print("SKIP: CUDA is not available; Round-F tree-delta Triton validation requires a GPU.")
    sys.exit(0)

try:
    from vllm.triton_utils import tl, triton
except Exception as exc:
    print(f"SKIP: vLLM/Triton validation dependencies are not available: {exc}")
    sys.exit(0)


@triton.jit(do_not_specialize=["N"])
def _tree_delta_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    state,
    idx,
    init_idx,
    parent,
    scale,
    N: tl.int64,
    HK: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BN: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_value: tl.constexpr,
    stride_state_key: tl.constexpr,
    stride_idx: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_hk = i_hv // (HV // HK)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_v[:, None] & m_k[None, :]

    head_off = (
        i_hv * stride_state_head
        + o_v[:, None] * stride_state_value
        + o_k[None, :] * stride_state_key
    )
    prefix_idx = tl.load(init_idx).to(tl.int64)
    prefix_h = tl.load(
        state + prefix_idx * stride_state_token + head_off,
        mask=m_h,
        other=0.0,
    ).to(tl.float32)

    for i in tl.static_range(0, BN):
        if i < N:
            if i == 0:
                parent_actual = tl.full((), -1, tl.int64)
            else:
                parent_raw = tl.load(parent + i).to(tl.int64)
                parent_actual = tl.where(parent_raw < 0, 0, parent_raw + 1)
            parent_safe = tl.maximum(parent_actual, 0)
            parent_write = tl.load(idx + parent_safe * stride_idx).to(tl.int64)
            parent_h = tl.load(
                state + parent_write * stride_state_token + head_off,
                mask=m_h,
                other=0.0,
            ).to(tl.float32)
            h = tl.where(parent_actual >= 0, parent_h, prefix_h)

            q_i = tl.load(
                q + (i * HK + i_hk) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            k_i = tl.load(
                k + (i * HK + i_hk) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            q_i = q_i * tl.rsqrt(tl.sum(q_i * q_i) + 1e-6) * scale
            k_i = k_i * tl.rsqrt(tl.sum(k_i * k_i) + 1e-6)

            g_i = tl.load(g + i * HV + i_hv).to(tl.float32)
            beta_i = tl.load(beta + i * HV + i_hv).to(tl.float32)
            v_i = tl.load(
                v + (i * HV + i_hv) * V + o_v,
                mask=m_v,
                other=0.0,
            ).to(tl.float32)

            h = h * tl.exp(g_i)
            delta_v = (v_i - tl.sum(h * k_i[None, :], axis=1)) * beta_i
            h = h + delta_v[:, None] * k_i[None, :]
            o_i = tl.sum(h * q_i[None, :], axis=1)

            tl.store(out + (i * HV + i_hv) * V + o_v, o_i, mask=m_v)
            write_idx = tl.load(idx + i * stride_idx).to(tl.int64)
            tl.store(
                state + write_idx * stride_state_token + head_off,
                h,
                mask=m_h,
            )


def _torch_ref(q, k, v, g, beta, state, idx, init_idx, parent):
    q = q.squeeze(0).float()
    k = k.squeeze(0).float()
    v = v.squeeze(0).float()
    g = g.squeeze(0).float()
    beta = beta.squeeze(0).float()
    n, hk, key_dim = q.shape
    hv = v.shape[1]
    repeat = hv // hk
    q = torch.nn.functional.normalize(q, dim=-1) * (key_dim**-0.5)
    k = torch.nn.functional.normalize(k, dim=-1)
    q = q.repeat_interleave(repeat, dim=1)
    k = k.repeat_interleave(repeat, dim=1)

    out = torch.empty((n, hv, v.shape[-1]), device=q.device, dtype=torch.float32)
    state_ref = state.clone().float()
    prefix = state_ref[int(init_idx[0])]
    for i in range(n):
        if i == 0:
            h = prefix.clone()
        else:
            parent_raw = int(parent[i])
            parent_actual = 0 if parent_raw < 0 else parent_raw + 1
            h = state_ref[int(idx[parent_actual, 0])].clone()
        h = h * torch.exp(g[i])[:, None, None]
        proj = torch.einsum("hvk,hk->hv", h, k[i])
        delta_v = (v[i] - proj) * beta[i, :, None]
        h = h + torch.einsum("hv,hk->hvk", delta_v, k[i])
        out[i] = torch.einsum("hvk,hk->hv", h, q[i])
        state_ref[int(idx[i, 0])] = h
    return out.unsqueeze(0), state_ref


def main() -> int:
    torch.manual_seed(0)
    max_out = 0.0
    max_state = 0.0
    cases = [
        (4, [-2, -1, 0, 1]),
        (6, [-2, -1, -1, 0, 1, 2]),
    ]
    for n, parent_list in cases:
        hk, hv, key_dim, value_dim = 2, 6, 32, 32
        slots = n + 3
        q = torch.randn(1, n, hk, key_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, n, hk, key_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, n, hv, value_dim, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, n, hv, device="cuda", dtype=torch.float32) * 0.1
        beta = torch.rand(1, n, hv, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(
            slots, hv, value_dim, key_dim, device="cuda", dtype=torch.bfloat16
        )
        state0 = state.clone()
        idx = torch.arange(1, n + 1, device="cuda", dtype=torch.int32).view(n, 1)
        init_idx = torch.tensor([0] + [1] * n, device="cuda", dtype=torch.int32)
        parent = torch.tensor(parent_list, device="cuda", dtype=torch.int32)
        out = torch.empty_like(v)

        bk = triton.next_power_of_2(key_dim)
        bv = 32
        bn = triton.next_power_of_2(n)
        _tree_delta_kernel[(triton.cdiv(value_dim, bv), hv)](
            q,
            k,
            v,
            g,
            beta,
            out,
            state,
            idx,
            init_idx,
            parent,
            key_dim**-0.5,
            n,
            hk,
            hv,
            key_dim,
            value_dim,
            bk,
            bv,
            bn,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            idx.stride(0),
            num_warps=4,
        )
        out_ref, state_ref = _torch_ref(q, k, v, g, beta, state0, idx, init_idx, parent)
        torch.cuda.synchronize()
        out_err = (out.float() - out_ref.float()).abs().max().item()
        state_err = (state.float() - state_ref.float()).abs().max().item()
        print(f"N={n} out_max={out_err:.6g} state_max={state_err:.6g}")
        max_out = max(max_out, out_err)
        max_state = max(max_state, state_err)
    if max_out > 5e-2 or max_state > 5e-2:
        raise SystemExit(f"validation failed: out={max_out} state={max_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
