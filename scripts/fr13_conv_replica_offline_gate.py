"""FR13 conv realization replica — OFFLINE bit-match gate (GB10, in-container).

Goal: make OUR tree causal-conv (fused_tree_conv_taps_acc + triton_ex2_silu_bf16)
BIT-EXACT (raw int16 view) to native vLLM ``causal_conv1d_update`` on identical
window inputs, at width=4.

Native is the COMPARISON ORACLE ONLY. We never call it to PRODUCE our answer.

Run INSIDE the live vLLM image (host venv is CPU-only; GB10 is container-only):

  docker run --rm --gpus all \
    -v <worktree>:/workspace -w /workspace \
    -e PYTHONPATH=/workspace/src \
    vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
    python scripts/fr13_conv_replica_offline_gate.py

No model boot. Small random tensors. Seconds per iteration.
"""

from __future__ import annotations

import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover
    print(f"FATAL: triton import failed: {exc}")
    sys.exit(1)

from lumo_flywheel_serving.fr13_ex2_silu import triton_ex2_silu_bf16
from lumo_flywheel_serving.fr13_tree_conv_fused import fused_tree_conv_taps_acc

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update


# ---------------------------------------------------------------------------
# Candidate replica Triton kernels (prototyped here; the winner is promoted
# into the two module files behind `replica`).
# ---------------------------------------------------------------------------


@triton.jit
def _replica_mac_kernel(
    window_ptr,  # [rows, width, dim] bf16
    w_ptr,  # [dim, width] bf16
    bias_ptr,  # [dim] or dummy
    out_ptr,  # [rows, dim] fp32 (MAC only) OR bf16 (fused silu)
    rows,
    dim,
    stride_win_row: tl.int64,
    stride_win_col: tl.int64,
    stride_win_dim: tl.int64,
    stride_w_dim: tl.int64,
    stride_w_width: tl.int64,
    stride_out_row: tl.int64,
    stride_out_dim: tl.int64,
    HAS_BIAS: tl.constexpr,
    WIDTH: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx_feats < dim
    if HAS_BIAS:
        acc = tl.load(bias_ptr + idx_feats, mask=mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = w_ptr + idx_feats * stride_w_dim
    win_base = window_ptr + row * stride_win_row + idx_feats * stride_win_dim
    # native op-order: acc += matrix_x * matrix_w ascending col0..col{W-1}
    for c in tl.static_range(WIDTH):
        w_c = tl.load(w_base + c * stride_w_width, mask=mask, other=0.0)
        x_c = tl.load(win_base + c * stride_win_col, mask=mask, other=0.0)
        acc += x_c * w_c
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(out_ptr + row * stride_out_row + idx_feats * stride_out_dim, acc, mask=mask)


def replica_conv_mac(
    *,
    window: torch.Tensor,  # [rows, width, dim] bf16
    conv_weights: torch.Tensor,  # [dim, width] bf16
    bias: torch.Tensor | None,
    silu: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    rows = int(window.size(0))
    width = int(window.size(1))
    dim = int(window.size(2))
    out = torch.empty((rows, dim), dtype=out_dtype, device=window.device)
    BLOCK_N = 256
    grid = (rows, triton.cdiv(dim, BLOCK_N))
    bias_arg = bias if bias is not None else window  # dummy ptr when no bias
    _replica_mac_kernel[grid](
        window,
        conv_weights,
        bias_arg,
        out,
        rows,
        dim,
        window.stride(0),
        window.stride(1),
        window.stride(2),
        conv_weights.stride(0),
        conv_weights.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        WIDTH=width,
        SILU=silu,
        BLOCK_N=BLOCK_N,
    )
    return out


# Diagnostic variants: force product rounding to isolate what native does.
@triton.jit
def _diag_mac_kernel(
    window_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,  # fp32
    rows,
    dim,
    stride_win_row: tl.int64,
    stride_win_col: tl.int64,
    stride_win_dim: tl.int64,
    stride_w_dim: tl.int64,
    stride_w_width: tl.int64,
    stride_out_row: tl.int64,
    stride_out_dim: tl.int64,
    HAS_BIAS: tl.constexpr,
    WIDTH: tl.constexpr,
    MODE: tl.constexpr,  # 0=native expr, 1=round prod bf16, 2=round prod fp32
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx_feats < dim
    if HAS_BIAS:
        acc = tl.load(bias_ptr + idx_feats, mask=mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = w_ptr + idx_feats * stride_w_dim
    win_base = window_ptr + row * stride_win_row + idx_feats * stride_win_dim
    for c in tl.static_range(WIDTH):
        w_c = tl.load(w_base + c * stride_w_width, mask=mask, other=0.0)
        x_c = tl.load(win_base + c * stride_win_col, mask=mask, other=0.0)
        if MODE == 0:
            acc += x_c * w_c
        elif MODE == 1:
            prod = (x_c * w_c).to(tl.bfloat16).to(tl.float32)
            acc += prod
        else:
            prod = x_c.to(tl.float32) * w_c.to(tl.float32)
            acc += prod
    tl.store(out_ptr + row * stride_out_row + idx_feats * stride_out_dim, acc, mask=mask)


def diag_conv_mac(*, window, conv_weights, bias, mode) -> torch.Tensor:
    rows = int(window.size(0))
    width = int(window.size(1))
    dim = int(window.size(2))
    out = torch.empty((rows, dim), dtype=torch.float32, device=window.device)
    BLOCK_N = 256
    grid = (rows, triton.cdiv(dim, BLOCK_N))
    bias_arg = bias if bias is not None else window
    _diag_mac_kernel[grid](
        window,
        conv_weights,
        bias_arg,
        out,
        rows,
        dim,
        window.stride(0),
        window.stride(1),
        window.stride(2),
        conv_weights.stride(0),
        conv_weights.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        WIDTH=width,
        MODE=mode,
        BLOCK_N=BLOCK_N,
    )
    return out


# ---------------------------------------------------------------------------
# int-view compare helpers.
# ---------------------------------------------------------------------------


def intview_compare(name: str, native: torch.Tensor, ours: torch.Tensor) -> int:
    """Bitcast bf16->int16 and compare raw bits. Returns mismatch count."""
    assert native.shape == ours.shape, (native.shape, ours.shape)
    assert native.dtype == torch.bfloat16 and ours.dtype == torch.bfloat16
    n = native.contiguous().view(torch.int16).to(torch.int32)
    o = ours.contiguous().view(torch.int16).to(torch.int32)
    diff = (n - o).abs()
    nmis = int((diff != 0).sum().item())
    total = int(diff.numel())
    maxd = int(diff.max().item()) if total else 0
    ex = ""
    if nmis:
        flat_n = n.reshape(-1)
        flat_o = o.reshape(-1)
        flat_d = diff.reshape(-1)
        idxs = torch.nonzero(flat_d != 0, as_tuple=False).reshape(-1)[:6].tolist()
        pairs = []
        for i in idxs:
            fv = native.reshape(-1)[i].item()
            pairs.append(
                f"(nat={int(flat_n[i]) & 0xFFFF:#06x} ours={int(flat_o[i]) & 0xFFFF:#06x} nat_val={fv:.6g} d={int(flat_d[i])})"
            )
        ex = " ; ".join(pairs)
    verdict = "BIT-EXACT" if nmis == 0 else "MISMATCH"
    print(f"  [{name:28s}] {verdict}: {nmis}/{total} mismatched, max_int_delta={maxd}")
    if ex:
        print(f"      e.g. {ex}")
    return nmis


# ---------------------------------------------------------------------------
# Native drivers.
# ---------------------------------------------------------------------------


def _padded_state_and_indices(conv_state0):
    """vLLM reserves cache-line 0 as the NULL block (padding); a row whose
    conv_state_index == null_block_id is SKIPPED by the kernel (out=raw x).
    Prepend a dummy line-0 and map row i -> line i+1 so every active row is
    a real, non-null block (mirrors live serving, where slot 0 is reserved)."""
    batch = int(conv_state0.size(0))
    dummy = torch.zeros_like(conv_state0[:1])
    cs = torch.cat([dummy, conv_state0.clone()], dim=0)  # [batch+1, ...]
    idxs = torch.arange(1, batch + 1, device=conv_state0.device, dtype=torch.int32)
    return cs, idxs


def run_native_simple_decode(*, conv_state0, x1, weight, bias, activation):
    # NOTE: causal_conv1d_update overwrites x IN PLACE (out = x). Clone x so the
    # caller's tensor stays pristine for window building.
    xin = x1.clone()
    cs, idxs = _padded_state_and_indices(conv_state0)
    out = causal_conv1d_update(
        xin, cs, weight, bias, activation=activation, conv_state_indices=idxs
    )
    return out.to(torch.bfloat16)


def run_native_spec_decode(*, conv_state0, x_seq, weight, bias, num_accepted, activation):
    xin = x_seq.clone()  # native overwrites x in place
    cs, idxs = _padded_state_and_indices(conv_state0)
    out = causal_conv1d_update(
        xin,
        cs,
        weight,
        bias,
        activation=activation,
        conv_state_indices=idxs,
        num_accepted_tokens=num_accepted,
    )
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Our-side window builders (mirror how native reads the window).
# ---------------------------------------------------------------------------


def build_window_simple(*, conv_state0, x1, width):
    batch = int(x1.size(0))
    dim = int(x1.size(1))
    prior = conv_state0[:, :, : width - 1]
    win = torch.empty((batch, width, dim), dtype=x1.dtype, device=x1.device)
    for c in range(width - 1):
        win[:, c, :] = prior[:, :, c]
    win[:, width - 1, :] = x1
    return win


def build_windows_spec(*, conv_state0, x_seq, width, num_accepted):
    """Native spec-decode per-token windows.

    offset = num_accepted-1. The kernel seeds col0..col{w-2} from
    conv_state[offset .. offset+w-2] then slides one x-token in per output.
    So with S = conv_state0[b, :, offset:offset+w-1] ++ x_seq[b] (along tokens),
    window(t) = S[t : t+width]  (a clean sliding window over S)."""
    batch = int(x_seq.size(0))
    dim = int(x_seq.size(1))
    seqlen = int(x_seq.size(2))
    wins = []
    for b in range(batch):
        off = int(num_accepted[b].item()) - 1
        prior = conv_state0[b, :, off : off + width - 1].transpose(0, 1)  # [w-1,dim]
        xb = x_seq[b].transpose(0, 1)  # [seqlen, dim]
        S = torch.cat([prior, xb], dim=0)  # [w-1+seqlen, dim]
        for t in range(seqlen):
            wins.append(S[t : t + width])  # [width, dim]
    return torch.stack(wins, dim=0)


# ---------------------------------------------------------------------------
# Our-side candidate paths.
# ---------------------------------------------------------------------------


def ours_current(*, window, conv_weights, bias, out_dtype):
    acc = fused_tree_conv_taps_acc(window=window, conv_weights=conv_weights, bias=bias)
    return triton_ex2_silu_bf16(acc, out_dtype=out_dtype)


def ours_replica_mac_plus_oursilu(*, window, conv_weights, bias, out_dtype):
    acc = replica_conv_mac(
        window=window, conv_weights=conv_weights, bias=bias, silu=False,
        out_dtype=torch.float32,
    )
    return triton_ex2_silu_bf16(acc, out_dtype=out_dtype)


def ours_replica_fully_fused(*, window, conv_weights, bias, out_dtype):
    return replica_conv_mac(
        window=window, conv_weights=conv_weights, bias=bias, silu=True,
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available (must run inside the vLLM container).")
        sys.exit(1)
    dev = "cuda"
    torch.manual_seed(1234)
    width = 4
    activation = "silu"

    print("=" * 78)
    print("FR13 CONV REPLICA OFFLINE GATE  (width=4, activation=silu)")
    print("=" * 78)

    overall_pass = True
    v0_all_exact = True

    # --- Section A: simple decode (seqlen=1), multiple dims/rows/seeds ---
    for seed in (1234, 7, 99999):
        torch.manual_seed(seed)
        for dim in (128, 256, 512):
            for batch in (32, 96):
                for has_bias in (True, False):
                    conv_state0 = torch.randn(
                        batch, dim, width - 1, dtype=torch.bfloat16, device=dev
                    )
                    x1 = torch.randn(batch, dim, dtype=torch.bfloat16, device=dev)
                    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=dev)
                    bias = (
                        torch.randn(dim, dtype=torch.bfloat16, device=dev)
                        if has_bias else None
                    )

                    native = run_native_simple_decode(
                        conv_state0=conv_state0, x1=x1, weight=weight, bias=bias,
                        activation=activation,
                    )
                    window = build_window_simple(
                        conv_state0=conv_state0, x1=x1, width=width
                    )
                    print(
                        f"\n-- simple decode  seed={seed} dim={dim} "
                        f"batch={batch} bias={has_bias} --"
                    )
                    v0 = ours_current(
                        window=window, conv_weights=weight, bias=bias,
                        out_dtype=torch.bfloat16,
                    )
                    n_v0 = intview_compare("V0 current (pytorch mul)", native, v0)
                    v1 = ours_replica_mac_plus_oursilu(
                        window=window, conv_weights=weight, bias=bias,
                        out_dtype=torch.bfloat16,
                    )
                    n_v1 = intview_compare("V1 triton-MAC + our silu", native, v1)
                    v2 = ours_replica_fully_fused(
                        window=window, conv_weights=weight, bias=bias,
                        out_dtype=torch.bfloat16,
                    )
                    n_v2 = intview_compare("V2 fully-fused triton", native, v2)
                    v0_all_exact = v0_all_exact and (n_v0 == 0)
                    overall_pass = overall_pass and (n_v1 == 0 or n_v2 == 0)
    torch.manual_seed(1234)

    # --- Section B: diagnostic — product-rounding mode sweep ---
    print("\n" + "=" * 78)
    print("DIAGNOSTIC: product-rounding mode (triton MAC->fp32 acc, then our silu)")
    print("MODE0=native-expr(acc+=x*w)  MODE1=round-prod-bf16  MODE2=round-prod-fp32")
    print("=" * 78)
    dim, batch = 256, 96
    conv_state0 = torch.randn(batch, dim, width - 1, dtype=torch.bfloat16, device=dev)
    x1 = torch.randn(batch, dim, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=dev)
    bias = torch.randn(dim, dtype=torch.bfloat16, device=dev)
    native = run_native_simple_decode(
        conv_state0=conv_state0, x1=x1, weight=weight, bias=bias, activation=activation
    )
    window = build_window_simple(conv_state0=conv_state0, x1=x1, width=width)
    for mode in (0, 1, 2):
        acc = diag_conv_mac(window=window, conv_weights=weight, bias=bias, mode=mode)
        out = triton_ex2_silu_bf16(acc, out_dtype=torch.bfloat16)
        intview_compare(f"MODE{mode}", native, out)
    acc_pt = fused_tree_conv_taps_acc(window=window, conv_weights=weight, bias=bias)
    out_pt = triton_ex2_silu_bf16(acc_pt, out_dtype=torch.bfloat16)
    intview_compare("pytorch-mul(V0)", native, out_pt)
    # Also exonerate the LEGACY OFF-arm (_fr11_conv_tap_product replica) so the
    # whole conv-compute family is covered, not just the FR13-fused arm.
    from lumo_flywheel_serving.fr13_tree_conv_fused import (
        legacy_tree_conv_taps_acc_reference,
    )

    x_new = window[:, width - 1, :].contiguous()  # the "current" token row
    acc_legacy = legacy_tree_conv_taps_acc_reference(
        window=window, conv_weights=weight, bias=bias, x=x_new, tap_dtype=window.dtype
    )
    out_legacy = triton_ex2_silu_bf16(acc_legacy, out_dtype=torch.bfloat16)
    intview_compare("legacy OFF-arm ref", native, out_legacy)

    # --- Section C: spec decode (deep-accept, num_accepted>=3) ---
    print("\n" + "=" * 78)
    print("SPEC DECODE (num_accepted>=3, seqlen>1) — deep-accept column read")
    print("=" * 78)
    dim = 256
    batch = 16
    seqlen = 4
    state_len_alloc = width - 1 + (seqlen - 1) + 4
    conv_state0 = torch.randn(
        batch, dim, state_len_alloc, dtype=torch.bfloat16, device=dev
    )
    x_seq = torch.randn(batch, dim, seqlen, dtype=torch.bfloat16, device=dev)
    num_accepted = torch.randint(1, seqlen + 1, (batch,), dtype=torch.int32, device=dev)
    num_accepted[0] = 3
    num_accepted[1] = 4
    try:
        native_sp = run_native_spec_decode(
            conv_state0=conv_state0, x_seq=x_seq, weight=weight, bias=bias,
            num_accepted=num_accepted, activation=activation,
        )
        wins = build_windows_spec(
            conv_state0=conv_state0, x_seq=x_seq, width=width, num_accepted=num_accepted
        )
        v1 = ours_replica_mac_plus_oursilu(
            window=wins, conv_weights=weight, bias=bias, out_dtype=torch.bfloat16
        ).view(batch, seqlen, dim).transpose(1, 2).contiguous()
        v2 = ours_replica_fully_fused(
            window=wins, conv_weights=weight, bias=bias, out_dtype=torch.bfloat16
        ).view(batch, seqlen, dim).transpose(1, 2).contiguous()
        v0 = ours_current(
            window=wins, conv_weights=weight, bias=bias, out_dtype=torch.bfloat16
        ).view(batch, seqlen, dim).transpose(1, 2).contiguous()
        print(f"  num_accepted[:8]={num_accepted[:8].tolist()}")
        intview_compare("SPEC V0 current", native_sp, v0)
        intview_compare("SPEC V1 triton-MAC", native_sp, v1)
        intview_compare("SPEC V2 fully-fused", native_sp, v2)
    except Exception as exc:
        import traceback

        print(f"  SPEC decode driver failed (interface): {exc}")
        traceback.print_exc()

    # --- Section D: adversarial red-team of the "bit-exact" claim ---
    print("\n" + "=" * 78)
    print("RED-TEAM D1: is the num_accepted offset actually ENGAGED? "
          "(different offsets must give different native outputs, else vacuous)")
    print("=" * 78)
    dim = 128
    batch = 8
    seqlen = 4
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=dev)
    bias = torch.randn(dim, dtype=torch.bfloat16, device=dev)
    state_len_alloc = width - 1 + (seqlen - 1) + 4
    conv_state0 = torch.randn(batch, dim, state_len_alloc, dtype=torch.bfloat16, device=dev)
    x_seq = torch.randn(batch, dim, seqlen, dtype=torch.bfloat16, device=dev)
    na_a = torch.full((batch,), 1, dtype=torch.int32, device=dev)
    na_b = torch.full((batch,), 4, dtype=torch.int32, device=dev)
    out_a = run_native_spec_decode(
        conv_state0=conv_state0, x_seq=x_seq, weight=weight, bias=bias,
        num_accepted=na_a, activation=activation,
    )
    out_b = run_native_spec_decode(
        conv_state0=conv_state0, x_seq=x_seq, weight=weight, bias=bias,
        num_accepted=na_b, activation=activation,
    )
    n_engaged = int((out_a.view(torch.int16) != out_b.view(torch.int16)).sum().item())
    print(f"  offset=0 vs offset=3 native outputs differ in {n_engaged}/{out_a.numel()} "
          f"elements -> offset {'ENGAGED (test is meaningful)' if n_engaged else 'NOT ENGAGED (VACUOUS!)'}")
    # and OURS tracks the offset too
    w_a = build_windows_spec(conv_state0=conv_state0, x_seq=x_seq, width=width, num_accepted=na_a)
    w_b = build_windows_spec(conv_state0=conv_state0, x_seq=x_seq, width=width, num_accepted=na_b)
    ours_a = ours_current(window=w_a, conv_weights=weight, bias=bias, out_dtype=torch.bfloat16).view(batch, seqlen, dim).transpose(1, 2).contiguous()
    ours_b = ours_current(window=w_b, conv_weights=weight, bias=bias, out_dtype=torch.bfloat16).view(batch, seqlen, dim).transpose(1, 2).contiguous()
    intview_compare("D1 offset=0  ours vs native", out_a, ours_a)
    intview_compare("D1 offset=3  ours vs native", out_b, ours_b)

    print("\n" + "=" * 78)
    print("RED-TEAM D2: production window-build (fused_tree_conv_source + "
          "index_select) must be BYTE-preserving vs a direct build")
    print("=" * 78)
    from lumo_flywheel_serving.fr13_tree_conv_fused import (
        build_tree_conv_window_source_indices,
        fused_tree_conv_source,
    )

    # A tiny linear-chain tree (parent[i]=i-1): node i's window = last `width`
    # of (prior[0..w-2] ++ path tokens). Build via the production path and via
    # a direct index and confirm byte-identical windows -> identical conv out.
    dim = 128
    tree_n = 6
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=dev)
    bias = torch.randn(dim, dtype=torch.bfloat16, device=dev)
    parent = [-1, 0, 1, 2, 3, 4]  # chain
    prior_window = torch.randn(dim, width - 1, dtype=torch.bfloat16, device=dev)
    xb = torch.randn(tree_n, dim, dtype=torch.bfloat16, device=dev)
    zero_row = torch.zeros(1, dim, dtype=torch.bfloat16, device=dev)
    source_flat = build_tree_conv_window_source_indices(parent=parent, width=width, device=dev)
    source_z = fused_tree_conv_source(prior_window=prior_window, x=xb, zero_row=zero_row)
    window_prod = source_z.index_select(0, source_flat.reshape(-1)).view(tree_n, width, dim)
    acc_prod = fused_tree_conv_taps_acc(window=window_prod, conv_weights=weight, bias=bias)
    out_prod = triton_ex2_silu_bf16(acc_prod, out_dtype=torch.bfloat16)
    # native oracle on the SAME windows: feed each node's window as a simple
    # decode (prior=window[:, :width-1], x=window[:, width-1]).
    cs_nat = window_prod[:, : width - 1, :].transpose(1, 2).contiguous()  # [tree_n,dim,w-1]
    x_nat = window_prod[:, width - 1, :].contiguous()  # [tree_n, dim]
    nat_prod = run_native_simple_decode(
        conv_state0=cs_nat, x1=x_nat, weight=weight, bias=bias, activation=activation
    )
    intview_compare("D2 production-window conv", nat_prod, out_prod)

    print("\n" + "=" * 78)
    print(f"V0 (CURRENT production code) bit-exact to native, all section-A "
          f"configs: {'YES' if v0_all_exact else 'NO'}")
    print(
        f"OVERALL (V1/V2 replica bit-exact everywhere): "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
