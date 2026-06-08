"""Triton SiLU helper that matches native causal_conv1d_update ex2.approx."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - host-side tools may not have Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fr13_ex2_silu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x / (1.0 + tl.exp(0.0 - x))
        tl.store(out_ptr + offs, y, mask=mask)


def triton_ex2_silu_bf16(x: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
    """Apply SiLU using Triton's `tl.exp`, which lowers to `ex2.approx.f32`.

    Native vLLM `causal_conv1d_update` lowers its activation to:
    `mul.f32 -x, log2(e)` -> `ex2.approx.f32` -> `1+` -> `div.full.f32`
    -> `cvt.rn.bf16`. This helper keeps tree causal-conv on our compute path
    while using the same hardware activation instruction.
    """

    if triton is None or not x.is_cuda:
        return torch.nn.functional.silu(x).to(dtype=out_dtype)
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, dtype=out_dtype)
    n = x_contig.numel()
    _fr13_ex2_silu_kernel[(triton.cdiv(n, 256),)](
        x_contig,
        out,
        n,
        BLOCK=256,
    )
    return out
