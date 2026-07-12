"""Decisive GB10 microbench: are the L0 fp8 in_proj GEMMs (qkv/z/out_proj) M-keyed and/or BIASED?

The bf16 in_proj_a/b are M-keyed (kernel switch at M>=9) but REFUTED as the garble cause.
The fp8 GEMMs (in_proj_qkv/z, out_proj) are block-scaled fp8 (deepseek-style, weight_scale_inv 128x128).
If the ACTIVATION quantization is per-TENSOR (scale over the whole M-row batch) instead of per-TOKEN
(per-row), the tree's M=9 rows shift the scale vs native's M=1 -> row-0 output changes with M -> a
batch-dependent SYSTEMATIC seed. Per-token quant => row-0 M-invariant => refuted.

Uses the ACTUAL vLLM block-scaled fp8 path. Run INSIDE the vllm container (GPU). Deterministic, no server.
Reports row-0 output for M=1..12 vs M=1 baseline: max|diff| (M-keying) and signed_mean (bias).
"""
import torch, glob
from safetensors import safe_open

dev = "cuda"
LAYER = "/models/qwen3.6-27b-fp8/layers-0.safetensors"

# --- load the three fp8 in_proj weights + their block scales ---
W = {}
with safe_open(LAYER, framework="pt", device="cpu") as sf:
    for nm, wkey in [("qkv", "in_proj_qkv"), ("z", "in_proj_z"), ("out", "out_proj")]:
        w = sf.get_tensor(f"model.language_model.layers.0.linear_attn.{wkey}.weight")
        s = sf.get_tensor(f"model.language_model.layers.0.linear_attn.{wkey}.weight_scale_inv")
        W[nm] = (w.to(dev), s.to(dev).float())
        print(f"{wkey:14s} w={tuple(w.shape)} {w.dtype}  scale_inv={tuple(s.shape)}", flush=True)

# in features: qkv/z consume 5120 (the residual stream); out_proj consumes 6144
IN_FEAT = {"qkv": 5120, "z": 5120, "out": 6144}

# try the real vLLM block-scaled fp8 path; report which quant granularity it uses
quant_fn = None
matmul_fn = None
try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8, w8a8_triton_block_scaled_mm,
    )
    quant_fn = per_token_group_quant_fp8
    matmul_fn = w8a8_triton_block_scaled_mm
    print("USING vLLM per_token_group_quant_fp8 + w8a8_block_fp8_matmul", flush=True)
except Exception as e:
    print("vLLM fp8 import failed:", e, flush=True)


def fp8_linear(nm, M, x_full):
    w, ws = W[nm]
    x = x_full[:M].contiguous()
    # per-token-group quant: group=128 along the in-feature dim, scale PER ROW-GROUP.
    xq, xs = quant_fn(x, 128)
    out = matmul_fn(xq, w, xs, ws, [128, 128], output_dtype=torch.bfloat16)
    return out  # [M, out_feat]


def main():
    if quant_fn is None:
        print("Cannot run: vLLM fp8 utils unavailable", flush=True)
        return
    g = torch.Generator(device=dev).manual_seed(1313)
    print("\n=== fp8 in_proj row-0 M-keying + bias (row-0 held fixed, M rows total) ===", flush=True)
    for nm in ("qkv", "z", "out"):
        infeat = IN_FEAT[nm]
        # fixed row-0 + random other rows (branches). Row-0 IDENTICAL across all M.
        row0 = (torch.randn(1, infeat, generator=g, device=dev, dtype=torch.bfloat16) * 0.1)
        others = (torch.randn(15, infeat, generator=g, device=dev, dtype=torch.bfloat16) * 0.1)
        x_full = torch.cat([row0, others], 0)
        base = fp8_linear(nm, 1, x_full)[0].float()
        cells = []
        worst_signed = 0.0
        for M in range(1, 13):
            r0 = fp8_linear(nm, M, x_full)[0].float()
            diff = (r0 - base)
            mx = diff.abs().max().item()
            sm = diff.mean().item()
            if abs(sm) > abs(worst_signed):
                worst_signed = sm
            cells.append(f"M{M}:{mx:.1e}")
        print(f"  in_proj_{nm:3s} row0 max|d|: " + " ".join(cells), flush=True)
        print(f"           worst signed_mean(BIAS) over M = {worst_signed:+.3e}  "
              f"(nonzero+same-sign => systematic M-keyed seed)", flush=True)

    # co-residency: does row-0 change when the OTHER rows' CONTENT changes (M fixed=9)?
    print("\n=== fp8 CO-RESIDENCY: M=9 fixed, row-0 fixed, vary rows 1..8 content ===", flush=True)
    for nm in ("qkv", "z", "out"):
        infeat = IN_FEAT[nm]
        row0 = (torch.randn(1, infeat, generator=g, device=dev, dtype=torch.bfloat16) * 0.1)
        gA = torch.Generator(device=dev).manual_seed(7)
        gB = torch.Generator(device=dev).manual_seed(99)
        oA = torch.randn(8, infeat, generator=gA, device=dev, dtype=torch.bfloat16) * 0.1
        oB = torch.randn(8, infeat, generator=gB, device=dev, dtype=torch.bfloat16) * 0.1
        r0A = fp8_linear(nm, 9, torch.cat([row0, oA], 0))[0].float()
        r0B = fp8_linear(nm, 9, torch.cat([row0, oB], 0))[0].float()
        d = (r0A - r0B)
        print(f"  in_proj_{nm:3s} row0(neighborsA) - row0(neighborsB): max|d|={d.abs().max().item():.3e} "
              f"signed_mean={d.mean().item():+.3e}  (nonzero => row-0 CONTAMINATED by co-resident branches)",
              flush=True)


if __name__ == "__main__":
    main()
