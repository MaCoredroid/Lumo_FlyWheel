from __future__ import annotations

import ast
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle.py"


def test_cuda_source_is_strict_k64_m1_full_warp_r32_pair4bits() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kLanes = 32;" in source
    assert "constexpr int kRowsPerCta = 32;" in source
    assert "constexpr int kElementsPerLoad = 4;" in source
    assert "constexpr int kQuads = kHidden / kElementsPerLoad;" in source
    assert "static_assert(kLanes * kRowsPerCta == 1024);" in source
    assert "static_assert(kCtas == 2048);" in source
    assert "const dim3 block(kLanes, kRowsPerCta, 1);" in source
    assert "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>" in source
    assert "__syncthreads" not in source
    assert "extern __shared__" not in source


def test_cuda_source_uses_packed_quad_loads_and_fp32_accumulation() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "float accumulator = 0.0f;" in source
    assert "#pragma unroll 1" in source
    assert "for (int quad = lane; quad < kQuads; quad += kLanes)" in source
    assert "reinterpret_cast<const uint2*>(input)" in source
    assert "reinterpret_cast<const uint2*>(weight)" in source
    assert source.count("__uint_as_float(") == 8
    for half in ("x.x", "x.y", "w.x", "w.y"):
        assert f"{half} << 16" in source
        assert f"{half} & 0xffff0000u" in source
    assert source.count("accumulator = __fmaf_rn(x") == 4
    assert source.count("__shfl_down_sync(") == 5
    assert source.count("__fadd_rn(") == 5
    for stride in (16, 8, 4, 2, 1):
        assert f", {stride}, kLanes)" in source
    for threshold in (16, 8, 4, 2):
        assert f"if (lane < {threshold})" in source
    assert "if (lane == 0)" in source
    assert "const float sum = __fmaf_rn(alpha, reduced_sum, beta);" in source
    assert "1.0f, 0.0f);" in source
    assert "output[row] = __float2bfloat16_rn(sum);" in source
    assert "atomicAdd" not in source


def test_bf16_bit_expansion_matches_fp32_for_every_non_nan_pattern() -> None:
    raw = torch.arange(1 << 16, dtype=torch.int32)
    bf16 = raw.to(torch.int16).view(torch.bfloat16)
    actual = bf16.float().view(torch.int32)
    expected = raw << 16
    is_nan = ((raw & 0x7F80) == 0x7F80) & ((raw & 0x007F) != 0)

    assert torch.equal(actual[~is_nan], expected[~is_nan])


def test_width32_shuffle_assigns_one_row_per_warp() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr unsigned kFullWarpMask = 0xffffffffu;" in source
    assert "static_cast<int>(threadIdx.x)" in source
    assert "static_cast<int>(threadIdx.y)" in source
    assert source.count("kFullWarpMask, accumulator") == 5
    assert source.count("kLanes);") >= 5


def test_cuda_op_is_out_variant_with_strict_k64_geometry() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert (
        "gemvx_m1_warp32_r32_pair4bits_out(Tensor(a!) output, Tensor input, "
        "Tensor weight) -> ()" in source
    )
    assert "input.sizes() == at::IntArrayRef({1, kHidden})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({1, kVocab})" in source
    assert "weight must be contiguous [65536,5120]" in source
    assert "pair4bits inputs must be 8-byte aligned" in source
    assert "alignof(uint2)" in source
    assert "at::cuda::getCurrentCUDAStream()" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source
    assert "TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_head, library)" in source


def test_builder_is_pinned_default_off_and_claims_no_qualification() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="ascii"))
    assert tree is not None
    source = BUILDER.read_text(encoding="ascii")

    assert 'EXPECTED_TORCH = "2.11.0+cu130"' in source
    assert 'EXPECTED_CUDA = "13.0"' in source
    assert 'EXPECTED_ARCH = "12.1a"' in source
    assert '"/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"' in source
    assert '(CUDA_PACKAGE_INCLUDE / "cusparse.h").is_file()' in source
    assert 'f"-I{CUDA_PACKAGE_INCLUDE}"' in source
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"byte_equality_claim": False' in source
    assert '"resource_claim": False' in source
    assert '"performance_measurement": False' in source
    assert '"production_default_enabled": False' in source
    assert '"grid": [2048, 1, 1]' in source
    assert '"block": [32, 32, 1]' in source
    assert '"output_rows_per_cta": 32' in source
    assert '"k_partition_lanes": 32' in source
    assert '"elements_per_load": 4' in source
    assert '"lane_load_iterations": 40' in source
    assert '"lane_fma_iterations": 160' in source
    assert '"packed_unpack": "BF16 bits shifted/masked into exact FP32 bits"' in source
