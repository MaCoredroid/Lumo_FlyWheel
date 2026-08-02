from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle.py"


def test_cuda_source_is_strict_k64_m1_and_halves_r16_cta_grid() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kLanes = 16;" in source
    assert "constexpr int kRowsPerCta = 32;" in source
    assert "static_assert(kLanes * kRowsPerCta == 512);" in source
    assert "static_assert(kCtas == 2048);" in source
    assert "const dim3 block(kLanes, kRowsPerCta, 1);" in source
    assert "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>" in source
    assert "__syncthreads" not in source
    assert "extern __shared__" not in source


def test_cuda_source_preserves_scalar_accumulation_and_reduction_order() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "float accumulator = 0.0f;" in source
    assert "#pragma unroll 1" in source
    assert "for (int k = lane; k < kHidden; k += kLanes)" in source
    assert "accumulator = __fmaf_rn(x, w, accumulator);" in source
    assert source.count("__shfl_down_sync(") == 4
    assert source.count("__fadd_rn(") == 4
    for stride in (8, 4, 2, 1):
        assert f", {stride}, kLanes)" in source
    for threshold in (8, 4, 2):
        assert f"if (lane < {threshold})" in source
    assert "if (lane == 0)" in source
    assert "const float sum = __fmaf_rn(alpha, reduced_sum, beta);" in source
    assert "1.0f, 0.0f);" in source
    assert "output[row] = __float2bfloat16_rn(sum);" in source
    assert "atomicAdd" not in source


def test_width16_shuffle_keeps_two_rows_per_warp_independent() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr unsigned kFullWarpMask = 0xffffffffu;" in source
    assert "static_cast<int>(threadIdx.x)" in source
    assert "static_cast<int>(threadIdx.y)" in source
    assert source.count("kFullWarpMask, accumulator") == 4
    assert source.count("kLanes);") >= 4


def test_cuda_op_is_out_variant_with_strict_k64_geometry() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert (
        "gemvx_m1_shuffle_r32_out(Tensor(a!) output, Tensor input, "
        "Tensor weight) -> ()" in source
    )
    assert "input.sizes() == at::IntArrayRef({1, kHidden})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({1, kVocab})" in source
    assert "weight must be contiguous [65536,5120]" in source
    assert "at::cuda::getCurrentCUDAStream()" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source
    assert "TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_head, library)" in source


def test_builder_is_pinned_default_off_and_claims_no_qualification() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="ascii"))
    assert tree is not None
    source = BUILDER.read_text(encoding="ascii")

    assert 'EXPECTED_TORCH = "2.10.0+cu130"' in source
    assert 'EXPECTED_CUDA = "13.0"' in source
    assert 'EXPECTED_ARCH = "12.1a"' in source
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"byte_equality_claim": False' in source
    assert '"resource_claim": False' in source
    assert '"performance_measurement": False' in source
    assert '"production_default_enabled": False' in source
    assert '"grid": [2048, 1, 1]' in source
    assert '"block": [16, 32, 1]' in source
    assert '"output_rows_per_cta": 32' in source
