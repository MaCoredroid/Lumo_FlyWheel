from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
R32_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
R64_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle_r64.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle_r64.py"


def _arithmetic_body(source: str) -> str:
    start = source.index("  float accumulator = 0.0f;")
    end = source.index("\n  }\n}\n", start) + len("\n  }\n}")
    return source[start:end]


def test_cuda_source_is_strict_k64_m1_with_maximum_thread_cta() -> None:
    source = R64_CUDA.read_text(encoding="ascii")

    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kLanes = 16;" in source
    assert "constexpr int kRowsPerCta = 64;" in source
    assert "static_assert(kLanes * kRowsPerCta == 1024);" in source
    assert "static_assert(kCtas == 1024);" in source
    assert "const dim3 block(kLanes, kRowsPerCta, 1);" in source
    assert "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>" in source
    assert "__syncthreads" not in source
    assert "extern __shared__" not in source


def test_r64_changes_only_row_ownership_outside_per_row_arithmetic() -> None:
    r32 = R32_CUDA.read_text(encoding="ascii")
    r64 = R64_CUDA.read_text(encoding="ascii")

    assert _arithmetic_body(r64) == _arithmetic_body(r32)
    assert "#pragma unroll 1" in r64
    assert "for (int k = lane; k < kHidden; k += kLanes)" in r64
    assert "accumulator = __fmaf_rn(x, w, accumulator);" in r64
    assert r64.count("__shfl_down_sync(") == 4
    assert r64.count("__fadd_rn(") == 4
    for stride in (8, 4, 2, 1):
        assert f", {stride}, kLanes)" in r64
    assert "const float sum = __fmaf_rn(alpha, reduced_sum, beta);" in r64
    assert "output[row] = __float2bfloat16_rn(sum);" in r64
    assert "atomicAdd" not in r64


def test_width16_shuffle_keeps_two_rows_per_warp_independent() -> None:
    source = R64_CUDA.read_text(encoding="ascii")

    assert "constexpr unsigned kFullWarpMask = 0xffffffffu;" in source
    assert "static_cast<int>(threadIdx.x)" in source
    assert "static_cast<int>(threadIdx.y)" in source
    assert source.count("kFullWarpMask, accumulator") == 4
    assert source.count("kLanes);") >= 4


def test_cuda_op_is_separate_out_variant_with_strict_k64_geometry() -> None:
    source = R64_CUDA.read_text(encoding="ascii")

    assert (
        "gemvx_m1_shuffle_r64_out(Tensor(a!) output, Tensor input, "
        "Tensor weight) -> ()" in source
    )
    assert "gemvx_m1_shuffle_r32_out" not in source
    assert "input.sizes() == at::IntArrayRef({1, kHidden})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({1, kVocab})" in source
    assert "weight must be contiguous [65536,5120]" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source
    assert "TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_head, library)" in source


def test_builder_is_pinned_default_off_and_claims_no_qualification() -> None:
    builder = BUILDER.read_text(encoding="ascii")

    assert ast.parse(builder) is not None
    assert 'EXPECTED_TORCH = "2.11.0+cu130"' in builder
    assert 'EXPECTED_CUDA = "13.0"' in builder
    assert 'EXPECTED_ARCH = "12.1a"' in builder
    assert '"/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"' in builder
    assert '(CUDA_PACKAGE_INCLUDE / "cusparse.h").is_file()' in builder
    assert 'f"-I{CUDA_PACKAGE_INCLUDE}"' in builder
    assert '"status": "BUILT_UNQUALIFIED"' in builder
    assert '"byte_equality_claim": False' in builder
    assert '"resource_claim": False' in builder
    assert '"performance_measurement": False' in builder
    assert '"production_default_enabled": False' in builder
    assert '"grid": [1024, 1, 1]' in builder
    assert '"block": [16, 64, 1]' in builder
    assert '"threads_per_cta": 1024' in builder
    assert '"output_rows_per_cta": 64' in builder


def test_candidate_has_no_runtime_selector_or_performance_claim() -> None:
    combined = R64_CUDA.read_text(encoding="ascii") + BUILDER.read_text(
        encoding="ascii"
    )

    assert "FR13_DRAFT_HEAD_M1_R64" not in combined
    assert "timing_eligible" not in combined
    assert "performance_claim" not in combined
