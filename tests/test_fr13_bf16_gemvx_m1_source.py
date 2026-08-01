from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_m1.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_m1.py"


def test_cuda_source_preserves_stock_gemvx_launch_geometry() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kVocab = 248320;" in source
    assert "constexpr int kLanes = 16;" in source
    assert "constexpr int kRowsPerCta = 8;" in source
    assert "constexpr int kSharedRowStride = 17;" in source
    assert "static_assert(kCtas == 31040);" in source
    assert "const dim3 block(kLanes, kRowsPerCta, 1);" in source
    assert "kRowsPerCta * kSharedRowStride * sizeof(float)" in source
    assert "fr13_bf16_gemvx_m1_kernel<<<kCtas, block, shared_bytes" in source


def test_cuda_source_preserves_stock_scalar_arithmetic_order() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "float accumulator = 0.0f;" in source
    assert "#pragma unroll 1" in source
    assert "for (int k = lane; k < kHidden; k += kLanes)" in source
    assert "accumulator = __fmaf_rn(x, w, accumulator);" in source
    assert source.count("__fadd_rn(") == 4
    for stride in (8, 4, 2):
        assert f"if (lane < {stride})" in source
    assert "if (lane == 0)" in source
    assert "output[row] = __float2bfloat16_rn(sum);" in source
    assert "grid-stride" not in source.lower()
    assert "atomicAdd" not in source


def test_cuda_op_is_out_variant_with_strict_full_head_geometry() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "gemvx_m1_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()" in source
    assert "input.sizes() == at::IntArrayRef({1, kHidden})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({1, kVocab})" in source
    assert "at::cuda::getCurrentCUDAStream()" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source


def test_builder_is_pinned_and_does_not_claim_qualification() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="ascii"))
    assert tree is not None
    source = BUILDER.read_text(encoding="ascii")

    assert 'EXPECTED_TORCH = "2.10.0+cu130"' in source
    assert 'EXPECTED_CUDA = "13.0"' in source
    assert 'EXPECTED_ARCH = "12.1a"' in source
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"byte_equality_claim": False' in source
    assert '"performance_measurement": False' in source
    assert '"production_default_enabled": False' in source
