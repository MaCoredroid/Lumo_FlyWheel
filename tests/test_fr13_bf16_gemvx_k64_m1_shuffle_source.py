from __future__ import annotations

import ast
import re
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle.py"


def test_cuda_source_is_strict_k64_m1_warp4_globalx_pair8bits() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kLanes = 32;" in source
    assert "constexpr int kWarpsPerCta = 8;" in source
    assert "constexpr int kRowsPerWarp = 4;" in source
    assert "constexpr int kRowsPerCta = kWarpsPerCta * kRowsPerWarp;" in source
    assert "constexpr int kThreadsPerCta = kLanes * kWarpsPerCta;" in source
    assert "constexpr int kElementsPerLoad = 8;" in source
    assert "constexpr int kOctets = kHidden / kElementsPerLoad;" in source
    assert "static_assert(kRowsPerCta == 32);" in source
    assert "static_assert(kThreadsPerCta == 256);" in source
    assert "static_assert(kCtas == 2048);" in source
    assert "static_assert(kOctets == 640);" in source
    assert "static_assert(alignof(uint4) == 16);" in source
    assert "(kHidden * sizeof(__nv_bfloat16)) % alignof(uint4) == 0" in source
    assert "__launch_bounds__(kThreadsPerCta)" in source
    assert "const dim3 block(kLanes, kWarpsPerCta, 1);" in source
    assert "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>" in source


def test_each_warp_loads_one_hidden_octet_for_four_rows() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "const auto* input_octets = reinterpret_cast<const uint4*>(input);" in source
    assert "const uint4 x = input_octets[octet];" in source
    assert source.count("input_octets[octet]") == 1
    assert "__shared__" not in source
    assert "__syncthreads" not in source
    assert "extern __shared__" not in source


def test_each_warp_accumulates_four_rows_in_the_pair8_order() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "warp * kRowsPerWarp" in source
    for row in range(4):
        assert f"const auto* weight{row}" in source
        assert f"float accumulator{row} = 0.0f;" in source
        assert f"const uint4 w{row} = weight{row}[octet];" in source
        assert len(
            re.findall(rf"accumulator{row}\s*=\s*__fmaf_rn\(", source)
        ) == 8
        assert f"fr13_reduce_full_warp(accumulator{row}, lane)" in source
    assert source.count("const uint4 x = input_octets[octet];") == 1
    assert source.count("__uint_as_float(") == 40
    assert source.count("__shfl_down_sync(") == 5
    assert source.count("__fadd_rn(") == 5
    for stride in (16, 8, 4, 2, 1):
        assert f", {stride}, kLanes)" in source
    assert source.count("__float2bfloat16_rn(") == 4
    assert "atomicAdd" not in source


def test_static_work_reduces_input_loads_and_warps_without_weight_duplication() -> None:
    octets = 5120 // 8
    ctas = 65536 // 32
    baseline_warps_per_cta = 32
    candidate_warps_per_cta = 8
    rows_per_warp = 4
    lanes = 32
    lane_iterations = octets // lanes

    baseline_input_loads_per_cta = baseline_warps_per_cta * lanes * lane_iterations
    candidate_input_loads_per_cta = candidate_warps_per_cta * lanes * lane_iterations
    candidate_weight_loads_per_cta = (
        candidate_warps_per_cta * rows_per_warp * lanes * lane_iterations
    )

    assert baseline_input_loads_per_cta == 20480
    assert candidate_input_loads_per_cta == 5120
    assert baseline_input_loads_per_cta // candidate_input_loads_per_cta == 4
    assert candidate_weight_loads_per_cta == 20480
    assert ctas * baseline_warps_per_cta == 65536
    assert ctas * candidate_warps_per_cta == 16384


def test_bf16_bit_expansion_matches_fp32_for_every_non_nan_pattern() -> None:
    raw = torch.arange(1 << 16, dtype=torch.int32)
    bf16 = raw.to(torch.int16).view(torch.bfloat16)
    actual = bf16.float().view(torch.int32)
    expected = raw << 16
    is_nan = ((raw & 0x7F80) == 0x7F80) & ((raw & 0x007F) != 0)

    assert torch.equal(actual[~is_nan], expected[~is_nan])


def test_cuda_op_is_out_variant_with_strict_k64_geometry() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert (
        "gemvx_m1_warp4_globalx_pair8bits_out(Tensor(a!) output, "
        "Tensor input, Tensor weight) -> ()" in source
    )
    assert "input.sizes() == at::IntArrayRef({1, kHidden})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({1, kVocab})" in source
    assert "weight must be contiguous [65536,5120]" in source
    assert "warp4 global-x inputs must be 16-byte aligned" in source
    assert "alignof(uint4)" in source
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
    assert '"block": [32, 8, 1]' in source
    assert '"static_shared_bytes": 0' in source
    assert '"output_rows_per_cta": 32' in source
    assert '"warps_per_cta": 8' in source
    assert '"output_rows_per_warp": 4' in source
    assert '"input_global_loads_per_cta": 5120' in source
    assert '"lane_input_global_iterations": 20' in source
    assert '"lane_weight_load_iterations": 80' in source
    assert '"lane_fma_iterations": 640' in source
    assert '"packed_unpack": "BF16 bits shifted/masked into exact FP32 bits"' in source
