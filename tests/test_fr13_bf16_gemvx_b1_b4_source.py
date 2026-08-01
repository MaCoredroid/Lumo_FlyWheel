from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_m1.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_b1_b4.py"


def _candidate_source() -> str:
    source = CUDA.read_text(encoding="ascii")
    return source[source.index("// B1-B4 uses") :]


def test_b1_b4_kernel_reuses_each_weight_across_request_rows() -> None:
    source = _candidate_source()

    assert "template <int kBatch>" in source
    assert "static_assert(kBatch >= 1 && kBatch <= kMaxBatch);" in source
    k_loop = source.index("for (int k = lane; k < kHidden; k += kLanes)")
    weight_load = source.index(
        "const float w = __bfloat162float(weight[row * kHidden + k]);",
        k_loop,
    )
    batch_loop = source.index(
        "for (int batch = 0; batch < kBatch; ++batch)", weight_load
    )
    assert weight_load < batch_loop
    assert "input[batch * kHidden + k]" in source[batch_loop:]
    assert (
        "accumulators[batch] = __fmaf_rn(x, w, accumulators[batch]);"
        in source[batch_loop:]
    )
    assert source.count("atomicAdd") == 0


def test_b1_b4_kernel_keeps_stock_order_per_output_logit() -> None:
    source = _candidate_source()

    assert "#pragma unroll 1\n  for (int k = lane;" in source
    assert source.count("__fadd_rn(") == 4
    for stride in (8, 4, 2):
        assert f"if (lane < {stride})" in source
    assert "const float reduced_sum =" in source
    assert "__fmaf_rn(alpha, reduced_sum, beta)" in source
    assert (
        "output[batch * kVocab + row] = __float2bfloat16_rn(sum);"
        in source
    )


def test_b1_b4_op_is_one_batch_specialized_launch_per_head() -> None:
    source = _candidate_source()

    assert (
        "gemvx_b1_b4_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()"
        in source
    )
    assert "input.size(0) >= 1" in source
    assert "input.size(0) <= kMaxBatch" in source
    assert "input.stride(0) == kHidden" in source
    assert "output.stride(0) == kVocab" in source
    for batch in (1, 2, 3, 4):
        assert f"launch_b1_b4<{batch}>(output, input, weight);" in source
    assert "<<<kCtas, block, shared_bytes" in source


def test_b1_b4_builder_is_pinned_and_explicitly_unqualified() -> None:
    source = BUILDER.read_text(encoding="ascii")
    assert ast.parse(source) is not None
    assert 'EXPECTED_TORCH = "2.10.0+cu130"' in source
    assert 'EXPECTED_CUDA = "13.0"' in source
    assert 'EXPECTED_ARCH = "12.1a"' in source
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"byte_equality_claim": False' in source
    assert '"performance_measurement": False' in source
    assert '"production_default_enabled": False' in source
    assert '"supported_batch_sizes": [1, 2, 3, 4]' in source
    assert '"logical_weight_element_loads_per_head": 1271398400' in source
    assert '"candidate_launches_per_head": 1' in source
    assert '"cuda_graph_batch_specialization": True' in source
