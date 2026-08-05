from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc" / "fr13_bf16_gemm_k64_tc16x256x64_s2.cu"
BUILDER = ROOT / "scripts" / "fr13_build_bf16_gemm_k64_tc16x256x64_s2.py"
CHECKER = (
    ROOT / "scripts" / "fr13_check_bf16_gemm_k64_tc16x256x64_s2_codegen.py"
)


def _builder():
    spec = importlib.util.spec_from_file_location("fr13_tc_head_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checker():
    spec = importlib.util.spec_from_file_location("fr13_tc_head_checker", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_identity_and_exact_fixed_shapes() -> None:
    builder = _builder()
    source = SOURCE.read_text(encoding="ascii")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == builder.SOURCE_SHA256
    for declaration in (
        "constexpr int kHidden = 5120;",
        "constexpr int kVocab = 65536;",
        "constexpr int kBatch1 = 1;",
        "constexpr int kBatch4 = 4;",
        "constexpr int kThreadblockM = 16;",
        "constexpr int kThreadblockN = 256;",
        "constexpr int kThreadblockK = 64;",
        "constexpr int kWarpM = 16;",
        "constexpr int kWarpN = 64;",
        "constexpr int kWarpK = 64;",
        "constexpr int kStages = 2;",
        "static_assert(kLogicalCtas == 256);",
        "static_assert(kThreadsPerCta == 128);",
        "static_assert(kM1SharedStorageBytes == 69632);",
    ):
        assert declaration in source


def test_tensor_core_weight_layout_needs_no_conversion() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert "cutlass::arch::OpClassTensorOp" in source
    assert "cutlass::arch::Sm80" in source
    assert "cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling" in source
    assert "cutlass::gemm::GemmShape<kInstructionM, kInstructionN, kInstructionK>" in source
    assert "using WeightLayout = cutlass::layout::ColumnMajor;" in source
    assert "{weight_ptr, kHidden}" in source
    assert 'weight.strides() == at::IntArrayRef({kHidden, 1})' in source
    assert "permute" not in source.lower()
    assert "transpose" not in source.lower()
    assert "quant" not in source.lower()


def test_b1_b4_use_distinct_kernel_types_and_strict_ops() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert "struct M1IdentitySwizzle" in source
    assert "struct M4IdentitySwizzle" in source
    assert "using M1TensorHead = FixedK64TensorHead<M1IdentitySwizzle>;" in source
    assert "using M4TensorHead = FixedK64TensorHead<M4IdentitySwizzle>;" in source
    assert "fr13_launch_tensor_head<M1TensorHead, kBatch1>" in source
    assert "fr13_launch_tensor_head<M4TensorHead, kBatch4>" in source
    assert "gemm_m1_tc16x256x64_s2_out" in source
    assert "gemm_m4_tc16x256x64_s2_out" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source


def test_a_and_weight_keep_128_bit_vectors() -> None:
    source = SOURCE.read_text(encoding="ascii")
    gemm = source[source.index("using FixedK64TensorHead") : source.index("using M1TensorHead")]
    assert "kStages,\n    8,\n    8>;" in gemm


def test_traffic_model_preserves_mandatory_weight_once_and_reduces_input() -> None:
    model = _builder().traffic_model()
    weight = 65536 * 5120 * 2
    assert model["weight_bytes_per_call"] == weight == 671088640
    assert model["b1"]["duplicated_input_read_bytes"] == 2621440
    assert model["b4"]["duplicated_input_read_bytes"] == 10485760
    assert model["b1"]["global_read_bytes"] == 673710080
    assert model["b4"]["global_read_bytes"] == 681574400
    pair8_reads = {"b1": 838860800, "b4": 1342177280}
    assert 1.0 - model["b1"]["global_read_bytes"] / pair8_reads["b1"] == (
        pytest.approx(0.196875)
    )
    assert 1.0 - model["b4"]["global_read_bytes"] / pair8_reads["b4"] == (
        pytest.approx(0.4921875)
    )


def test_padded_compute_is_explicitly_modeled_not_measured() -> None:
    model = _builder().compute_model()
    assert model["tensor_core_executed_flops_per_call"] == 10737418240
    assert model["useful_flops"] == {"b1": 671088640, "b4": 2684354560}
    assert model["padding_factor"] == {"b1": 16, "b4": 4}
    assert "modeled only" in model["claim"]


def test_candidate_is_proposal_only_and_does_not_touch_target_sampling() -> None:
    combined = SOURCE.read_text(encoding="ascii") + BUILDER.read_text(encoding="ascii")
    assert '"proposal_only": True' in combined
    assert '"target_authority_changed": False' in combined
    assert "rejection_sampler" not in combined
    assert "target_logits" not in combined
    assert '"production_default_enabled": False' in combined
    assert '"runtime_wired": False' in combined


def _synthetic_sass() -> str:
    checker = _checker()
    operations: list[str] = []
    for operation, count in checker.EXPECTED_COUNTS.items():
        operations.extend([operation] * count)
    operations.extend(["NOP"] * (checker.EXPECTED_STATIC_INSTRUCTIONS - len(operations)))
    bodies = []
    for batch in (1, 4):
        lines = [
            f"\t\tFunction : fake_M{batch}IdentitySwizzle_kernel",
            '\t.headerflags\t@"EF_CUDA_ACCELERATORS EF_CUDA_SM121"',
        ]
        lines.extend(
            f"        /*{index:04x}*/                   {operation} R0 ;"
            for index, operation in enumerate(operations)
        )
        bodies.append("\n".join(lines))
    return "\n".join(("\t.target\tsm_121a", *bodies))


def _synthetic_resource() -> str:
    records = []
    for batch in (1, 4):
        records.append(
            "\n".join(
                (
                    f" Function fake_M{batch}IdentitySwizzle_kernel:",
                    "  REG:168 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:1264",
                )
            )
        )
    return "\n".join(("arch = sm_121a", *records))


def test_codegen_checker_accepts_exact_fixed_tensor_core_shape() -> None:
    audit = _checker().audit(_synthetic_sass(), _synthetic_resource())
    assert audit["status"] == "STATIC_CODEGEN_PASS_UNQUALIFIED"
    assert audit["resources"]["b1"]["registers_per_thread"] == 168
    assert audit["resources"]["b4"]["launch_dynamic_shared_bytes"] == 69632
    assert audit["sass"]["b1_selected_instruction_counts"][
        "HMMA.16816.F32.BF16"
    ] == 32
    assert audit["sass"]["b1_static_instructions"] == 760
    assert audit["sass"]["b1_selected_instruction_counts"][
        "LDG.E.LTC128B.128"
    ] == 34
    assert audit["only_alpha_epilogue_codegen_delta"][
        "static_instruction_reduction_fraction"
    ] == pytest.approx(1.0 - 760 / 952)
    assert audit["logical_global_traffic_model"]["b1"][
        "pair8_global_read_reduction_fraction"
    ] == pytest.approx(0.196875)
    assert audit["logical_global_traffic_model"]["b4"][
        "pair8_global_read_reduction_fraction"
    ] == pytest.approx(0.4921875)


def test_codegen_checker_rejects_hmma_and_resource_drift() -> None:
    sass = _synthetic_sass()
    mutated_sass = sass.replace("HMMA.16816.F32.BF16", "FFMA", 1)
    with pytest.raises(RuntimeError, match="B1 HMMA"):
        _checker().audit(mutated_sass, _synthetic_resource())

    resource = _synthetic_resource()
    mutated_resource = re.sub(r"REG:168", "REG:169", resource, count=1)
    with pytest.raises(RuntimeError, match="B1 resources"):
        _checker().audit(sass, mutated_resource)
