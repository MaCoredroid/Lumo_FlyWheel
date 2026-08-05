from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu"
CHECKER = REPO / "scripts" / "fr13_check_bf16_gemvx_k64_m1_shuffle_r64_u8_codegen.py"
ARTIFACT = (
    REPO / "results" / "fr13_fixed32_dfwd_k64_m1_r64_u8_sm121a_codegen_20260805"
)


def load_checker():
    spec = importlib.util.spec_from_file_location("fr13_r64_u8_codegen", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_is_strict_fixed32_k64_m1_r64_u8() -> None:
    source = CUDA.read_text(encoding="ascii")
    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kLanes = 16;" in source
    assert "constexpr int kRowsPerCta = 64;" in source
    assert "static_assert(kLanes * kRowsPerCta == 1024);" in source
    assert "static_assert(kCtas == 1024);" in source
    assert "const dim3 block(kLanes, kRowsPerCta, 1);" in source
    assert "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>" in source
    assert "FR13_DEVICE_CODEGEN_ONLY" in source


def test_u8_visits_the_incumbent_lane_partition_in_exact_order() -> None:
    for lane in range(16):
        incumbent = list(range(lane, 5120, 16))
        candidate = [
            k + step * 16
            for k in range(lane, 5120, 16 * 8)
            for step in range(8)
        ]
        assert candidate == incumbent
        assert len(candidate) == 320


def test_u8_keeps_one_ordered_accumulator_and_reduction_tree() -> None:
    source = CUDA.read_text(encoding="ascii")
    assert "#pragma unroll 1" in source
    assert "k += kLanes * 8" in source
    assert "for (int step = 0; step < 8; ++step)" in source
    assert source.count("float accumulator = 0.0f;") == 1
    assert source.count("accumulator = __fmaf_rn(x, w, accumulator);") == 1
    assert source.count("__shfl_down_sync(") == 4
    assert source.count("__fadd_rn(") == 4
    for stride in (8, 4, 2, 1):
        assert f", {stride}, kLanes)" in source
    assert "const float sum = __fmaf_rn(alpha, reduced_sum, beta);" in source
    assert "output[row] = __float2bfloat16_rn(sum);" in source
    assert "atomicAdd" not in source
    assert "__syncthreads" not in source


def test_op_is_separate_default_off_out_variant() -> None:
    source = CUDA.read_text(encoding="ascii")
    assert (
        "gemvx_m1_shuffle_r64_u8_out(Tensor(a!) output, Tensor input, "
        "Tensor weight) -> ()" in source
    )
    assert "input.sizes() == at::IntArrayRef({1, kHidden})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({1, kVocab})" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source
    assert "TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_head, library)" in source
    assert "FR13_DRAFT_HEAD_M1_R64_U8" not in source
    assert "PRODUCTION" not in source


def test_codegen_checker_accepts_pinned_u8_and_u1_shapes() -> None:
    module = load_checker()
    result = module.audit(
        (ARTIFACT / "candidate_sass.txt").read_text(encoding="ascii"),
        (ARTIFACT / "candidate_resource.txt").read_text(encoding="ascii"),
        (ARTIFACT / "baseline_r64_u1_sass.txt").read_text(encoding="ascii"),
        (ARTIFACT / "baseline_r64_u1_resource.txt").read_text(encoding="ascii"),
    )
    assert result["dynamic_loop_instruction_delta"] == -1600
