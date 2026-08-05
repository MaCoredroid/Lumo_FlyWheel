from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu"


def test_source_is_strict_fixed32_k64_m4_r64_u8() -> None:
    source = CUDA.read_text(encoding="ascii")
    for declaration in (
        "constexpr int kHidden = 5120;",
        "constexpr int kVocab = 65536;",
        "constexpr int kBatch = 4;",
        "constexpr int kLanes = 16;",
        "constexpr int kRowsPerCta = 64;",
        "static_assert(kLanes * kRowsPerCta == 1024);",
        "static_assert(kCtas == 1024);",
    ):
        assert declaration in source
    assert "const dim3 block(kLanes, kRowsPerCta, 1);" in source
    assert "<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>" in source
    assert "FR13_DEVICE_CODEGEN_ONLY" in source


def test_u8_preserves_each_request_lane_sequence() -> None:
    for lane in range(16):
        incumbent = list(range(lane, 5120, 16))
        candidate = [
            k + step * 16
            for k in range(lane, 5120, 16 * 8)
            for step in range(8)
        ]
        assert candidate == incumbent
        assert len(candidate) == 320


def test_four_requests_share_weights_but_not_accumulators() -> None:
    source = CUDA.read_text(encoding="ascii")
    loop = source[source.index("#pragma unroll 1") : source.index("#define FR13_REDUCE_STEP")]
    assert loop.count("const float w =") == 1
    for batch in range(4):
        assert f"const float x{batch} =" in loop
        assert (
            f"accumulator{batch} = __fmaf_rn(x{batch}, w, accumulator{batch});"
            in loop
        )
    assert loop.count("__fmaf_rn(") == 4
    assert "float accumulators[" not in source
    assert "__syncthreads" not in source
    assert "atomicAdd" not in source


def test_reduction_and_output_are_exactly_per_request() -> None:
    source = CUDA.read_text(encoding="ascii")
    for stride in (8, 4, 2, 1):
        assert f"FR13_REDUCE_STEP({stride});" in source
    for batch in range(4):
        assert source.count(f"accumulator{batch}") == 7
        assert f"peer{batch} = __shfl_down_sync" in source
        assert f"accumulator{batch} = __fadd_rn(accumulator{batch}, peer{batch});" in source
    assert "output[row] =" in source
    assert "output[kVocab + row] =" in source
    assert "output[2 * kVocab + row] =" in source
    assert "output[3 * kVocab + row] =" in source


def test_op_is_separate_default_off_exact_b4_variant() -> None:
    source = CUDA.read_text(encoding="ascii")
    assert (
        "gemvx_m4_shuffle_r64_u8_out(Tensor(a!) output, Tensor input, "
        "Tensor weight) -> ()" in source
    )
    assert "input.sizes() == at::IntArrayRef({kBatch, kHidden})" in source
    assert "output.sizes() == at::IntArrayRef({kBatch, kVocab})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source
    assert "PRODUCTION" not in source
