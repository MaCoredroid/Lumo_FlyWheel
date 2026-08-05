from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shared_r64_u8.cu"


def test_shared_u8_retains_the_fast_lane_accumulation_order() -> None:
    source = CUDA.read_text(encoding="ascii")
    for lane in range(16):
        incumbent = list(range(lane, 5120, 16))
        candidate = [
            k + step * 16
            for k in range(lane, 5120, 16 * 8)
            for step in range(8)
        ]
        assert candidate == incumbent
    assert source.count("float accumulator = 0.0f;") == 1
    assert source.count("accumulator = __fmaf_rn(x, w, accumulator);") == 1


def test_shared_u8_matches_the_incumbent_reduction_association() -> None:
    source = CUDA.read_text(encoding="ascii")
    assert "constexpr int kRowsPerCta = 64;" in source
    assert "constexpr int kSharedRowStride = 17;" in source
    assert "kRowsPerCta * kSharedRowStride * sizeof(float)" in source
    assert source.count("__syncthreads();") == 4
    assert source.count("__fadd_rn(") == 4
    assert "row_partials[lane + 8]" in source
    assert "row_partials[lane + 4]" in source
    assert "row_partials[lane + 2]" in source
    assert "row_partials[0], row_partials[1]" in source
    assert "__shfl_down_sync" not in source
    assert "output[row] = __float2bfloat16_rn(sum);" in source


def test_shared_u8_is_a_separate_default_off_sm121_op() -> None:
    source = CUDA.read_text(encoding="ascii")
    assert "FR13_DEVICE_CODEGEN_ONLY" in source
    assert "static_assert(kLanes * kRowsPerCta == 1024);" in source
    assert "static_assert(kCtas == 1024);" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source
    assert (
        "gemvx_m1_shared_r64_u8_out(Tensor(a!) output, Tensor input, "
        "Tensor weight) -> ()" in source
    )
    assert "FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE" not in source
    assert "PRODUCTION" not in source
