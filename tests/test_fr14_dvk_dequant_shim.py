"""FR14 arm B: the DVK Phase-1 dequant-at-slice shim.

``_fr13_dvk_prepare`` lives inside a ~5,000-line replacement STRING in
``scripts/fr10_phase4_patch_vllm_tree_gdn.py`` that the patcher injects into
vLLM's model runner inside the container. ``python -m py_compile`` on the
patcher therefore proves nothing about the injected code -- it only proves the
string literal is well formed. These tests compile the fragment for real, which
is the only thing standing between a typo and an ~8-minute boot that dies at
first forward.

They also pin the parts of the shim that are load-bearing for the FLOOR rather
than for correctness-in-general: the arm's whole premise is that the five K64
draft-head reads are 671,088,640 B of BF16, which is only true if the sliced
NVFP4 rows are dequantised exactly once, at boot, and nothing downstream can
still see a stale scale.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
BANNER = "[FR14_DVK_DEQUANT] phase1 nvfp4->bf16 at slice"


def _injected_fragment() -> str:
    """The string literal containing _fr13_dvk_prepare, as it will be injected."""
    source = PATCHER.read_text(encoding="utf-8")
    marker_line = next(
        (i for i, line in enumerate(source.splitlines(), 1) if BANNER in line),
        None,
    )
    assert marker_line is not None, f"{BANNER!r} not found in the patcher"
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING and token.start[0] <= marker_line <= token.end[0]:
            value = eval(token.string)  # noqa: S307 - a plain literal from our own repo
            assert isinstance(value, str)
            return value
    raise AssertionError("the DVK banner is not inside a string literal any more")


def test_injected_fragment_is_valid_python() -> None:
    """py_compile on the patcher cannot see this; a boot 8 minutes later can."""
    fragment = _injected_fragment()
    # The fragment is a method body at 8-space indent.
    compile(
        "class _W:\n    def _m(self):\n" + fragment + "\n        pass\n",
        "<fr10_injected_fragment>",
        "exec",
    )


def test_dequant_uses_vllms_own_kernel_math_not_a_reimplementation() -> None:
    """The BF16 rows must be the numbers the NVFP4 GEMM computes against.

    ``run_nvfp4_emulations`` -- vLLM's own emulation kernel -- dequantises with
    exactly this call. Reimplementing the e2m1 lookup and the global-scale
    convention by hand is how a silent scalar factor gets in.
    """
    fragment = _injected_fragment()
    assert "nvfp4_emulation_utils import" in fragment
    assert "dequantize_to_dtype as _fr13_dvkq_dequant" in fragment
    # swizzle=True, because FlashInferCutlassNvFp4LinearKernel swizzled the
    # scale. The emulation kernel passes False only because it never swizzles.
    assert "swizzle=True," in fragment
    assert "weight_global_scale" in fragment


def test_shim_is_left_as_a_bf16_k64_head_with_no_stale_quant_state() -> None:
    fragment = _injected_fragment()

    # The quant-only attributes are deleted, not merely ignored: a swizzled
    # scale sitting next to a BF16 weight is how a scale gets applied twice.
    for attribute in (
        '"weight_scale",',
        '"weight_global_scale",',
        '"input_global_scale",',
        '"input_global_scale_inv",',
        '"alpha",',
        '"weights_padding_cols",',
    ):
        assert attribute in fragment, attribute
    assert "delattr(_fr13_dvkq_sh, _fr13_dvkq_a)" in fragment

    # The declared widths follow the weight, and are cross-checked against it.
    assert "_fr13_dvkq_sh.output_size_per_partition = (" in fragment
    assert "_fr13_dvkq_sh.logical_widths = [_fr13_dvk_configured]" in fragment
    assert "!= [_fr13_dvkq_sh.output_size_per_partition]" in fragment
    assert "!= _fr13_dvkq_sh.weight.shape[0]" in fragment

    # The sealed FR13 sub-arms assert on the method's class NAME, so the shim
    # must actually carry an UnquantizedEmbeddingMethod, not just a BF16 weight.
    assert "UnquantizedEmbeddingMethod as _fr13_dvkq_um" in fragment
    assert "_fr13_dvkq_sh.quant_method = _fr13_dvkq_um()" in fragment


def test_dequant_is_fail_closed_not_a_silent_fallback() -> None:
    """A dequant failure must kill the boot, not quietly re-read the full head.

    If the shim dies, the run falls back to ``self.model.compute_logits``, i.e.
    five reads of the FULL 715,161,608 B head instead of five 671,088,640 B K64
    reads -- a different byte profile from the pinned floor. A wrong number is
    worse than no boot.
    """
    fragment = _injected_fragment()
    start = fragment.index("# FR14 ARM B -- DVK PHASE 1")
    end = fragment.index("if (\n                    _fr13_dh_fp8", start)
    block = fragment[start:end]

    assert "_fr13_dvk_dead = True" not in block
    assert "except" not in block
    assert block.count("raise RuntimeError(") >= 5

    # Non-NVFP4 heads are refused rather than dequantised on a guess, and a
    # head with no weight_scale at all (arm A, or the FP8-3.8 baseline) skips
    # the block entirely instead of failing.
    assert 'hasattr(self._fr13_dvk_shim, "weight_scale")' in fragment
    assert '!= "ModelOptNvFp4LinearMethod"' in block

    # CUTLASS row/col padding would materialise columns that are not weights.
    assert "_fr13_dvkq_pad != 0" in block


def test_lookup_table_is_moved_to_the_device_before_the_dequant() -> None:
    """BIRTH DEFECT, first device run (boot probe 20260817T011303Z).

    ``break_fp4_bytes`` indexes a MODULE-LEVEL ``kE2M1`` tensor that ships on
    CPU. The only thing that ever moves it is
    ``EmulationNvFp4LinearKernel.process_weights_after_loading`` -- and this arm
    runs the FlashInfer kernel, so that hook never fires and the table stays on
    CPU while the indices are on CUDA:

        RuntimeError: indices should be either on cpu or on the same device as
        the indexed tensor (cpu)

    The shim now does what the emulation kernel does. Upstream's own comment
    says why it must happen at prepare time and not inside the dequant:
    ``.to(device)`` is illegal during CUDA graph capture -- hence the explicit
    capture assert, which turns a confusing capture-time failure into a named
    one.
    """
    fragment = _injected_fragment()
    assert "kE2M1ToFloat_handle as _fr13_dvkq_lut" in fragment
    assert "_fr13_dvkq_lut.val.device != _fr13_dvkq_w.device" in fragment
    assert "_fr13_dvkq_lut.val.to(" in fragment
    assert "torch.cuda.is_current_stream_capturing()" in fragment

    # The move must precede the first dequant call, or it fixes nothing.
    assert fragment.index("_fr13_dvkq_lut.val.to(") < fragment.index(
        "_fr13_dvkq_bf16[_fr13_dvkq_lo:_fr13_dvkq_hi] = ("
    )


def test_chunking_respects_the_128_row_tile_the_slice_depends_on() -> None:
    fragment = _injected_fragment()
    assert "_fr13_dvkq_chunk = 8192" in fragment
    assert "_fr13_dvk_configured % _fr13_dvkq_chunk != 0" in fragment
    # 8192 is a whole number of the 128-row tiles the swizzle inversion needs,
    # and divides the K64 draft vocabulary.
    assert 8192 % 128 == 0
    assert 65_536 % 8192 == 0


def _swizzle(scale: np.ndarray) -> np.ndarray:
    """numpy replica of vllm nvfp4_utils.swizzle_blockscale (2-D path)."""
    m, k = scale.shape
    mp, kp = -(-m // 128) * 128, -(-k // 4) * 4
    padded = np.zeros((1, mp, kp), dtype=scale.dtype)
    padded[0, :m, :k] = scale
    padded = padded.reshape(1, mp // 128, 4, 32, kp // 4, 4)
    return np.ascontiguousarray(
        np.transpose(padded, (0, 1, 4, 3, 2, 5))
    ).reshape(mp, kp)


def _unswizzle(sw: np.ndarray, m: int, k: int, block: int = 16) -> np.ndarray:
    """numpy replica of nvfp4_emulation_utils.convert_swizzled_to_linear."""
    m_tiles = (m + 127) // 128
    f = block * 4
    k_tiles = (k + f - 1) // f
    tmp = sw.reshape(1, m_tiles, k_tiles, 32, 4, 4)
    tmp = np.transpose(tmp, (0, 1, 4, 3, 2, 5))
    return tmp.reshape(m_tiles * 128, k_tiles * f // block)[0:m, 0:k]


def test_sliced_and_chunked_deswizzle_equals_the_reference() -> None:
    """The claim the whole Phase-1 design rests on, at a cheap shape.

    The full-shape [248320, 320] proof with the real 512-block gather, the
    bitwise dequant comparison and both must-fail controls is
    results/fr14_nvfp4_port_20260816/radixark_dvk_dequant_check.py (20 s,
    ~10 GB); it is deliberately not run here.
    """
    rng = np.random.default_rng(0)
    vocab, hidden, chunk = 2_560, 512, 512
    scale = rng.integers(0, 255, size=(vocab, hidden // 16), dtype=np.uint8)
    sw = _swizzle(scale)
    linear = _unswizzle(sw, vocab, hidden)

    blocks = rng.choice(vocab // 128, size=8, replace=False)
    idx = (blocks[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)

    sliced = _unswizzle(np.ascontiguousarray(sw[idx]), idx.size, hidden)
    assert np.array_equal(sliced, linear[idx])

    chunked = np.empty_like(sliced)
    for lo in range(0, idx.size, chunk):
        chunked[lo : lo + chunk] = _unswizzle(
            np.ascontiguousarray(sw[idx][lo : lo + chunk]), chunk, hidden
        )
    assert np.array_equal(chunked, linear[idx])

    # Control: a gather off the 128-row grid must NOT commute, or the above
    # asserts nothing about why the block map's granularity matters.
    misaligned = np.arange(64, 192)
    assert not np.array_equal(
        _unswizzle(np.ascontiguousarray(sw[misaligned]), misaligned.size, hidden),
        linear[misaligned],
    )


def test_phase1_byte_count_matches_the_pinned_floor_term() -> None:
    """The shim's output IS the ledger's SUBSET_HEAD_BYTES, not a coincidence."""
    from scripts.fr13_hardware_floor_ledger import (
        DRAFT_VOCAB_ROWS,
        SUBSET_HEAD_BYTES,
    )

    assert DRAFT_VOCAB_ROWS == 65_536
    assert SUBSET_HEAD_BYTES == 65_536 * 5_120 * 2 == 671_088_640

    fragment = _injected_fragment()
    # The banner reports the realised byte count, so a boot log can be checked
    # against the floor without inference.
    assert "bytes={_fr13_dvkq_bf16.numel() * 2}" in fragment
