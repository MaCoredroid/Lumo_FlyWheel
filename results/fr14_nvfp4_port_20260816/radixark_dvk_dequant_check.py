#!/usr/bin/env python3
"""FR14 arm B: does the CHUNKED, SLICED NVFP4 dequant equal the reference?

`radixark_dvk_swizzle_check.py` proved the first half: a 128-id-aligned row
slice commutes with `swizzle_blockscale`, so `_fr13_dvk_prepare`'s existing
`index_select` is correct as-is against the NVFP4 head. This script proves the
two things the PHASE-1 dequant adds on top:

1. **De-swizzle after slicing.** The shim hands
   `nvfp4_emulation_utils.dequantize_to_dtype(..., swizzle=True)` a tensor that
   was swizzled at the FULL [248320, 320] shape and then row-sliced to
   [65536, 320].  `convert_swizzled_to_linear` will re-derive the tile
   structure from the SLICED `m`, not the original one.  That is only sound if
   the swizzle is per-128-row-tile and identical in every tile -- which it is,
   but the slice makes the claim non-obvious enough to be worth a number.

2. **Chunking.** The shim dequantises in 8192-row chunks so a whole-head pass
   cannot spike several GB of transient int64 on a unified-memory pool that
   already holds the 46 GiB KV reservation.  Each chunk is de-swizzled
   independently, so chunk boundaries must also fall on 128-row tiles.

Both are checked numerically at the real [248320, 320] / [248320, 2560] shapes
with pure numpy replicas of vLLM's own `swizzle_blockscale`,
`convert_swizzled_to_linear` and `break_fp4_bytes` -- no GPU, no torch, no model
weights.  Controls that MUST fail are included: without them the checks prove
nothing.

    python3 radixark_dvk_dequant_check.py
"""

import sys

import numpy as np

VOCAB = 248_320
HIDDEN = 5_120
BLOCK = 16
DRAFT_K = 65_536
CHUNK = 8_192

# vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py:18-20
kE2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def round_up(x, a):
    return -(-x // a) * a


def swizzle_blockscale(scale: np.ndarray) -> np.ndarray:
    """Replica of vllm nvfp4_utils.swizzle_blockscale, 2-D path."""
    m, k = scale.shape
    mp, kp = round_up(m, 128), round_up(k, 4)
    p = np.zeros((1, mp, kp), dtype=scale.dtype)
    p[0, :m, :k] = scale
    p = p.reshape(1, mp // 128, 4, 32, kp // 4, 4)
    s = np.transpose(p, (0, 1, 4, 3, 2, 5))
    return np.ascontiguousarray(s).reshape(mp, kp)


def convert_swizzled_to_linear(sw: np.ndarray, m, k, block_size) -> np.ndarray:
    """Replica of vllm nvfp4_emulation_utils.convert_swizzled_to_linear."""
    m_tiles = (m + 127) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = sw.reshape(1, m_tiles, k_tiles, 32, 4, 4)
    tmp = np.transpose(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def break_fp4_bytes(packed: np.ndarray) -> np.ndarray:
    """Replica of vllm nvfp4_emulation_utils.break_fp4_bytes."""
    m, n = packed.shape
    flat = packed.reshape(-1)
    high = (flat & 0xF0) >> 4
    low = flat & 0x0F
    combined = np.stack((low, high), axis=1).reshape(-1)
    signs = (combined & 0x08).astype(bool)
    abs_vals = (combined & 0x07).astype(np.int64)
    values = kE2M1[abs_vals] * np.where(signs, -1.0, 1.0).astype(np.float32)
    return values.reshape(m, n * 2)


def dequantize(packed: np.ndarray, sw_scale: np.ndarray, global_scale: float):
    """Replica of dequantize_to_dtype(..., swizzle=True), 2-D path."""
    m, packed_k = packed.shape
    k = packed_k * 2
    f32 = break_fp4_bytes(packed).reshape(-1, k // BLOCK, BLOCK)
    linear_sf = convert_swizzled_to_linear(sw_scale, f32.shape[0], k, BLOCK)
    sf = linear_sf.astype(np.float32) * global_scale
    return (f32 * sf[..., None]).reshape(m, k)


def main() -> int:
    rng = np.random.default_rng(20260816)
    ok_all = True

    # ---------------------------------------------------------------- setup
    # e4m3 block scales are stored as raw bytes; any byte pattern is a legal
    # scale, so random uint8 is the right stress here (we never interpret the
    # e4m3 bits, only that the same bytes come back in the same places).
    full_scale = rng.integers(0, 255, size=(VOCAB, HIDDEN // BLOCK), dtype=np.uint8)
    full_packed = rng.integers(0, 255, size=(VOCAB, HIDDEN // 2), dtype=np.uint8)
    sw_full = swizzle_blockscale(full_scale)
    assert sw_full.shape == (VOCAB, HIDDEN // BLOCK), sw_full.shape

    # The DVK gather: 512 measured 128-id blocks -> 65,536 rows.
    blocks = rng.choice(VOCAB // 128, size=DRAFT_K // 128, replace=False)
    idx = (blocks[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)
    assert idx.size == DRAFT_K

    # ------------------------------------------------- 1. de-swizzle a slice
    # Reference: de-swizzle the WHOLE head, then take the rows.
    linear_full = convert_swizzled_to_linear(
        sw_full, VOCAB, HIDDEN, BLOCK
    )
    reference_rows = linear_full[idx]
    # What the shim actually does: slice the swizzled tensor, then hand the
    # slice to convert_swizzled_to_linear with the SLICED m.
    sliced_sw = sw_full[idx]
    sliced_linear = convert_swizzled_to_linear(sliced_sw, DRAFT_K, HIDDEN, BLOCK)
    ok = np.array_equal(sliced_linear, reference_rows)
    ok_all &= ok
    print(
        f"1. deswizzle(slice(sw))  == slice(deswizzle(sw))   "
        f"[{DRAFT_K} rows of {VOCAB}] : {ok}"
    )

    # ------------------------------------------------------ 2. chunk-by-chunk
    chunked = np.empty_like(sliced_linear)
    for lo in range(0, DRAFT_K, CHUNK):
        hi = lo + CHUNK
        chunked[lo:hi] = convert_swizzled_to_linear(
            sliced_sw[lo:hi], CHUNK, HIDDEN, BLOCK
        )
    ok = np.array_equal(chunked, reference_rows)
    ok_all &= ok
    print(
        f"2. chunked({CHUNK}) deswizzle == whole-slice deswizzle          "
        f"        : {ok}"
    )

    # ------------------------------------------ 3. full dequant, chunked vs whole
    # Restrict the dequant comparison to the sliced head (65,536 x 5,120 f32 is
    # 1.3 GB; the whole 248,320-row head would be 5 GB and proves nothing more).
    sliced_packed = full_packed[idx]
    global_scale = np.float32(0.0173)
    whole = dequantize(sliced_packed, sliced_sw, global_scale)
    chunked_dq = np.empty_like(whole)
    for lo in range(0, DRAFT_K, CHUNK):
        hi = lo + CHUNK
        chunked_dq[lo:hi] = dequantize(
            sliced_packed[lo:hi], sliced_sw[lo:hi], global_scale
        )
    ok = np.array_equal(chunked_dq, whole)
    ok_all &= ok
    print(f"3. chunked dequant       == whole-slice dequant  (bitwise)     : {ok}")

    # The dequant must also agree with dequantising the FULL head and then
    # taking the rows -- i.e. the slice is legal all the way through, not just
    # self-consistent. Done on a small stand-in head so the reference fits.
    small_vocab = 1_280
    small_scale = rng.integers(
        0, 255, size=(small_vocab, HIDDEN // BLOCK), dtype=np.uint8
    )
    small_packed = rng.integers(
        0, 255, size=(small_vocab, HIDDEN // 2), dtype=np.uint8
    )
    small_sw = swizzle_blockscale(small_scale)
    small_idx = (
        rng.choice(small_vocab // 128, size=4, replace=False)[:, None] * 128
        + np.arange(128)[None, :]
    ).reshape(-1)
    ref = dequantize(small_packed, small_sw, global_scale)[small_idx]
    got = dequantize(small_packed[small_idx], small_sw[small_idx], global_scale)
    ok = np.array_equal(got, ref)
    ok_all &= ok
    print(f"4. dequant(slice(head))  == slice(dequant(head))  (bitwise)    : {ok}")

    # ------------------------------------------------------------- controls
    # A chunk boundary off the 128-row tile grid must BREAK, or checks 2-3
    # prove nothing about why 8192 was chosen.
    bad_chunk = 100
    try:
        bad = np.empty_like(sliced_linear[: bad_chunk * 4])
        for lo in range(0, bad_chunk * 4, bad_chunk):
            bad[lo : lo + bad_chunk] = convert_swizzled_to_linear(
                np.ascontiguousarray(sliced_sw[lo : lo + bad_chunk]),
                bad_chunk,
                HIDDEN,
                BLOCK,
            )
        control_ok = np.array_equal(bad, reference_rows[: bad_chunk * 4])
        control_note = "produced a WRONG result"
    except ValueError:
        # Stronger than a wrong answer: a chunk that is not a whole number of
        # 128-row tiles cannot even be reshaped into the tile layout, so the
        # de-swizzle refuses outright rather than mis-mapping rows.
        control_ok = False
        control_note = "refused (reshape into the 128-row tile layout is impossible)"
    print(
        f"control: chunk={bad_chunk} (not a multiple of 128) reproduces it"
        f"     : {control_ok} (must be False -- {control_note})"
    )
    ok_all &= not control_ok

    # A misaligned row GATHER must break too (the swizzle-check's control,
    # repeated here so this script stands alone).
    mis = np.arange(64, 192)
    mis_ok = np.array_equal(
        convert_swizzled_to_linear(
            np.ascontiguousarray(small_sw[mis]), mis.size, HIDDEN, BLOCK
        ),
        convert_swizzled_to_linear(small_sw, small_vocab, HIDDEN, BLOCK)[mis],
    )
    print(
        f"control: non-128-aligned row gather commutes                   "
        f": {mis_ok} (must be False)"
    )
    ok_all &= not mis_ok

    print("VERDICT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
