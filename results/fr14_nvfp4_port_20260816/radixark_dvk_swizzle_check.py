#!/usr/bin/env python3
"""Does the FR13 DVK row-slice survive the NVFP4 block-scale swizzle?

WHY THIS HAD TO BE ASKED. The staged design note claimed the DVK slice is "row
aligned by construction, because the scales are [out, in/16] -- index_select on
rows works for weight AND scale". That is true of the tensor ON DISK, but it is
NOT the tensor `_fr13_dvk_prepare` walks. Between load and slice,
`ModelOptNvFp4LinearMethod.process_weights_after_loading` hands the layer to
`FlashInferCutlassNvFp4LinearKernel.process_weights_after_loading`
(`model_executor/kernels/linear/nvfp4/flashinfer.py:42-50`), which does::

    layer.weight_scale = swizzle_blockscale(layer.weight_scale.data)
    layer.weight, layer.weights_padding_cols = pad_nvfp4_weight_for_cutlass(...)

`swizzle_blockscale`
(`model_executor/layers/quantization/utils/nvfp4_utils.py:13-53`) keeps the
LOGICAL shape [248320, 320] but interleaves rows::

    padded = padded.reshape(B, M//128, 4, 32, K//4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous()

so a naive `index_select(0, rows)` on the swizzled scale is only valid if the
permutation never moves a row across a 128-row tile boundary.

THE ANSWER: it never does. Dim 1 (`M//128`) stays in position 1 through the
permute, and each m-tile contributes exactly `K_padded x 128` elements = 128
rows of the flattened result. So swizzled rows `[t*128, (t+1)*128)` are exactly
original tile `t`, permuted within itself, and the within-tile permutation is
identical for every tile.

That means the FR13 DVK block map's **128-id granularity -- chosen for fp8
block-scale alignment -- is exactly what makes the NVFP4 swizzled scale
sliceable**. `_fr13_dvk_prepare`'s existing
`index_select(0, blk[:,None]*128 + arange(128))` is CORRECT AS-IS against the
NVFP4 head; no de-swizzle/re-swizzle step is needed. A slice at any other
granularity would be silently wrong.

This script proves it numerically at the real [248320, 320] shape rather than
by reading the source, using a pure-numpy replica of the permutation (no GPU,
no torch, no model weights).

    python3 radixark_dvk_swizzle_check.py
"""
import sys

import numpy as np


def round_up(x, a):
    return -(-x // a) * a


def swizzle(scale: np.ndarray) -> np.ndarray:
    """Replica of vllm nvfp4_utils.swizzle_blockscale, 2-D path."""
    M, K = scale.shape
    Mp, Kp = round_up(M, 128), round_up(K, 4)
    p = np.zeros((1, Mp, Kp), dtype=scale.dtype)
    p[0, :M, :K] = scale
    p = p.reshape(1, Mp // 128, 4, 32, Kp // 4, 4)
    s = np.transpose(p, (0, 1, 4, 3, 2, 5))
    return np.ascontiguousarray(s).reshape(Mp, Kp)


def main() -> int:
    rng = np.random.default_rng(0)
    ok_all = True

    # The real lm_head block-scale geometry: [vocab, hidden/16].
    for M, K in ((1280, 320), (248320, 320)):
        full = rng.integers(0, 255, size=(M, K), dtype=np.uint8)
        sw = swizzle(full)
        assert sw.shape == (M, K), sw.shape

        n_blocks = min(512, M // 128)
        blocks = rng.choice(M // 128, size=n_blocks, replace=False)
        idx = (blocks[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)

        ok = np.array_equal(sw[idx], swizzle(full[idx]))
        ok_all &= ok
        print(
            f"M={M:>6} K={K}  {n_blocks:>4} x 128-id blocks  "
            f"slice(swizzle(X)) == swizzle(slice(X)) : {ok}"
        )

    # Control: a slice that straddles a 128-row tile boundary must FAIL, or the
    # test above proves nothing.
    full = rng.integers(0, 255, size=(1280, 320), dtype=np.uint8)
    bad = np.arange(64, 192)
    misaligned_ok = np.array_equal(swizzle(full)[bad], swizzle(full[bad]))
    print(f"control: non-128-aligned slice commutes            : {misaligned_ok} "
          "(must be False)")
    ok_all &= not misaligned_ok

    print("VERDICT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
