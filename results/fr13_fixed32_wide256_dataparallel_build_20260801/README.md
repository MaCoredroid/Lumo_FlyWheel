# B1 wide256 data-parallel CUTLASS candidate

Status: compiled and statically verified; real-task byte gate pending. This is
not a performance or acceptance result.

## Why forced Stream-K failed raw-byte equality

CUTLASS `ReductionMode::Deterministic` makes a split reduction repeatable; it
does not preserve the stock accumulator sequence. Forced
`DecompositionMode::StreamK` partitions an output tile's K tiles across CTAs.
The first partial is stored to workspace, intermediate CTAs reduce their
partial FP32 accumulators into that workspace, and the final CTA performs a
workspace `load_add` before the BF16 epilogue. This changes floating-point
association from stock's one-CTA K accumulation and therefore cannot generally
be raw-byte equal to stock.

`scheduler.splits = 1` disables explicit split-K, but it does not disable the
forced Stream-K decomposition. There is no scheduler flag that makes genuine
K-split Stream-K reproduce the stock single-CTA reduction order.

## Exact-safe replacement

The replacement keeps the B1 swapped-AB geometry and widens only its output
tile from `128x32x128` to `256x32x128`. It instantiates the original
`cutlass_3x_gemm_fp8_blockwise` template, so CUTLASS uses the stock
one-CTA-per-output-tile scheduler and `StageCountAutoCarveout`. Each CTA owns the
complete K range; there is no reduction workspace or cross-CTA fixup.

Selectors:

- Diagnostic: `wide256_dataparallel_byte_ab`
- Production after qualification: `wide256_dataparallel`

Candidate binary:

```text
/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.wide256_dataparallel_b1_gate_ready.abi3.so
sha256=5b921ab7b428f2c5cfeefc0daed0314ff903d73bb0d4f8a790b17234c9d60890
bytes=112787936
mode=0555
runpath=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64
```

Static `cuobjdump --dump-resource-usage` comparison found exact symbol and
resource matches for all six baseline stock kernels. The two new half/BF16
wide256 kernels both use `REG=168`, `STACK=0`, `SHARED=1024`, and
`CONSTANT[0]=2560`.

The next action is the one-real-SWE B1 raw-byte gate on
`astropy__astropy-12907`. Timing remains forbidden until that gate passes.
