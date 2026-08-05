# Fixed32 DFWD K64 B1/B4 Tensor Core candidate

Status: **default off, runtime unwired, static codegen pass**. This package
contains no GPU execution, real-task correctness, timing, acceptance, or
hardware-floor claim.

## Kernel

The candidate computes the exact fixed projection shapes
`[B, 5120] x [65536, 5120] -> [B, 65536]` for `B in {1, 4}` with BF16 CUTLASS
Tensor Core kernels. The threadblock, warp, and HMMA shapes are respectively
`16x256x64`, `16x64x64`, and `16x8x16`, with two pipeline stages, 128 threads
per CTA, and 256 logical CTAs. B1 and B4 have distinct compiled kernel symbols.

The existing row-major `[65536, 5120]` weight allocation is directly viewed as
the column-major CUTLASS B operand. No transpose, layout conversion, weight
copy, quantization, or reduced vocabulary is introduced. The kernel emits
proposal logits only and cannot alter target-authoritative rejection sampling.

`ScaleType::OnlyAlphaScaling` removes the generic `beta * C` epilogue path at
compile time because this projection always uses `beta=0`.

## Static result

| Metric | B1 | B4 |
| --- | ---: | ---: |
| Registers/thread | 168 | 168 |
| Launch dynamic shared bytes | 69,632 | 69,632 |
| Stack/local/spill bytes | 0 | 0 |
| Static instructions | 760 | 760 |
| BF16 HMMA instructions | 32 | 32 |
| Global-load instructions | 36 | 36 |
| Global-store instructions | 4 | 4 |
| Deferred barriers | 6 | 6 |

Compared with the separately compiled generic CUTLASS epilogue, the only-alpha
specialization reduced static instructions from 952 to 760, global loads from
42 to 36, global stores from 8 to 4, barriers from 10 to 6, and the linked
extension from 314,520 to 248,984 bytes. These are compiler-output deltas, not
performance measurements.

## Traffic model

The mandatory weight read is 671,088,640 bytes per call. With 256 vocabulary
tiles, the modeled global reads are 673,710,080 bytes for B1 and 681,574,400
bytes for B4. Relative to the prior pair8 model, those are reductions of
19.6875% and 49.21875%. The Tensor Core tile executes 10,737,418,240 padded
FLOPs per call, a 16x B1 and 4x B4 padding factor. All byte and FLOP figures are
algorithmic models; they are not measured DRAM traffic or achieved throughput.

## Reproducibility

Two independent CUDA 13.0 `sm_121a` builds from pinned CUTLASS commit
`da5e086dab31d63815acafdac9a9c5893b1c69e2` produced byte-identical cubins,
SASS, and resource reports. The linked host containers have equal size but
different hashes, so linked-ELF reproducibility is explicitly false. Both
extensions registered the B1 and B4 Torch ops with `CUDA_VISIBLE_DEVICES`
empty.

## Next gate

Wire a source-authenticated, default-off selector, then run real SWE-Verified
B1 and B4 proposal/target-authority gates. Only candidate-served exact4 and
exact16 measurements may establish throughput or hardware-floor progress.

Source checkpoint: `a618b3ad1342929adb8470529ac8bdb85d67d2c5`.
Integrated main checkpoint: `addc1af4cf86ad24f70d367b68e0afecbd4f87d8`.
