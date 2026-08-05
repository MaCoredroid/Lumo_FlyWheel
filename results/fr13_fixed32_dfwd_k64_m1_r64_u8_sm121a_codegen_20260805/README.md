# Fixed32 DFWD K64 M1 R64-U8 static codegen

Status: **static SM121a codegen pass, default off and runtime unwired**. No GPU
kernel execution, Docker workload, real task, byte gate, timing, acceptance,
or production admission was performed.

Historical real-task attribution ranks the five BF16 K64 drafter heads at
about 26.227 ms/event against a 12.291 ms weight-read floor. That leaves about
13.936 ms/event of head excess, substantially larger than the approximately
2.289 ms/event projection-scheduler excess. These values only rank the kernel
work; they are not current candidate measurements.

## Kernel change

The R64-U8 candidate retains the B1 K64 M1 geometry, 1,024 CTAs, 1,024 threads
per CTA, 64 output rows per CTA, 16 lanes per output row, one FP32 accumulator,
and the width-16 `8+4+2+1` shuffle reduction. Each lane still executes the same
320 `__fmaf_rn` operations over exactly the same K indices in exactly the same
order. The only arithmetic-loop change is an explicit eight-step body: it
exposes eight independent weight loads before the dependent FMA chain and pays
the loop backedge once per eight products.

The source test enumerates all 16 lane index sequences and proves equality with
the U1 sequence. The source also uses explicit round-to-nearest FMA, add, and
BF16 conversion intrinsics. This makes byte equality the expected first live
gate, although only real execution can establish it.

## SM121a codegen

The exact source was compiled device-only with CUDA 13.0.88 for `sm_121a`, then
decoded with CUDA 13.0.85 `nvdisasm`. No GPU device or Docker was used.

| Metric | R64 U1 static baseline | R64 U8 candidate |
| --- | ---: | ---: |
| Registers/thread | 18 | 29 |
| Stack/local/shared bytes | 0/0/0 | 0/0/0 |
| Steady loop instructions | 11 | 48 |
| Steady loop iterations/row | 320 | 40 |
| Dynamic loop instructions/row | 3,520 | 1,920 |
| Weight-load window/warp | 1 | 8 |

The dynamic instruction model therefore falls by 1,600 instructions per row,
or 45.45%. Candidate SASS contains no CTA barrier, local load/store, call, or
atomic instruction. The 29-register footprint still fits the same 1,024-thread
CTA resource envelope, but only live profiling can establish occupancy and
performance.

The comparator is the earlier default-off R64 U1 static artifact from source
checkpoint `fc744b40009c88f873a53edb2c7e37c2bf81154f`. It was not GPU- or
task-qualified and is not asserted to be the current production route. This
package uses it only to isolate the loop-unroll codegen delta.

## Qualification boundary

Before any promotion or performance claim:

1. Replay the source checkpoint onto the then-current pushed `main` and rebuild
   a linked extension in the pinned deployed Torch/CUDA environment.
2. Add a default-off B1-only selector authenticated to the exact source and
   binary; keep production and all other candidate selectors off.
3. Run one real SWE-Verified B1 task and compare all 65,536 BF16 logits at root
   and MTP depths 1 through 4 while serving the incumbent output.
4. If all five sites are byte-equal, run clean real-task production timing on
   the standing exact4 set and confirm on exact16. If bytes differ, classify it
   as a quality-changing lossless drafter candidate and require the full
   acceptance/TPS curve before deciding whether to retain it.
5. This M1 kernel does not cover B4. B4 needs a separately qualified M4 head
   route and clean exact4/exact16 evidence.

Source checkpoint: `06f45f4bde9bfaf9e8e84eb9d2867d81ba55857d`.

