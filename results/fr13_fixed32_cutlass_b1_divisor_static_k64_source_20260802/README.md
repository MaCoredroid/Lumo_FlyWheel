# Fixed32 B1 K64 divisor-balanced static CUTLASS handoff

Status: source, host compile/link, and static binary audit pass. The candidate
is default off and is not wired for production, byte qualified, timed, or
acceptance valid. No GPU kernel, Docker container, synthetic probe, or task was
run for this handoff.

## Candidate

`divisor_static_stocktile` preserves the byte-clean B1 static candidate's
complete-output-tile math:

- physical M: 32;
- tile: `128x32x128`, cluster: `1x1x1`;
- scale granularity: `128x1x128`;
- cooperative SM120 mainloop, FP32 accumulation, and stock epilogue;
- full ordered K traversal per output tile;
- no split K, workspace, reduction, or fixup kernel.

Only the persistent launch width changes. Starting from CUTLASS's at-most-48
CTA static grid, the scheduler chooses the widest exact divisor of the logical
tile count that is at least 28 CTAs. For the pinned real K64 projection shapes:

| Output N | Logical tiles | Incumbent grid | Candidate grid | Tiles per candidate CTA |
| ---: | ---: | ---: | ---: | ---: |
| 5,120 | 40 | 40 | 40 | 1 |
| 14,336 | 112 | 48 | 28 | 4 |
| 16,384 | 128 | 48 | 32 | 4 |
| 34,816 | 272 | 48 | 34 | 8 |

This removes partially occupied final waves without changing a tile's
arithmetic. It also lowers instantaneous CTA concurrency on three shapes, so a
real workload must determine whether 28-34 CTAs still saturate memory bandwidth.
No speedup is claimed from the grid model alone.

## Static audit

The BF16 and FP16 candidate kernels each use 168 registers/thread, zero stack,
zero local memory, 1,024 bytes static shared memory, and 2,688 bytes of constant
bank 0. These resources exactly match the already byte-clean generic static
stock-tile kernel in the same binary.

Candidate and baseline also each contain 936 SASS instructions: 32 QMMA, 32
FFMA, 24 FMUL, 24 LDSM, four STSM, 45 branches, and 38 SYNCS instructions.
Neither contains local-memory instructions or calls. The grid policy is host
code, so the device kernel math body remains unchanged.

The immutable candidate is:

```text
/home/mark/fr13_cutlass_divisor_balanced_build_20260802/bin/_C_stable_libtorch.divisor_static_stocktile_k64_root_338e89d062c2b1ac.abi3.so
SHA256 338e89d062c2b1ac40909dbc8d64d4ab6b0def9fd86988c9e395e8244606a9f6
bytes 113837288, mode 0444
```

It targets `sm_121a`, has no symbol requirement newer than `GLIBC_2.32`, and
retains the pinned torch/CUDA runpath.

## Next gate

Wire the immutable identity under a distinct selector, then run the same
authenticated one-task SWE-Verified K64/root B1 shadow comparison that qualified
the generic static kernel. All 320 comparisons and all five projection shapes
must be present and byte equal while stock remains served. Only then may this
candidate enter matched real-task timing. B1 diagnostics are not acceptance;
final acceptance remains the canonical exact4 B4 or exact16 campaign.

