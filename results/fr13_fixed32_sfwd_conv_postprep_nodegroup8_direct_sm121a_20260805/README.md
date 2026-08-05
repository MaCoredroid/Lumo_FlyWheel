# Fixed32 SFWD direct nodegroup8 candidate

This package records the source and offline SM121a resource gate for
`fixed32_sfwd_conv_postprep_nodegroup8_direct_v1` at source commit
`b73d78f681d0cea8487b97a75eaf2ac44d3bc8ec`.

## What changed

The default-off selector splits the fixed32 SFWD producer into four fixed node
groups of eight. Each group has 40 `BLOCK_C=256` channel programs, for 160
channel programs per request. Standalone gating adds four programs; embedded
gating remains on the first four channel programs. The layer still uses one
kernel launch.

The 32 convolution source triples are extracted mechanically from
`fixed32_descriptorless_sources`. Each node is scalar-unrolled with the exact
BF16 product boundary, ordered FP32 adds, SiLU, recurrence stores, and its own
source-stage store. Only group 0 writes the three prior edges and final zero
edge. There is no source descriptor in the generated kernel, row gather,
shared tile, reduction, or barrier. The incumbent generated kernel function is
byte-identical when the selector is off.

## Offline codegen

All builds use CUDA 13.0.85, Triton 3.6.0, SM121a, `BLOCK_C=256`, four warps,
and three stages. `CUDA_VISIBLE_DEVICES` was empty. Every pre-build
`MemAvailable` sample exceeded the 20 GiB guard.

| Schedule | Kernel | Registers | Stack/local/shared | LDG | STG |
|---|---|---:|---:|---:|---:|
| standalone | incumbent | 56 | 0 / 0 / 0 | 85 | 336 |
| standalone | nodegroup8 direct | 46 | 0 / 0 / 0 | 131 | 336 |
| embedded | incumbent | 56 | 0 / 0 / 0 | 85 | 336 |
| embedded | nodegroup8 direct | 48 | 0 / 0 / 0 | 131 | 336 |

B1 and B4 compile to the same resources for a given schedule because batch is
not part of per-CTA work. The candidate reduces register pressure by 8-10
registers/thread while exposing four times as many channel CTAs. It also raises
static load sites by 46 and increases code size. Offline resource data cannot
predict whether added parallelism repays the repeated reads.

## Prior rowgroup result

The real B1 `fixed32_sfwd_state_fusion_rowgroup8_v3` diagnostic regressed by
2.282206918 ms/step (0.941944235%). That measured kernel used an 8x256 row
tensor, dynamic `source_flat` descriptors, masked prior/x selection, and
row-dependent addressing. It is not this direct scalar-unrolled design. The
result remains a warning that more row parallelism is not automatically faster.

## Status

This is source-only, default-off evidence. It contains no GPU execution,
wall-time measurement, throughput result, or hardware-floor acceptance claim.
Promotion requires an authenticated real B1 diagnostic and the standing exact4
B4 timing/byte gate.
