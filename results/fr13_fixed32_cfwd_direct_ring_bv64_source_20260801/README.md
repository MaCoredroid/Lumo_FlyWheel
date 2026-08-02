# Fixed32 CFWD direct-ring BV64 candidate

Status: **default off; source/static verified; pinned-image GPU compile,
resource inspection, real byte qualification, and timing pending**.

## Change

The direct-ring 48-layer CFWD candidate previously split each 128-element
value-state dimension into four `BV=32`, four-warp programs. This revision uses
two `BV=64`, eight-warp programs. The FP32 recurrence, accepted-length bound,
ring inputs, state destination, and final-store behavior are unchanged.

For the pinned `K=V=128` geometry, doubling both the value tile and warp count
keeps 32 FP32 state elements per thread before compiler allocation effects. It
halves the number of programs while preserving the same total warp count. The
candidate contract now requires `value_tile=64`, `kernel_warps=8`, and two
programs per layer/request/value-head, so an older BV32 process cannot satisfy
the observer.

## Static work model

These are logical-work counts, not DRAM or latency measurements. With 48
layers and 48 value heads:

| Occupancy | BV32 programs/event | BV64 programs/event | Total warps |
| --- | ---: | ---: | ---: |
| B1 | 9,216 | 4,608 | 36,864 |
| B4 | 36,864 | 18,432 | 147,456 |

Each eliminated program avoids a duplicated 128-element BF16 K-vector load on
every live recurrence step, plus duplicated accepted-path and gate metadata.
At the prior Hydra27 mean of 5.753885 root-inclusive steps, the logical
K-vector load reduction is 6.473121 MiB/event at B1 and 25.892482 MiB/event at
B4. At the reachable maximum of 12 steps, it is 13.5 MiB/event and 54 MiB/event.
State and V traffic are tiled rather than removed.

No speedup is claimed. The wider tile must compile in the pinned vLLM image and
pass register, stack, local-memory, and spill checks; it must then satisfy the
same real SWE-Verified raw-byte coverage and same-process timing boundary as
the direct-ring candidate.

## Static verification

- Focused CFWD lifecycle/committer suite: `84 passed`.
- Python byte compilation: pass.
- `git diff --check`: pass.
- No GPU command or synthetic performance probe was run.

This directory contains source-level aggregate metadata only. It contains no
prompts, responses, patches, traces, raw logs, process/container identities,
credentials, or timing samples.
