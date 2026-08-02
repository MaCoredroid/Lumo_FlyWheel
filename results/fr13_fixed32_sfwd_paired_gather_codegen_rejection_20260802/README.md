# Fixed32 SFWD paired local-gather experiment

Status: **OFFLINE_CODEGEN_REJECTED_UNCHANGED_SHARED_SYNCHRONIZATION**.

This experiment combines the tap-0 and tap-1 row indices, performs one local
gather, then splits the gathered values back into explicit tap-0 and tap-1
values. The kernel keeps the row-major global pointer expressions unchanged
and accumulates tap 0 before tap 1 with the existing BF16 product rounding.

The candidate compiles reproducibly for B1 and B4 on the fixed row32/C64/W16
geometry. It does not meet the declared objective: both `BAR` and `STS` remain
unchanged at 3 and 6. It is therefore rejected without a GPU launch.

| Metric | Fixed-SSI anchor | Tap-mask source | Paired gather | Pair vs tap-mask |
|---|---:|---:|---:|---:|
| Reported registers/thread | 54 | 49 | 50 | +1 |
| Allocated registers/thread | 56 | 56 | 56 | 0 |
| Static SASS | 382 | 371 | 369 | -2 |
| Encoded SASS | 400 | 392 | 384 | -8 |
| LDG / STG | 19 / 12 | 19 / 12 | 19 / 12 | 0 / 0 |
| LDS / STS | 6 / 6 | 6 / 6 | 6 / 6 | 0 / 0 |
| BAR | 3 | 3 | 3 | 0 |
| Launch shared bytes | 4,096 | 4,096 | 4,096 | 0 |

Triton 3.6 `tl.cat` is not a valid ordered implementation for this shape: its
implementation accepts only rank-1 operands and requires reordering to be
allowed. The prototype instead uses `join`, order-preserving `reshape`,
`permute`, and `split`; a CPU semantic test proves the reconstructed first and
second results equal independent tap-0 and tap-1 gathers in that order.

No GPU, Docker, service, task, timing, or acceptance run was used. This reduced
package excludes binaries, compiler IR, raw SASS/PTX, logs, task or model
content, requests, responses, environment values, credentials, process
identifiers, and secrets.
