# Fixed32 GDN static-descriptor SM121a rejection

Status: **rejected and reverted**.

The experiment replaced the ordered single-launch kernel's fixed32 root,
group, path-length, and branch-node descriptor loads with integer decoding of
the already-validated 32-row topology. It did not change launch count, node
order, recurrence math, output stores, replay-ring stores, K64/root1 policy, or
Tail23/Hydra27 validity masks.

Offline compilation targeted the deployed TP4 GDN shape (`KH=4`, `VH=12`,
`DK=128`, `DV=128`, `BV=8`, eight warps) with CUDA 13.0 and Triton 3.6.0. The
generated PTX names `sm_121a`. No GPU workload, probe, synthetic timing, or
acceptance run was used.

| Metric | Incumbent | Candidate | Delta |
| --- | ---: | ---: | ---: |
| registers/thread | 99 | 112 | +13 |
| stack bytes | 0 | 0 | 0 |
| local bytes | 0 | 0 | 0 |
| shared bytes | 1024 | 1024 | 0 |
| SASS instructions | 3552 | 3296 | -256 |
| static `LDG` sites | 62 | 47 | -15 |
| cubin bytes | 136840 | 130320 | -6520 |

The lower descriptor traffic and instruction count are real, but the register
increase violates the no-resource-regression retention gate. The experiment is
preserved in commit `9d13f1ef8`; commit `2500e5ed1` restores the incumbent
source. This artifact makes no timing, acceptance, or hardware-floor claim.
