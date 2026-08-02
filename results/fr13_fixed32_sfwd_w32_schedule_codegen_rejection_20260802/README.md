# Fixed32 SFWD W32 schedule experiment

Status: **OFFLINE_CODEGEN_REJECTED_WARP_WEIGHTED_WORK_AND_RESIDENCY**.

The selected fixed-stride kernel source at
`0e3d33ad22b111a7dae025085b2c6509cd3a4be7` was compiled at the same
row32/C64 geometry with 32 warps instead of 16. Per-warp counters shrink, but
the CTA executes twice as many warps and consumes more allocated registers.

| Metric | W16 selected | W32 experiment | Delta |
|---|---:|---:|---:|
| Threads/CTA | 512 | 1,024 | +512 |
| Reported / allocated registers/thread | 54 / 56 | 38 / 40 | -16 / -16 |
| Allocated registers/CTA | 28,672 | 40,960 | +12,288 |
| Static / encoded SASS | 382 / 400 | 244 / 264 | -138 / -136 |
| Warp-weighted static / encoded | 6,112 / 6,400 | 7,808 / 8,448 | +1,696 / +2,048 |
| LDG / warp-weighted LDG | 19 / 304 | 11 / 352 | -8 / +48 |
| STG / warp-weighted STG | 12 / 192 | 8 / 256 | -4 / +64 |
| BAR / warp-weighted BAR | 3 / 48 | 3 / 96 | 0 / +48 |

Register-budget residency falls from two CTAs to one CTA per 65,536-register
SM, before considering the doubled thread footprint. Two fresh-cache B1/B4
builds reproduce the W32 result without spills or calls.

W32 is rejected and was not runtime-bound or launched. No GPU, Docker,
service, task, timing, or acceptance run was used. This package excludes raw
compiler output, binaries, IR, logs, task/model content, requests, responses,
environment values, credentials, process identifiers, and secrets.
