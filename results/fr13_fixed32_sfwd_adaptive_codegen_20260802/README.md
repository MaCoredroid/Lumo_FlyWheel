# Fixed32 SFWD adaptive B1/B4 codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_GATES_REQUIRED**.

The source-bound
`fixed32_sfwd_channel_serial_r32_b1c128w4_bxc256w8_u32x2_v1` candidate uses
C128/W4 for B1 and C256/W8 for B2-B4. Both schedules retain 320 warps per
request. C128 gives B1 80 CTAs, covering the 48-SM device; at B4, C256 gives
160 total CTAs and halves launch programs relative to C128 without leaving an
SM-coverage deficit.

Fresh-cache SM121a builds reproduce exactly. Both geometries contain zero
shared memory, barriers, spills, local memory, stack, or calls. B1 C128 reports
60 registers and 80 CTAs per launch. B4 C256 reports 64 registers and 160 CTAs
per launch.

| Metric | B1 C128/W4 | B4 C256/W8 |
|---|---:|---:|
| CTAs/request | 80 | 40 |
| CTAs/launch | 80 | 160 |
| Warps/request | 320 | 320 |
| Registers/thread | 60 | 64 |
| Static / encoded SASS | 903 / 920 | 888 / 904 |
| LDG / STG | 38 / 68 | 38 / 68 |
| LDS / STS / BAR | 0 / 0 / 0 | 0 / 0 / 0 |
| Cubin bytes | 63,496 | 62,856 |

This is a compiler/resource result only. B1 and B4 must each pass the required
real SWE-Verified correctness and isolated timing campaigns before any speed,
hardware-floor, or production claim. No GPU run was launched for this
candidate.

No raw SASS, PTX, compiler IR, binary, task/model content, request, response,
environment value, process/container identifier, credential, or secret is
included.
