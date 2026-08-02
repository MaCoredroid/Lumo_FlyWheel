# Fixed32 SFWD C128 two-pair codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_BYTE_GATE_REQUIRED**.

The source-bound `fixed32_sfwd_channel_serial_r32_c128_w4_u32x2_v1`
candidate carries the observed live channel-major convolution-state layout and
loads the four contiguous BF16 weights as two aligned 32-bit pairs. C128/W4
keeps 320 warps per request while reducing B1 programs from 160 to 80 relative
to the repaired C64/W2 candidate.

Offline SM121a code generation reports 60 registers, zero shared memory, zero
barriers, and no local-memory, stack, or call path. B1 and B4 generate the same
binary, and an independent fresh-cache rebuild reproduces every reported hash.

| Metric | Repaired C64/W2 u64 | C128/W4 u32x2 | Delta |
|---|---:|---:|---:|
| Threads/CTA | 64 | 128 | +64 |
| CTAs/request | 160 | 80 | -80 |
| Warps/request | 320 | 320 | 0 |
| Registers/thread | 66 | 60 | -6 |
| Static / encoded SASS | 932 / 944 | 903 / 920 | -29 / -24 |
| LDG / STG | 37 / 68 | 38 / 68 | +1 / 0 |
| LDS / STS / BAR | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Cubin bytes | 64,336 | 63,496 | -840 |

These are compiler/resource observations, not throughput measurements or
hardware-floor evidence. This candidate remains default-off and is not
production eligible until it passes a real SWE-Verified byte gate and later
isolated timing under the standing task-set rule.

No raw SASS, PTX, compiler IR, binary, task/model content, request, response,
environment value, process identifier, container identifier, credential, or
secret is included.
