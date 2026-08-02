# Fixed32 SFWD split20 codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_GATES_REQUIRED**.

The source-bound
`fixed32_sfwd_channel_serial_r32_b1c128w4_bxc256w8_u32x2_s20_v1`
candidate explicitly specializes all 32 fixed-tree node calculations and
places one input-load boundary after node 19. Rows 0-19 are loaded and consumed
before rows 20-31 are loaded. This shortens register live ranges without
adding global traffic or changing arithmetic order.

An offline split-position sweep selected row 20. At B1 C128/W4 it improves the
adaptive two-pair baseline from 60 to 48 registers and from 903 to 897 static
SASS instructions. At B4 C256/W8 it improves 64 to 48 registers while changing
static SASS from 888 to 895. Both retain 38 `LDG`, 68 `STG`, zero shared
memory, zero barriers, and no spills, local memory, stack, or calls.

| Metric | B1 C128/W4 | B4 C256/W8 |
|---|---:|---:|
| CTAs/request | 80 | 40 |
| CTAs/launch | 80 | 160 |
| Registers/thread | 48 | 48 |
| Static / encoded SASS | 897 / 912 | 895 / 912 |
| LDG / STG | 38 / 68 | 38 / 68 |
| LDS / STS / BAR | 0 / 0 / 0 | 0 / 0 / 0 |

The test suite also checks every explicit tap operand against the fixed32
descriptorless source table, including current-node inputs, and verifies the
single split boundary.

This is compiler/resource evidence only. The candidate remains default-off and
has not consumed a GPU slot. Real SWE-Verified correctness and isolated timing
are required before any speed, hardware-floor, or production claim.

No raw SASS, PTX, compiler IR, binary, task/model content, request, response,
environment value, process/container identifier, credential, or secret is
included.
