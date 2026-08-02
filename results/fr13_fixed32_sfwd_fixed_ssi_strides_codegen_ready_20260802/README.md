# Fixed32 SFWD fixed state-index strides

Status: **OFFLINE_CODEGEN_PASS_REAL_BYTE_GATE_REQUIRED**.

Source commit `0e3d33ad22b111a7dae025085b2c6509cd3a4be7` extends the
fixed convolution-stride kernel by removing both runtime state-index stride
arguments. Fixed32 uses 31 physical drafts plus root, and the launcher now
requires a contiguous int32 state-index tensor with exactly 32 columns. The
first dimension may exceed the active batch; only its first `B` rows are read.

Two isolated fresh-cache SM121a builds reproduced identical B1/B4 cubin, PTX,
SASS, and resource reports. Relative to the fixed-convolution-stride source,
the C64/W16 kernel removes one static instruction and two kernel arguments:

| Metric | Fixed conv strides | Fixed SSI strides | Delta |
|---|---:|---:|---:|
| Registers/thread | 54 | 54 | 0 |
| Static SASS | 383 | 382 | -1 |
| Encoded SASS | 400 | 400 | 0 |
| LDG / STG | 19 / 12 | 19 / 12 | 0 / 0 |
| LDS / STS / BAR | 6 / 6 / 3 | 6 / 6 / 3 | 0 / 0 / 0 |
| Launch / ELF shared bytes | 4,096 / 1,024 | 4,096 / 1,024 | 0 / 0 |

Stack, local memory, spill loads/stores, and calls remain zero. Focused source
tests passed 22 cases, including exact width/contiguity rejection and absence
of runtime convolution/state-index stride arguments in the active kernel.

This is static codegen evidence only. The source is not runtime-bound and has
not run a GPU kernel or real task. It requires a fresh reference-served real
SWE-Verified B1 byte gate before timing and is ineligible for floor acceptance.

The reduced package excludes raw binaries, PTX, SASS, IR, compiler logs,
task/model/request/response content, patches, environment values, credentials,
process identifiers, container identifiers, and secrets.
