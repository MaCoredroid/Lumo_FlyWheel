# Fixed32 SFWD fixed convolution strides

Status: **OFFLINE_CODEGEN_PASS_REAL_BYTE_GATE_REQUIRED**.

Source commit `725b40b6c2b39918b2de0026500a7e973083b0a8` removes the
three runtime convolution-state stride arguments from the active prior-pair,
quad-weight, packed x-gather kernel. The launcher already requires an exact
contiguous BF16 `[bank, 10240, 34]` tensor, so its element strides are fixed at
`[348160, 34, 1]`. State bank addressing remains int64.

Two isolated fresh-cache SM121a builds reproduced identical B1/B4 cubin, PTX,
SASS, and resource reports. Relative to parent `6254582fc8b2000f00bc3fa425e8d55df11b3216`,
the C64/W16 specialization improves:

| Metric | Parent | Fixed strides | Delta |
|---|---:|---:|---:|
| Registers/thread | 55 | 54 | -1 |
| Static SASS | 391 | 383 | -8 |
| Encoded SASS | 408 | 400 | -8 |
| Cubin bytes | 36,472 | 35,344 | -1,128 |
| LDG / STG | 19 / 12 | 19 / 12 | 0 / 0 |
| LDS / STS / BAR | 6 / 6 / 3 | 6 / 6 / 3 | 0 / 0 / 0 |
| Launch / ELF shared bytes | 4,096 / 1,024 | 4,096 / 1,024 | 0 / 0 |

Stack, local memory, spill loads/stores, and calls remain zero. Focused source
tests passed 22 cases, including the exact fixed-stride contract and absence
of runtime stride arguments in both the kernel and launcher.

This is static codegen evidence only. The source is not runtime-bound and has
not run a GPU kernel or real task. It requires a fresh reference-served real
SWE-Verified B1 byte gate before timing and is ineligible for floor acceptance.

The reduced package excludes raw binaries, PTX, SASS, IR, compiler logs,
task/model/request/response content, patches, environment values, credentials,
process identifiers, container identifiers, and secrets.
