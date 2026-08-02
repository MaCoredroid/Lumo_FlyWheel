# SFWD prior-reuse descriptorless topology codegen

Status: **offline SM121a codegen improves the dense-int32 C64 prior-reuse
kernel; live correctness and runtime performance remain unqualified**.

This package audits source commit
`4ca86e3beddd9a69a7b3471ccc22ba5c2708b029` at the fixed B1/B4
specialization: 32 physical rows per request, row group 32, 10,240 channels,
`BLOCK_C=64`, convolution width 4, state length 34, eight warps, and three
stages. CUDA visibility was explicitly empty. No GPU kernel, service, SWE task,
request, timing run, or acceptance run was launched.

## Change

The `b108b8819` baseline loads three `int32` source descriptors for each fixed
row. The candidate removes the descriptor pointer and derives the three ordered
ancestor rows from the fixed32 node number. A compact piecewise parent function
is applied three times; CPU tests prove that it reproduces all 32 parent edges
and all 96 non-final source rows. Convolution products remain ordered BF16
products accumulated by FP32 adds, followed by the unchanged current-node tap,
activation, output, current-row, prior-row, and zero-row stores.

The source-bound real B1 layout uses a padded `x` row stride of 16,384
elements. At the corresponding B4 maximum, the largest `x` element offset is
2,091,007; dense output and source-stage maxima are 1,310,719 and 1,474,559.
All remain below the signed-int32 maximum of 2,147,483,647.

The module remains offline-only. It has no serving launcher, selector, or
production integration. A source-bound real SWE B1 reference-served byte gate
is required before timing or promotion.

## Result

| Metric | `b108b8819` | Descriptorless | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported / allocated registers per thread | 56 / 56 | 40 / 40 | -16 / -16 |
| Allocated registers per CTA | 14,336 | 10,240 | -4,096 |
| Register-budget CTAs / warps per SM | 4 / 32 | 6 / 48 | +2 / +16 |
| Static / encoded SASS | 912 / 928 | 885 / 896 | -27 / -32 |
| Warp-weighted static / encoded SASS | 7,296 / 7,424 | 7,080 / 7,168 | -216 / -256 |
| LDG / STG / LDS / STS | 64 / 20 / 0 / 0 | 37 / 20 / 0 / 0 | -27 / 0 / 0 / 0 |
| Launch / ELF shared bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 64,040 | 60,720 | -3,320 |

The register-budget ceiling uses the project SM121 allocation model of 65,536
registers per SM and the exact per-CTA allocations above. It is not an achieved
occupancy measurement. The B1 launch has 160 CTAs, only 3.33 CTAs per SM on the
48-SM project hardware ledger, so the larger register ceiling primarily
removes a resource limit; runtime scheduling still requires measurement.

B1 and B4 have different Triton compile hashes because batch is a compile
constant. Their cubin, PTX, SASS, resource report, and non-launch metrics are
otherwise identical. A second build with a separate fresh cache and output
tree reproduced every generated file byte for byte.

The focused descriptorless, dense-int32, and state-fusion suites passed 24
tests, including the live padded B4 address maximum. This is not a real-task
correctness result and contains no latency, TPS, acceptance, hardware-floor,
or production claim.

The package contains only source hashes and derived summaries. Cubin, PTX,
SASS, IR, model/task content, requests, responses, patches, environment dumps,
credentials, process identities, container identities, and raw logs are
excluded.
