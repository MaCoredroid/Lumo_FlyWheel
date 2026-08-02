# SFWD prior-reuse int32 address codegen

Status: **offline SM121a codegen improves the int32-descriptor candidate;
live correctness and runtime performance remain unqualified**.

This package audits source commit
`b108b8819bed9424c28ff4c9c4bb33bd495994cc` at the fixed B1/B4
specialization: 32 physical rows per request, row group 32, 10,240 channels,
`BLOCK_C=64`, width 4, state length 34, eight warps, and three stages. CUDA
visibility was explicitly empty. No GPU kernel, Docker service, SWE task,
request, timing run, or acceptance run was launched for this package.

## Change

The candidate preserves the contiguous int32 source descriptor and exact
ordered BF16-product/FP32-add convolution math. It narrows only dense-buffer
element-offset arithmetic for `x`, `out`, and `source_stage` from int64 to
int32. Conv-state bank addressing remains int64 because its selected bank row
is runtime state.

The host-side contract accepts only B1-B4, requires a valid row stride, and
rejects any specialization whose maximum dense-buffer element offset exceeds
signed int32. At the exact dense B4 shape, the maximum `x`/`out` offset is
1,310,719 elements and the maximum source-stage offset is 1,474,559 elements,
both below 2,147,483,647.

The module is offline-only and has no launcher or production selector. It
cannot be engaged by the serving stack. A source-bound real B1
reference-served byte gate is required before timing or production work.

## Result

| Metric | C64 prior reuse | Int32 descriptor | Int32 dense offsets |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 160 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 160 / 640 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 8 / 256 |
| Reported registers/thread | 62 | 64 | 56 |
| Allocated registers/thread | 64 | 64 | 56 |
| Allocated registers/CTA | 16,384 | 16,384 | 14,336 |
| Static / encoded SASS | 993 / 1,008 | 981 / 992 | 912 / 928 |
| Warp-weighted static / encoded SASS | 7,944 / 8,064 | 7,848 / 7,936 | 7,296 / 7,424 |
| LDG / STG / LDS / STS | 64 / 20 / 0 / 0 | 64 / 20 / 0 / 0 | 64 / 20 / 0 / 0 |
| Launch / ELF shared bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| Cubin bytes | 69,088 | 67,752 | 64,040 |

Relative to the prior int32-descriptor candidate, reported and allocated
registers fall by eight per thread, static SASS falls by 69 instructions,
encoded SASS falls by 64 instructions, and cubin size falls by 3,712 bytes.
The register reduction creates allocation headroom but does not increase the
four-CTA register-residency ceiling at this launch geometry.

B1 and B4 have distinct compile hashes because batch is a compile constant.
Their cubin, PTX, SASS, resource report, and non-launch metrics are otherwise
identical. A second build with a separate cache and output tree reproduced all
generated files byte for byte.

The focused state-fusion and descriptor tests passed 18 tests. They cover the
existing fixed topology and source contract, every descriptor entry, exact
ordered convolution math, the final-current-node invariant, the exact B4
address maximum, and rejection of an unsafe row stride. This is not a real-task
correctness result and contains no latency, TPS, acceptance, floor, or
production claim.

The package contains only source hashes and derived summaries. Cubin, PTX,
SASS, IR, model/task content, requests, responses, patches, environment dumps,
credentials, process identities, container identities, and raw logs are
excluded.
