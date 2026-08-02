# SFWD prior-reuse int32 descriptor codegen

Status: **offline SM121a codegen improves the C64 prior-reuse kernel; live
correctness and runtime performance remain unqualified**.

This package audits source commit
`8a3e77792ecdfd7df798ab24715aefbca11d3c93` at the exact B1/B4 kernel
specialization: 32 physical rows per request, row group 32, 10,240 channels,
`BLOCK_C=64`, width 4, state length 34, eight warps, and three stages. CUDA
visibility was explicitly empty. No GPU kernel, Docker service, SWE task,
request, timing run, or acceptance run was launched.

## Change

The qualified-next prior-reuse kernel reads three non-final source descriptors
per physical row as `int64`; the final tap is already specialized to the
current node. This candidate preserves the same three entries and exact tap
order in a separate contiguous `int32` descriptor. The kernel still widens an
entry before address arithmetic, so selected state and `x` rows are unchanged.
The descriptor's logical payload falls from 768 to 384 bytes per CTA. Actual
memory transactions and cache behavior were not measured.

The module is offline-only and has no launcher or production selector. It
cannot be engaged by the serving stack. A source-bound real B1 reference-served
byte gate is required before any timing or production work.

## Result

| Metric | C64 prior reuse | Int32 descriptor | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported registers/thread | 62 | 64 | +2 |
| Allocated registers/thread | 64 | 64 | 0 |
| Allocated registers/CTA | 16,384 | 16,384 | 0 |
| Static / encoded SASS | 993 / 1,008 | 981 / 992 | -12 / -16 |
| Warp-weighted static / encoded SASS | 7,944 / 8,064 | 7,848 / 7,936 | -96 / -128 |
| LDG / STG / LDS / STS | 64 / 20 / 0 / 0 | 64 / 20 / 0 / 0 | 0 |
| Launch / ELF shared bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 69,640 | 67,752 | -1,888 |

B1 and B4 have distinct compile hashes because batch is a compile constant.
Their cubin, PTX, SASS, resource report, and non-launch metrics are otherwise
identical. A second build with a separate cache and output tree reproduced all
generated files byte for byte.

The focused prior-reuse and descriptor tests passed 16 tests. They cover the
fixed topology, every descriptor entry, exact ordered BF16-product/FP32-add
math, the final-current-node invariant, and the existing row32/C64 source
contract. This is not a real-task correctness result and contains no latency,
TPS, acceptance, floor, or production claim.

The package contains only source, hashes, and derived summaries. Cubin, PTX,
SASS, IR, model/task content, requests, responses, patches, environment dumps,
credentials, process identities, container identities, and raw logs are
excluded.
