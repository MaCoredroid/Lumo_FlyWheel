# FR13 fixed32 CFWD native contiguous-K v8 host proof

## Classification

`READY_AFTER_V5_FOR_REAL_B1_BYTE_GATE`

This is a host-only source, exact-expression, compile, and regression receipt.
It is not a correctness, timing, B1, B4, exact4, exact16, or hardware-floor
result.

## Change

V8 retains the v7 warp-per-step normalization schedule, v6 fused gate, and v5
fixed-16 suffix-zero repair. It remaps each recurrence lane from four strided K
columns to one aligned contiguous `float4`.

Each eight-lane subgroup now owns one incumbent 32-element K quad. Subgroup
xor-4, xor-2, and xor-1 stages reproduce the incumbent logical xor-16, xor-8,
and xor-4 stages. Local component adds reproduce logical xor-2 then xor-1. The
four quad roots are combined with warp xor-16 then xor-8, preserving the exact
`(quad0 + quad2) + (quad1 + quad3)` expression tree.

The same mapping is used for K normalization and the state dot product. It
allows 128-bit state loads and stores plus 128-bit shared K publication and
consumption. State decay, residual formation, rank-one FMA update, head order,
step order, and final signed-zero normalization are unchanged.

## Static delta from v7

| Surface | V7 | V8 |
| --- | ---: | ---: |
| Static SASS instructions | 1,232 | 1,016 |
| Total shuffle sites | 217 | 157 |
| `SHFL.BFLY` | 200 | 140 |
| `SHFL.IDX` | 17 | 17 |
| `FADD` | 264 | 207 |
| `FFMA` | 84 | 84 |
| `LDG.E.128` state sites | 0 | 8 |
| `STG.E.128` state sites | 0 | 8 |
| `LDS.128` K sites | 0 | 1 |
| `STS.128` K sites | 0 | 1 |
| Registers per thread | 64 | 64 |
| Source shared bytes | 6,488 | 6,488 |
| Cuobjdump shared bytes | 7,512 | 7,512 |
| Spill loads / stores | 0 / 0 | 0 / 0 |

The static instruction body is 17.5% smaller, and static shuffle sites are
27.6% lower. Static copies are not dynamic instruction counts or a latency
claim. Only real-task timing can determine the wall-time effect.

## Verification

- Source commit: `40a956e868f3ba2221b53716698c6e5787a9396f`
- CUDA source SHA-256:
  `cc0ddb8d5aab11e1f6156f434492302bae6b6481d0fd7f38282f70e28687e09f`
- Host object SHA-256:
  `26205f50a84155dc0984fb24b4c615fc83b6401395c567e55724f5a995e6e2dd`
- The committed structural unit proof constructs both complete FP32
  expression trees and requires exact equality.
- Focused source, binary gate, committer, boundary, and real-task-arm tests:
  170 passed.
- Ruff, Python byte compilation, patched-source hashes, codegen checker, and
  diff checks: pass.
- Exact SM121 compile: 64 registers, zero stack, local memory, spills, and
  calls; 64 signed-zero normalization `FADD` sites retained.
- No GPU execution and no container launch were used for this receipt.

## Gate order

Run the frozen v5 real SWE-Verified B1 raw-bank byte gate first. If v5 passes,
the fastest route is to rebuild and byte-gate v8 directly. If v8 rejects, run
v7 as the localization fallback for the contiguous-K remap. If v7 also rejects,
v6 localizes the warp-per-step schedule. Only a byte-clean candidate may enter
real-task timing or exact4 B4 qualification.
