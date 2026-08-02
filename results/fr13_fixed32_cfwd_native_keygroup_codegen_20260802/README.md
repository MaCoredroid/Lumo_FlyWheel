# Fixed32 native key-group CFWD codegen

Status: **SM121a source, math-tree, and resource contract passed; default off;
real-task raw-byte and timing qualification pending**.

## Kernel change

The diagnostic candidate maps one CTA to each
`(layer, request, key_head)` and processes all three value heads associated with
that key head. Compared with the native one-value-head candidate, this changes
the per-event grid from 2,304 to 768 CTAs at B1 and from 9,216 to 3,072 CTAs at
B4. Each CTA loads and normalizes K once per root-inclusive recurrence step for
all three value heads. Final FP32 state-bank HBM traffic is unchanged.

The zero-padding geometry is 384 threads / 12 warps, with each warp owning 32
value rows. Sixteen rows per warp remain in registers and sixteen use a
bank-conflict-free transposed shared-state layout.

## Compile result

An exact full-translation-unit SM121a compile passed the bound object checker:

| Resource | Result |
| --- | ---: |
| Registers/thread | 168 |
| Stack frame | 0 bytes |
| Spill stores / loads | 0 / 0 bytes |
| Local memory | 0 bytes |
| Source static shared memory | 98,868 bytes |
| Reported shared memory | 99,892 bytes |
| SASS `LDL` / `STL` / `CALL` | 0 / 0 / 0 |

The final spill removal narrows the contiguous gate-ring element offset to
checked int32 arithmetic. The wrapper proves the A/B ring is contiguous, exact
geometry, and has at most `INT32_MAX` elements before dispatch.

## Math contract

Both the register and shared-state halves use the incumbent-aligned operation
tree: XOR butterfly reductions, `0,2,1,3` K-group combination, separately
rounded decay/residual operations, and fused state update. `EX2`, reciprocal
square root, and full divide are pinned with inline PTX.

| Opcode | Count |
| --- | ---: |
| `MUFU.EX2` | 9 |
| `MUFU.RSQ` | 1 |
| `MUFU.RCP` | 4 |
| `SHFL.BFLY` | 654 |
| `SHFL.IDX` | 64 |
| `SHFL.DOWN` | 0 |
| `FFMA` | 294 |

CUDA `logf` retains the known FTZ/control-flow caveat from the one-value-head
candidate. Only authenticated real B1 and exact4 B4 all-bank raw-byte gates can
close that risk.

The direct `_C` operator also treats its two prevalidation booleans as trusted
internal preconditions. An arbitrary caller could lie about bank/path metadata;
the operator is not a safe general API and must remain internal, default-off,
and non-production until direct validation is added.

## Qualification boundary

- Focused source, selector, installer, and checker suite: `42 passed`.
- Ruff, Python byte compilation, and `git diff --check`: pass.
- Exact SM121a compile and source/object/toolchain-bound checker: pass.
- No kernel launch, synthetic/probe timing, or new performance measurement was
  performed for this candidate.
- Real SWE-Verified B1 and canonical exact4 B4 raw-byte gates remain pending.
- Timing and production authorization remain hard-disabled.
- The hardware-floor distance and acceptance claims are unchanged.

This directory contains reduced codegen facts only. It excludes object files,
PTX/SASS dumps, prompts, responses, patches, traces, raw logs, process/container
identities, credentials, and timing samples.
