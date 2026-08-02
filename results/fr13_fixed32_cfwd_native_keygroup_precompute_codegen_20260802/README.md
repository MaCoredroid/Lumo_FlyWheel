# Fixed32 native key-group precompute CFWD codegen

Status: **SM121a source, exact-math, and resource contract passed; default off;
real-task raw-byte and timing qualification pending**.

## Kernel design

The candidate keeps one CTA per `(layer, request, key_head)`, preserving the
threefold CTA and K-load reduction of the prior grouped candidate. Each CTA:

1. Uses up to 36 threads to precompute all active root-inclusive step nodes and
   three value-head gate scalar pairs.
2. Uses four four-warp groups to normalize four K vectors concurrently per
   wave. Three waves cover the fixed 12-step capacity. Per-slot inverse norms
   are separate from the partial-reduction array, so each wave needs two CTA
   barriers rather than three; one final barrier publishes all normalized K
   rows before recurrence.
3. Processes the three value heads sequentially. Sixteen warps cover eight
   value rows each, and every thread holds one 32-FP32-state register tile.
4. Loads, runs all active steps, and stores one head before reusing the same
   register tile for the next head.

The head loop is explicitly rolled with `#pragma unroll 1`, and the active-step
state recurrence remains rolled. Only the three fixed normalization waves are
unrolled. Therefore the static SASS arithmetic counts represent one state
recurrence body, not three heads times twelve steps.

## Shared-memory change

Persistent shared state is eliminated. Source static shared memory contains:

| Surface | Elements | Bytes |
| --- | ---: | ---: |
| Twelve normalized K vectors | 1,536 FP32 | 6,144 |
| Four-by-four norm partials | 16 FP32 | 64 |
| Four inverse norms | 4 FP32 | 16 |
| Twelve three-head gate pairs | 72 FP32 | 288 |
| Twelve nodes | 12 int32 | 48 |
| Step count and state index | 2 int32 | 8 |
| **Total source static shared** | | **6,568** |

`cuobjdump` reports 7,592 bytes because SM121 adds a 1,024-byte target reserve.
The prior grouped candidate used 98,868 source bytes and 99,892 reported bytes
for its half-register, half-shared persistent state layout.

## Exact compile result

The exact full translation unit at source commit
`ecfff80b32063d575d1e9ab5006bac2bdd525055` produced frozen object SHA256
`8a20199ee6ad357f6188aed5551fa2697c0185ca86edf63f4fa2d8b8f29649b0`.
The bound checker passed:

| Resource | Result |
| --- | ---: |
| Registers/thread | 64 |
| Stack frame | 0 bytes |
| Spill stores / loads | 0 / 0 bytes |
| Local memory | 0 bytes |
| Source static shared memory | 6,568 bytes |
| Reported shared memory | 7,592 bytes |
| SASS `LDL` / `STL` / `CALL` | 0 / 0 / 0 |

This satisfies the requested two-CTA launch-bound register ceiling and removes
the prior grouped candidate's 168-register and one-CTA resource shape.

## Codegen shape

| Opcode | Static count |
| --- | ---: |
| `MUFU.EX2` | 3 |
| `MUFU.RSQ` | 3 |
| `MUFU.RCP` | 2 |
| `SHFL.BFLY` | 202 |
| `SHFL.IDX` | 16 |
| `SHFL.DOWN` | 0 |
| `FFMA` | 82 |

The state recurrence retains the incumbent-aligned FMA-first XOR reduction,
`0,2,1,3` group combination, separately rounded state decay and residual, and
fused update. Moving K/gate work ahead of the state passes does not reorder
operations within any step or value head.

CUDA `logf` retains the known FTZ/control-flow caveat. This compile does not
establish raw-byte equality. Authenticated real SWE-Verified B1 and canonical
exact4 B4 all-bank byte gates remain mandatory before timing or production.

## Verification boundary

- Focused selector, source, installer, and codegen-checker suite: `43 passed`.
- Python byte compilation and `git diff --check`: pass.
- Read-only barrier/index/state-ownership review: no defect found.
- Exact source/object/toolchain-bound SM121a checker: pass.
- No GPU query, CUDA launch, container mutation, synthetic/probe timing,
  real-task campaign, or timing measurement was performed for this candidate.
- Default-on, timing eligibility, and production authorization remain false.
- No hardware-floor or acceptance claim changed.

This directory contains reduced codegen facts only. It excludes object files,
PTX/SASS dumps, prompts, responses, traces, raw logs, process/container
identities, credentials, and timing samples.
