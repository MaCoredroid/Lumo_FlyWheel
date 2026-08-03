# Fixed32 committer sticky-guard SM121a codegen

Status: **offline SM121a codegen passes; keep default-off for a real byte and
timing gate**.

The retained physical32 owner-path guard writes one bool per layer/request and
then launches `guard_flags.all()` before the async assertion. The default-off
`FR13_FIXED32_COMMITTER_STICKY_GUARD=1` route preserves the exact validation
body but publishes failure to one process-lifetime int32 scalar. Valid programs
exit without a global result write; an invalid program atomically changes the
scalar from one to zero, and it is never reset. Passing the scalar directly to
the async assertion removes the per-event scalar-reduction launch.

The candidate is available only with direct persistent committer metadata. Its
one-shot lease binds the guard pointer, shape, stride, dtype, and device along
with the accepted-input pointers, batch, and CUDA stream. With the arm absent,
the incumbent bool-vector guard remains unchanged.

## Codegen result

| Metric | B1 incumbent / candidate | B4 incumbent / candidate |
|---|---:|---:|
| Registers/thread | 18 / 16 | 16 / 16 |
| Stack/local bytes | 0 / 0 | 0 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Shared bytes / BAR | 0 / 0 | 0 / 0 |
| Encoded SASS | 144 / 144 | 168 / 168 |
| Static non-control SASS | 129 / 130 | 149 / 150 |
| Static LDG / STG | 8 / 1 -> 8 / 0 | 8 / 1 -> 8 / 0 |
| Failure-only global atomic | 0 / 1 | 0 / 1 |
| Valid-event guard-result stores | 48 / 0 | 192 / 0 |
| Scalar-reduction launches/event | 1 / 0 | 1 / 0 |

The candidate SASS places its single `ATOMG.EXCH` after a predicated early
exit, so a valid event does not execute the atomic or its failure-path memory
barriers. The two independent fresh-cache builds produced identical summaries,
cubins, PTX, SASS, and resource records.

All builds used the deployed guard specialization: 48 layers, physical node
domain 32, path capacity 16, maximum accepted length 11, alias width 3, B1/B4,
four warps, one stage, and target `sm_121a`. K64/root1 affect the fixed drafter
configuration and are unchanged by this CFWD guard candidate.

## Decision

Retain the source for a real SWE-Verified byte gate and paired B1/B4 full-step
timing. This package proves source-visible launch elimination and static codegen
quality only. It contains no GPU execution, service run, task, timing,
acceptance, TPS, or hardware-floor evidence.

The checked-in package contains reduced summaries and reproduction code only.
It excludes cubin, PTX, SASS, compiler caches, raw logs, task/model/request/
response/patch content, credentials, environment dumps, process IDs, and
container IDs.
