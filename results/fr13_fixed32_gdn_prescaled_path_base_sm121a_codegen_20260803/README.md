# Fixed32 GDN pre-scaled path-base SM121a codegen

Status: **offline SM121a codegen passes; retain default-off for a real byte
gate and task timing**.

Commit `8959f328ce6b5e36c5eb6bbb1cb53c3c6e5f5bbe` adds an exact
K64/root1 physical32 specialization to the retained ordered one-launch GDN
kernel. The incumbent descriptor contains compact path indices, so every
reachable branch path dynamically forms `path_index * MAX_PATH_LEN`. The
candidate stores those five-by-three descriptor entries as pre-scaled bases
and pads the length descriptor at the corresponding bases.

The candidate keeps the group count and path length as device-loaded loop
bounds. It does not statically expand either ordered loop, change root/group/
path order, alter the GDN recurrence, change node or output stores, or change
rejection sampling. Both descriptor tensors are immutable, preseeded, and
validated against the exact fixed32 topology before launch.

## Codegen result

| Metric | B1 incumbent / candidate | B4 incumbent / candidate |
|---|---:|---:|
| Registers/thread | 99 / 99 | 99 / 99 |
| Stack/local bytes | 0 / 0 | 0 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| SASS addressed lines | 3552 / 3520 | 3552 / 3520 |
| Static SASS instructions | 1776 / 1760 | 1776 / 1760 |
| Cubin bytes | 136864 / 136560 | 136864 / 136560 |
| Static LDG sites | 62 / 62 | 62 / 62 |
| Path-base scales per program | 11 / 0 | 11 / 0 |

The candidate removes six `IADD`, three `IMAD`, three `LEA`, and three `SHF`
instructions from the static opcode census while leaving the 62 LDG sites
unchanged. This is an address-generation reduction, not a descriptor-traffic
claim. The source hashes for the node body and recurrence are identical across
arms. The compiler also reschedules a small number of non-address operations,
so source invariance and byte validation, rather than an arithmetic opcode
census, remain the correctness authority.

Two independent fresh-cache builds produced identical summaries, cubins, PTX,
and SASS for B1 and B4. Compilation used the deployed physical32 geometry,
K64/root1, BF16 inputs, four key heads, twelve value heads, DK=DV=128, BV=8,
ring and flag export, eight warps, and target `sm_121a`.

## Rejected variants

The exploration also compiled and rejected the following source-only variants:

| Variant | Rejection reason |
|---|---|
| Static root descriptor | 119 registers |
| Static group count | 16-byte stack plus LDL/STL |
| Static root plus group count | 117 registers |
| Packed path index and length | 8-byte stack plus LDL/STL |
| Static path length | 8-byte stack plus LDL/STL |
| Packed group count | 16-byte stack plus LDL/STL |
| Contiguous packed path record | 99 registers but no SASS reduction |

## Decision

Keep the route default-off behind
`FR13_FIXED32_GDN_PRESCALED_PATH_BASE=1` or its boot sidecar. Offline source,
resource, and reproducibility gates pass. No GPU kernel, serving process, SWE
task, timing sample, acceptance sample, full-step TPS measurement, or
hardware-floor measurement was run by this artifact. A real B1 and B4
SWE-Verified byte gate followed by the standing 4-task/16-task timing campaign
is required before promotion or any speed claim.

The checked-in package contains reduced summaries and reproduction code only.
It excludes cubin, PTX, SASS, compiler caches, raw logs, task/model/request/
response/patch content, credentials, environment dumps, process IDs, and
container IDs.
