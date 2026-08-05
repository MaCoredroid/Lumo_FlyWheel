# Fixed32 GDN GQA3 B4 node-domain specialization

Status: **offline SM121a codegen gate PASS; default-off candidate; no
performance promotion**.

This artifact compares origin/main revision
`9091ddae2046f42fc5e754f976c3493a033785ac` with candidate revision
`8c85135cb6092f01230d93c55b1c6f3fcf7336f3`. It compiles the exact B4,
physical32, K64/root1, 16-key-head, 48-value-head, BV8 GDN GQA3 kernel for
`sm_121a`. Both the base closure and the K-norm/gate/decay committer-stack
closure are covered.

## Source change

The baseline recomputes the node-domain predicate, nonnegative clamp, and
global row four times per logical node visit: once in the grouped key-head
helper and once in each of three value-head helpers. The candidate computes
the global row once and passes it to the sibling helpers. The exact fixed32
launcher selects a compile-time trusted node domain only after the internal
preseed caller supplies the validated 32-node execution SHA. Descriptor
extent, dtype, and layout checks remain fail closed. No device readback or
per-step descriptor scan was added.

At B4 the unchanged kernel grid visits 1,572,864 logical nodes per 48-layer
event. The source removes 6,291,456 guard sites and 6,291,456 clamp sites per
event. These are source-level work counts, not dynamic SASS counts or timing.
The four independent request recurrences, 1,024 CTAs/layer, 49,152 CTAs/event,
and all required state/V work remain unchanged.

## Codegen result

| Profile | Variant | Registers/thread | Non-control SASS | LDG / STG | Stack / local | LDL / STL / calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Baseline | 120 | 2,174 | 85 / 54 | 0 / 0 | 0 / 0 / 0 |
| Base | Node-domain | 116 | 2,052 | 85 / 54 | 0 / 0 | 0 / 0 / 0 |
| Committer | Baseline | 126 | 2,280 | 85 / 82 | 0 / 0 | 0 / 0 / 0 |
| Committer | Node-domain | 122 | 2,155 | 85 / 82 | 0 / 0 | 0 / 0 / 0 |

The candidate removes four registers/thread in both profiles. Static
non-control instruction sites fall by 122 (5.61%) in the base profile and 125
(5.48%) in the committer profile. Encoded instruction sites fall by 120 in
both profiles. Global load/store counts are unchanged, and no spill, local
memory, or device call appears. Two independent fresh-cache builds are
byte-identical across PTX, cubin, SASS, compiler IR, resource dumps, and JSON
summaries.

## Decision

The candidate passes this offline static codegen/resource gate and remains
credential-gated. It is not performance-promoted. No GPU kernel was launched,
no serving task ran, and no acceptance, TPS, full-step latency, or
hardware-floor measurement was collected. The next authority is a real
SWE-Verified B4 byte-equivalence gate and the standing real four-task timing
gate. B1 performance also remains governed by its real workload gate; this
artifact contains no B1 timing evidence.

Only sanitized summaries and reproduction code are checked in. Cubin, PTX,
SASS, compiler IR/cache, raw logs, task/model/request/response/patch content,
credentials, environment dumps, process IDs, and container IDs are excluded.
