# Fixed32 GDN GQA3 B1+B4 static schedule

Status: **offline SM121a codegen/resource gate PASS; default-off candidate;
no performance promotion**.

This artifact compares the fixed32 node-domain candidate
`6c28fc58992e495bd8d4c8640370cc82f17316ee` with the static-schedule
candidate `a5174ed5e8ac2d5768a4a9e0fda16786c564e40a`. It compiles exact B1 and
B4 physical32, K64/root1, 16-key-head, 48-value-head, BV8 GDN GQA3 kernels
for `sm_121a`. Both the base closure and the K-norm/gate/decay committer
closure are covered.

## Source change

The baseline kernel reads five immutable physical32 descriptor arrays:
roots, per-root path counts, path IDs, path lengths, and branch nodes. The
candidate removes all five device pointer arguments and reconstructs the
same five roots, eleven executed paths, and twenty-seven branch nodes with
exact integer formulas. The host still fail-closes on the authenticated
descriptor execution SHA, tensor/device/dtype/layout checks, and exact
descriptor extents before launching the candidate.

This removes 59 executed descriptor loads per CTA. The unchanged grids have
12,288 CTAs per B1 48-layer event and 49,152 CTAs per B4 event, giving
724,992 and 2,899,968 removed descriptor loads respectively. These are
deterministic source/runtime operation counts, not timing estimates.

## Codegen result

B1 and B4 resolve to the same resource and instruction counts:

| Profile | Variant | Registers/thread | Non-control SASS | LDG / STG | Stack / local | LDL / STL / calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Node-domain baseline | 116 | 2,052 | 85 / 54 | 0 / 0 | 0 / 0 / 0 |
| Base | Static schedule | 116 | 2,012 | 74 / 54 | 0 / 0 | 0 / 0 / 0 |
| Committer | Node-domain baseline | 122 | 2,155 | 85 / 82 | 0 / 0 | 0 / 0 / 0 |
| Committer | Static schedule | 118 | 2,119 | 74 / 82 | 0 / 0 | 0 / 0 / 0 |

Static LDG sites fall by 11 in both profiles. Non-control instruction sites
fall by 40 (1.95%) in the base profile and 36 (1.67%) in the committer
profile. The base register count is unchanged; the committer closure removes
four registers/thread. Stores are unchanged and no spill, local-memory, or
device-call instruction appears. Eleven static LDG sites encode the 59
executed dynamic descriptor loads removed by the looped schedule.

Two independent fresh-cache builds are byte-identical across all generated
files, including PTX, cubin, SASS, compiler IR, resource dumps, and summaries.

## Decision

The candidate passes this offline static codegen/resource gate and remains
credential-gated. It is not performance-promoted. No GPU kernel was launched,
no serving task ran, and no acceptance, TPS, full-step latency, or
hardware-floor measurement was collected. Promotion requires real
SWE-Verified byte-equivalence and the standing real B1/B4 workload timing
gates.

Only sanitized summaries and reproduction code are checked in. Cubin, PTX,
SASS, compiler IR/cache, raw logs, task/model/request/response/patch content,
credentials, environment dumps, process IDs, and container IDs are excluded.
