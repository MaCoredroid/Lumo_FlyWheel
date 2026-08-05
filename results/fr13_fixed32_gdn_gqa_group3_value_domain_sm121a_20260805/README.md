# Fixed32 GDN GQA3 value-domain specialization

Status: **offline SM121a codegen/resource gate PASS; default-off candidate;
no performance promotion**.

This artifact compares static-schedule revision
`cbca5f65a5af17364e356045a3e633f885908d11` with value-domain revision
`1d08d3952d806306816de12988e5aa1258620566` for exact B1 and B4 physical32
GDN GQA3 base and committer profiles.

The exact grid has 16 value tiles of width 8 over `DIM_V=128`, so every
generated value offset is in `[0,127]`. The candidate removes redundant value
bounds from three root-state loads and every node's three sibling value
load/ring-store/output-store paths: 291 mask sites per CTA, or 3,575,808 B1
and 14,303,232 B4 sites per 48-layer event. Existing B1/B4 geometry and tensor
shape checks remain fail closed.

The candidate also restricts the invocation-counter atomic to the unique
`pid_batch == 0`, `pid_kh == 0`, `pid_v == 0` writer. This preserves the
reference's exact one-increment contract and removes three redundant B4
global atomics per event.

| Profile | Variant | Registers/thread | Non-control SASS | LDG / STG | LDL / STL / calls |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Static-schedule baseline | 116 | 2,012 | 74 / 54 | 0 / 0 / 0 |
| Base | Value-domain | 108 | 1,972 | 74 / 54 | 0 / 0 / 0 |
| Committer | Static-schedule baseline | 118 | 2,119 | 74 / 82 | 0 / 0 / 0 |
| Committer | Value-domain | 118 | 2,078 | 74 / 82 | 0 / 0 / 0 |

B1 and B4 have identical resource/opcode counts. Two independent fresh-cache
builds are byte-identical across all eight generated build trees. No GPU
kernel, Docker container, serving task, acceptance run, or timing measurement
was used. Real SWE-Verified byte equivalence and B1/B4 timing remain required.

Only sanitized summaries and reproduction code are checked in; cubin, PTX,
SASS, compiler IR/cache, raw task data, and credentials are excluded.
