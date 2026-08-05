# Fixed32 committer alias-group guard SM121a codegen

Status: **offline SM121a codegen passes; keep default-off for real SWE-Verified
byte and timing gates**.

The incumbent sticky guard assigns one program to each layer/request. The
candidate assigns one program to the three layers that exactly alias one state
bank for one request. It retains the full physical-32 row-domain checks,
accepted path/length checks, exact alias membership validation, destination
uniqueness, and the process-lifetime fail-closed scalar.

The candidate is armed only by
`FR13_FIXED32_COMMITTER_ALIAS_GROUP_GUARD=1` or its boot arm file. It also
requires the existing sticky-guard and direct-metadata routes. With the arm
absent, the incumbent owner-program guard is unchanged.

## Static result

| Metric | B1 incumbent / candidate | B4 incumbent / candidate |
|---|---:|---:|
| Programs/event | 48 / 16 | 192 / 64 |
| Peer running-row values | 144 / 48 | 2304 / 768 |
| Physical SSI row values | 1536 / 1536 | 6144 / 6144 |
| Registers/thread | 16 / 32 | 16 / 30 |
| Stack/local bytes | 0 / 0 | 0 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Shared bytes / BAR | 0 / 0 | 0 / 0 |
| Static SASS/program | 130 / 239 | 150 / 262 |
| Aggregate static SASS | 6240 / 3824 | 28800 / 16768 |
| Aggregate static LDG | 384 / 272 | 1536 / 1088 |

The program is larger because it validates three layers and the compact alias
table, but the grid is three times smaller. Aggregate static instruction work
falls 38.7% at B1 and 41.8% at B4; aggregate static LDG issue count falls
29.2% at both batches. Both fresh-cache builds are byte-identical and spill
free.

## Qualification gate

This artifact is static evidence only. Production enablement still requires:

1. Exact byte/state equality on the standing real SWE-Verified 4-task set at
   every reachable accepted length.
2. Paired B1 full-step timing on that same 4-task set.
3. The standing 16-task confirmation if B1 improves without regressions.
4. B4 byte and full-step timing before any B4 claim.

No GPU execution, service run, task run, acceptance, TPS, timing, or
hardware-floor claim is included here.

The package contains reduced summaries and reproduction code only. It excludes
cubin, PTX, SASS, compiler caches, raw logs, task/model/request/response/patch
content, credentials, process IDs, and container IDs.
