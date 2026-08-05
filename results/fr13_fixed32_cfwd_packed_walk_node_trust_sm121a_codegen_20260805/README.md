# Fixed32 CFWD packed-walk node trust: offline SM121a audit

This artifact binds the served packed physical-slot walk at
`8c59e28f4afa56edb526562df78fd72c39561616` against the default-off trusted-node
candidate at `ed66c077bd543f90ad18a78ea974325227a21d7d`. The candidate targets the
fixed32 Tail23/Hydra27 contract for K64 B1 and B4. It is not wired into the
served overlay.

The credentialed CFWD v3 producer already proves that row zero is the root and
that every accepted child is one of physical rows 1 through 31. The candidate
consumes that proof in the fixed 12-level packed walk. It removes 48 redundant
node-domain clamps and 24 redundant leaf-domain comparisons per request. It
also masks the self-token load to the one possible leaf instead of issuing it
at every level, and removes the unreachable root bonus-token fallback. The loop
bound remains 12, so topology work does not grow with the logical Tail23 or
Hydra27 tree while both use the same physical32 layout.

Offline `sm_121a` codegen produced the same result for B1 and B4:

| Metric | Served base | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Registers | 46 | 44 | -2 |
| Static LDG | 35 | 24 | -11 |
| Static STG | 41 | 41 | 0 |
| Static non-control SASS | 509 | 496 | -13 |
| Encoded SASS | 520 | 512 | -8 |
| Cubin bytes | 46,320 | 43,584 | -2,736 |

Both builds have zero stack, local, LDL, STL, CALL, and shared-memory use. Two
independent cold-cache builds produced byte-identical summaries. CPU oracle
tests cover both modes, B1/B4, rejection and full-depth boundaries, and 64
random seeds per mode/batch against the credentialed packed-walk oracle.

This is static and CPU evidence. It does not establish device byte equality or
runtime speed. The next gate is lossless byte equality on one real
SWE-Verified task for B1 and B4, followed by exact4 and exact16 full-step timing
if the byte gate passes.
