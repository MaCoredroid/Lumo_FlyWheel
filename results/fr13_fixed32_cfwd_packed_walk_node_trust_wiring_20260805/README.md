# Fixed32 packed-walk node trust: default-off CFWD shadow wiring

This artifact binds source commit
`3c282de0698c4e53aa4e991010b39f8660178e6f`. It wires the trusted-node packed
walk into the reviewed CFWD v3 runtime without changing the credentialed CFWD
producer or comparator source.

`FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB` defaults to `0` and accepts only
literal `0` or `1`. With `0`, the wrapper leaves the installed packed-v3 walk
function unchanged. With `1`, it requires the existing CFWD diagnostic shadow,
Hydra27, physical32 geometry, and batch 1 or 4. The candidate writes into the
existing candidate walk buffers; `_fr13_cfwd_logit_direct_compare` compares
those buffers against the incumbent result, and the incumbent remains served.

The checked-in B1 runner is a source-bound front end to the canonical real
SWE-Verified one-task CFWD byte gate. It fixes K64/root1 and Hydra27 physical32
through the base gate and explicitly arms only the new walk selector. It is not
a probe or performance run. B4 is admitted by the runtime selector, but its
runtime gate must use the standing exact4 task set before exact16 acceptance.

The kernel source is unchanged from the offline SM121a artifact in
`results/fr13_fixed32_cfwd_packed_walk_node_trust_sm121a_codegen_20260805`:
both B1 and B4 reduce registers from 46 to 44, static LDG from 35 to 24, and
static non-control SASS from 509 to 496. Those are static codegen results, not
runtime speed measurements.

No GPU, Docker, service, request, response, real task, timing, or acceptance
path ran while producing this wiring artifact. The next required step is the
real one-task B1 byte gate, followed by exact4 B4 byte coverage and exact4/16
full-step timing only after zero mismatches.
