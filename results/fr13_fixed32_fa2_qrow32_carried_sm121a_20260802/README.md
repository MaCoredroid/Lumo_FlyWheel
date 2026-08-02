# Fixed32 FA2 qrow32 carried-page SM121a audit

Status: **the carried-page edit passes host compilation and static resource
checks, but the full qrow32 route remains rejected at the 255-register
ceiling**.

The committed carried-page source (`61cee753c`) had not been compiled for
SM121a. The only available qrow32 object was built from an older header and
therefore could not qualify this edit. This checkpoint regenerates and
compiles three source states with the same CUDA 13.0 command:

- the retained direct-page qrow32 incumbent (`adc96dd0e`);
- the carried edit's immediate fused-initial-page predecessor (`d01ec11d1`);
- the carried-page candidate (`61cee753c`).

## Carried-edit result

Against its immediate predecessor, the carried candidate keeps 255 registers,
zero stack, zero local memory, zero spills, and zero SASS `LDL`, `STL`, or
`CALL` instructions. Static SASS changes are:

- instructions: 4,115 to 4,082, down 33 (0.802%);
- `LDG`: 69 to 68, down one load site;
- `LEA`: 237 to 202, down 35;
- ordered attention work is unchanged: 512 BF16 HMMAs, 132 FFMAs, 264 FMULs,
  176 `LDGSTS`, 288 `LDSM`, and 38 global stores.

Because this was an isolated header transformation under identical compile
flags, the one fewer static `LDG` site is consistent with the intended reuse
of the K-advance page address by the following V copy. Runtime dynamic savings
still depend on sequence length; no performance claim is made here.

## Admission result

The paired retained direct-page source reproduces its 254-register resource
tuple. The full carried candidate uses 255 registers, a regression of one
register per thread and the architectural ceiling. It is 728 static SASS
instructions smaller than that retained source, while the mandatory attention
math and memory pipeline counts remain unchanged, but the established qrow32
offline admission rule rejects a register regression to 255 before GPU
qualification. Recompiling the candidate with ptxas register-usage level 3
also reports 255 registers and does not recover headroom.

The candidate remains hidden, default-off, byte-unqualified,
timing-ineligible, and unauthorized for production. The next useful source
step is to recover at least one register without restoring the removed page
load; only then should the canonical real SWE-Verified exact4 B4 byte gates be
run for Tail23 and Hydra27.

No GPU code, container, synthetic probe, real task, task data, byte gate, or
timing measurement was used. This directory contains aggregate host-only
metadata and hashes, not objects, cubins, raw compiler logs, raw SASS, task
content, prompts, responses, patches, model traffic, credentials, process
identities, or environment dumps.
