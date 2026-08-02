# Fixed32 CFWD channel zero-elision static checkpoint

Status: **default off; source and host tests pass; compiled and real-task
qualification pending**.

This checkpoint adds a lower-CTA alternative to the flat fixed32 conv
committer. It preserves the incumbent launch geometry of 10 programs per
layer/request. For the deployed `C=10240`, `L=34`, BF16 contract, columns 0-2
read their selected source rows and columns 3-33 write literal BF16 zero.

Preseed verifies that every node maps columns 3-33 to sentinel source row 35.
The candidate is fail-closed and selected only by
`FR13_FIXED32_CONV_CHANNEL_ZEROELIDE_COMMIT=diagnostic`. It is mutually
exclusive with the flat zero-elision candidate. Captured launches use only the
preseeded route and do not read the environment.

## Static traffic accounting

| Batch | Incumbent source reads | Candidate source reads | Removed source reads | Destination writes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 33,423,360 B | 2,949,120 B | 30,474,240 B | 33,423,360 B |
| 4 | 133,693,440 B | 11,796,480 B | 121,896,960 B | 133,693,440 B |

This removes 31/34, or 91.176%, of semantic source-read bytes while leaving
destination-write bytes unchanged. Launches remain 480 CTAs/event at B1 and
1,920 CTAs/event at B4, versus 16,320 and 65,280 for the flat candidate.

## Qualification boundary

Host verification passed 98 tests; one CUDA-only module was skipped because no
GPU runtime was used. Ruff, Python bytecode compilation, and `git diff --check`
passed. This checkpoint provides no compiled-codegen, GPU execution,
byte-equivalence, timing, TPS, or hardware-floor claim. Real SWE-Verified B1
qualification is required before B4 or timing.

This directory excludes tasks, prompts, responses, patches, raw logs,
environment dumps, process/container identities, credentials, binaries,
PTX/SASS dumps, and timing samples.
