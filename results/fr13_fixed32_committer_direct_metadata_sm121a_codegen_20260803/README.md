# Fixed32 direct committer metadata SM121a codegen

Status: **offline SM121a codegen passes; retain default-off for a real byte gate**.

The fixed32 TAW publisher already owns persistent capacity-sized
`accepted_paths` and `accepted_lens` buffers whose storage addresses are pinned
during all-B graph preseed. Commit `0e2f3b940ee7076e7818da4e048206a978236f04`
adds a default-off route that captures both the native reference and the
one-launch 48-layer committer directly against the B-specific views of those
buffers. An exact pointer/batch/stream one-shot lease binds the preceding
physical32 row guard to the replay and fails closed if the order or storage
changes.

This lets treeconv select `_fr13_fixed32_conv_direct_col0_kernel` instead of
`_fr13_fixed32_conv_direct_col0_metadata_kernel`. The candidate removes the
duplicate 16 path values plus one length value per request. It does not change
tree topology, rejection sampling, accepted lengths, recurrence math, state
destinations, or output publication.

## Codegen result

| Metric | B1 incumbent / candidate | B4 incumbent / candidate |
|---|---:|---:|
| Registers/thread | 34 / 34 | 36 / 36 |
| Stack/local bytes | 0 / 0 | 0 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Encoded SASS | 744 / 728 | 776 / 744 |
| Static non-control SASS | 458 / 445 | 488 / 459 |
| Static LDG / STG | 12 / 274 -> 11 / 272 | 12 / 274 -> 11 / 272 |
| Metadata loads per event | 16 / 0 | 64 / 0 |
| Metadata stores per event | 17 / 0 | 68 / 0 |
| Intermediate copied elements | 17 / 0 | 68 / 0 |

The two independent fresh-cache builds produced identical summaries, cubins,
PTX, and SASS for the same full source revision. All builds used the deployed
physical32 BF16 geometry: 48 layers, C=10240, state length 34, source rows 36,
zero-tail live columns 3, channel block 1024, four warps, and target `sm_121a`.

## Decision

Keep the route default-off behind
`FR13_FIXED32_COMMITTER_DIRECT_METADATA=1` or its boot arm. Host/source closure
and SM121a resource gates pass, but no GPU kernel, service, task, timing, or
acceptance run was launched. A real SWE-Verified byte gate across reachable
accepted lengths remains mandatory before the candidate may serve or support a
speed claim.

The checked-in package contains reduced summaries and reproduction code only.
It excludes cubin, PTX, SASS, compiler caches, raw logs, task/model/request/
response/patch content, credentials, environment dumps, process IDs, and
container IDs.
