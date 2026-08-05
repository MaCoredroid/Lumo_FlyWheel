# Fixed32 CFWD packed-walk active-depth SM121a audit

This artifact binds the reviewed packed-v3 node-trust committer at
`ed66c077bd543f90ad18a78ea974325227a21d7d` to the default-off active-depth
candidate at `cd1398aee`. The candidate is exact for the fixed physical32
Tail23/Hydra27 B1/B4 contract and is not wired into serving.

The served node-trust kernel statically emits all 12 walk bodies. Each body is
predicated after rejection, but its instructions remain in the straight-line
kernel. The candidate emits one body behind a device loop. It exits on the
first rejection or leaf and retains the same hard 12-iteration cap. The tree's
logical node count does not control the loop bound. Output and accepted-path
initialization, packed token/row decoding, and all five published products are
unchanged.

Offline SM121a codegen is identical for B1 and B4:

| Metric | Node-trust base | Active-depth | Delta |
| --- | ---: | ---: | ---: |
| Registers | 44 | 31 | -13 |
| Static LDG | 24 | 2 | -22 |
| Static STG | 41 | 8 | -33 |
| Static non-control SASS | 496 | 81 | -415 |
| Encoded SASS | 512 | 96 | -416 |
| Cubin bytes | 43,584 | 14,920 | -28,664 |
| BRA | 1 | 2 | +1 loop branch |

Both kernels have zero stack, local, LDL, STL, CALL, and shared-memory use.
Two independent cold-cache builds produced byte-identical summaries with
SHA256 `b4d16e1d553048d68133496f9d3fd8748220acde7bbc6fbf75b8e2bf90a67d3f`.
The CPU oracle suite covers Tail23 and Hydra27, B1 and B4, root rejection,
maximum depth, and 64 randomized packed-event sets per mode/batch.

This is source, CPU semantics, and static codegen evidence only. It does not
establish GPU byte equality or runtime speed. A real SWE-Verified B1/B4 byte
gate must precede production, followed by the standing exact4 and exact16
full-step measurements if it passes.
