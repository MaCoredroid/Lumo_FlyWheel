# Fixed32 B4 two-M scheduler-state rejection

This reduced artifact records three host-only SM121a code-generation attempts
on top of the default-off B4 two-M one-dimensional scheduler. All candidates
preserve the fixed32 K64/root1 projection shapes, the `64x128x128` tile,
ping-pong mainloop, identity epilogue, two-stage pipeline, and even X-only
persistent grids.

## Result

Splitting the persistent linear index into separate M and N state introduced an
8-byte stack frame plus three `LDL` and one `STL` instruction per dtype. A
stateless form that advanced from CUTLASS `WorkTileInfo` had the same stack
traffic and expanded from 1,032 to 1,192 SASS instructions per dtype. Both are
rejected.

A final one-word N-state form restored zero stack and local traffic, but still
compiled to exactly 1,032 instructions. It exchanged two uniform branches and
two uniform comparisons for scalar equivalents without reducing register use,
tensor math, or total instructions. This is not a defensible kernel win and is
also rejected.

The branch was restored to the reviewed one-dimensional scheduler source. No
GPU kernel, synthetic workload, real task, timing campaign, or hardware-floor
measurement was run. Raw objects, cubins, SASS, resource dumps, generated
dispatch files, and build logs are not published; only reduced identities and
counts appear here.

## Files

- `manifest.json`: source binding, toolchain identity, and final verdict.
- `kernel_resources.tsv`: reduced resource comparison for both output dtypes.
- `sass_summary.tsv`: reduced instruction comparison.
- `verification.txt`: checks and explicit scope boundary.
- `SHA256SUMS`: checksums for this reduced artifact.
