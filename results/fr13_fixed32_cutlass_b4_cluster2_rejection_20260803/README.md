# Fixed32 B4 cluster-2 multicast rejection

This reduced artifact records a host-only SM121a compile and code-generation
audit of a default-off B4 CUTLASS candidate. The candidate used a
`64x128x128` ping-pong tile, `2x1x1` CTA cluster, identity epilogue, explicit
two-stage pipeline, and CUTLASS's generic cluster-aware scheduler. At physical
M=128, the two cluster-M CTAs cover the two fixed 64-row tiles and multicast
the shared B weight tile.

## Result

The candidate compiles for FP16 and BF16, but each kernel has a 128-byte
per-thread stack frame. Focused SASS contains 69 `LDL`, 79 `STL`, and two
`CALL` instructions per dtype. It expands to 2,064 instruction slots, versus
1,040 slots, zero stack, zero local traffic, and zero calls for the retained
two-M kernel emitted in the same object.

The compiler also emits an SM121a advisory that `.multicast::cluster` is
expected to have substantially reduced performance on this target. The
candidate therefore fails the offline code-generation gate and was removed
from the branch. It was not linked, imported, installed, or sent to a GPU.

This is compile evidence, not a speed measurement. No synthetic workload,
real task, timing run, or hardware-floor measurement was used. The object,
cubin, raw SASS, raw resource dump, generated dispatch, and build logs are not
published; their reduced identities and counts are recorded here.

## Files

- `manifest.json`: pinned source/build identities and rejection verdict.
- `kernel_resources.tsv`: candidate and retained-kernel resource comparison.
- `sass_summary.tsv`: reduced instruction evidence.
- `verification.txt`: host-only checks and scope boundary.
- `SHA256SUMS`: checksums for the reduced artifact.
