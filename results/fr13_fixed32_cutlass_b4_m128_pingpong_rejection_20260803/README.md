# Fixed32 B4 full-M ping-pong rejection

This reduced artifact records a host-only SM121a compile and resource audit of
a default-off B4 CUTLASS candidate combining a `128x128x128` tile, ping-pong
mainloop, identity epilogue, explicit two-stage pipeline, and the existing
static scheduler.

## Result

The candidate compiles for both FP16 and BF16, but each kernel has a 512-byte
per-thread stack frame. Focused SASS contains 241 `LDL` and 236 `STL`
instructions per dtype. It is rejected and was removed from the branch.

The retained cooperative-M128 identity Stage2 kernel has zero stack and 1,352
SASS slots per dtype. The retained 64-row two-M ping-pong kernel also has zero
stack and 1,040 slots. The rejected full-M ping-pong kernel expands to 2,608
slots and doubles the static QMMA and FFMA counts from 128 to 256.

CUTLASS's pinned SM120 blockwise example uses a 64-row tile for ping-pong to
avoid register spilling. The compile result confirms that guidance for this
route. A `128x256x128` ping-pong variant was not compiled: full-M ping-pong
already spills 512 bytes per thread at N=128, while the prior 64-row N=256
experiment spilled 488 bytes per thread. It is not a viable next candidate.

No GPU kernel, Docker service, synthetic workload, real task, timing run, or
hardware-floor measurement was used. The object, cubin, raw SASS, raw resource
dump, generated dispatch, and build logs are not published; their reduced
identities and counts are recorded here.

## Files

- `manifest.json`: pinned source/build identities and rejection verdict.
- `kernel_resources.tsv`: candidate and retained-kernel resource comparison.
- `sass_summary.tsv`: reduced instruction evidence.
- `verification.txt`: host-only checks and scope boundary.
- `SHA256SUMS`: checksums for the reduced artifact.
