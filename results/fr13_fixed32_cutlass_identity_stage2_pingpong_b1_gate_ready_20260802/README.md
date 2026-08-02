# CUTLASS identity-stage2 ping-pong B1 gate readiness

This artifact records a host-only SM121a build and gate-readiness audit for a
default-off Tail23/Hydra27 K64/root B1 CUTLASS hybrid.

## Candidate

The candidate keeps the divisor-balanced complete-tile scheduler, identity
epilogue, explicit two-stage mainloop, `128x32x128` tile, ordered full-K
traversal, and physical 32 rows. It changes only the consumer schedule on the
three wider projections:

- `N=14336`, `16384`, and `34816` use CUTLASS ping-pong consumers. Their exact
  divisor grids assign 4, 4, and 8 complete output tiles per CTA.
- Both `N=5120` projections retain the cooperative identity-stage2 kernel
  because their grid assigns one tile per CTA and offers nothing to overlap.

The ping-pong-eligible projections represent 73.5632% of the matrix elements
across the five audited projection shapes. This is a selection calculation,
not performance evidence.

## Result

- The BF16 and FP16 ping-pong kernels compile at 168 registers, 1,024 bytes
  shared memory, and zero stack/local memory.
- The loadable extension imports and passes its exact binary-identity verifier.
- Relative to the production divisor extension, all 1,303 defined dynamic
  symbols are retained, seven are added, and all 183 undefined symbols,
  dependencies, and runtime search paths remain equal.
- The one-real-SWE K64/root B1 byte gate is wired end to end, serves stock, and
  keeps direct production disabled until an exact-byte pass.

Ping-pong contains two consumer instruction paths, so its static QMMA/FFMA
counts are doubled. That is not doubled arithmetic per output tile; each tile
is owned by one consumer warp group. Only a real workload can determine whether
the intended mainloop/epilogue overlap improves latency.

No GPU kernel, Docker service, real task, timing, acceptance, or hardware-floor
measurement was run for this artifact. The binary is retained outside Git and
is pinned in `manifest.json`. No task/model identifier, request/response
content, raw log, credential, environment dump, PID, or container identity is
included.
