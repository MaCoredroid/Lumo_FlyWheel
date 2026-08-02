# CUTLASS identity stage-two loadable gate readiness

This artifact records the host-only link, ABI audit, and default-off B1 gate
wiring for the fixed32 CUTLASS identity-plus-two-stage candidate on the
Tail23/Hydra27 K64/root route.

## Result

- The SM121a candidate object was linked into an importable AArch64 ELF shared
  object and pinned by SHA-256 and byte size.
- The production divisor binary's 1,303 defined dynamic symbols are retained;
  the candidate adds five symbols and removes none.
- All 183 undefined dynamic symbols, all nine `DT_NEEDED` entries, and the
  runtime search path are unchanged relative to that production reference.
- The binary verifier accepts only the pinned identity-stage2 shared object for
  the diagnostic selector.
- The K64/root one-real-SWE B1 gate path is wired end to end. It serves stock,
  records candidate comparisons, and keeps the direct selector unavailable
  until byte equality passes.

This is readiness evidence only. No GPU kernel was executed, no Docker service
was started, and no real workload, timing, acceptance, or hardware-floor claim
was made. The binary is retained outside Git and is identified by the manifest.
This artifact contains no task identifier, model identifier, request/response
content, raw log, credential, environment dump, PID, or container identity.

## Files

- `manifest.json`: source, binary, ABI, and gate identities and verdicts.
- `abi_summary.tsv`: reduced ABI comparison against the production divisor
  binary.
- `test_results.txt`: host-only verification summary.
- `SHA256SUMS`: checksums for the reduced files.
