# CUTLASS identity stage-two source audit

This artifact records a host-only SM121a audit of a default-off fixed32
CUTLASS candidate for the Tail23/Hydra27 K64/root route. It combines the
identity epilogue with an explicit two-stage mainloop, the legal minimum for
the retained cooperative schedule.

## Result

- B1 retains the `128x32x128` swap-AB tile, divisor-balanced static scheduler,
  and ordered full-K traversal.
- B4 retains the `128x128x128` normal-layout tile, static scheduler, and
  ordered full-K traversal.
- All four FP16/BF16 B1/B4 instantiations compile with 168 registers, 1,024
  bytes shared memory, zero stack, and zero local memory.
- B1 falls from 896 identity-epilogue SASS slots to 864, a further 32-slot and
  512-byte reduction. B4 is unchanged at 1,352 slots.
- QMMA, FFMA, FMUL, LDSM, STSM, conversion, constant-load, and synchronization
  counts are unchanged relative to the identity-epilogue candidate.

The B1 result passes the offline codegen gate. The B4 result is neutral and is
not promoted on static evidence alone. Both selectors remain default off, and
real byte equality is mandatory before timing because pipeline staging changes
the generated implementation even though tile order and math are preserved.

This artifact contains no GPU timing, synthetic probe, raw disassembly,
binary, task identifier, model identifier, request/response content, log,
credential, PID, or container identity.

## Files

- `manifest.json`: pinned source/build identities and verdicts.
- `kernel_resources.tsv`: candidate and matched resource usage.
- `sass_summary.tsv`: reduced instruction counts.
- `test_results.txt`: host-only verification summary.
- `SHA256SUMS`: checksums for the reduced files.
