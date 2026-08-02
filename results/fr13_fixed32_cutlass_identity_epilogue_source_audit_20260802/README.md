# CUTLASS identity-epilogue source audit

This artifact records the offline gate for a default-off fixed32 CUTLASS
candidate covering the existing Tail23/Hydra27 K64/root route at B1 and B4.
It contains no GPU timing, synthetic probe, raw log, task identifier, model
identifier, request/response content, production credential, or binary.

## Candidate

- B1 keeps the divisor-balanced static scheduler, `128x32x128` tile,
  swap-AB layout, and ordered full-K mainloop.
- B4 keeps the static persistent scheduler, `128x128x128` tile, normal layout,
  and ordered full-K mainloop.
- The epilogue feeds the accumulator through CUTLASS `Identity` before the
  normal round-to-nearest FP16/BF16 output conversion.
- The change removes the explicit FP32 multiply-by-one from the fixed-alpha
  candidate. It does not change the drafter, K64 vocabulary profile, tree,
  mainloop, projection shapes, output type, or rejection sampling.

The selectors `identity_static` and `identity_static_byte_ab` are additive,
default off, and restricted to M=32 or M=128 real projection shapes.

## Offline result

The final isolated pinned SM121a translation unit compiled without warnings.
All four identity instantiations use 168 registers, 1,024 bytes shared memory,
zero stack, and zero local memory, matching their B1/B4 baselines.

For FP16 and BF16 alike, B1 decreases from 936 to 896 SASS slots and B4 from
1,440 to 1,352. Relative to the fixed-alpha candidate, identity removes a
further 16 B1 slots and 64 B4 slots. The identity path has eight FMUL opcodes
in both shapes; QMMA, FFMA, LDSM, STSM, conversion, and synchronization counts
match the baseline. There are no calls or local-memory instructions.

This passes source, compile, resource, and static-codegen gates. Removing a
floating-point operation can change bytes even when it is algebraically an
identity, so real byte equality is mandatory before timing. B1 may then be
used only as a one-task diagnostic; performance acceptance requires the pinned
exact4 or exact16 real SWE-Verified campaign.

## Files

- `manifest.json`: pinned source/build identities and gate verdicts.
- `kernel_resources.tsv`: candidate and matched-baseline resource usage.
- `sass_summary.tsv`: reduced instruction counts; no raw disassembly.
- `test_results.txt`: host-only verification summary.
- `SHA256SUMS`: checksums for the reduced artifact files.
