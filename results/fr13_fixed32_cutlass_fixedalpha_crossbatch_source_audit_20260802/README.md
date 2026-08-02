# Fixed-alpha cross-batch CUTLASS source audit

This artifact records the offline gate for a default-off fixed32 CUTLASS
candidate covering the existing Tail23/Hydra27 K64/root route at B1 and B4.
It contains no GPU timing, synthetic probe, raw log, task identifier, model
identifier, or production credential.

## Candidate

- B1 keeps the divisor-balanced static scheduler, `128x32x128` tile,
  swap-AB layout, and ordered full-K mainloop.
- B4 keeps the static persistent scheduler, `128x128x128` tile, normal layout,
  and ordered full-K mainloop.
- Both replace the generic alpha/beta epilogue visitor state with an empty
  argument object and a fixed FP32 scalar value of one.
- The FP32 multiply and normal FP16/BF16 output conversion remain in the
  generated instruction stream. The patch does not bypass rejection sampling
  or change the drafter, K64 vocabulary profile, tree, mainloop, projection
  shapes, or output type.

The selectors `fixedalpha_static` and `fixedalpha_static_byte_ab` are additive,
default off, and restricted to M=32 or M=128 real projection shapes.

## Offline result

The isolated pinned SM121a translation unit compiled without warnings. All
four candidate instantiations use 168 registers, 1,024 bytes shared memory,
zero stack, and zero local memory, exactly matching their B1/B4 baselines.

For FP16 and BF16 alike, B1 decreases from 936 to 912 SASS slots and B4 from
1,440 to 1,416. Each candidate removes eight constant-load opcodes and 384
text bytes. QMMA, FFMA, FMUL, LDSM, STSM, F2FP, and synchronization counts are
unchanged; there are no calls or local-memory instructions.

This passes the source, compile, resource, and static-codegen gates. It is not
a performance acceptance result. Runtime value must be established later on
the standing real SWE-Verified workload: B1 may be used only as a one-task
diagnostic, while acceptance requires the pinned exact4 or exact16 campaign.

## Files

- `manifest.json`: pinned source/build identities and gate verdicts.
- `kernel_resources.tsv`: resource usage for candidate and matched baselines.
- `sass_summary.tsv`: reduced SASS counters and canonical instruction hashes.
- `test_results.txt`: host-only verification summary.
- `SHA256SUMS`: checksums for the reduced artifact files.
