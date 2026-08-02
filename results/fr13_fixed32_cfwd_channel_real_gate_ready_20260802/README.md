# Fixed32 CFWD Channel Commit Gate Readiness

This artifact records the host-side and offline-codegen readiness of the
`fixed32_channel_zeroelide_source_col0` convolution committer candidate.

## Scope

- Code commit: `75caa9cf67d98eb0b711d374139903a745c65039`
- Candidate route: `fixed32_channel_zeroelide_source_col0`
- Launch shape: 10 CTAs per layer and request
- Qualification selector: `FR13_FIXED32_CONV_CHANNEL_ZEROELIDE_COMMIT=diagnostic`
- Qualification scope: one-task B1 diagnostic only
- Default and formal paths remain strict and unchanged

The real-event gate reference-serves previously unseen accepted lengths,
restores the pre-event state, runs the candidate, and compares every destination
convolution byte while also checking that companion SSM rows remain unchanged.
Coverage, remaining lengths, launch counts, and served-event counts are retained
in the terminal boundary. The older flat-commit selector is not launchable.

## Offline Evidence

- B1 and B4 SM121a compile contracts pass.
- Both variants use 48 registers per thread, zero stack/local/shared bytes, and
  contain no `CALL`, `LDL`, or `STL` instructions.
- Semantic source reads are 2,949,120 bytes per B1 event and 11,796,480 bytes
  per B4 event. Destination writes are 33,423,360 and 133,693,440 bytes.
- 72 focused host tests passed.
- The work-census self-test passed all 167 tamper cases.
- Independent review found no remaining correctness blocker.

## Limits

No GPU execution, real SWE-Verified task, timing measurement, acceptance result,
or hardware-floor result is claimed here. The real B1 gate is still required.
The gate is not B4-ready: its current attempt-to-coverage invariant is exact for
B1 but must be generalized before a B4 event may expose multiple new lengths.
