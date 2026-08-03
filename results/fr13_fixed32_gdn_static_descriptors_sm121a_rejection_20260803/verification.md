# Verification

- Focused fixed32 GDN tests: 21 passed, 1 CUDA-only test skipped.
- Candidate and selector-off incumbent compiled offline for `sm_121a`.
- Selector-off SASS is byte-identical to the pre-experiment incumbent SASS.
- Both arms have zero stack bytes, zero local bytes, and no `LDL`, `STL`, or
  `CALL` instructions.
- Candidate removes 15 static `LDG` sites and 256 SASS instructions.
- Candidate raises registers/thread from 99 to 112, so it is rejected.
- Source is reverted at the branch tip; no production or live-gate selector is
  retained.
- No GPU timing, SWE-Verified acceptance, or hardware-floor claim is made.
