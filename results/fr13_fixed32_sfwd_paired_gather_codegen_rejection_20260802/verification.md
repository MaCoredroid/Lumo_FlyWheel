# Verification

- Source commit: `34352e2ae3b6eb5f4773de50e7d9c4d4b3d444e6`.
- Candidate B1 and B4 emit the same cubin, PTX, and SASS hashes.
- A second compile from a fresh cache reproduces both batch builds exactly.
- The tap-mask parent was compiled separately to attribute the paired-gather
  delta instead of crediting it with the earlier tap-mask improvement.
- The paired-gather semantic and source contracts pass with the existing
  descriptorless and byte-order coverage: 26 focused tests passed.
- Ruff, Python compilation, and Git whitespace validation passed.
- Global `x`, output, and source-stage pointer expressions are unchanged.
- Tap-0 multiplication and FP32 accumulation precede tap-1 multiplication and
  FP32 accumulation in both source and semantic-order coverage.
- `BAR=3` and `STS=6` are unchanged, so both target gates fail.
- No runtime correctness, performance, full-step TPS, or hardware-floor claim
  is made.

## Next shared-stage target

Static IR/SASS inspection attributes the first shared phase to converting the
coalesced row-major `current_x` load layout into the row-gather layout. The
second shared phase converts activated values back to the coalesced row-major
store layout. Direct stores from the gather layout preserve logical addresses
but create lane-to-lane row strides and are not a credible memory path.

A bounded W8 schedule compile reduces `LDS/STS` from 6/6 to 1/1, but keeps
`BAR=3`, raises `LDG/STG` from 19/12 to 35/20, reaches 64 reported registers,
and raises static SASS from 369 to 627. Schedule reduction alone is rejected.

The next bounded candidate is a channel-serial, register-resident x tile: load
each physical row as a coalesced channel vector, retain rows per channel, and
gather ancestors locally before coalesced row stores. It is viable only if
offline codegen removes at least one shared phase and one barrier without
increasing LDG/STG, exceeding 64 reported registers, or introducing spills.
