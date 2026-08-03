# FR13 fixed32 CUTLASS two-M Hydra27 K64 live gate attempt A

Status: `REJECTED_INVALID_TERMINAL_TRACE`.

This real SWE-Verified exact4 B4 attempt exercised the fixed32
`identity_twom_b4` route for Hydra27 at K64/root1. All four canonical tasks
reached terminal evaluation: two resolved and two failed. The stock-served
kernel comparator recorded 320 calls at physical M=128 across all five audited
projection shapes. Every BF16 output byte matched stock.

The campaign did not qualify. One task's SDK trace ended after 57 turns with a
nominal success result but no nonempty final assistant text. The fail-closed
campaign provenance reducer rejected that terminal response group, the wrapper
exited nonzero, and no live or production credential was issued. The run must
not be combined with another campaign or used for timing.

Launch and end runtime manifests were identical, as were launch and end
external manifests. The source was pinned at `30f8437d2`; the candidate binary
was 117,488,608 bytes with SHA-256
`c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29`.

No task identifiers, prompts, responses, patches, raw logs, environment,
process identifiers, or container identifiers are published here.

## Scope

This is rejection and diagnostic evidence only. It is not a byte-correctness
qualification, timing, TPS, one-sided-U95, exact16, or hardware-floor
acceptance result. An unchanged fresh exact4 Hydra27 campaign is required.
