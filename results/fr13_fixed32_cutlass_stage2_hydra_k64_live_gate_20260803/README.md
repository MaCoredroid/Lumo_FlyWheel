# FR13 fixed32 CUTLASS Stage2 Hydra27 K64 live gate

Status: `QUALIFIED_BYTE_EXACT`.

This real SWE-Verified exact4 B4 diagnostic qualified the fixed32
`identity_stockshape_stage2_b4` route for Hydra27 at K64/root1. The diagnostic
always served the stock result and collected no timing samples.

All four canonical tasks reached terminal evaluation: two resolved and two
failed. The kernel comparator exercised all five audited target projection
shapes for 320 calls. Every BF16 output byte matched stock: zero mismatching
comparisons and zero differing bytes.

The launch and end runtime manifests were identical, as were the launch and
end external manifests. The source was pinned at `4db00ec4c`; the candidate
binary was 117,488,608 bytes with SHA-256
`c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29`.

## Closeout repair

The four-task campaign and manifest closeout completed, but the original
unprivileged reducer could not read the container-created, root-owned mode-0400
real-event marker. The exact source-pinned reducer was rerun with read
privilege against the unchanged evidence, then issued the production
credential. Commit `c506dcc34` fixes the runner by keeping the immutable marker
root-owned and granting privilege only to the closed-over reducer.

No task identifiers, prompts, responses, patches, raw logs, environment,
process identifiers, or container identifiers are published here.

## Scope

This is a byte-correctness and real-task provenance result. It is not a timing,
TPS, one-sided-U95, exact16, or hardware-floor acceptance result.
