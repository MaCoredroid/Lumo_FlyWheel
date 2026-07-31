# FR13 fixed32 unified-attention BM8 production source readiness

Status: `SOURCE_READY_ONLY`.

Source commit `08a5fde9cbc548be6f7b00477e1bc695e4879c02` adds a
default-off, attested production selector for the B1 MTP drafter's four
`kernel_unified_attention_2d` calls. It is pushed on
`agent/fixed32-bm8-production-ready`.

The selector consumes the existing real SWE-Verified B1 byte-pass credential
from `astropy__astropy-12907`. That run dispatched the BM8 candidate four times
at sequence lengths 22,872 through 22,875 and found zero raw-byte mismatches in
12,288 output bytes per call. The production sidecar binds that result to the
qualified emitted unified-attention source SHA-256
`3baccaa1a83907e15561b1cf807f15a41bd4764513bb43c4046b434937c3274b`.

Production startup fails closed unless the exact sidecar, source hash, fixed32
B1 FULL graph, and final capture geometry all match. The candidate selector is
scoped around only the four exact drafter calls and is cleared in `finally`.
The target fixed32 decode remains on the required FA2 tree-bias route. The
capture is published as `ENGAGED` only after its first measured replay is
successfully installed; failure removes the graph from the cache.

CPU-only verification covered the BM8/fixed32/FA2/launcher/ingress/publication
surface: 90 tests passed. Independent review additionally executed the emitted
branch behavior and found no blocking issue. No GPU or Docker was used for this
source-readiness step.

This artifact is not a timing result, a production-return result, a B4 result,
or formal exact4/exact16 acceptance. The last valid Hydra27 exact4 baseline
remains 239.026634 ms/event wall and 222.558408 ms/event GPU component versus
the corrected 119.658015 ms/event floor and 137.606718 ms/event acceptance cap.
The earlier real-SWE Nsight attribution places the entire unified-attention
group at only 6.967564 ms/event, so BM8 alone cannot close the remaining floor
gap even under an impossible full-group deletion.

The next required step is one real SWE-Verified B1 production-return task. It
must produce the `ENGAGED` capture sidecar and resolve cleanly before any timing
claim. If that passes, measure the standing real-task exact4 set, followed by
exact16 for acceptance evidence. B4 remains unmeasured.
