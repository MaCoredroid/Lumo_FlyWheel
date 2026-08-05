# Fixed32 K64 packed draft-head selector readiness

Status: `STATIC_SELECTOR_PASS_LIVE_UNMEASURED`

This checkpoint wires the packed-load `warp4_pair8` draft-head binary into the
exact Hydra27 physical32 route for both B1 and B4. The selector is default off,
source authenticated, and fail closed. It directly serves proposal logits;
there is no incumbent BF16 shadow call and no drafter-quality equality gate.

The target rejection sampler and target logits remain authoritative and were
not modified. The K64 block map is retained, so selected row indices are mapped
back to real vocabulary token IDs before they enter the speculative tree.

## Exact route

- Mode: `hydra27_fixed32`
- Physical drafts: 31; active nodes: 27
- Draft vocabulary: 65536 rows; root reduction enabled
- B1 op: `gemvx_m1_warp4_pair8_out`
- B4 op: `gemvx_m4_warp4_pair8_out`
- Graph contract: one or more root selections and exactly four captured loop
  selections, with zero fallback calls
- Enable: `FR13_DRAFT_HEAD_B14_WARP4_PAIR8=1`
- Credential: `FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT=$(git rev-parse HEAD)`

## Evidence boundary

CPU/source gates passed for B1 and B4 and the linked candidate identities were
revalidated. No GPU, Docker, SWE-Verified task, throughput, or acceptance
measurement was run for this selector checkpoint. It is ready for the next
real B1/B4 candidate-served gates, but is not acceptance evidence.

