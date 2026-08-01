# B4 fixed32 batched-GDN audit at 080c417ed

Status: no existing candidate is production-qualified for the exact current
source and both Tail23/Hydra27 K64 routes. No GPU, Docker, synthetic probe, or
new performance run was used for this audit.

## Best existing candidate

`fixed32_batch_gdn_bv8_v1` is already present in the current source behind a
fail-closed production credential. It folds request identity into path-grid
axis 2 while preserving the fixed physical-32 parent tree, the two dependency
levels `[1, 11]`, BV8, node order, recurrent node visits, rings, flags, and
output stores.

At physical B4 the structural change is:

| Scope | Per-request reference | Batched BV8 | Reduction |
| --- | ---: | ---: | ---: |
| Launches/layer | 8 | 2 | 75% |
| Launches/48-layer event | 384 | 96 | 288 |
| Paths/request/layer | 12 | 12 | 0 |
| Recurrent nodes/request/layer | 32 | 32 | 0 |

This is a launch-coalescing optimization. It does not reduce CTA count,
recurrent work, or issued q/k/v work at a fixed BV8 geometry.

## Evidence that transfers only informatively

A prior authenticated real SWE-Verified exact4 B4 K64/ROOT1 graph-shadow gate
passed under the old `tail6_fixed32` label at execution source `cc91aa304` and
kernel source SHA-256
`8d0b4c592921880a000c2a19ef37745310c90183b8d4c464f52befde53203886`.
All 48 layers passed nine candidate/reference byte surfaces, covering
5,009,179,200 candidate/reference bytes and 177,340,800 graph-baseline bytes.
The reference was restored and served. The candidate was not timed.

That source had 21 active drafts and valid mask `0x7a9ce73f`. Current Tail23
retains the same mode label but has 23 active drafts and mask `0x7a9ce7ff`.
The prior pass therefore covers neither current Tail23 nor Hydra27.

The candidate recurrence helper, batched kernel, launch contract, and batched
launcher have identical AST hashes between `cc91aa304` and `080c417ed`. The
selector changed and unrelated code was added to the same module, however, so
the whole-file source credential does not transfer.

The reviewed coefficient-staging branch `f5ccbdfdd` contains an offline SM121
build of the B4 batched kernel with zero stack, zero local memory, and zero
spill instructions. Its B4 candidate specializations use 79 registers at
level 0 and 64 at level 1. That build is bound to source SHA-256
`16fde18ebf4ace9893d2f8890294c894c71222b85d7c9cdc4bc7789cf5afff4e`
and `COUNT_INVOCATION=False`; it is not a resource qualification for the
current `080c417ed` production specialization.

## Blocking facts

1. Current kernel source SHA-256 is
   `01af99bdec789b21de3f68334d640f22bd6ebcb7bfce11f6650814f915143b3c`,
   not the passed `8d0b...` source. The production validator deliberately
   rejects this drift.
2. The prior real byte gate is pre-Tail23 Tail21. Current source permits the
   old credential under either fixed32 mode because all modes use the same
   physical parent tree, but there is no authenticated Tail23 or Hydra27 byte
   comparison on this source.
3. No zero-spill build artifact exists for the exact current source,
   `COUNT_INVOCATION=True`, B4 BV8 specialization.
4. Existing BV8 runs are graph-shadow correctness diagnostics with the
   reference served. They provide no eligible candidate wall TPS or phase
   timing.
5. Wide BV64 is excluded: the real exact4 B4 gate at `99d3869d6` failed on
   one `out` byte and 195,944 `state_export_compact` bytes. BV16, BV32, and
   BV128 have no authenticated byte PASS.

## Work-reduction companion

The separate level-0 coefficient-staging candidate keeps two launches and 32
recurrent node visits but reduces repeated work per request/layer:

- q/k normalization instances: 24,576 -> 4,352 (`82.2917%` reduction);
- raw-gate fold instances: 24,576 -> 5,376 (`78.125%` reduction).

It has reviewed zero-spill offline builds, but no live B4 byte gate or timing.
It therefore cannot be stacked into production yet.

## Ceiling and next gate

The maximum production-plausible reduction from the existing exact-capable
batch topology is 8 -> 2 launches/layer, or 384 -> 96 launches/event. A
one-launch topology could reach 48 launches/event, but the existing feasibility
work either increases recurrent work from 32 to 62 nodes/request or serializes
the critical chain from 12 to 32; neither route is byte-gated or performance
qualified.

The next valid sequence is: build the exact `080c417ed` BV8 B4 specialization
and record resources, run source-bound reference-served exact4 graph byte gates
for Tail23 and Hydra27, then run same-source exact4 stock/candidate timing. The
candidate must remain default-off until those gates pass.

Current-source static verification passed 79 focused batched-GDN, wide-BV,
graph-gate, and BV8-production tests.
