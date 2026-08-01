# Fixed32 Tail23 and all-parent integration

Status: **source/static integration only; no GPU timing or acceptance claim**.

## Integrated changes

The fixed32 Tail mode now activates the two already-generated depth-1 rescue
rows at draft-local nodes 6 and 7, paths `(1, 0)` and `(2, 0)`. This raises
logical active drafts from 21 to 23 without changing the physical topology:

- Tail mode: 23 logical active drafts, mask `0x7a9ce7ff`.
- Hydra mode: 27 logical active drafts, mask `0x7abdffff`.
- Both modes: 31 physical drafts and 32 root-inclusive tree rows.
- Both modes retain the same physical parent vector, tree-attention bias, fixed
  drafter schedule, GDN schedule, and committer capacities.

The fixed32 all-parent TAW candidate is also integrated. It batches the fixed
union of 13 self rows and 17 target-parent rows, uses two full-vocabulary
softmax calls, and performs one integer-only exact commit launch per request.
The candidate is default off and requires a new live byte-equivalence gate.

## Non-scaling contract

Changing the logical validity mask between Tail23 and Hydra27 does not change
physical work. The static B1 census for either mode is:

- Drafter: 4 MTP forwards, 3 Arctic lookups, 12 requested lookup tokens, and a
  `[1, 31]` packed draft output.
- Tree attention: 16 calls, 512 query rows, and a fixed `[32, 32]` bias.
- GDN scan: 48 scans, 96 launches, 576 path programs, 3,936 padded slots, and
  a fixed critical path of 12.
- KV committer: fixed path capacity 16, 48 layer calls, 4 ring gathers, 5
  neutralization operations, and zero host flag/lens readbacks.
- All-parent TAW candidate: 13 self rows, 17 target rows, 2 full-vocabulary
  softmax calls, and 1 exact commit launch per request.

B4 multiplies request-row/program counts by four where appropriate, but Tail23
and Hydra27 remain physically identical at the same batch size. Work does not
scale with the logical active count while the fixed32 route is selected.

## Static verification

- Fixed32 semantics: 7 tests passed across both modes.
- All-parent focused suite: 34 passed, 1 CUDA test skipped on this CPU-only
  host.
- Geometry, work-census ledger, floor propagation, and pretask metrics: 29
  passed.
- Mask ownership selection: 4 passed.
- Final preseed suite: 54 passed.
- Python compilation and `git diff --check`: passed.

The full ownership/preseed collection additionally reached 80 passing tests;
8 CUDA/Triton collection cases cannot run because this host has no Triton
installation. Those are environment skips, not real-task qualification.

## Required live gates

No container, GPU, real SWE-Verified task, timing, TPS, acceptance, or hardware
floor measurement was run for this integration artifact. The next gates are:

1. Real SWE-Verified exact4 B1 reference-returning byte diagnostic for the
   all-parent candidate and Tail23 acceptance diagnostic.
2. Real SWE-Verified exact4 B4 byte-equivalence gate.
3. Source-bound production timing on the standing exact4 and exact16 task sets,
   reporting full-wall TPS and full-step GPU-component TPS for both B1 and B4.
4. Nsight confirmation of the expected 24-to-2 softmax and 12-to-1 exact
   commit launch reductions, with no topology/work drift or fallback.

The source model of 2-6 ms/event savings for all-parent TAW remains a forecast,
not a measurement. It must not be used for hardware-floor acceptance.
