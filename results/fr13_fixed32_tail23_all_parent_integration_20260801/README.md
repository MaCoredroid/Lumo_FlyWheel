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

The production gate is now fail-closed to this exact Tail23 source. Individual
PASS v2 records bind both valid masks, the physical topology digest, the 13/17
source-row identities, target-parent slots, and both uniform-level schedules.
A production bundle requires independent B1 and B4 records, so a B4 replay
cannot qualify B1. Unqualified B2/B3 calls explicitly use the exact reference
path; optional B2/B3 candidate use requires their own replay records. Any
B2/B3/B4 record must carry a canonical exact4/exact16 campaign SHA marker;
B1 may still qualify on one real SWE-Verified task.

The standard real-task harness can now produce this evidence. B4 uses one
campaign-scoped marker across all concurrent workers, bound to the canonical
subset path, SHA, ordered task IDs, and task count. Its artifact is mandatory
in the atomic campaign auto-commit. A missing B4 replay leaves the bundle
partial and production remains disabled.

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
- All-parent focused suite: 53 passed, 1 CUDA test skipped on this CPU-only
  host.
- B1/B4 real-task harness and canonical campaign binding: 36 tests passed.
- Fixed-work, preseed, and ownership selection: 109 passed, 8 environment-only
  tests deselected.
- Work-census self-test: passed all 162 tamper cases.
- Full depth reducer self-test, including synthetic real-task provenance:
  passed.
- Python compilation and `git diff --check`: passed.

The campaign reducer also accepts the authorized mixed route: independently
qualified batches use the candidate while unqualified batches use reference,
with one signature per batch and exact Tail/Hydra equality at each shared
batch. The broader trace-provenance suite passed 150 tests and skipped one,
with one environment-only failure because this worktree has no pinned
SWE-Verified dataset cache blob. The depth fixture now emits the required
`runner_metadata.json` and campaign proof; production validation remains
fail-closed.

## Required live gates

No container, GPU, real SWE-Verified task, timing, TPS, acceptance, or hardware
floor measurement was run for this integration artifact. The next gates are:

1. Run the source-v7 byte diagnostic on either a single real SWE-Verified B1
   task or a canonical exact4/exact16 B4 campaign. The latter keeps one
   campaign-bound marker active across all workers and can produce separate
   exact-batch records as real B1-B4 occupancies occur.
2. Confirm the emitted bundle contains independently validated B1 and B4
   records. It remains partial and cannot arm production if either is absent.
3. Source-bound production timing on the standing exact4 and exact16 task sets,
   reporting full-wall TPS and full-step GPU-component TPS for both B1 and B4.
4. Nsight confirmation of the expected 24-to-2 softmax and 12-to-1 exact
   commit launch reductions, with no topology/work drift or fallback.

The source model of 2-6 ms/event savings for all-parent TAW remains a forecast,
not a measurement. It must not be used for hardware-floor acceptance.
