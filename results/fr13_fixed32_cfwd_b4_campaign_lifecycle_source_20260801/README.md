# Fixed32 CFWD B4 campaign lifecycle source review

Classification: source-only qualification lifecycle; not a performance result.

This change extends the process-local CFWD layer-batch byte gate from the B1
task bracket to a canonical SWE-Verified exact4/16 B4 campaign bracket. It does
not publish a durable production pass and does not authorize a floor claim.

## Implemented contract

- One campaign marker covers all overlapping B4 tasks. Its bytes are
  `swe_verified:campaign{4|16}_{canonical_subset_sha256}\n`.
- The marker is created atomically in the owned mode-0700 logs directory and is
  a mode-0400, single-link regular file.
- Campaign identity is bound to the validated canonical exact4/16 subset and
  concurrency 4. Per-task CFWD markers are forbidden for B4.
- A global flush snapshot is completed before marker publication. Marker
  teardown completes before the post-campaign flush snapshot.
- Reachable accepted lengths remain exactly 0 through 11 (`0x0fff`) for each
  captured batch size B1 through B4.
- B4 coverage math accounts for one event exposing up to B accepted lengths:
  every successful attempt adds at least one and at most B new coverage bits.
- Task artifacts are classified as campaign members and make no task-local
  qualification claim.
- Complete B1-through-B4 coverage creates only an in-memory timing handoff. The
  handoff rechecks the same live server binding, a later flush generation, arm
  absence, unchanged gate attempts, and unchanged full coverage.
- The serialized handoff is explicitly non-replayable, non-durable, and not a
  timing credential. Generic B4 auto-publication is disabled for this
  qualification so raw campaign output is not swept into git.

## Remaining blocker

The engine and proxy ingress state machines currently allow only
`preflight -> campaign -> finalized`; finalization is terminal. A clean timing
campaign on the same server therefore still needs a paired authenticated
ingress-phase contract. Timing execution remains explicitly unimplemented.

No GPU timing, throughput, acceptance, or hardware-floor result is claimed by
this artifact.
