# Oracle — inventory-ops variant

This directory contains the authored oracle used to prove that the broken
`inventory-ops` repo is solvable and that the hidden tests stay aligned with
the intended behavior. Never mount these files into an agent container.

## Files

- `solution.patch` adds the visible Markdown feature and updates the usage
  docs.
- `solution_followup.patch` enables the dormant watchlist path so Markdown
  handoffs keep on-shift owners visible even when their queues are empty, and
  it fixes the shared zero-count wording used by the summary line.

## Why two patches?

`inventory-ops` is the small-investigative benchmark in this family. The
broken repo already carries most of the Markdown layout in `report_app/`,
but the CLI/docs path was never wired up. That is the round-1 fix.

Round 2 exists for the latent watchlist behavior documented in
`docs/handoff.md`: when operators review an empty queue, they still want the
on-shift owners to remain visible so "quiet" does not look like "missing".
The broken repo even exposes an `include_known_owners` flag, but it is a dead
parameter. The follow-up patch threads that flag through the shared summary
layer and fixes the `0 queued items` wording at the shared formatter.

## Expected states

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | round-1 slices pass, follow-up slices fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |
