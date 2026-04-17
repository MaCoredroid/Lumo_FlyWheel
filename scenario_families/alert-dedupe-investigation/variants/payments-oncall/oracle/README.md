This directory contains the authored oracle for the richer
`alert-dedupe-investigation/payments-oncall` benchmark uplift.

Round 1 (`solution.patch`) repairs the visible regression by:

- canonicalizing environment aliases and 5-minute window buckets in the parser
- rebuilding the dedupe fingerprint around service, environment, window, and
  payment lane
- restoring occurrence-count and first/last-seen aggregation for the handoff

Round 2 (`solution_followup.patch`) fixes the latent payments-specific bug:
processor-noisy page titles must prefer the stable `dedupe_hint` when one is
available so the incident family stays stable without losing the earliest
human-readable display title.
