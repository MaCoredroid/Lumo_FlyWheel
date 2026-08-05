# B1 Gate A no-split attempt12 PASS

Status: **PASS for the authenticated one-task combined graph byte gate**.

Attempt12 ran from exact, pushed source commit
`6b13ff859cb5e532d43b5ab34ea83e764acd5fe9`. The fixed contract was Hydra27,
physical32, K64/root1, B1, FULL graph mode, and the canonical SWE-Verified task
`astropy__astropy-12907`.

The task resolved cleanly and the v3 traffic audit passed all ten checks. Proxy
and engine each completed ten requests with exact attempt parity, no failed or
aborted requests, and 382 authenticated work-census events.

The combined gate issued three source-bound PASS credentials:

- Qrow32 no-split matched the truthful qrow16 reference across all 16 target
  tree-attention layers with zero BF16 output and FP32 LSE byte mismatches. The
  qrow16 reference remained served during this diagnostic.
- GDN GQA-group3 matched the two-launch reference on ten authenticated
  comparator events. Reference state was restored and served.
- DFWD K64 top3 emitted its ready, engaged, and FULL-graph-captured markers.
  It removes at least 45 reduction launches per event. Its drafter proposal
  semantics intentionally differ, so it makes no drafter byte-equality claim;
  the target verifier and rejection sampling remain authoritative.

This PASS is diagnostic and non-production. Every issued credential records
`production_enabled=false`, `performance_measurement=false`, and
`floor_acceptance_eligible=false`. Source commit `6b13ff859` admits no-split as
a live gate only; its production selector and downstream composed credential
validator remain split2-only. No-split must be explicitly promoted and tested
before it can be served in the production stack.

This attempt makes no timing, TPS, acceptance, production, or hardware-floor
claim.

This directory is sanitized. It excludes prompts, responses, task outputs,
request identifiers, raw credentials, environment dumps, raw logs, process and
container identifiers, binaries, tensors, and source/runtime manifests. Each
sanitized credential summary binds the exact excluded credential by SHA-256.
