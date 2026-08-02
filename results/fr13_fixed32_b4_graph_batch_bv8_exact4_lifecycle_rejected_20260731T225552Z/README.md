# Rejected B4 batched-BV8 lifecycle: 2026-07-31T22:55:52Z

This is a real SWE-Verified exact4 B4 correctness diagnostic. The batched-BV8
GDN kernel passed its live graph-shadow byte gate, but the campaign is rejected
because task-boundary snapshot generation 5 raced an in-flight drafter proposal.
The failed snapshot prevented terminal work-census materialization and the
formal validator never ran. This artifact contains no valid timing, TPS,
acceptance, or hardware-floor result.

## Result

- Source: `524f469cdc47aca24c4600a99d0f2ee391db984b`
- Workload: canonical real SWE-Verified exact4, B4, concurrency 4
- Tasks: `astropy__astropy-12907`, `astropy__astropy-13033`,
  `astropy__astropy-13236`, `astropy__astropy-13398`
- Geometry: Tail6 fixed32, 32 physical rows per request
- Gate event: authenticated `swe_verified:astropy__astropy-13033`
- Reference: per-request BV8, 8 physical launches per layer at B4
- Candidate: batched BV8, 2 physical launches per layer at B4
- Kernel gate: PASS, 48 of 48 layers, nine byte surfaces per layer
- Graph baseline: byte-identical
- Candidate versus reference: byte-identical
- Reference state: restored and served
- Campaign lifecycle: REJECTED

The source and launcher intentionally mark this run as a non-timing diagnostic
and `production_eligible=0`. The kernel PASS is evidence for the BV8
implementation only; it is not formal campaign acceptance.

## Lifecycle rejection

At 23:49:38Z, while the engine reported three running requests, snapshot
generation 5 returned:

```
RuntimeError('fixed32 flush saw an incomplete drafter proposal')
```

The error ack contains counters for 640 complete events and 640 forward steps.
Those are the stale counters observed at the failed snapshot, not final campaign
counters. No later snapshot succeeded, so final SFWD, DFWD, CFWD, wall-step,
work-census, acceptance, and TPS values are unmeasured.

Traffic continued after the failure and both authenticated ingress ledgers later
finalized with 151 accepted and 151 completed model requests, zero active
requests, and four task-evidence entries. This shows the proposal condition was
transient; it does not repair the failed boundary or make the campaign valid.
The orchestrator returned 1 with an empty task summary, the terminal flush
returned 2, and the formal gate validator did not run.

## Diagnosis

The failure is a lifecycle synchronization defect, not a kernel byte mismatch.
The host-side sample/proposal lifecycle can remain active outside the execution
lock used by the flush worker. Generation 5 reached proposal reconciliation
during that interval and failed closed. Later proposal traffic proves the
singleton did not remain permanently stranded: another proposal begin would
have failed if the earlier singleton had remained non-`None`.

The production repair must wait for the sample-pending condition to clear before
snapshot reconciliation, preserve fail-closed handling for sample failures, and
be requalified on a fresh exact4 B4 run from the repaired source. This diagnostic
cannot be retroactively promoted.

## Cleanup

The preserved container was retained until the failed request/ack, full kernel
gate, ingress finalization, runtime identities, and selected log evidence were
copied. Its immutable ID was then verified and only that container was removed.
Host-memory recovery returned 0. Final state: zero Docker containers, zero GPU
compute processes, 105.6 GiB host memory available, and zero swap used.

## Evidence

- `verdict.json`: machine-readable rejection and claim boundaries.
- `kernel/`: complete 48-layer byte-gate JSONL and PASS summary.
- `flush/`: immutable generation-5 request/ack and terminal retry error.
- `ingress/`: engine/proxy final summaries and ledger finalize records.
- `container_lifecycle_excerpt.txt`: physical-B4 PASS, failure, later traffic,
  and idle-finalization lines selected from the preserved Docker log.
- `arm/`: runner/orchestrator lifecycle and rejected health summary.
- `runtime/`: byte-identical launch/end runtime and external manifests.
- `attestation/`: runtime, container, and forked-FA2 identities.
- `subset_b4_four.json`: exact canonical task set.
- `cleanup_status.txt`: immutable removal and final host/GPU state.

Raw prompts, task dumps, environment dumps, ingress secrets, process
environments, and full Docker/ingress logs are intentionally not published.
Selected unpublished-file SHA-256 identities remain in `verdict.json`.
