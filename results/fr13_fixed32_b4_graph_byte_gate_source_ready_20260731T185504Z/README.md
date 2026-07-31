# FR13 fixed32 B4 graph byte-gate source readiness

Status: **SOURCE READY**

This artifact records source and CPU-test readiness only. It is not a GPU
result, timing result, hardware-floor result, or production-eligibility PASS.
No container or GPU workload was launched because the exact4 B1 campaign owned
the device during implementation.

## Source identity

- Branch: `agent/fixed32-b4-livegate-ready`
- Base rejection artifact commit: `dfc783bc9b98db471a3a1a29386ac5539e0cc0ff`
- Graph-gate source commit: `374aed180b6b7474770eb485f2b55ca38c6b18e4`
- Candidate: combined fixed32 batch GDN `BV=64`, reference `BV=8`
- Real campaign: canonical SWE-Verified exact4 task set only

## Closed execution contract

- `FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB` is a separate exact `0|1`,
  default-off selector.
- Eager and graph diagnostics are mutually exclusive. Graph mode requires
  fixed32 B4, `MAX_NUM_SEQS=4`, metrics, K/V/A/B ring export, in-kernel flags,
  `BV=8` served geometry, and `ENFORCE_EAGER=0`.
- The graph PID1 contract remains the default 47 arguments. The rejected eager
  diagnostic remains a distinct 48-argument `--enforce-eager` contract.
- Only final exact-B4 FULL capture opens collection. It freezes exactly 48
  unique `A_log` layer identities against the full-graph ID and structural
  signature.
- The served graph remains the established four-request BV8 route. BV64 is not
  captured or served by the diagnostic graph.
- After an authenticated real-task B4 replay, each layer runs explicit BV8 and
  BV64 shadows from the same graph-produced snapshot. Output, K/V/A/B rings,
  compact export, export tail, flags, and invocation counter are byte-compared.
- The shape/device-global export scratch is not used as a per-layer graph
  baseline. It is still covered by compact reference/candidate comparison,
  untouched-tail comparison, and full restoration.
- Every shared surface is restored and byte-verified in `finally`. A mismatch,
  zero carrier, signature drift, record drift, restore failure, or PASS-file
  publication failure cannot produce an in-memory or on-disk PASS.
- Authenticated ingress accepts exactly one eager or graph enabled sidecar and
  publishes the shared real-event marker before invoking the admitted request.
- The runner requires a graph/task/signature-bound 48-layer JSONL and PASS
  record. It remains explicitly non-timing and floor-ineligible.

## Verification

- Fixed32 suite: `633 passed, 7 skipped`.
- Focused graph/ingress/launcher/process slice: `119 passed`.
- Python compilation: passed for the kernel, ingress, patcher, and contract.
- Shell syntax: passed for launcher, serving orchestrator, and B4 runner.
- `git diff --check`: passed before source commit.
- Independent review found the shared-export blocker; the fix and a shared-cache
  regression test are included. No acceptance-integrity blockers remained.

One supplemental legacy FR10 wiring file still has three stale source-string
assertions already absent at the base commit. Those failures are unrelated to
this graph gate and are not part of the fixed32 readiness suite.

## Next gate

When the GPU is free, run `scripts/fr13_run_b4_gdn_wide_live_gate.sh` on the
canonical exact4 task set. A real run must produce the graph-specific PASS
artifact before any production arm or hardware-floor timing campaign can begin.

