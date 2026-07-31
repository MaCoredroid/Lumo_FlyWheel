# Rejected B4 graph-byte gate: 2026-07-31T22:27:50Z

This is a real SWE-Verified exact4 B4 correctness rejection. It proves that
the 40 GiB KV-cache configuration reaches physical four-request replay, but
the diagnostic BV64 GDN candidate is not byte-exact with the served BV8
reference. The run is intentionally non-timing and contains no valid TPS,
acceptance, or hardware-floor result.

## Result

- Source: `99d3869d6e5438e0a1c8c0ee9321ab1c9e525d02`
- Workload: canonical real SWE-Verified exact4, B4, concurrency 4
- Tasks: `astropy__astropy-12907`, `astropy__astropy-13033`,
  `astropy__astropy-13236`, `astropy__astropy-13398`
- Geometry: Tail6 fixed32, 32 physical rows per request
- Manual KV cache: `42,949,672,960` bytes
- KV capacity: `153,600` tokens
- PIECEWISE captures: 8 of 8
- FULL captures: 4 of 4
- Graph memory: `9.12 GiB`
- Live scheduler evidence: 4 running, 0 waiting, 79.5% KV usage
- Gate: post-replay shadow on an authenticated real-task event
- Reference: BV8, 8 physical launches per layer at B4
- Candidate: BV64, 2 physical launches per layer at B4

## Primary failure

The first graph-shadow record failed closed:

```
RuntimeError: FR13 fixed32 B4 graph GDN byte mismatch at record 0:
graph=[] candidate=['out', 'state_export_compact']
```

The replayed graph and the explicit BV8 reference were byte-identical on every
stable graph surface. That validates the replay snapshot/restore baseline for
this record. The BV64 candidate then differed from BV8 as follows:

- `out`: 1 differing byte of 1,572,864
- `state_export_compact`: 195,944 differing bytes of 62,914,560
- K/V/A/B rings, untouched export tail, flags, and invocation counter:
  byte-identical

The served graph state was restored before the mismatch raised. The engine
then exited by design, so no campaign timing or task result can be admitted.
BV64 is rejected from production unless a new implementation passes the same
exact-byte gate; numerical tolerance must not replace this contract.

## Diagnosis and next kernel

The evidence points with high confidence to low-bit Triton codegen/reduction
layout drift, not request or node indexing:

- The first compact-state difference is request 0, root node 0, where batch
  stride and parent-slot addressing cannot explain the difference.
- The shared node update reduces K=128 tensors at both the state correction and
  output dot. Changing `BLOCK_V` from 8 to 64 changes the containing tensor
  layout and can change Triton's thread-level reduction mapping.
- The output difference is exactly one low byte in one BF16 element, an adjacent
  BF16 encoding. The state differences begin in the low byte of an FP32 value.
- The source already documents an analogous Triton layout change that reshaped
  `tl.sum` trees and produced low-ULP drift.

The current A/B combines the new batched kernel with the BV64 width, so it
cannot attribute every difference to width alone. The next exact gate must
compare batched BV8 against per-request BV8. Batched BV8 already issues two
physical kernel launches per layer independent of B, so it can satisfy the
launch-count requirement without a wide reduction layout. BV64 must not be
timed or enabled in production.

## Capacity and cleanup

The 40 GiB KV pin fixed the previous 20 GiB B4 capacity failure: all four
canonical requests were simultaneously running before the gate fired. The
failed container was preserved until its immutable identity and logs were
captured, then removed. Final cleanup shows no Docker containers, no GPU
compute process, 105 GiB host memory available, and zero swap in use.

## Evidence

- `verdict.json`: machine-readable classification and byte counts.
- `gate/fr13_fixed32_batch_gdn_byte_ab.jsonl`: complete one-record gate output.
- `container_failure_excerpt.txt`: graph, physical-B4, and fail-closed lines.
- `diagnosis.md`: byte-offset analysis, source evidence, and next action.
- `swe_orchestrator.log`: exact dataset and four task starts.
- `launcher_meta.txt`: source, subset, runner, FA2, BV, KV pin, and exit code.
- `arm/runlog.txt`: runner lifecycle and downstream boundary failure.
- `runtime/`: byte-identical launch/end runtime and external manifests.
- `attestation/`: runtime identity, stock FA2 identity, and gate selectors.
- `cleanup_status.txt`: immutable removal and final host/GPU state.

The raw Docker inspect, full Docker input dump, process identity, container
environment, and ingress secret files are intentionally not published.
Their selected SHA-256 identities are retained in `verdict.json`.
