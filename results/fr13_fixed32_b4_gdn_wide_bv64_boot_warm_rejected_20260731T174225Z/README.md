# FR13 fixed32 B4 GDN BV64 eager boot-warm rejection

## Verdict

REJECTED before model traffic. This run does not classify the BV64 candidate
and contains no timing, TPS, acceptance, or hardware-floor result.

The run used source `b75fe43ce501dc640d893805ba375fe72b55d814`,
the canonical real SWE-Verified exact4 subset, B4/concurrency 4, Tail6
fixed32 physical-32 geometry, stock FA2 SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`,
and the explicit eager BV64 byte A/B diagnostic selector. The runner was
non-timing and floor-ineligible.

## Failure

The prior PID1 contract blocker is closed: process attestation passed with the
exact 48-argument B4 eager diagnostic command, including the final
`--enforce-eager`, and the runtime published its generation-0 ready
acknowledgement.

The harness then proved a zero-traffic baseline and opened the exact4 campaign.
At the first task boundary, before sending a model request, the mandatory
snapshot flush failed:

```
RuntimeError('fixed32 boot-warm evidence is missing')
```

The orchestrator had scheduled the four canonical task IDs, but no request
crossed either authenticated ingress. Engine and proxy finalization both show
zero accepted, active, or completed requests for every task; the flush shows
zero complete work-census events and zero pure-decode forward steps. The work
census, real-event arm, B4 live-PASS, and task results are absent.

This is an eager lifecycle/readiness failure, not a BV64 byte mismatch. Eager
mode disables CUDA graph capture, and this run reached readiness without the
boot-warm evidence required by boundary snapshots. The exact source-path fix
is deliberately not claimed by this result artifact.

The initially preserved container was logged and inspected, then the exact
named container was stopped and removed. The immediate post-removal GPU
compute-process query was empty.

## Evidence

- `launcher_meta.txt`: source, subset, runner, FA2, BV, and serve rc 15.
- `process_argv.json`: sanitized successful PID1 and EngineCore argv capture.
- `ready_ack.json` and `pretask_zero_traffic.json`: generation-0 readiness and
  zero generation probes, drafts, tokens, and work-census bytes.
- `engine_ingress_begin.json` and `engine_ingress_finalize.json`: campaign
  lifecycle with zero accepted/completed model requests.
- `failed_snapshot_ack.json`: generation-1 snapshot error and zero work.
- `swe_orchestrator.log`: first pre-boundary traceback before task execution.
- `runtime_manifest.json`, `external_manifest.json`, and
  `subset_b4_four.json`: immutable source/runtime/task identities.

Raw Docker inspect and full process-environment captures are not published
because they contain host/runtime environment values. Their hashes are kept in
`verdict.json`; sanitized evidence retains the fields needed for this verdict.
