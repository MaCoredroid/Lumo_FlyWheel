# FR13 fixed32 B4 GDN BV64 eager-hook startup rejection

## Verdict

REJECTED before model traffic. This run does not classify the BV64 candidate
and contains no timing, TPS, acceptance, or hardware-floor result.

The run used source `cd377b687d67e95688b904cd89900032fcb2d6ba`, the
canonical real SWE-Verified exact4 subset, B4/concurrency 4, Tail6 fixed32
physical-32 geometry, stock FA2 SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`,
and the explicit eager BV64 byte A/B diagnostic selector. The runner was
non-timing and floor-ineligible.

## Failure

PID1 process attestation passed with the exact 48-argument B4 eager diagnostic
command, including the trailing `--enforce-eager`. vLLM resolved the runtime to
`CUDAGraphMode.NONE`, completed engine initialization, and published its
generation-0 ready acknowledgement.

The source included the proposed eager boot hook in
`GPUModelRunner.capture_model()`, but that hook did not execute in real eager
startup. The complete container log has no stock `capture_model()` eager-skip
warning, the hook published no boot-warm evidence, and the healthy server then
failed the first mandatory boundary snapshot at generation 1 with:

```
RuntimeError('fixed32 boot-warm evidence is missing')
```

The harness proved a zero-traffic baseline before opening the exact4 campaign.
The orchestrator scheduled the four canonical task IDs, but the first pre-task
boundary failed before any authenticated model request. Engine and proxy
finalization show zero accepted, active, or completed requests for every task;
the flush shows zero complete work-census events and zero pure-decode forward
steps. The work census, authenticated real-event arm, B4 live-PASS, and task
results are absent.

This is a startup lifecycle-routing failure, not a BV64 byte mismatch or kernel
timing result. No further eager hook or GPU run is claimed by this artifact.
The exact named container was stopped and removed; a post-removal check found
zero matching containers and zero GPU compute processes.

## Evidence

- `launcher_meta.txt`: source, subset, runner, FA2, BV, and serve rc 15.
- `process_argv.json`: sanitized successful PID1 and EngineCore argv capture.
- `ready_ack.json` and `pretask_zero_traffic.json`: generation-0 readiness and
  zero generation probes, drafts, tokens, and work-census bytes.
- `engine_ingress_*.json` and `proxy_ingress_*.json`: campaign lifecycle with
  zero accepted or completed model requests.
- `failed_snapshot_ack.json`: generation-1 snapshot error and zero work.
- `container_failure_excerpt.txt`: eager startup, server readiness, and the
  first boundary failure.
- `swe_orchestrator.log`: first pre-boundary traceback before task execution.
- `runtime_manifest.json`, `external_manifest.json`, and
  `subset_b4_four.json`: immutable source/runtime/task identities.

The full process-environment capture is not published because it contains
runtime environment values. Its SHA-256 is retained in `verdict.json`; the
sanitized evidence preserves the fields needed for this verdict.
