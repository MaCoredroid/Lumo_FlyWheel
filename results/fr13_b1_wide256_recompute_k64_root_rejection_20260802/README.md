# FR13 B1 wide256 recompute K64/root rejection

Status: **candidate rejected on exact bytes; no timing or hardware-floor claim**.

An authenticated real SWE-Verified B1 diagnostic ran with physical row count
32, draft-vocabulary root reduction enabled, and `K=65536`. The task resolved,
the server and task orchestrator both returned zero, and the diagnostic served
the stock result while comparing the candidate against stock.

## Kernel verdict

- Candidate family: `streamk_force_wide256`
- Candidate SHA-256: `503277a2dca6784502b709007adfe45f42d0f1a1851107e7b913e1e85a00de5a`
- Gate source commit: `6e06bcb2e341f119b56c5e7eef46e338be327bbc`
- Fixed physical rows: `32`
- Comparison cap and observed comparisons: `320`
- Projection shapes covered: all five required real shapes
- Mismatching comparisons: `320/320`
- Compared BF16 bytes: `311,951,360`
- Differing bytes: `14,910` (`47.795913 ppm`)
- Candidate output served: no; stock output was served
- Production enabled: no

The failure is decisive: changing to the wide256 recompute/Stream-K
accumulation schedule does not preserve exact BF16 output bytes. Timing this
candidate cannot produce acceptance evidence.

## Lifecycle and recovery

The task lifecycle completed before the first host-side reducer attempt. That
attempt then hit `PermissionError` because the root container created the
non-secret binary attestation as `0600`. No task rerun was needed. The existing
bounded comparator records were unchanged, the attestation was made readable,
and the same reducer produced the expected gate exit code 4 and the formal
`fail` verdict.

Commit `fad40550b` fixes the ownership boundary by writing future CUTLASS
attestations atomically as `0444`. The focused binary/gate tests pass.

## Evidence scope

This directory deliberately contains only reduced facts and hashes. It does
not contain prompts, generated patches, task traces, request payloads, model
logs, or raw comparator records. `verdict.json`, `comparison_summary.json`,
and `lifecycle.json` bind the result to the original evidence by SHA-256.

The run is a one-task B1 byte diagnostic, not a 4-task or 16-task acceptance
campaign. It establishes kernel rejection only. It does not measure full-step
TPS, acceptance rate, B4 behavior, latency U95, or distance to the hardware
floor.
