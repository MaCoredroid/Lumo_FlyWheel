# FR13 fixed32 B4 GDN BV64 live-gate rejection

## Verdict

REJECTED before real-task execution. This run does not classify the BV64
candidate and contains no timing, TPS, acceptance, or hardware-floor result.

The run used source `86de4d8337f1ddb591de35d55ee95e1d5f15a45c`,
the canonical real SWE-Verified exact4 subset, batch size 4, concurrency 4,
Tail6 fixed32 physical-32 geometry, stock FA2 SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`,
and the diagnostic-only BV64 byte A/B selector. The runner was explicitly
non-timing and floor-ineligible.

## Primary failure

The server became healthy, then the fail-closed process attestation rejected
PID1 before the ready acknowledgement or any SWE task traffic. The actual
command had 48 arguments. Its only difference from the canonical 47-argument
graph-mode command was the final argument:

```
--enforce-eager
```

The diagnostic runner deliberately pins `ENFORCE_EAGER=1`, because the wide-BV
byte A/B hook is eager-only. The launcher therefore appended
`--enforce-eager`, but `expected_process_pid1_argv()` still required the
graph-mode command exactly. The runner exited with serve rc 3 on
`fixed32 PID1 argv mismatch`.

This is a process-contract mismatch in the diagnostic path, not a GDN kernel
byte mismatch. No candidate/reference comparison reached a real event. The
authenticated engine ingress ledger was empty, the terminal flush reported
zero complete work-census events and zero pure-decode forward steps, and the
real-event arm, work census, B4 live-PASS, ready acknowledgement, and SWE
results were all absent.

## Secondary teardown failure

After the process attestation exited, the terminal flush returned
`error:RuntimeError` with `fixed32 boot-warm evidence is missing`. This is
downstream fallout: the run left before the readiness path could establish
boot-warm evidence. It is not the primary failure and is not candidate
correctness evidence.

The initially preserved container was logged and inspected, then the exact
named container was stopped and removed. The immediate post-removal GPU
compute-process query was empty.

## Evidence

- `verdict.json`: machine-readable rejection and exact identities.
- `launcher_meta.txt`: source, subset, runner, FA2, BV, and serve rc.
- `process_argv.json`: sanitized PID1 and EngineCore argv capture.
- `runlog.txt`: primary attestation failure and teardown result.
- `final_flush_ack.json`: zero-work counters and secondary error status.
- `container_failure_excerpt.txt`: boot setup and terminal EngineCore error.
- `runtime_manifest.json` and `external_manifest.json`: immutable launch
  closures; launch and end copies were byte-identical.
- `subset_b4_four.json`: exact four-task SWE-Verified contract.

The raw Docker inspect and full process-environment capture are intentionally
not published because they contain host/runtime environment values. Their
SHA-256 identities are retained in `verdict.json`; the sanitized argv evidence
contains the complete data needed for this diagnosis.
