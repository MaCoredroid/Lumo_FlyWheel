# Caller v3.6 four-finding repair recheck

**PASS for the bounded F1–F4 source repair.** No remaining blocker was found in these four repaired behaviors. This does not approve a workload launch, the final package freeze, or unfinished runtime-producer integration. GATE-WP remains closed and attempts remain **0/4**.

Reviewed only the parent-repaired source delta against `workload-runtime-v36-independent-review.md`. Original submitted bytes remain in the earlier review snapshot and parent's `identity/codex-caller-v36-pre-repair/`. Repaired input snapshot: `papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-runtime-v36-repair-reviewed-20260928T202857Z`. No production files were edited by this reviewer; no Docker, SSH, server, proxy, tunnel, model, evaluator, GPU, or workload action occurred.

| Finding | Bounded closure evidence |
|---|---|
| F1: unset names omitted from file environment | `settings_observation_v1.py:234–238` applies the frozen names to the union of container, dotenv, and JSON-env observations. Actual collector controls now refuse `QWEN_HOME` from container, home dotenv, or JSON env; empty file-supplied values also refuse. The no-override positive path passes. |
| F2: cancellation lost the patch receipt | `PatchFetch.fetch` catches `BaseException`, retains `transport_failed` with the interruption reason and available container identity, then re-raises cancellation. Injected `KeyboardInterrupt` and `SystemExit` at the actual archive read each retain an immutable receipt, restore the alarm, and create no local patch. Calling fetch again returns the same failed receipt without repeating SDK calls. Honest transfer still passes. |
| F3: kill return called an observed exit | The actual caller's nested callback now maps `wait_returned` to `agent_exit_boundary` and control return to `agent_control_return`. AST-extracted unchanged callback controls exercise both mappings, other-CID exclusion, and duplicate suppression. A control return leaves the exit field null. The already-correct after-evaluation label is preserved. |
| F4: timeout only checked between yielded chunks | The shared archive reader now owns a POSIX main-thread interval deadline around SDK entry, iteration, and stream cleanup. Blocking controls interrupt all three stages. Combined blocked iteration + blocked close returns failure in **0.0803 s** for a 0.03 s transfer deadline plus the 0.05 s cleanup interrupt interval. A later observation sees no remaining alarm. An existing active timer is refused without replacing its handler or interval; a non-main-thread context refuses before SDK entry; a successful read restores an inactive custom handler. No background helper is introduced. |

**20 CPU controls passed**, retained in `REVIEW-controls.attempt1.txt`; `REPLAY-controls.py` is executable using the configured Python with `-B`. Process creation, helper-thread start, and socket connection entry points are blocked by the reviewer harness. The deadline tests use only local injected blocking functions and short POSIX timers, not a live Docker client. No full remote-suite result is asserted here; the parent's reported corrected fixture and remote-suite status are separate evidence.

Source bindings (full SHA256):

- `patch_transport_v1.py`: `a8d07736e9f278a0b7e9c112ab44992e675b2acc0491fc74cdf501d997e88052`
- `settings_observation_v1.py`: `6453881bec7b7abd7b0c9dec38155a98f0b90231099daabe0a979a5d3adf7196`
- `sole_executor_v3_6.py`: `d1e167b8a8967a24249ec550c1e1afa52f5fa0eea5364d40199d9d67e98f111d`
- `test_workload_caller_v3_6.py`: `517ee07b0ef6d09e2980cf02d66ae24cc381d86224bb87328217862a1620a120`

`INPUTS.json` binds the exact reviewed sources; `REVIEW-MANIFEST.json` binds the report and controls. These were byte-compared to the current local files when sealing. Parent remains responsible for source freeze, runtime producer integration, and admission; no additional task, seed, cap, retry, or experiment is requested.
