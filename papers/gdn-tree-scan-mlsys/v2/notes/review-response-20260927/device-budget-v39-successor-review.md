# Device budget v2 / executor v3.9 bounded successor review

**Disposition: previous F1/F2 repaired; hold for one reproduced signal-delivery race in `stop()`.** Review scope is CPU source and injected controls only. No Docker, network, GPU, model, task, or evaluator execution; no implementation or gate edits.

| Reviewed file, relative to campaign `workload-plan/tools/attempt-runtime` | SHA-256 |
| --- | --- |
| `device_budget_v2.py` | `5d691fdf8af6f9efa5f293174e713d7f118266ba89717796348eb969fed0efd7` |
| `sole_executor_v3_9.py` | `4d5a71867bf414db803771a01d6710348b84171061f9160e71c414135c33b1f2` |
| `test_device_budget_v2.py` | `4dc80f518e293bbe7ada4facb93ad8b151c83c988ffca75710ad5b8b4bbcfa94` |

## Reproduced remaining blocker: process-directed signal can interrupt shutdown

`device_budget_v2.py:71–73` sends `os.kill(os.getpid(), SIGUSR1)`. The watchdog inherits a blocked mask, but another existing thread may have SIGUSR1 unblocked; admission does not prohibit this. Blocking only the main thread at `stop():113` therefore does not ensure that the process-directed signal remains kernel-pending for the main thread to drain. A different unblocked thread can receive it, and CPython can dispatch the Python handler on main while `stop()` is still executing. `_expire` then raises before old-handler restoration and receipt finalization at lines 122–128.

**Actual CPU reproduction:** start an unrelated thread with the original unblocked mask. Start the actual .02-second budget. Wrap only `D.os.kill` to wait at the natural scheduling gap after the watchdog's `Event.wait` has timed out and before the send. Once that gap is reached, call actual `stop('exception_cleanup')`; the injected cancellation seam calls the original `cancel.set`, releases the watchdog's send, and briefly yields. The original `os.kill` sends the real signal. Observed:

```json
{"stop_interrupted":true,"original_handler_restored":false,"ended_recorded":false,"state":"EXPIRED"}
```

No production operation was involved. The review performed a second cleanup afterward, joined both test threads, and verified restoration of the original handler and mask. This is a shutdown race, not merely an expiry being reported: the first `stop()` exits without completing its restoration/receipt contract. `sole_executor_v3_9.py:826` calls it before the exception branch's owned-actor/server cleanup; an interrupt there can escape to the outer handler before that cleanup.

**Minimum correction:** direct the watchdog signal to the recorded main-thread identity, e.g. `signal.pthread_kill(main_thread_ident, SIGUSR1)`, and explicitly require that API. Retain cancellation, join, pending drain, and handler/mask restoration. Repeat the same existing-worker/send-gap control. Do not replace or suspend the independent archive alarm.

## Closed findings and successful controls

The supplied command `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_device_budget_v2` passed **7 tests**. Additional independent controls established:

1. **Previous F1 closed:** the actual archive reader returns `status='ok'` under an active v2 budget. Its own timeout also remains separate, as covered by the supplied test. Budget expiry during a blocked archive read escapes and restores both signal handlers.
2. **Pending main-thread signal drains correctly in the single-reader case:** start the actual .025-second budget, block SIGUSR1 in main, wait .07 s, verify it is pending, then call `stop`. It joins the watchdog, drains the signal, records EXPIRED, restores the original handler, and leaves no pending or late signal after unmasking. This positive control does not cover delivery to another unblocked thread; that distinction exposes the blocker above.
3. **Admission refusal:** an already-blocked SIGUSR1 refuses before takeover and preserves the mask. The supplied test similarly verifies refusal of an already-owned handler.
4. **Previous F2 closed at the caller boundary:** an AST control compiled the exact four statements at executor lines 794–798 and the entire original exception handler at 825–869, injecting only surrounding transport/lifecycle seams. It invokes the actual imported `Runtime.execute`, with an agent wait interrupted by the real budget. The caller re-raises `DeviceBudgetExpired`, records `REFUSED_OR_FAILED`, and never enters normal measurement.
5. **Deadline cleanup and immutable closure:** in that same actual-caller control, an incomplete first closure leads to observed running-agent `TIMED_OUT` control, terminal observation, exact-CID release, then reclosure. Trace: `close:1 → inspect:agent → control:agent:TIMED_OUT → collect_agent_terminal → stop_proxy:exception → release_exact_cid → close:2 → retain_result`. With an already-written immutable closure and an observed stopped actor, there is no duplicate control or second close; the original closure bytes remain untouched, expiry is re-raised, and the executor failure is retained. The supplied actor-control test also checks no duplicate control when a prior control-issued record exists.

The source checks expiry immediately after `execute` and again after stopping the budget before normal release. The retained result and actual closure-file flag prevent the old swallowed-expiry success path. Receipt-error final persistence and next-arm refusal remain inherited from v3.8 without a changed logic path.

## Measurement and authority boundary

For a delivered expiry, the record now ends `elapsed_wall_seconds` at the observed expiry timestamp and separately records watchdog-stop time and the subsequent interval. In the pending-drain branch, expiry is observed at the drain; this is not evidence of exact kernel signal-send time. `post_expiry_cleanup_seconds` describes time until watchdog stop, not total subsequent actor/server cleanup duration. A three-hour prospective operational wall deadline is neither an ETA nor GPU utilization/accounting; Python signal dispatch can have scheduling latency. Agent/evaluator limits and scientific/model settings remain unchanged.

This review preserves the prior failure note and closes its two demonstrated integration findings without reopening unrelated accepted bundles. A successor must close the one shutdown race before bounded source acceptance; WP and all runtime gates remain parent-owned.
