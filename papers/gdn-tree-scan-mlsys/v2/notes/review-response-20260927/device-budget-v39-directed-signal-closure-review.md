# Device-budget v3.9 directed-signal closure

**Disposition: the remaining reproduced shutdown race is closed for the reviewed caller; bounded CPU/source acceptance.** The failed-version findings and reproductions remain in `device-budget-v39-successor-review.md` (SHA `f3b6b7f247afa7831d1ded4ea9ed9887aa54b8f3c816187ad819f8188df3991d`). This is not a workload launch, runtime qualification, or gate approval.

| Current source, campaign `workload-plan/tools/attempt-runtime` | SHA-256 |
| --- | --- |
| `device_budget_v2.py` | `1d885c8e05907a94514642319ebf80a4341b4fa52031da7394c2b4807805d9c7` |
| `sole_executor_v3_9.py` | `4d5a71867bf414db803771a01d6710348b84171061f9160e71c414135c33b1f2` |
| `test_device_budget_v2.py` | `4dc80f518e293bbe7ada4facb93ad8b151c83c988ffca75710ad5b8b4bbcfa94` |

The parent preserved the previous three files in `p0/monitor/review-response-20260927/device-budget-v39-preshutdown-race-fix/`; their hashes match the preceding review, including the earlier budget SHA `5d691fdf8af6f9efa5f293174e713d7f118266ba89717796348eb969fed0efd7`.

The correction requires `signal.pthread_kill`, records the construction thread identity, and targets that thread instead of the whole process (`device_budget_v2.py:47,61,74`). In the reviewed sole-executor path, construction and start occur on the main thread and the main-thread admission check precedes arming. The existing block/cancel/join/drain/restore sequence remains intact, and SIGALRM remains owned by the archive reader.

## Independent recheck

- **Same previously failing race, now passes:** retain an unrelated unblocked worker; pause only the natural scheduling gap after the watchdog wait expires and before its send; release that send while the real `stop('exception_cleanup')` is cancelling. The new send uses actual `pthread_kill`. `stop()` is not interrupted, records EXPIRED and its end clock, joins the watchdog, drains the pending signal, and restores the old handler. No pending or late signal remains. The auxiliary test thread is joined afterward.
- **Pending directed signal, passes:** start a .02-second budget, block SIGUSR1 in main, wait .06 s, verify it is pending, and call actual `stop`. Expiry is recorded; the signal is drained; the watchdog is joined; the original handler/mask are restored. Unmasking and waiting does not deliver a late signal.
- **Supplied regression controls:** `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_device_budget_v2` passes **7/7** on these exact bytes. These retain archive/deadline independence, actual `Runtime.execute` retained-expiry detection, idempotent stop, handler-ownership refusal, bounded actor control, and receipt integrity checks.

The previous independent actual-caller AST controls remain applicable because the executor bytes are unchanged: expiry is re-raised before normal measurement, running owned actors are controlled before a permitted reclosure, and an existing immutable closure is preserved. Prior failed-receipt final persistence and next-arm refusal are unchanged. No further defect was identified within this bounded delta.

Three hours remains a prospective operational wall deadline. The record distinguishes observed expiry from watchdog stop; this is neither exact signal-send timing nor total later cleanup duration, GPU utilization, or ETA. Numerical/model settings and separate agent/evaluator limits are unchanged. The parent must bind the corrected source hash into the eventual prospective authority; this review neither updates it nor opens WP. No live operations or scientific results were produced.
