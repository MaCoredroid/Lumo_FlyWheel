# Device budget v1 / sole executor v3.8 independent CPU review

**Disposition: hold this implementation for two concrete integration blockers.** Six supplied CPU controls pass, but they do not exercise the existing archive deadline or the actual runtime's `BaseException` handling. Parent has acknowledged both findings and is preparing a versioned successor; that successor is outside this review. No launcher, container, network, GPU, model, evaluator, or workload execution occurred. WP remains closed, workload attempts 0/4.

## Reviewed bytes

Paths below are relative to campaign `workload-plan/tools`.

| File | SHA-256 |
| --- | --- |
| `attempt-runtime/device_budget_v1.py` | `e4965e140315a3f6ca6d001dfceb7fa02d054d27ca70d1781cf4dd93e8fc728b` |
| `attempt-runtime/sole_executor_v3_8.py` | `1a6d1fa232cd5aaee37b11ca621e40aab740d134067bd52c85cd07dac2b04cab` |
| Preserved `attempt-runtime/sole_executor_v3_7.py` | `4fe88d9bd4a23cca788487fdec61a9fb17318f7407e691df9b54b99a24a4a101` |
| `attempt-runtime/test_device_budget_v1.py` | `2968f044d60a7f367b2c5a8c46c7de0e8dee1b4d0cd55ee4a150d1a78fed5924` |
| Imported `attempt-runtime/runtime_v3_3.py` | `3c2da10b0c2325b51522f99d2a42c8d3783920ecebbbbc1c137caa85771a498b` |
| `attempt-runtime/patch_transport_v1.py` | `a8d07736e9f278a0b7e9c112ab44992e675b2acc0491fc74cdf501d997e88052` |
| `attempt-runtime/two_host_v1.py` | `9af64943fa3c1398856e4af2c0382f8500226a617e7c5221e4322e792ae860f8` |
| Actual imported `attempt-closure/closure_v3_3.py` | `791a983b6cc0f7600b5554cad10fa3f9ef42d650a8de2619c5911a1ed4372d23` |

## F1 — attempt alarm blocks the normal archive reader

`device_budget_v1.py:80` arms `ITIMER_REAL` for the attempt. Existing `patch_transport_v1.py:127–145` refuses an already-active real-time timer rather than replacing it. `settings_observation_v1.py:208` calls that reader before agent start; the patch-fetch path also uses it. Thus a healthy attempt with an active budget cannot pass the ordinary settings archive read.

**CPU reproduction:** a fake container's `get_archive` increments a counter and returns `iter([b'fixture']), {'size': 7}`. Calling the actual `PT.bounded_archive(container, '/fixture', 100, .1, Absent, time.monotonic)` without the attempt timer returns `status='ok'`, with one read. Starting actual `D.DeviceBudget({'seconds': 1})` first makes the same call return:

```json
{"status":"failed","reason":"archive deadline refused: another real-time alarm is active"}
```

The fake container receives **zero** reads and the attempt budget remains armed. Both controls use the real reader, without Docker or network. The timer and original handler were restored afterward.

**Required correction:** use a nonconflicting attempt-deadline mechanism, or explicitly compose independently bounded deadlines while preserving timeout ownership, archive cleanup, and the earlier absolute deadline. Simply suspending the attempt deadline during remote reads would leave a hole in the stated wall budget. Retest a positive settings/patch archive read with the attempt budget active, plus expiry and ordinary archive-timeout cases.

## F2 — actual runtime swallows expiry before the caller's cleanup branch

Although `DeviceBudgetExpired` inherits directly from `BaseException`, imported `runtime_v3_3.Runtime.execute` catches **BaseException** at line 181, retains its type, and returns a normal result at line 192. Its finalization handler at lines 190–191 also catches BaseException. The executor does not inspect `budget.record['expired']` after `runtime.execute` returns (`sole_executor_v3_8.py:769–793`), so it can enter normal measurement/release and record `state='EXECUTED'` after expiry. The alarm has already disarmed itself, leaving no second deadline to interrupt continued work.

**CPU reproduction:** invoke the actual `X.R.Runtime.execute(fake, {}, 'unused')`. The fake provides no-op owner/gate methods, `actors={'agent': None}`, a failure trace, `run_agent` sleeping for .2 s, and `close` returning `{'status':'CLOSABLE','schedule_advanced':False}`. Arm the actual budget for .02 s. Observed output:

```json
{"returned_normally":true,"result":{"operation_error_type":"DeviceBudgetExpired","schedule_advanced":false,"status":"CLOSABLE"},"timer_state_after_return":"EXPIRED","handler_still_budget_handler":true,"trace":["owned","gate:launch","agent_wait","attempt_execution:DeviceBudgetExpired","close"]}
```

This uses the real imported method, not a rewritten approximation. The fake closure is deliberately a simple return seam, not evidence that a live running actor would receive a valid closure. It demonstrates the exception swallowing and normal caller continuation. The review explicitly stopped the budget and verified handler restoration after the control.

**Required correction:** immediately detect recorded expiry on return from `execute`, before measurement or normal success. Enter deadline-specific exact-owned cleanup and preserve failure evidence. For a running recorded actor, the existing agent-host sweep is insufficient: `two_host_v1.py:536–559` refuses to kill running agents/evaluators and removes only already-stopped owned actors. Observe identity and state through the existing checks, apply typed control only to a proven running actor, observe the result, and retain unverified cleanup as incomplete. Do not infer cancellation of absent, unobserved, or never-started actors.

The closure constraints matter: `closure_v3_3.py:499–509` permits additional incomplete observation revisions but makes a successful `attempt-closure.json` immutable. Re-close after verified control only when no successful closure already exists. Preserve an already-written terminal closure and record a later budget failure in the executor ledger; do not rewrite its history. Existing `Runtime.control` has per-role write-once control-issued records and a single current `refs['control']`, and agent control invokes patch-fetch hooks. Avoid duplicate control and test those actual seams rather than only checking the exception class.

## Verified portions and timing limits

- Ran `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_device_budget_v1` from `attempt-runtime`: **6 tests passed**. They cover interruption, normal disarm, existing-timer refusal, direct exception inheritance, invalid spec rejection, and bound/corrupt/error receipt validation.
- `sole_executor_v3_8.py:676–680` starts the budget inside acquired ownership and before the launch gate. Preflight/host identity validation happens before this clock; the budget does not measure that earlier work.
- `verify_budget_receipt` binds the attempt, freeze, producer, scope, seconds, receipt bytes, RESULT payload, and finite ordered clocks. `prior_evidence_map:307–308` requires `terminal is True` and the verified receipt before admitting the next arm. Missing/failed receipt evidence cannot silently advance the schedule.
- An additional AST control compiled the **unchanged exact nested** `stop_budget`, `release`, and `retain_result` definitions into a fixture factory. A preexisting `device-budget.json` caused `FileExistsError`; the exact-owned cleanup callback still ran. After deliberately setting `outcome['terminal']=True`, `retain_result` saved it as **false**. Timer disarm and old-handler restoration both succeeded. This confirms the last persistence check preserves a failed receipt despite later success updates.
- The proposed **three hours per attempt** is an operational wall deadline, not an ETA, observed GPU utilization, kernel time, or performance result. The v3.7-to-v3.8 delta does not change the separate agent/evaluator limits or scientific/model settings.
- `cleanup_excluded=True` needs a precise boundary in the successor. The normal executor calls `stop_budget` before server release, but lower-level boot cancellation may already quiesce its owned launcher before raising `BootFailure`; runtime finalization may also precede the caller's stop. The recorded `elapsed_wall_seconds` currently ends at `stop()`, not at the earlier expiry timestamp, so it can include such inner cleanup. Preserve expiry and stop timestamps and label the interval accurately; do not describe it as an exact GPU allocation lifetime or a hard cap on cleanup.

The pending prospective decision's old caller hash was not treated as final authority. No accepted bundle, implementation, source gate, or campaign result was changed by this review.
