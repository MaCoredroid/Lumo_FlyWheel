# E3 attempt-closure independent review — 2026-09-27

**Disposition: seal and CPU tests verified; four bounded consistency repairs required before accepting this builder.** Normal official-outcome and failed-attempt retention paths work, but contradictory evidence can still produce a closure or pass the closed-prefix check. These are local contract fixes, not requests for additional experiments or real instrumentation. No Docker subprocess, remote operation, model/server request, GPU work, evaluator task or workload ran. No accepted bundle, implementation, gate or paper was edited; only this note was written.

## Verification

The independent hashes match the requested pins:

- `MANIFEST.json`: `766df06b14d369c3c92fd04ce2c511ba9514f4442c12e10bafaedbd3acd92521`.
- `closure.py`: `7dd81ca01444cd9d1e7ebc24fb2aef90fb6f919d0defa3565f4b08b85ba8b1ce`.

With the configured Python, reran `-B -m unittest discover -s . -p test_closure.py -q` and `-B verify_bundle.py` from `experiments/review-response-20260927/workload-plan/tools/attempt-closure`. **87 injected tests passed**. Verification covered **10 payload files**, **8 local source bindings**, and unchanged accepted dependencies: **29 identity-adapter files** and **25 runtime-collector files**. The two unavailable grader/reporting bodies are correctly identified as inventory-only; no claims about their full schemas were inferred.

The retained `run_evaluation.py` at SHA `6959f0b4e4eaf979771f529b88e3e9df1daa7fe86bc4291feec2e7d320bf7f2e` confirms that the official report is written before `eval_completed=True`, while the `finally` return can supply fallback `resolved=False` without a completed report. The builder correctly preserves this distinction.

## Independent controls and reproduction setup

Each probe below used a fresh `test_closure.ClosureTests` instance, `setUp()`, then `doCleanups()`; all bytes and Docker/clock observations were temporary injected fixtures. `build()` calls the evidence builder and `save()` adds immutable local retention. `edit_event()` creates new internally consistent raw/envelope fixture hashes, so these checks test cross-evidence consistency rather than broken hashing.

| Independent control | Observed result |
|---|---|
| `evaluation(resolved=True)` | `COMPLETED`, official completed `true`, resolved `true`. |
| `evaluation(resolved=False)` | `COMPLETED`, official completed `true`, resolved `false`. |
| `evaluation(completed=False,resolved=False,report='MISSING')` | `INFRASTRUCTURE_FAILED`, official completed `false`, resolved `null`. |
| `failed_collection()` with exit 137 and no prediction | Saved `FAILED` closure; prediction `MISSING`, resolved `null`; original failure journal retained. |
| Delete `refs['started']`, then save and verify prefix | Saved `INCOMPLETE`, no closure file; prefix refuses the unclosed reservation. |

The passing suite also covers source/image/task/prediction mismatches, stale reports/quiescence, missing raw files, nonzero exits, missing/unreadable/malformed reports, unapplied control, unknown directories, holes, open reservations, retry refusal, and exclusive-create retention. The findings below are additional cases absent from those checks.

## F1 — failure receipts discard an available raw start time

**Source:** `closure.py:131–143` and `:204–212`.

```python
x.failed_collection()  # accepted collector observes exit 137; raw StartedAt is 00:00:15
x.edit_event('started', lambda o: o.update(
    observed_at_utc='2026-09-27T00:00:17+00:00'))
record, closure = x.build()
```

**Result:** `CLOSABLE` / `FAILED`, with closure start `00:00:17`, despite the retained failed-agent Docker observation proving start `00:00:15`. The failure branch extracts `FinishedAt` but returns `started_at_utc=None`, so the consistency check applied to normal terminals is skipped. This affects the actual-start record and any later non-overlap validation.

**Minimal repair:** when the accepted failure receipt contains an observed started/exited container, preserve and validate its nonzero raw `StartedAt`, then require the lifecycle start event to agree. Keep start unobserved when it truly is unavailable; do not manufacture it from collection time. Add this missing-prediction mismatch case alongside the normal-terminal start mismatch test.

## F2 — evaluator identity is compared with an unverified packet field

**Source:** `closure.py:147–162`; `validate_packet` at `:54–74` does not validate the retained evaluator admission observation.

```python
x.evaluation()
x.packet['evaluation']['container_id'] = 'd' * 64
x.edit_event('evaluator_terminal', lambda o: o['payload'].update(container_id='d' * 64))
x.edit_event('quiescent', lambda o: o['payload']['actors']['evaluator'].update(container_id='d' * 64))
record, closure = x.build()
```

**Result:** `CLOSABLE` / `COMPLETED`, resolved `true`, although the unchanged retained evaluator admission receipt binds container `'b' * 64`. Independently calling the accepted `e.observation(...,'evaluator_container',subject)` passes before the edit and refuses afterward with `observation subject binding differs from bound identity`; the closure builder still accepts the edited packet.

**Minimal repair:** bind the evaluator event to the retained, verified admission subject/gate packet, including its actual container, prediction and accepted agent completion. Require actual evaluator start to follow that admission gate. The existing adapter observation verifier is reusable; this does not require editing the accepted adapter or requiring evaluator evidence on paths where evaluation never occurred. Update normal fixtures so the retained admission actually refers to the accepted collector terminal/prediction used by closure.

## F3 — an applied timeout can target an actor recorded as never started

**Source:** `closure.py:218–233` and `:241–250`.

```python
x.normal_agent()
x.refs['control'] = x.event('EXECUTION_CONTROL', 17, {
    'action': 'TIMED_OUT', 'target': 'evaluator', 'applied': True})
record, closure = x.build()
```

**Result:** `CLOSABLE` / `TIMED_OUT`, although the same evidence says evaluator `NOT_STARTED`, with no evaluator terminal or observed evaluator start. The control validates only a role name and Boolean; it is not cross-checked with the target's lifecycle. This silently permits a different interpretation of timeout from the documented applied execution-control observation.

**Minimal repair:** bind control to the actual target identity/lifecycle and reject a process timeout against `NOT_STARTED`. If cancellation before launch is intended, define it explicitly as a separate supported control meaning; do not infer it from this generic process-control payload. Keep missing target evidence incomplete/refused rather than inventing a target lifetime. Add the contradiction case and a valid observed-target control case.

## F4 — closed-prefix verification accepts changed official outcome fields

**Source:** `closure.py:321–350`, particularly `:339–346`.

```python
x.evaluation(resolved=True)
x.save()
path = x.directory / 'attempt-closure.json'
obj = k.e.load(path)
obj['resolved'] = False
path.write_bytes(k.e.canonical(obj))  # temporary fixture corruption only
verdict = k.verify_closed_prefix(x.directory.parent, x.f.freeze, x.packet)
```

**Result:** `CLOSED_PREFIX_VERIFIED_NO_LAUNCH`, next ordinal **2**, although the unchanged hash-bound attempt record still says resolved `true`. The prefix function hashes the current closure afresh and checks classification/identity/lifecycle, but never compares `resolved` or `official_evaluation_completed` against the bound record. Exclusive creation protects writes through the builder; it does not make this read-side mutation check complete.

**Minimal repair:** compare all derived closure outcome fields against the verified record (including exact Boolean/null distinctions). Also compare the closure record timestamp and require the expected closure producer when validating this package's output; reconstructing the expected closure projection from the record is a compact approach. Keep the original exclusive-create/reclassification refusal. Add corruption tests for both official outcome fields; no new cryptographic signing scheme is needed.

## Scope after repair

The proposed lifecycle/evaluator producer and real sole-executor insertion remain explicitly pending. It is appropriate for this CPU builder to accept source-bound injected observations while real clock/transport, launch inventory and control-event authenticity are handled in the later reviewed wrapper. Those future obligations do not excuse contradictions already visible among supplied records, and fixing the four cases above needs no live task.

Missing predictions and incomplete evaluations must continue to retain their attempted rows, with unknown official outcomes distinct from official unresolved results. No rate, token-count, timing estimator, denominator, seed policy or workload authorization follows from this review. Parent gates remain unchanged.
