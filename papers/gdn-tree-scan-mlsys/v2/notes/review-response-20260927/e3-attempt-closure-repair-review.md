# E3 attempt-closure v2 repair review — 2026-09-27

**Disposition: approve the sealed CPU builder/protocol repair within the requested scope. F1–F4 are addressed; no blocker remains in those deltas.** This is not runtime or workload launch approval. No Docker subprocess, remote operation, model/server request, GPU work or evaluator task ran. Only this review note was written; all test mutations were isolated temporary fixtures and cleaned up. Parent gates remain unchanged.

## Exact seal and verification

Independently computed and verified:

| File | SHA-256 |
|---|---|
| `MANIFEST.json` | `a2259089f2da6852e64f4ed01d9a09f1a92c5cda0b7f2c151734a49e4b578449` |
| `MANIFEST-v2.json` | `a2259089f2da6852e64f4ed01d9a09f1a92c5cda0b7f2c151734a49e4b578449` |
| `closure.py` | `4b15211060171818467dcf655b9f9b68ca6d9c85527a61b2a9d18a0f19ae6a7c` |

Reviewed `REPAIR-REVIEW.md`, the actual source diff against preserved v1, migrated fixture helpers, all 32 targeted test additions, reproducer and verification code. Ran these commands from `experiments/review-response-20260927/workload-plan/tools/attempt-closure` with `/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`:

```text
-B -m unittest discover -s . -p test_closure.py -q
-B verify_bundle.py
-B reproduce_review.py --expected repaired
```

All commands exited successfully. **119 tests passed**. An independent AST comparison confirms all **87** original test names remain, with **32** additions. The verifier checked **30 payload files**, **9 local source bindings**, all **29** accepted identity-adapter members and all **25** accepted runtime-collector members. Both accepted bundles remain unchanged.

The preserved original manifest still hashes to `766df06b14d369c3c92fd04ce2c511ba9514f4442c12e10bafaedbd3acd92521`; all **10** original payload hashes/sizes verify against it. The two current manifests are byte-identical. Original and v2 logs are retained separately. These are CPU preparation checks, not experimental results.

## Four original reproductions

In addition to the supplied reproducer, independently reran each original mutation using a fresh `test_closure.ClosureTests` instance, `setUp()`, and `doCleanups()`. The same helper sequences are recorded in the original review; only the repaired fixture's explicit admission/control bindings differ.

| Finding and unchanged contradiction | Current result |
|---|---|
| F1: exit 137/missing prediction, raw Docker start `00:00:15`, lifecycle start changed to `00:00:17` | Refuses: `actual start vs raw agent start differs from bound identity`. |
| F2: change packet evaluator ID and matching terminal/quiescence event IDs from `b…b` to `d…d`, retaining the original admission | Refuses: `observation subject binding differs from bound identity`. |
| F3: applied evaluator timeout while evaluator is `NOT_STARTED` | Refuses: `applied process control targets a never-started actor`. |
| F4: save an official resolved closure, change only its persisted `resolved` from `true` to `false`, then verify prefix | Refuses: `authoritative retained closure projection differs from bound identity`. |

The four recorded refusals arise at the intended consistency checks, not from missing fixture prerequisites.

## Source-level disposition

**F1 — addressed (`closure.py:131–148`).** The failure path now preserves an available nonzero raw `StartedAt`, checks the frozen window and start/end ordering, and returns it to the existing lifecycle continuity check. Missing/zero raw start remains explicitly unavailable; it is not replaced by collection time. The independent positive exit-137 case retained raw/lifecycle start `00:00:15`, prediction `MISSING`, its original failure journal and a `FAILED` closure.

**F2 — addressed (`closure.py:163–177`).** The builder verifies the retained admission through the accepted `e.observation` checker, binds the event to that receipt, and checks exact actual completion, prediction, container/image/TestSpec, task and attempt. Evaluator start must follow its admission gate. The fixture now binds admission to the accepted collector's actual terminal/prediction, fixing the positive test setup as requested. No evaluator admission is demanded on a path where evaluation never occurred. Existing accepted adapter bytes are unchanged.

**F3 — addressed (`closure.py:267–282`).** A control cannot target `NOT_STARTED`; it requires an observed target start and matching target container/image/start fields. Its timestamp must lie within the observed target lifetime. Missing target evidence refuses. The suite covers missing/wrong bindings and time before target start; valid agent and evaluator controls remain supported. No prelaunch cancellation meaning was added.

**F4 — addressed (`closure.py:204–211`, `:302`, `:377`).** Builder and prefix verifier now share `closure_projection(record, source)`. Full-object comparison binds outcome Boolean/null fields, timestamp, raw references and exact producer to the verified retained record. Existing exclusive-create retention and reclassification refusal remain. Targeted tests also reject changed completion, missing outcome, integer/Boolean substitution, changed timestamp and another approved producer.

## Independent positive controls

Each case was independently built, saved and passed through `verify_closed_prefix`:

| Evidence | Closure | Official completed | Resolved |
|---|---|---|---|
| Agent exit 137, missing prediction, coherent observed start | `FAILED` | `null` | `null` |
| Official completed/resolved report and matching return | `COMPLETED` | `true` | `true` |
| Official completed/unresolved report and matching return | `COMPLETED` | `true` | `false` |
| Incomplete runner with fallback false, missing report | `INFRASTRUCTURE_FAILED` | `false` | `null` |
| Applied timeout during the admitted evaluator's observed lifetime | `TIMED_OUT` | `false` | `null` |

Every positive prefix returned next ordinal 2 while retaining `schedule_advanced=false`; none launched work. Thus the repair preserves legitimate closures and the distinction between official unresolved results and unknown outcomes.

Real lifecycle/evaluator producers, enforcing wrapper/recovery call sites, clock/transport qualification and measured-request instrumentation remain the already disclosed future work. Their absence does not block acceptance of this bounded CPU repair and does not establish runtime readiness. No unrelated concerns or additional experiments were added.
