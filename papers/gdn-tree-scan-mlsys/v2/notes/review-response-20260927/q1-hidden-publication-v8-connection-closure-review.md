# Hidden publication v8 connection — bounded repair closure

Disposition: **F1 and F2 closed at source/CPU-control level for the exact bytes below.** No additional concrete blocker was found in the repaired connection. This is not GPU qualification, permission to launch, or approval of later wrapper/gate/source-freeze wiring. The initial report and both initial source snapshots remain unchanged.

| Reviewed source | SHA256 |
| --- | --- |
| `tools/q1_candidate_hooks_v8.py` | `d6977ec69324c68fd82492da18472430c268808ea062b5608255cd03390638d3` |
| `tools/q1_patch_candidate_v8.py` | `73ea046c8fa32eef3af919c3ce22890c26822df1267cb5335fed88ac0e0349d2` |
| `tools/q1_hidden_event_binding_v1.py` | `9ea8e10f4eb290840a908f4b78f38c5f29c0144d0e40dcab5b3750650c1550e4` |
| `tools/q1_candidate_raw_audit_v7.py` | `d65225715f06d0d80855e1af46e41192df5559a7f44d967266848d0962bc01e4` |

Paths in the table are relative to `experiments/review-response-20260927/`.

## F1 — actual completed-census join

The new event binder at7–21 uses exactly the production hash encoding: ASCII canonical JSON of the request list and each request's UTF8 digest. It reads these from `drafter_runtime`, where pinned generated GDN `_fr13_fixed32_observed_build_record:8226–8454` emits them, rather than requesting a nonexistent top-level raw request list. It also checks the complete census schema, observed mode/B1/forward step/event index, producer PID, event ID and zero-valued integer failure evidence.

Hooks662–679 retain a JSON copy of the actual census and join it back to the same-case bridge owner, live runner/drafter and retained proposal/pending/payload identities. Census length and pending-clear/complete-event checks bind the stage to exactly one newly completed event. The raw consumer147–157 independently joins the retained event to document/job/request/PID and publication before/after counts, runs the accepted operand-byte oracle, and rechecks the production census through the same source-pinned binder. The new binder hash is included by job v8:28, runtime initialization93, raw validation47 and raw source inventory252.

The independent positive executes the production constructor's exact record AST and the exact production request-hash expressions, using a synthetic Unicode/escaped request ID. Both event binder and actual seal method accept the real source shape without fabricated `request_ids`. Wrong request hashes, PID, event ID, schema, completion, mode/B1/step/index and absent/nonzero/noninteger failure evidence refuse. The actual raw consumer refuses foreign document/request/census identity and mismatched publication counts. Already accepted bridge byte validation is explicitly stubbed only in these new event-join controls; these controls do not claim a full production census-builder execution or new operand numerical evidence.

## F2 — original-operation and hidden-seal failure retention

Patcher92–111 preserves the exact original `drafter.propose` call and keyword AST inside one `hidden_proposal_scope`; the independent control checks their AST equality after applying the real edits to the pinned Runner. The producer-entry callback remains immediately before that call, and already has its own retention handler. The scope encloses the original Eagle copy, first model call, metadata operations and gathers inside `propose`; it does not assert coverage of unrelated outer Runner operations.

Hooks569–578 capture whether this is an active admitted case before yielding. An escaping `BaseException` latches unusable through the existing best-effort persistence method and raises `ProcessUnusable` with the original cause. Disabled/no-active-case behavior rethrows the same exception object. Hidden event-seal failures at692–696 use the same latch; the explicit exception arm710–711 prevents the outer existing S4 handler from swallowing that signal.

Independent controls execute the actual patched Runner context statement with a stub `propose` that executes the actual patched first-forward statements. The failure comes from the original model-call stub, after the before callback: `RuntimeError`, `KeyboardInterrupt` and `SystemExit` all retain the partial record, mark the process unusable, and preserve the causal exception. The matching inactive controls keep the original exception unchanged with no marker or receipt. An actual S4 call with a foreign census PID now latches and raises. A separate I/O-failure control confirms failed marker and receipt writes do not prevent the in-memory unusable latch or current-call exception.

## Bounded verification and remaining boundary

Independent `closure/controls.py` completed **32 grouped controls: 13 positive and19 expected refusal**, including the unchanged28-call disabled/no-case/sealed/prefill matrix. All three in-memory generated patches compile, original proposal arguments remain unchanged, and already-patched input is rejected. This used the configured local Python, stdlib AST extraction and explicit stubs only; no Torch, model, remote host, container or GPU was used.

The retained parent log `tools/test_log.q1_hidden_v8_connected.attempt1.txt` reports seven real-Torch CPU tests passing in0.604s, including pinned actual copy/gather ASTs, full bridge/raw join, mismatch refusal and partial retention. It contains ResourceWarnings from existing unclosed-file helpers, with no test failure. That log was read and retained, not independently rerun locally; local Torch is unavailable. Test source SHA is `7e22f608e5b7443740251789ac0b943373b8438095641eeedce695969ad37816`; retained log SHA is `700a6faa1fc81dfc9b583f07bb908f932a03c75876421fb328b743d4df530df5`.

Evidence directory: `p0/monitor/review-response-20260927/hidden-v8-connection-independent-20260929/closure/`. It preserves reviewed sources, executable controls, exact source hashes and results. The initial report SHA remains `0d396ab6fafb866cefe132ca565910f61e1843c4f8d743a23e3641614717bc88`; the original seal-failure and original-operation counterexamples remain available there in the parent directory.

The remaining real-runtime boundary is unchanged: correct final job/helper hashes and generated-source installation must be frozen and observed by the parent; the nine actual operand stages plus real production census must appear in a collected record and pass the raw audit. This review supplies no live provenance or numerical MTP claim and asks for no extra workload or qualification scope.
