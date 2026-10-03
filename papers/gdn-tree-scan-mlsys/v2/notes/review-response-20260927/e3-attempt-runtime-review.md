# E3 attempt-runtime independent review — 2026-09-27

**Disposition: two concrete CPU integration repairs required before acceptance.** The sealed implementation and 73 injected tests verify, but the documented sequence cannot enter the real agent gate, and the evaluator does not bind its supplied test specification to the selected pinned dataset row. These are reproducible source/control-flow gaps, distinct from the explicitly deferred real-host qualification. No live Docker, SSH, GPU, model/server, evaluator task or workload operation was performed. No implementation, accepted bundle or gate was changed.

## Seal, tests and separate audit outputs

Independently verified the requested SHA-256 pins:

| File | SHA-256 |
|---|---|
| `MANIFEST.json` | `2f15cfc427e081001d934b868c82b63973f449bea63f504a2a6a6683a60170f1` |
| `runtime.py` | `1dd34ef9d30dec24c1b594c148928499e10bb655222f8c3c67b342cda4745d45` |
| `evaluator.py` | `71ee0a8fda59d3bf1f57912a1de359b4931639dcafd11e8fff7f218abdb84882` |

Reran only the new package's suite and verifier with configured Python: **73 tests passed**; **28 payload files**, **11 source bindings**, and the preserved 13-file pre-handoff payload verified. Accepted dependencies remain unchanged: identity adapter **29**, runtime collectors **25**, attempt closure **30** members. Their full suites were not rerun.

Independent audit files are outside the sealed package:

- [Executable CPU reproductions](e3-attempt-runtime-review-audit/reproduce.py), SHA `c8b3240efe94b18e37bcd01dac5674094156fd524f6f5993736eb2db8d692288`.
- [Observed audit results](e3-attempt-runtime-review-audit/results.json), SHA `e046fe86103e4fb4fac497ba04bdf3890a13e827a21f7425fdad665f8e69184e`.

From the paper v2 directory, reproduce with:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B notes/review-response-20260927/e3-attempt-runtime-review-audit/reproduce.py
```

The script verifies the reviewed pins before testing, uses fresh temporary fixtures, and cleans them up. Its transport calls are injected; a counted `start` is never a real container launch. The test-spec probe uses the existing fake official evaluator and mocks the installed-harness check, exactly as the submitted integration tests do. It demonstrates argument/admission behavior, not an actual benchmark outcome.

## F1 — the executable launch-to-agent transition is missing

**Source:** `runtime.py:130–133`; `test_runtime.py:36–45`; accepted `runtime-collectors/integration.py:77–82` and `collectors.py:198–202`.

`Runtime.run_agent` calls `before_agent_request` while retaining the launch packet's `stage='launch'`. It also does not acquire/populate the actual server `boot_started_at_utc` required by the real collector. The test-only `PrecollectedAgentGates` changes the stage and supplies both boot/gate timestamps, so it hides this missing production transition.

**Reproduction:** use the existing small raw model/source/probe fixtures, but replace `PrecollectedAgentGates` with the unchanged accepted `GateHarness` and run the actual accepted `Collector.runtime`. Begin with the documented launch packet, acquire `own()`, pass `launch_gate()`, expose the injected already-running server and call `run_agent` with its actual fixture identities/files.

**Result:** launch passes, then `Refusal: actual agent hook phase differs from bound identity`; **zero injected starts**. A diagnostic changing only the stage still refuses with missing `boot_started_at_utc`. A positive control that supplies both the stage and the actual injected server Docker start timestamp allows the unchanged collector/gates to pass, retains a terminal, and produces a closure. This isolates the missing transition rather than a fixture/source mismatch.

**Minimal repair:** implement an explicit source-bound launch-to-agent transition in the runtime/setup interface. Observe and retain the actual server boot identity/time, preserve the passing launch packet, and set the agent stage before invoking the real collector; let that collector close the gate at its actual clock. Do not move synthetic fixture timestamps into production. Add one integration test using the real `GateHarness` and collector with injected physical inputs, covering the documented caller sequence and refusal on wrong boot identity. No accepted dependency edit or live server run is needed.

## F2 — a correct task ID admits an unbound evaluator TestSpec

**Source:** `evaluator.py:12–19`, `:27–35`, `:101–116`, `:155`; retained official `test_spec/test_spec.py` defines the delegated evaluation script and test lists, and `run_evaluation.py` consumes them during evaluation/grading.

`OfficialEvaluator.run` accepts a caller-created `standard_spec` and checks only its `instance_id`. `PinnedSpec` replaces image selection but delegates repository/version, `eval_script`, `FAIL_TO_PASS` and `PASS_TO_PASS` unchanged. `selected_record` correctly checks the dataset when called, but it is not part of the enforcing evaluator path and there is no retained binding from that row to the supplied spec.

**Reproduction:** retain the expected task ID, but set the caller spec to `repo='wrong/repository'`, `version='wrong-version'`, `eval_script='UNBOUND_EVAL_SCRIPT_SENTINEL'`, and `FAIL_TO_PASS=['unbound_test']`. Spy on the existing injected official body and make any call to `selected_record` fail visibly. Invoke the unchanged evaluator wrapper and closure path.

**Result:** every substituted field reaches the evaluator body; `selected_record` is called **zero** times. The injected completed/true return plus fixture report produces a `CLOSABLE`, resolved-true closure. No real tests were executed. The material issue is that pinned source bytes and image identity do not establish that the evaluator ran the frozen dataset's tests.

**Minimal repair:** make the enforcing evaluator entry construct the standard spec through the pinned official factory from the exact hashed selected row, with the already approved architecture/options, or require and verify an equivalent source-bound construction artifact covering all evaluation-relevant fields/scripts/test lists. Retain the dataset-row/spec identity in the invocation evidence. Keep full test-bearing data coordinator/evaluator-local and retain the existing agent-field allowlist. Add a negative case for altered script/test lists with unchanged task ID, plus a positive pinned-row construction case. This is a CPU binding repair, not a request to run SWE-bench or choose new evaluator settings.

## Supported behavior and limits

The reviewed source and passing suite support the following bounded behavior:

- Phase ownership checks the accepted closed prefix before reserving, uses exclusive reservation/state records, and refuses existing open or unknown attempts. Start requires retained launch/actor gates, the admitted actor and a never-started owned container. The identified transition bug fails before start; it does not silently launch ungated work.
- Actual start, terminal and quiescence evidence comes from injected Docker observations rather than intent alone. Typed absence is distinct from transport failure; unapplied controls and live/unobserved actors do not establish quiescence. Clock rollback/reboot and checkpoint/return corruption are tested.
- The evaluator image path uses immutable local lookup/create with no pull/build fallback, and its proxy gates start before test execution. Official completed false/true and incomplete fallback false remain distinct through the accepted closure builder. F2 concerns the missing dataset-to-spec binding within this otherwise explicit path.
- An independent missing-prediction recovery control closed from observations with **no additional start, wait or kill calls**. No official evaluator callback was used during that recovery.
- Independent patch controls retained an observed zero-byte `patch.diff` as an empty candidate, while a missing patch raised and created no replacement prediction. The submitted suite also checks stale patches, existing destinations and conversion refusal preventing the normal execute path's evaluator fallback.

Real transport/clock qualification, genuine model/runtime probes, measured-request seed/accounting evidence, source-qualified deployment settings and the parent's scientific/runtime gates remain deferred as documented. Those are not additional findings. The two repairs above are required to make the claimed executable CPU integration complete before that later qualification. Parent gates remain unchanged.
