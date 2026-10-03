# Four-attempt runtime successor v2: bounded independent review

Reviewed 2026-09-28 UTC. Verdict: **CPU component progress accepted; connected caller/reducer has the concrete defects below. WP remains closed and actual workload progress remains 0/4.** No live commands, containers, model requests, evaluator runs, or source/gate edits were performed. This review adds no tasks, tuning, retries, or experiment scope.

## Bound source and preserved closures

Snapshot: `p0/monitor/review-response-20260927/workload-runtime-v2-reviewed-20260928T0521Z`. All **36** payload members were independently checked for exact SHA256 and byte length against `MANIFEST-runtime-v2.json`, SHA256 `069fe7139737c37df307ef8fc53570f4b608976720e92c40b1526f2b9ad9eab3`. Paths below are relative to that snapshot's `repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/`.

Authoritative prospective decisions: SHA256 `e0835abed7061b7d264ccc2144c3073f6dc0037f4c8d88ce5062d981585c861a`. The successor adapter now calls the pinned schedule authority and requires exactly `scikit-learn__scikit-learn-9288`, in AR → CHAIN_MTP → SGLANG_EAGLE → LUMOTREE order, one configuration/attempt each and zero added tuning. Old pilot/confirmation files do not supply the rows (`identity-adapter/e3_preflight_v3.py:195–242`, under `workload-plan/tools/`).

Seed absence is explicitly bound to the decision file; VERIFIED, root injection, paired-seed claims and independent-draw claims are refused (`e3_preflight_v3.py:244–261`). The root remains a scheduling label. `closure_v2.py:212–217` separately records an observed engine seed flag, or explicitly says its default is unrecorded. This declaration is compatible with the chosen mode; it is not empirical proof of delivered request contents.

Comparing accepted versus successor sources shows `runtime_v2.py` changes only dependency/module imports; `evaluator_v2.py` changes only its dependency import. Thus the accepted F1 actual boot/stage-transition checks and F2 internally constructed, pinned-row official TestSpec checks are retained, not reopened by this review.

## Concrete fixes

### R1 — Restore retained evidence before checking the next closed row

`workload-plan/tools/attempt-runtime/sole_executor_v2.py:58–59` passes `{'freeze_sha256': freeze_sha, 'evidence_files': {}}` to `closure_v2.verify_closed_prefix`. The latter adds only the closure file before calling `e.terminal` (`closure_v2.py:373–374`); terminal validation correctly requires the approved collector manifest, collector source members and raw evidence (`e3_preflight_v3.py:166`, `82–85`). Consequently the new caller cannot admit row 2 after a valid row-1 closure.

**CPU reproduction:** exact AST-extracted `closure_projection`/`verify_closed_prefix` functions and the unmodified, imported adapter, with temporary files only. A valid retained first-row record with a complete evidence map returns `next_ordinal=2`. The identical journal with the caller's empty evidence map refuses: `collectors manifest needs a locally readable bound evidence file`.

**Narrow repair:** reconstruct/load the hash-verified retained evidence map before preflight prefix validation, then use the same validated packet/evidence population in `Runtime.own`. Preserve all closure checks. Add a real row-1-closed → row-2-preflight positive test; an empty-journal ordinal test does not exercise this transition.

### R2 — Enclose boot/setup in owned failure observation and server teardown

`sole_executor_v2.py:80` can raise a subprocess timeout before `boot-observed.json` is written. On the ordinary return path, lines 93–95 retain only stdout/stderr hashes, not their bytes. `run_attempt` boots and creates the agent before entering `Runtime.execute` (lines 124–127); its exception handler writes an executor failure ledger and raises, without a boot/server observation or cleanup pass.

The actual `Runtime.own` finalizer only clears `directory` and releases the phase lock (`runtime_v2.py:50`). Its actor inventory contains agent/evaluator only (line 15); quiescence/close therefore do not cover the serving container (lines 218–243). A boot that starts a server and then times out/fails can leave that server running, its actual identity/ownership and boot output unretained. Successful `enter_agent_stage` does validate owned server image/boot chronology (lines 130–150), but it is never reached for this failure.

**Narrow repair:** retain stdout/stderr bytes, including available timeout partial output, and a terminal boot/setup observation on every return/exception path. Bind the actually created server ID/image/attempt ownership before cleanup, and stop/observe only that owned server while still retaining the attempt lock/journal context. Unknown ownership must remain incomplete and must never trigger a name-based kill of an unrelated container. No rerun, silent reservation removal, or schedule advancement. Cover timeout-after-server-start and agent-creation failure with injected transports.

### R3 — Refuse each zero-output completed request before pooling

`workload-case-study-v1/pooled_rate_v2.py:57` permits zero output tokens. Lines 117–121 only check `sum(tokens) - completed_count >= 0`; another request can mask the invalid first-token subtraction.

**CPU reproduction:** two matched completed rows, `output_tokens=[10,0]`, `ttft_s=[1,1]`, `latency_s=[2,2]`, produce `REDUCED`, numerator 8, denominator 2, rate **4.0 tokens/s**. The zero-output row has no first generated token to subtract.

**Narrow repair:** any completed row lacking at least one generated output token and a real first-token observation must leave the attempt rate `NOT_REDUCED`, with that row/reason retained. Do not silently drop it or let other requests mask it. A one-token response can contribute zero decode tokens if its timing is valid; zero generated tokens cannot.

### R4 — Connect rate capture/reduction to complete retained request accounting

There is no `pooled_rate_v2`, `reduce_files`, or first-token timing call in the actual `attempt-runtime/*v2.py` caller chain. The new reducer consumes supplied JSON; it does not observe a first-token event. Its `streamed` boolean is a label, not such an observation. Current `producer_sha256` validation checks hash syntax only, and `reduce_files` does not retain/bind raw-file hashes. These are unfinished producer/integration facts, not measured rate evidence.

The provided API also makes `expected_completed_ids` optional (`pooled_rate_v2.py:67,86–88,128`). **CPU reproduction:** two matched requests with 10 output tokens each and decode intervals 1 s and 10 s produce 1.63636 tokens/s. Removing the slow request from **both** lists still produces `REDUCED=9.0`; the paired lists alone cannot prove the attempt population is complete. Supplying the complete expected-ID inventory correctly refuses the subset. Missing TTFT returns `NOT_REDUCED`, and mismatched server/client IDs correctly refuse.

**Narrow repair:** make the sole caller retain the actual server token records and client/proxy first-token/finish observations, with source and raw-file bindings, then invoke reduction against the retained complete attempt request inventory/closure. Include compaction, abort/error and retry accounting in that inventory; incomplete capture stays missing rather than reporting a subset rate. Do not infer TTFT from full-response arrival, a configured value, or historical console summaries. The immutable attempt closure remains authoritative for task completion; rates do not substitute for it.

## Readiness that is explicitly still unobserved

`CONFIGURATIONS-v2.json` honestly marks all four configurations BLOCKED. Its common values match the prospective 3600 s agent / 1800 s evaluator limits, context 131072, visible cap 32768, concurrency 1 and sampling (0.6, 0.95, 20, 0, 1); it records source-derived compaction 98304 and leaves device-hour ceiling unset. This is configuration preparation, not a final frozen/runtime-observed record. Actual cross-host clock transport, source/precision/template/route observations and rate producers remain unqualified. Parent is already driving their implementation; they are not additional findings or new experiment requests here. Before the real invocation, bind the final common values to the authoritative decisions and source-derived compaction receipt rather than treating positivity checks as decision equality.

The official evaluator path remains opt-in through `--evaluate` (`sole_executor_v2.py:144–157`); the eventual authorized four-attempt invocation must select it. If unavailable/failing, preserve missing official outcome rather than claiming task completion. No current invocation or workload outcome was observed in this review.

## Validation boundary

Used the configured bundled Python and standard-library-only temporary fixtures. Independently verified the 36-member seal; checked source diffs for accepted F1/F2 continuity; reproduced R1 and R3; demonstrated R4's matched-subset gap and positive rejection controls. R2 follows the actual caller/context-manager source path. No pytest suite rerun was claimed and no accepted immutable bundle was modified. This note requests only the narrow caller/reducer repairs above and a bounded successor review; it does not authorize WP or any workload launch.
