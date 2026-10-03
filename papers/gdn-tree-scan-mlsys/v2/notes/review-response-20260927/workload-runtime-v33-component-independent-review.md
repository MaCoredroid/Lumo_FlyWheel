# Workload caller v3.3 / proxy v2: bounded independent component review

**Verdict:** close the reproduced R2 cancellation/quiescence and R4 foreign-artifact reuse defects at the reviewed source/component level. The exact-CID created-agent cleanup addition passes. No new blocking defect was found within these changes. This is not final package or runtime admission: WP remains closed, with **0/4** workload attempts. The four-task-arm schedule is unchanged (one scikit-learn__scikit-learn-9288 attempt each: AR, CHAIN_MTP, SGLANG_EAGLE, LUMOTREE; no retry/tuning/extra attempt).

## Reviewed identity and limits

Read-only snapshot: `papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-runtime-v33-component-reviewed-20260928T172705Z`. The caller snapshot authenticated 236 references across 104 fetched files, including recursive closure/dependency manifests; the subsequent proxy snapshot authenticated its 8 payload members. Existing support files were copied from the earlier verified v3.2 snapshot without modifying it. No final root package successor existed at the bounded snapshot check; no polling followed. The parent’s accepted R1/R3, F1/F2, seed-absence decision and exception-class determination remain closed.

| Artifact | SHA256 |
|---|---|
| `attempt-runtime/MANIFEST-caller-v3_3.json` | `9eed1e3af8e748cba98ed698dfece2c62719686a516e86cbf7c995447cdbf705` |
| `attempt-closure/MANIFEST-successor-v4.json` | `21f74f960f835fc57b102559833e9273bd382ad1ec6984490ddaca7b81a0d80b` |
| `proxy-capture/MANIFEST-v2.json` | `51d29f60bb037d36fd10c755ac0ac0937ef7eac5ef5e3fd0eb67d4d5b39fc721` |
| Actual proxy v2 module | `33d8743496858a6d63b2e39874da8f40ca799afe3d2024ed711fe188a69cb571` |
| Independent evidence seal (361 files) | `0e3833beff05eb4cb6b9443f54e65cd2bf1d1f03c43a86ebc6b261b948685a5d` |

The earlier partial closure-v4 seal `0a4b2ed6…` remains preserved in the earlier partial snapshot; the current caller pins the final `21f74f96…` seal above.

## Closures

**R2 — closed for the reproduced race.** `boot_v3_3.py:84–176` catches timeout and every post-Popen BaseException, signals only its verified private process group, polls/reaps the leader and observes the group gone, escalating TERM to KILL within bounded waits. `boot_server` retains the launcher state before the failure inventory. `closure_v3_2.py:351–356` will not close setup failure when this proof is absent. `sole_executor_v3_3.py:177–180` also blocks the next arm if a successful launcher left its group alive. Connected injected controls cover cancellation before a server exists, TERM-resistant/KILL-success, surviving group, foreign pgid, missing runner proof, and successful launch with/without a leftover group. No real process was started.

**R4 — closed for the reproduced reuse attack.** The caller atomically claims a new attempt capture directory/run ID before boot, records/passes its environment, captures its own pre/post metric receipts around the agent and settled stream, and no longer accepts external artifact paths. `instrumentation_v3_2.py:107–135,138–200` authenticates claim, attempt, server CID, raw metric hashes, run ID and pinned producer before reduction. Reused files, foreign run IDs/producers/CIDs and changed metrics refuse. The source-bound proxy now stamps the same run ID and its actual module hash on every start/capture row.

Beyond the worker's stand-in producer tests, independent controls used the actual proxy-v2 hash with the real caller/collector/closure/reducer chain. Three further controls executed only the actual sealed proxy capture/profile helper AST, writing local raw JSONL into the caller-claimed directory during an injected agent wait. Those bytes passed through the real caller and reducer. Two completed fixture requests with repeated cumulative counts, first batches of two tokens and untimed final tails yielded exactly **25 / 4 = 6.25** fixture tokens/s; neither SSE count nor final untimed totals were substituted. Omitting either the first or last endpoint count produced NOT_REDUCED despite valid final usage. This is fixture arithmetic, not a performance result or live proxy deployment.

**Owned-agent cleanup — accepted bounded addition.** `boot_v3_3.py:559–656` binds the exact creation-return CID to name, pinned image and full attempt labels, then removes only that proven created/never-started container and verifies typed absence. Unknown CID discovery never authorizes deletion. The successor closure retains/binds these records, and incomplete cleanup prevents the next arm. Injected tests cover normal cleanup, creation exceptions before/after daemon creation, wrong name/labels, failed removal, started container and inspection failure. Normal executed-attempt behavior is unchanged.

## Validation and retained failures

The immutable worker caller log `tools/test_log.workload_caller_v3_3.attempt1.txt` (SHA `a27c7755a961782809fc2bae11dcb5bc874472fbda3e181455766bcc29cd9bbb`) reports **256 caller tests + 407 regression tests passed**, with no failure summary. Proxy log `tools/test_log.proxy_capture_v2.attempt1.txt` (SHA `4fb6fac4e51eeb73f58185db0bea9312f26fa76a2dae90ccb8d716b7cc93dcc6`) reports **20 passed**. These are retained worker logs, not locally rerun entire suites.

Independently ran **31 explicit injected/AST controls**, all passed. The configured Python lacks pytest; a small local decorator/assertion shim invoked selected unchanged test functions explicitly. No claim of a local pytest suite run. Final script `REVIEW-controls.py` SHA `9d0d6ffdbb4a00dea97e54f1f9e41336f1a16a373d18b11e4618dd4b0c9bbb1f`; log SHA `4081a2fdf0e165ab2541d3154e11278c190a73bff5c2a61b1f3515b4cdc986f3`. Two initial reviewer-control failures are preserved: one fixture default still stamped the v1 producer after updating its expected pin (correctly refused); one assertion incorrectly expected a repeated count when the first count was deliberately absent. Only reviewer fixtures/assertions were corrected. No submitted source changed.

## Remaining final-package dependencies

1. Seal the combined root package and bind the actual proxy-v2 producer hash, `measurement_binding.metrics_path`, closure/runtime successor roles, and final configuration/launcher bytes. The caller README openly uses a v1 producer stand-in in its own tests; the final freeze must use the actual v2 hash above.
2. Demonstrate in the final source/configuration that all four owned launchers run that module and propagate the caller’s capture directory/run ID plus `LUMO_PROXY_WORKLOAD_PROFILE=v2` to its process. Caller-to-launcher environment delivery and proxy-side consumption are reviewed separately; this final deployment connection remains unsealed here.
3. Retain the existing runtime admission checks for owned server/endpoint, actual metric counters, stream completion, serving settings and agent bundle. CPU fixtures do not establish those observations. A missing event count remains undefined; no substitute rate or extra workload attempt is authorized.

Common upstream continuous usage is supported by the already captured engine sources and the reviewed proxy transformation. It does not change model precision. The proxy preserves final usage behavior and strips injected per-choice usage when the agent did not request it; SGLang’s documented null-versus-absent field difference is retained in its feasibility memo. No newly observed source evidence requires reversing the parent’s option-1 preparation decision. This review does not reopen the separate numerical deployment admission.
