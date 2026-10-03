# Request/worker observer — bounded independent source review

2026-09-28. **PASS for the repaired source below; one initial SGLang admission defect was reproduced and closed.** This is draft source/CPU preparation, not serving qualification, workload admission or permission to launch. No live engine, tensor, GPU, container, API, workload or `/proc` process observation was executed. Original implementation and failing controls are preserved.

Paths below are relative to paper v2 unless stated otherwise. `R` denotes `experiments/review-response-20260927/workload-plan/tools/runtime-collectors/`. The inspected sources are under `workload-plan/inspections/codex-probe-hooks-20260928T230640Z` and `codex-vllm-idmap-20260928T231301Z` within the same campaign.

## F1: silent SGLang queue rejection produced an admission record — closed

Initial observer SHA `f541b1362042e532ae79470f94a1b749f59acc32b93e964237ec11f2d0bdf919`, lines 205–214, checked `finished_reason` only before calling `_add_request_to_queue`, then unconditionally emitted a sampling record and set `observed_sampling=True` on normal return. The inspected scheduler (SHA `26df94340228909d720f928e8da78eaf0c8068ef417070c9782835891a6e8cc3`, lines 2641–2645) can return normally after priority rejection or queue-limit rejection without admitting this request. Priority rejection does not require changing `req.finished_reason`.

Independent controls execute the exact inspected queue-method AST with synthetic collaborators. Both rejected paths originally emitted one record with an empty waiting queue and `observed_sampling=True`. This did not by itself establish successful HTTP completion, which remains a separate gate, but the sampling receipt overstated queue admission.

Repaired observer SHA `de881f0ab6302c4fee8d6d7ef2c8d6f7c512b69b34e5c3819056435e425d1451`, lines 212–214, now requires nondisaggregated mode, unfinished request state, and **same-object** membership in the live waiting queue after the original method returns. Both original rejection cases now refuse without any sampling record or observed flag. The positive queue path preserves the original return. Independent wrong-object, post-call finished-state and disaggregation controls also refuse. Only this three-line implementation guard changed in the repair.

## Verified source boundaries

- **vLLM ID join:** the inspected chat entry point constructs `chatcmpl-` plus its base request ID (chat serving lines 259–261; header/body selection in engine serving lines 580–589). `InputProcessor.assign_request_id` preserves that external ID and appends eight random characters (input processor lines 215–232). `AsyncLLM` calls it before subsequent request processing (lines 365–376). The observer wraps that static method once, preserves its result, disallows fanout, and writes the completed external/internal mapping before the caller proceeds. Worker association checks exact attempt/boot/request/source fields, the external ID and the eight-hex suffix; randomization disabled, malformed IDs and foreign mapping identities refuse. The worker records the mapping file digest, allowing the future consumer to bind the exact bytes it used.
- **Instantiated worker sampling:** the inspected runner assigns incoming sampling parameters to `CachedRequestState` (lines 1127–1173), and `execute_model` calls `_update_states` (line 3817). The observer reads that cached state only after the original update returns. It does not claim these scalar fields establish token-output correctness. Original update return values, including a deferred result, and execute arguments/return are preserved. Sampling is restricted to the one mapped probe; a mixed scheduled request set refuses.
- **SGLang sampling boundary:** tokenizer normalization occurs at tokenizer-manager line 1340; scheduler request capping occurs in `init_req_max_new_tokens`, lines 2118–2153, before queue insertion in the normal request path. Chat adaptation preserves `request.rid` at serving-chat line 1010. The repaired hook records the normalized/capped request parameters only after proven queue admission. It observes scheduler request settings, not a sampled device result.
- **Graph capture is not replay:** inspected `CUDAGraphWrapper.__call__` returns an underlying call on unavailable/mismatched context, captures when `entry.cudagraph is None` and returns the capture output, or calls `entry.cudagraph.replay()` and returns `entry.output` (lines 233–356). Repository patcher `scripts/fr10_phase4_patch_vllm_tree_gdn.py` lines 41958–42098 retains that final replay/return path and attaches the fixed32 capture signature. The observer takes the existing graph identity before calling the original method; a newly captured graph cannot produce its replay event. It checks FULL mode, unchanged graph/output identity, the signed capture payload and matching descriptor. `_fr13_fixed32_capture_end` (patcher lines 5814–6060) validates captured work and stores the signature/canonical manifest tuple the observer reads. The manifest supplies captured work identity, not a new live Python call count.
- **Only completed host dispatch:** graph events are buffered under the named request's `ContextVar` and written only after the enclosing original `execute_model` returns successfully. A later worker exception discards buffered events; the context is reset in `finally`. The record explicitly denies device-completion or timing inference. No synchronization, tensor `.item()`, tensor data read, CUDA query or new GPU event appears in this observer.
- **Unrelated requests:** external-ID filtering, exact worker-ID scheduling and the context/arm guard prevent unrelated requests from producing sampling or graph records. Original methods still execute with their arguments and return values. This is absence of records, not a claim of literally zero Python branch overhead.

## CPU evidence and identities

Initial snapshot: `p0/monitor/review-response-20260927/request-worker-observer-independent-v1/`. Repair snapshots: the siblings `request-worker-observer-independent-v1-repair1/` and final `request-worker-observer-independent-v1-repair2/`. Each contains source copies, `SNAPSHOT.json`, the independent control program and its JSON output. Canonical source/test copies equal the repaired snapshot at closure. `EVIDENCE-HASHES.json` records the inspected dependency hashes; all 13 successfully extracted source files match their two inspection receipts. Failed extraction paths in those receipts are not represented as evidence.

Commands, executed only in the independent snapshot:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_request_worker_observer_v1
PYTHONDONTWRITEBYTECODE=1 python3 independent_controls.py
```

The original 11 supplied tests passed despite F1. After repair, **12 supplied tests and 14 independent controls pass**, including the exact queue-AST rejection cases, ID-format negatives, late worker failure, context reset and unrelated request behavior. Bind-method and process-identity collaborators are mocked in synthetic execution; supplied source-binding rejection and static source tracing complement those tests. No test is claimed to establish an actual worker import or serving boot.

| Artifact | SHA256 |
|---|---|
| Final observer | `653088a9b40e7ec54cc249e80d80e8faa8ba9fcc249309aeeb967a9f157c9c14` |
| Repaired supplied tests | `7080c8fbdf7c96a16567b5b931584040561a87f080ddaf1608decb0ccda7a804` |
| Independent repaired controls | `1f346ab0a50f9a0de93b39422fa54a50709bcd19e25733dc971c8bce54c741b2` |
| Independent repaired results | `32b3c4f626be16f10dc127c1f8c12211d8a9a092617ba6bca88256ce4b55e995` |
| Inspected production patcher | `c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e` |

The final recording-only delta adds `capture_manifest_canonical` beside the parsed manifest (line 193). Exact diff inspection shows no changed dispatch or numerical control flow. The 12 supplied and 14 independent controls pass again on final SHA `653088a9…`; one additional synthetic check recomputes the capture signature from those preserved bytes and verifies their parsed object matches `capture_manifest` (`canonical-payload-control.json`). This avoids requiring the future consumer to guess JSON serialization.

## Explicitly pending integration

The helper has no caller or final freeze yet. A future owned boot must install hooks before the relevant methods run, use the reviewed source-only import policy, and bind actual **post-patch loaded** runner/cuda-graph/GDN modules rather than assuming the inspected unpatched image hash is the loaded Lumo route. The graph/context/manifest module arguments must be those selected loaded modules. The named pre-agent request must have the frozen body/header ID, sampler/cap/seed and isolated request schedule.

The host consumer must join mapping/API process and worker process receipts to the exact owned boot/PID epochs, freeze source/policy identities and complete record inventory, require observed sampling plus applicable dispatch evidence, compare capture work with the frozen expected route, and require successful matching HTTP completion. Mapping contents and a capture manifest alone are not sufficient runtime admission. No method here observes device completion, output parity, full workload coverage or task success. These remain explicit integration/qualification boundaries; no additional source blocker was found in this bounded review.
