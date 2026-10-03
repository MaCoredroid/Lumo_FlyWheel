# Candidate model-ID repair and preserved HTTP failure

**PASS for the bounded model-ID/HTTP repair.** No model, request-sampling, state, hook, numerical, or qualification criterion changed. A corrected fresh stage-one gate must bind the new driver and wrapper bytes below; this source review does not open that gate. A separate terminal-log membership caveat is preserved below.

## Failed run counts

For `q1-candidate-stage1-20260929T040755Z`, the saved engine started at 04:09:34 UTC, became healthy at 04:16:38 (424 seconds), and stopped at 04:16:42. There was **one engine boot, one attempted completion HTTP request, zero valid cases, zero case seals, zero raw objects/bytes**. The first request was rejected with HTTP 404 before the diagnostic case reached its hooks; the second requested repeat was never issued. Driver exit is 2; exact-owned engine cleanup records `stopped_and_removed`, cleanup rc0. The status string `COMPLETED_driver_rc=2_cleanup_rc=0` denotes terminal wrapper cleanup, not successful instrumentation or qualification. No numerical result exists.

The failed payload used `model="candidate"`; the unchanged engine log advertises `served_model_name=qwen3.8-27b-nvfp4-radixark` and records exactly one POST `/v1/completions` → 404 (`engine.log:54,214–216`). Its launch binding matches the preserved approved gate SHA `9866e64dd49d54136c0ffea259f8719bf8340e3f9ae5b4376fc15dcf075854c1`. The old driver did not retain the 404 response body, so the exact server error text is unavailable; the mismatched request/advertised model identities are directly source/log established.

## Repair verification

`q1_candidate_driver_v1_1.py` pins `SERVED_MODEL` to that actual deployed name. Before any completion/control loop, `run` invokes GET `/v1/models`, saves `served-model-discovery.json`, and requires exactly the expected advertised ID list. It does not choose an arbitrary advertised model. The request payload now uses the same constant. HTTP POST failures retain up to 65,536 body bytes, with an explicit truncation marker. Endpoint construction requires `/v1/completions`. All existing seal authentication, forcing, state diagnostics and stopping behavior remain unchanged.

The wrapper v2.5 diff contains only its explanatory version comment and the driver path substitution in both the executed default command and gate-bound REVIEWED map. No launch flags, readiness values, model settings, cleanup source, hook, fixture, or request criteria change.

I independently reran the three supplied actual-loopback HTTP CPU controls: correct discovery plus fixed-ID request wire, foreign advertised ID refusal with no POST, and retained HTTP-404 error body. All passed. These controls stub sealed model outputs; they validate transport wiring, not inference or qualification. Source snapshots and the complete test log are retained under `p0/monitor/review-response-20260927/candidate-model-id-repair-review/`.

## Terminal receipt caveat

70 of 71 indexed non-object files match the failed-run receipt exactly. `gpu_oom_guard.log` is an append-only exception: its first 201 bytes match the receipt hash exactly, followed by the 04:16:43 guard-exit line, one second after the receipt at 04:16:42.848787. The original receipt is preserved. This does not change the request/case/numerical counts, but the final directory cannot be described as wholly matching that receipt. Bind the later append in a supplemental terminal ledger; for future final receipts, ensure the owned guard has exited before indexing its final log, or explicitly seal that log separately after quiescence. Do not rewrite the original receipt or claim 71/71 equality.

## Exact repair hashes

- `tools/q1_candidate_driver_v1_1.py`: `269efee771cfd5032c21a3c53505d9ee2fad39d146f8e239d09ef1d345b183d4`
- `tools/run_q1_candidate_stage1_v2_5.sh`: `8b475bffbae5087a62580b8bf17f6d9115349021d9a59b6f286447e5611f9de2`

Only local source/receipt reads and loopback CPU controls were performed. No GPU, container, engine, remote mutation, retry, or gate change was made by this reviewer.
