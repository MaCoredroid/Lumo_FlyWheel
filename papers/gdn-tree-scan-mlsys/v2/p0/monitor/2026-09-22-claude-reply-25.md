# Reply 25 — F1 terminal seal, F2 snapshotted native reference, F3 frozen allocation, Q metadata corrections (settled hashes)

Written 2026-09-22T09:50:10Z (measured). No GPU launch; no new experiment. The idle-pane draft prompt was not treated as a verdict; the campaign stays unstarted until your concrete go after the reviewer's report (`e1-campaign-final-redteam.md` a562a452f7a1cf7c…).

## Repairs
- **F1 (terminal eligibility):** the summary now writes `cell_result.preliminary.json` only. The runner writes the durable `cell_result.json` with `terminal_seal` (gates, UTC, campaign SHA256SUMS hash) ONLY after verify + summary + (native) FINAL audit pass; any failed/aborted gate seals the attempt INVALID (`rate_not_eligible`, primary removed) and stops the campaign; the preliminary file is preserved separately. The aggregate counts only sealed VALID cells and REFUSES unsealed/preliminary results (listed as failed attempts); resume treats unsealed or INVALID attempts as failed (exit 46 unless `E1_RETRY_FAILED=1` → new attempt) and never re-runs a sealed measured cell. NOT_QUALIFIED entries are sealed too.
- **F2 (native reference):** `prefix_scores.json` (E7a native-score boot) is copied into the campaign snapshot (hashed in SHA256SUMS); the driver copies it into the cell snapshot and passes `--scores` to the LIVE check, the runner passes the same snapshot path to the FINAL check; `--scores` is required (no live default) and its sha256 is recorded in each preflight JSON.
- **F3 (frozen allocation):** `gpu_util` 0.6 / `max_model_len` 16384 are read from the cells order's `frozen_settings` (ambient `GPU_UTIL`/`MAX_MODEL_LEN` cannot reach the launcher) and verified in the actual container Cmd by `e1_cell_verify.py --gpu-util --max-model-len` (runner passes both from the cells order).
- **Q metadata:** `e1_qualification_manifest.v1.json`: T9 = served bf16 ULP16 (1), T10 = served output absolute (B1 2.36e-4; B4 ≤ 4.79e-4) — labels corrected; `reports_sha256` now includes `e2-b4-final-redteam.md` = 105d5eb78029e70a864cbefa3cb71a2886f94733d9a107055be59ef77d654788; batch order p072, p095, p017, p085; fidelity summary of record = `fidelity_per_request_summary.v2.json`.
- `E1_FREEZE.md`: dated section "Terminal eligibility and frozen allocation" (F1–F3 rules).

## Targeted negatives (all CPU; outputs in `experiments/e1/tests_out_*.json`)
- **F1 exact-runner (stubbed driver/verify/summary/preflight; final audit FAIL after verify+summary PASS):** runner exit 12; no qualification.json; terminal cell_result.json sealed INVALID, rate_not_eligible, no primary; gates {verify PASS, summary VALID, preflight_final FAIL}; cell_result.preliminary.json preserved (VALID 123); aggregate: rate None, status INVALID, contrast 'not estimable'; resume without E1_RETRY_FAILED: exit 46 (no new attempt); retry with passing final: attempt _a2 sealed VALID, native-5/B1 qualified, aggregate counts only _a2 (123.0); measured cell never re-run (exit 0, no _a3); tree cell sealed after verify+summary (gates without preflight_final)
- **aggregate:** cell_result.json VALID 123 WITHOUT terminal_seal → FAILED_ATTEMPT_UNSEALED (refused), rate None, contrast not estimable; sealed INVALID after final audit → not counted, preliminary preserved
- **verify (F3):** Cmd --gpu-memory-utilization '0.9' → FAIL cmd_gpu_memory_utilization_frozen; Cmd --max-model-len '8192' → FAIL cmd_max_model_len_frozen
- **campaign stub (F3/F2):** ambient GPU_UTIL=0.9 MAX_MODEL_LEN=8192 in the runner shell → launcher received 0.6 / 16384; campaign snapshot contains prefix_scores.json and hashes it in SHA256SUMS
- **summary:** writes cell_result.preliminary.json only; no cell_result.json

Test outcomes: {"e1/tests_out_e1_runner_terminal_gate.json": {"all_pass": true, "n": 9}, "e1/tests_out_e1_campaign_stub.json": {"all_pass": true, "n": 13}, "e1/tests_out_e1_cell_driver_stub.json": {"all_pass": true, "n": 30}, "e1/tests_out_e1_native_preflight_cpu.json": {"all_pass": true, "n": 17}, "e1/tests_out_e1_cell_summary_real_b1.json": {"all_pass": true, "n": 5}, "e1/tests_out_e1_aggregate_cpu.json": {"all_pass": true, "n": 7}}.

## Settled hashes (sha256[:16])
| File | sha |
|---|---|
| `e1/e1_run_cells.v2.sh` | `08e3aa21b08cdd5b` |
| `e1/e1_cell_driver.v1.sh` | `25aa6c2195957630` |
| `e1/e1_cell_summary.py` | `04a119321a2aa95a` |
| `e1/e1_cell_verify.py` | `e4c73148de56d75b` |
| `e1/e1_native_preflight.py` | `6dd79084a8a7dcec` |
| `e1/e1_aggregate.py` | `98beec9efac26f56` |
| `e1/e1_workload.py` | `bcc5af170fd16599` |
| `e1/e1_native_launch.v2.sh` | `5d86340e683a92c6` |
| `e7a/e7a_capture_launch.v7.sh` | `a5c398ffff43d1d8` |
| `e1/e1_make_cells.py` | `12bfe978c211549b` |
| `e1/e1_cells.v1.json` | `d5aa5b6eb3da19be` |
| `e1/e1_qualification_manifest.v1.json` | `c60a4a4c661e84a6` |
| `e1/E1_FREEZE.md` | `11f5dd00b8d8b707` |
| `e1/e1_recorder.py` | `1cf3f53552e008c4` |
| `e1/e1_event_recorder_shim.py` | `06344718f035949e` |
| `e1/e1_join.py` | `cbee2ae70947f1ad` |
| `e1/e1_api_tokens_from_capture.v2.py` | `6d36ee15d8b5b628` |
| `e1/test_e1_runner_terminal_gate.sh` | `81cd0db047f7b930` |
| `e1/test_e1_campaign_stub.sh` | `1c72d8d1bb7e55a7` |
| `e1/test_e1_cell_driver_stub.sh` | `e7c7b9b80d344c9f` |
| `e1/test_e1_native_preflight_cpu.py` | `e73cdffb20361a8b` |
| `e1/test_e1_cell_summary_real_b1.sh` | `f0462d3567ab8b8f` |
| `e1/test_e1_aggregate_cpu.py` | `6c5aaf74c71c2f38` |
