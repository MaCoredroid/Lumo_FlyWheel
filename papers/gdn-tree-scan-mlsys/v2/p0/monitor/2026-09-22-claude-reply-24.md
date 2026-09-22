# Reply 24 — settled E1 campaign sources (launch red-team C1–C6 repair pass), corrected B4 gate v3e, errata to reply-23

Written 2026-09-22T09:37:08Z (measured). Nothing has booted since the B4 pilot (08:49Z); GPU idle. The parent's route archive (7f127f0c…, 840 files) is complete; the campaign waits for the reviewer's final recheck of the hashes below and for the parent's go — no user approval is needed per the parent, but I will not launch before that recheck.

## 1. Errata to reply-23 (append-only; the earlier text stands as written)
- The duplicate-token sibling case at B4 forward 7 is request **p085** (recorder id …9234a7af = API `cmpl-822fbbe30317f121` = `req_2_p085`), not p095; the same request is the row 3 → row 0 compaction after p072 finished. Batch order of the B4 cohort was **p072, p095, p017, p085** (rows 0–3).
- The "served last-child tie-break" reading (gate v3b) is WITHDRAWN: the independent source finding is that the loaded sampler chooses among equal-overlap duplicate draft sources RANDOMLY, even under greedy; neither first- nor last-child is a contract. Gate v3c/v3d/v3e verify the published path against the actual drafts and the full-vocabulary argmax (valid chain; each accepted node's draft == argmax(parent); each output == argmax(path node) incl. the bonus; length rule; a non-terminal STOP while a matching child exists is a violation — v3e adds this premature-stop rule with a minimal negative; the real B4 has no such case). `capture_gate.v3.*` (first-child FAIL) and `.v3b.*` (last-child) are preserved as superseded.
- `fidelity_per_request_summary.json` had the T9/T10 labels swapped relative to the frozen classes; `fidelity_per_request_summary.v2.json` (values unchanged) carries T9 = served bf16 ULP16 ≤ 1 (all requests: 1), T10 = served output absolute ≤ 7.275e-4 (p072 2.41e-4, p095 4.63e-4, p017 4.73e-4, p085 4.79e-4), T6 ≤ 9.54e-7; v1 preserved.
- The B4 pilot's API identity is now DERIVED by the certified exact inversion (`e1_api_tokens.derived.json`, sidecar `b1c3e3f08c8208c7`; report `e2-b4-api-inversion-redteam.md` f3b37263801f0fae…; raw strings preserved; all four certified vectors equal the singleton recovery and the ledger prefixes). Every E1 cell requests token ids directly (workload client + cohort client v2); the cell summary REFUSES string-domain evidence.

## 2. Tree B1/B4 qualification — CLOSED (bounded) per the parent after the independent final review (`e2-b4-final-redteam.md` 105d5eb78029e70a…)
`e1/e1_qualification_manifest.v1.json` `ef79dd675b4318fe`: `tree/B1` and `tree/B4` qualified with the gate/report hashes; the four native `<arm>/<batch>:preflight` keys enable the in-boot untimed preflight. Gate of record: capture gate v3e `0f0aef4f7880427a` (helper `3983b57e0bb0e52c`, controls 15/15 `fc7e12f6f206512b`), operands gate v2 `f5f5ec29fe04c023`, continuation v2 `e00e8684facfb819`. `E1_FREEZE.md` line 38 and `E2_QUALIFICATION_PLAN.md` carry the dated closure with the report hashes.

## 3. Launch red-team C1–C6 → repaired (single pass), with CPU stubs/smokes
| Blocker | Repair | Evidence |
|---|---|---|
| C1 embedded Python does not compile | all dispatch Python is now `python3 - <<'PYC'` heredocs (no f-string escapes); every embedded block is compile()d by the tests; the real snapshot/re-exec paths are exercised with stub launchers | `test_e1_campaign_stub.sh` 12/12, `test_e1_cell_driver_stub.sh` |
| C2 heavy asymmetric capture | timed arms: `FR10_METRICS=0`, empty draft/LCP/sampler traces, no capture sinks (tree and native); verifier checks the ACTUAL container env; stubs assert the launcher-received env | verify `env_no_heavy_capture_metrics_or_traces`; stub `cell5_tree_…no_capture` |
| C3 API ids in an engine-id phase manifest | summary builds the unique API→engine map first and translates ALL phases (preflight/warm-up → joiner 'warmup', timed) — missing/ambiguous → INVALID | smoke on the real B1 pilot: phases translated; unmapped id → INVALID |
| C4 preflight after timing | native first boots: frozen UNTIMED preflight segment (B1: 8 pilot prompts; B4: cohorts 1–4, 5–8; 32 tokens) → `e1_native_preflight.py --live` pass/fail on the boot config + live ledger prefix (no seal claimed) + preflight responses → only on PASS the fixed warm-up and the timed workload run in the SAME boot → after shutdown `--final` re-audits the sealed ledger; qualification recorded only if live PASS + final PASS and the cell is not INVALID; a FAIL issues no warm-up/timed request | preflight CPU controls 15/15 (live/final; margin-gated greedy rule; ENGAGED, tree_mtp, missing rows, unsealed, occupancy) |
| C5 snapshot does not prevent change | ONE campaign snapshot executed from itself (runner re-exec, driver `E1_SOURCE_DIR`), SHA256SUMS + route-dependency identity (patcher, kernel module, decode modes, HEAD) re-verified before every cell (exit 45 on change); attempt dirs `_aK`; measured cells never re-run; failed attempts re-attempted only with `E1_RETRY_FAILED=1` (retained); driver refuses an existing cell dir (exit 44); exact SPEC_CONFIG/TREE per arm frozen in the cells order, passed explicitly and verified post-boot | stub: tamper → 45; existing dir → 44; rerun → 46; retry → `_a2`; exact SPEC_CONFIG received |
| C6 zero support = INVALID | complete, reconciled run with zero usable intervals (joiner rc 3, no invalid/refused) → INSUFFICIENT_SUPPORT (EOS/budget retained; campaign continues); instrumentation loss / refused evidence / malformed mapping / missing output → INVALID (stop) | smoke: synthetic one-interval cell → INSUFFICIENT_SUPPORT rc 0 |
Also: the driver's `precheck` no longer clobbers `cleanup`'s exit code (a launcher failure now exits non-zero — the same dynamic-scoping pattern exists in `serve_drivers.v12/v13.sh` `precheck`, noted, not edited); the runner's running-process guard ignores shell wrappers and its own subshells.

## 4. Settled hashes (sha256[:16])
| File | sha |
|---|---|
| `e1/e1_run_cells.v2.sh` | `c79655d2ba57563b` |
| `e1/e1_cell_driver.v1.sh` | `8d45c4439e78f274` |
| `e1/e1_workload.py` | `bcc5af170fd16599` |
| `e1/e1_cell_verify.py` | `dec15e5bf3111fe4` |
| `e1/e1_cell_summary.py` | `ae3016f05246f471` |
| `e1/e1_native_preflight.py` | `6c671e0ef7f3834c` |
| `e1/e1_native_launch.v2.sh` | `5d86340e683a92c6` |
| `e7a/e7a_capture_launch.v7.sh` | `a5c398ffff43d1d8` |
| `e1/e1_make_cells.py` | `12bfe978c211549b` |
| `e1/e1_cells.v1.json` | `d5aa5b6eb3da19be` |
| `e1/e1_qualification_manifest.v1.json` | `ef79dd675b4318fe` |
| `e1/E1_FREEZE.md` | `dbd1605eee3ee06c` |
| `e1/e1_recorder.py` | `1cf3f53552e008c4` |
| `e1/e1_event_recorder_shim.py` | `06344718f035949e` |
| `e1/e1_join.py` | `cbee2ae70947f1ad` |
| `e1/e1_api_tokens_from_capture.v2.py` | `6d36ee15d8b5b628` |
| `e1/e1_api_tokens_derived_sidecar.py` | `b1c3e3f08c8208c7` |
| `e2/cohort_request.v2.py` | `385176cd9f97e813` |
| `e1/test_e1_campaign_stub.sh` | `eb38e72dba64b022` |
| `e1/test_e1_cell_driver_stub.sh` | `e7c7b9b80d344c9f` |
| `e1/test_e1_native_preflight_cpu.py` | `8ffc9d43ada719f6` |
| `e1/test_e1_cell_summary_real_b1.sh` | `73352db9285d24cb` |
| `e2/e7b_b4_capture_gate.py` | `0f0aef4f7880427a` |
| `e2/e7b_ledger_align.py` | `3983b57e0bb0e52c` |
| `e2/e7b_operands_gate.py` | `f5f5ec29fe04c023` |
| `e2/test_e7b_b4_capture_gate_cpu.py` | `fc7e12f6f206512b` |
| `e2/e7b_state_continuation.v2.py` | `e00e8684facfb819` |
| `e2/E2_QUALIFICATION_PLAN.md` | `38cd8006cb556028` |
| `e7a/serve_drivers.v13.sh` | `93462f077d39e94b` |
| `e2/e7b_loop.v4.sh` | `5684d3e67699a950` |
Tests (all_pass): {"e1/tests_out_e1_campaign_stub.json": true, "e1/tests_out_e1_cell_driver_stub.json": true, "e1/tests_out_e1_native_preflight_cpu.json": true, "e1/tests_out_e1_cell_summary_real_b1.json": true, "e2/tests_out/e7b_b4_capture_gate_v3e_cpu_controls.json": true}; check counts {"e1/tests_out_e1_campaign_stub.json": 12, "e1/tests_out_e1_cell_driver_stub.json": 30, "e1/tests_out_e1_native_preflight_cpu.json": 15, "e1/tests_out_e1_cell_summary_real_b1.json": 4, "e2/tests_out/e7b_b4_capture_gate_v3e_cpu_controls.json": 15}. Reports referenced: {"e1-campaign-launch-redteam.md": "a4571ea0aaa59502", "e2-b4-final-redteam.md": "105d5eb78029e70a", "e2-b4-api-inversion-redteam.md": "f3b37263801f0fae", "e2-policyB-b1-redteam.md": "dd7de6d504fdd4da"}.

## 5. Native baseline label and remaining pre-launch items
Native cells = PATCHED-RUNNER NATIVE BASELINE (stock MTP method, `naive_mtp`, `FR10_ENABLE_TREE_GDN=0`, no tree descriptor; tree-branch guards exclude the baked paths; independent AST review) — not a global feature-OFF claim (cells order and freeze say so). Pending before the first retained run: (1) the reviewer's final recheck of the hashes above; (2) the parent's go; (3) idle preconditioning before each boot (the launchers' guarded `recover_host_memory` + the driver `precheck` MemAvailable/swap guard; recorded in each cell's `precheck_before.txt`, outside timing). Not blocking: the bs16 multi-block KV regression parameter (optional maintenance) and the final campaign bootstrap/aggregate script (paired whole-boot blocks, B1/B4 separately, native-5 mean-rate denominator) — to be written before any aggregate is reported, not before cells run.


## Addendum (2026-09-22T09:38:28Z) — campaign aggregate written before any cell runs
`e1/e1_aggregate.py` `8cead3a82210ec30` (frozen rule: measured VALID cells only; paired whole-boot blocks; B1 and B4 separately; bootstrap of the 3 paired blocks, 10 000 resamples, seed 20260921, 95 % percentile; precision = (U − L)/(2 · mean native-5 rate); target 0.10; a contrast with any unmeasured paired block is 'not estimable' — no unpaired replacement; INSUFFICIENT_SUPPORT / INVALID / NOT_QUALIFIED cells listed only; diagnostics carried, never in the primary). CPU controls `e1/test_e1_aggregate_cpu.py` `94dc8868b5cadd57` 5/5 (`e1/tests_out_e1_aggregate_cpu.json`). The optional bs16 multi-block KV regression parameter is deferred (the reviewer's six independent spot checks already pass; not a launch item).
