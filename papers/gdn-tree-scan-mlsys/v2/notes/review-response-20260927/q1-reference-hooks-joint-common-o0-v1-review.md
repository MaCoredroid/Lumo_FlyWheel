# Joint common-O0 runtime hook: bounded source review

**PASS for the reviewed source connection; no launch or numerical qualification approval.** No new blocking defect was found. The existing union importer, native mapper, source builder and imported-history reviews are reused, not reopened.

Reviewed hook: `tools/q1_reference_hooks_joint_common_o0_v1.py`, SHA-256 `6d4a95e06a95f8f00f708800a146ca92a6264a0e5d80e3210a97061866c18809` (6,804 bytes). Supplied test: `tools/tests/test_q1_reference_hooks_joint_common_o0_v1.py`, SHA-256 `023e9615ef4b1e13d27c60917ace3b3a5fd2d1ca4b44bc55e1568f13f4485279`. Full dependency snapshot is in the linked review evidence directory below; all snapshotted sources still matched when this review was sealed.

## Boundary and ordering

- Lines 34–52 validate the actual prepared root token and every position axis, then invoke the actual inherited target pre-forward. At its O0 seam the current destination MTP history must already exist (lines 58–60). Natural prefill remains destination model work.
- Lines 62–66 resolve the frozen source row and complete `SourceDriver.authenticate_seal` before either the bootstrap export or the union importer. The audited record digest must match the source-bound document. Invalid audit, changed record digest and interrupted audit all refused before the mock union write in the independent controls.
- Lines 67–77 export and preserve natural target/MTP bootstrap; require the live MTP prefix snapshot to equal the just-completed destination first-pass snapshot; call the accepted single union transaction; independently export target and MTP again; require both source digests and completed transaction; then adopt the source carry. The actual inherited pre-forward returns only after this succeeds, and `History.pre` follows it at line 49. This order ensures the subsequent root target forward consumes the imported state.
- History adoption retains the destination allocation, generation, request/observation and metadata serial. It separates the destination bootstrap records from subsequent destination continuation records. Source hidden/scratch/owner/metadata are not imported. The unchanged `on_target_hidden` forwards the actual destination tensor to `History.hidden`; unchanged `History.first` uses that step's cloned hidden rows in its native `target_hidden_states` argument (`q1_native_mtp_history_v2.py:82–87, 98–110`).
- Any exception or `BaseException` escaping the new seam leaves `unusable` latched. The actual legacy target hook catches ordinary exceptions and seals the case, but the outer hook explicitly rejects that sealed/invalid state, calls `_mtp_fail`, poisons the MTP owner and raises before `History.pre`. Independent controls exercised the real inherited method, rather than replacing it with a mock.
- `_mandatory_observation_problems` requires successful import and both history segments (84–89). `_write` (91–97) retains the distinct schema, bootstrap, import and continuation segments, seals those bytes, and invokes the base writer directly. Foreign-observation writes do not inherit another observation's history.

## CPU evidence

The supplied 8 seam controls passed. Ten additional independent controls passed using local CPU Torch 2.8.0. They cover the real inherited root pre-forward ordering and actual hidden argument; source audit rejection and mismatched audited record digest; an audit interruption; target and MTP post-import digest mismatches; an interrupted union import; a post-import destination-lease mismatch; a wrong second position axis; and foreign-observation history isolation/seal correctness. The bad-source/readback controls verify that the current forward stops and owner readiness becomes false even after the base catches the original exception.

These are orchestration tests: source selection/authentication outcomes, object exports and union copy are injected; the history adoption and inherited target pre-forward are real Python methods. They neither execute production model/GPU code nor re-establish the already-reviewed importer's numerical or raw-copy behavior. The independent positive preserves destination lease object and serial and invokes `History.pre` only after adoption.

Evidence: `p0/monitor/review-response-20260927/joint-common-o0-hooks-v1-review/` contains `SNAPSHOT.json`, exact source copies, `independent_controls.py`, both test logs and `REVIEW-SEAL.json`.

The module remains prospective. A future caller/driver/auditor must explicitly support this new schema and separate history segments, and its frozen dependency set must bind these exact bytes. No launcher, gate, model setting, raw experiment result or qualification count was changed by this review.
