# Q1.2a revision review, round 2

27 September 2026. Scope: closure of the six findings in `q1-2a-code-review.md`, for native-only characterization. Read-only SSH; no remote mutations, GPU work or gate edits.

## Current verdict

**APPROVED for the bounded native-only Q1.2a launch on freeze v3, SHA256 `4a506fdddda96d87e8e6338a9e1b7bfc95d5b275c14c648a2ed04a8896ca98ea`.** The three final v2 items are closed, all frozen source hashes match, and the refactored runner preserves the reviewed behavior. Approval is restricted to the scope in the final section below. Earlier hold/revision entries are retained chronologically. No candidate or full-Q1 approval follows from this review.

Reviewed hashes:

```text
645944992d116115976eea5a925f1ebf9a23562d8d8a3ebd693116daa0ec5d82 tools/q1_oracle.py
69089d5140cfb71c4d745bb35666978e54b1efc9be0aec49257400dd3c1b97dd tools/q1_native_baseline_runner.py
e06c548083ec32b15a1d429f1cc3971d409b5b49b34d5000b87f435f6a38da41 tools/q1_2a_reduce.py
91f954ad325d8dd0125df531c2685d471477c85751827c5676ff6f60fecf648f tools/run_q1_2a_native_baseline.sh
a535ae2326dfa9ef2c997e6d8b75a2e55a7be61182736d32fd9bdccf173a0b5e tools/tests/test_q1_2a_reduce.py
d4da018aa247177fecf118749533062179a08605a02a75df44e856609aacc6d3 q1_execution_manifest.q1.2a.json
```

The execution manifest currently declares 20 calibration fixtures × 3 native call variants × 2 processes × 8 repeats = 960 operator execution records. This is a native characterization denominator, not 960 LumoTree qualification cases. `T=12` remains the correct longest root-inclusive path length.

## First-round findings: resolution status

| Finding | Revision assessment |
| --- | --- |
| CPU/GPU metric mismatch | **Closed statically.** Oracle lines 147–148 move both operands to CPU float64. Runner lines 180–188 add an explicit CUDA-input/CPU-reference smoke check. |
| Reference/difference nonfinites hidden as zero | **Closed for the reported fault.** Oracle lines 153–175 separately count candidate/reference/difference nonfinites per head, invalidate metrics and emit null errors. The ignored `head_dim` parameter is removed. |
| Empty/subset/duplicate/incomplete evidence | **Substantially closed.** Reducer lines 64–76 require a nonempty manifest denominator, lines 120–170 reject duplicates, missing/extra fixture IDs, hash/metadata mismatches, missing variants and wrong repeated-array shapes. Execution-manifest binding at launch remains open below. |
| Integrity failures return success | **Closed for tested classes.** Runner lines 310–348 aggregates integrity and exits 3 on a failed completed characterization. Reducer returns 2/5/6 for malformed evidence, same-shape determinism failure, or nonfinite/invalid metrics. Finite cross-shape disagreements remain findings. |
| Incomplete per-repeat/per-token/A–B checks | **Closed for integrity coverage; one raw witness gap remains.** Runner records every repeat/token/variant, and reducer lines 179–217 recomputes within/across-process hashes and checks both A and B nonfinites/null rows. Process B's initial per-token raw tensor retention remains open below. |
| Unenforced source/launch provenance and failed-run receipts | **Substantially closed.** Runner enforces the native module hash, package export, fixture manifest and oracle identity; launcher lines 30–54 compares gate-reviewed source hashes and scope; retries refuse existing run directories; failure receipt/exit recording exists. Bind the execution manifest and restore fail-closed command handling below. |

## Four remaining closures

1. **Bind the execution manifest through launch and reduction.** `run_q1_2a_native_baseline.sh:40–43` pins fixture manifest and tools but not `q1_execution_manifest.q1.2a.json`; line 89 invokes the reducer with mutable default manifest paths. Add its hash to the gate, snapshot both manifests into the run directory and pass those snapshots explicitly to reduction. Verify the actual fixture-manifest bytes against the frozen hash in the execution manifest, not only a field copied into a process attestation. This closes the original completeness/provenance findings rather than adding a new experiment.
2. **Keep launcher failures fail-closed.** The launcher changed from `set -e` to `set -uo pipefail` to retain exit receipts. Its GPU contention query at line 72 and `docker ps` at line 74 now have unchecked failure statuses; an unavailable query can look like an empty GPU. Check those command statuses explicitly and abort with the retained receipt before launching. After reduction, preserve and return its nonzero code; the current last successful `write_receipt` can make the launcher itself exit zero despite a reducer failure. A failed first native process should not trigger another GPU process when the evidence can no longer constitute the approved two-process characterization.
3. **Retain process B's first per-token raw witness.** Launcher lines 80–81 still set `ARCH=0` for B, and runner lines 273–278 condition repeat-zero archival on that flag. If all B repeats agree internally but differ from A, no B per-token raw tensor is retained. Use `ARCH=1` for both processes, or implement a post-reduction retention mechanism that can recover those tensors without rerunning. Also retain a nonfinite repeat-zero witness unconditionally. This is a small amount of native-only diagnostic evidence, not a new case.
4. **Validate the actual per-token C2 reference.** Runner line 225 currently accepts `state_stack_sha_matches_host OR final-state/output within_bound`. Matching final state and outputs does not establish a bound for each intermediate recurrent state; forgetting can erase an earlier discrepancy. The following per-token errors at line 284 may therefore use an unvalidated intermediate reference while line 286 labels it validated. Require the frozen state-stack hash, or store/bound the actual host per-token states, or predeclare one consistent in-container reference with its own hash/provenance for all reported surfaces. Do not infer intermediate-state agreement from a final-state bound.

After these closures, freeze the updated source/fixture/execution hashes, run the affected CPU/static tests, and provide the exact dry-run command and source-bound CPU receipt for a final narrow launch verdict. No full-model hooks are required for this stage.

## Independent CPU verification

I fetched reducer `e06c548…` and tests `a535ae23…` read-only over SSH, loaded their source in memory locally, and executed the tests using the repository's pytest-capable Python without torch or GPU. All 23 supplied cases passed: valid pair and 22 malformed/determinism/nonfinite rejection cases. I added one in-memory case setting all finite primary/per-token/padded shape-relation findings false while preserving each arm's within/across-process repeatability. It returned code zero and reported `shape_relation_findings_all_true=false`, as intended. **24/24 CPU checks passed.** Temporary test directories were local and automatically removed.

This confirms the key distinction: same-shape repeated execution and cross-process repeatability are integrity rules; finite differences between primary, per-token and padded call shapes are characterization observations unless separately predeclared as an acceptance rule. No numerical tolerance was inferred from these tests.

## Freeze v2 final check

Reviewed `FREEZE-Q1_2A.json`, frozen at `2026-09-27T21:05:34.548960+00:00`, SHA256 `e0470c1554a52a15d572ea8f7ecdf84c5e09504f05b5b92fbecc5265ed86ef60`. Current core artifacts match the freeze:

```text
676dcc78f761a93b542243f75aa139165ef3e4b858c494034f4e0732282a4306 tools/q1_native_baseline_runner.py
84fbddbe577f4bae27fc3816efa7fad58d023a4775b54282f064fce129f51e37 tools/q1_2a_reduce.py
1db5abd611278c48662fa2af57e737ddbb6bd5d871d278c0209d140934b2261c tools/run_q1_2a_native_baseline.sh
07256c21c8647e109dc2bdc8dd42f628b094e5ccf5b44da129773f68649f238e fixtures/q1_2a/manifest.json
d4da018aa247177fecf118749533062179a08605a02a75df44e856609aacc6d3 q1_execution_manifest.q1.2a.json
3ac604002be91dd0da3cd635d0afb21ce75f1653a097d195d89d3ecd53f17499 tools/test_log.txt
09bf3a941fbdfd974e315d0849ce28d0d570b641f1e8d28569e24423245662a0 tools/launcher_dry_run.txt
```

All 20 calibration fixture file hashes were independently recomputed and match. The execution manifest is exactly the 960 unique tuples in the declared Cartesian product; its denominator and fixture-manifest hash match. The frozen log says **42 passed in 13.78s**. The dry run binds the correct immutable image, calibration block, eight repetitions, matching fixture hash, and per-token archival for both A and B.

Three closures landed: the launcher pins and snapshots the execution manifest, queries fail closed, and both processes retain per-token repeat-zero tensors. Runner line 228 now requires the actual host-recorded state-stack hash; a mismatch is invalid and its raw stack is preserved. These are accepted.

Remaining corrections are small and stay within the original integrity/provenance findings:

1. **Propagate executable failures.** Launcher lines 91–94 still record a nonzero A exit then launch B; lines 96–99 record a nonzero reducer exit then finish with a successful receipt-write command. Capture the process/reducer status in variables, retain the failure receipt, stop after a failed first process, and explicitly exit with the final nonzero status. This preserves the reducer's integrity decision at the launch boundary.
2. **Verify snapshot bytes directly.** The reducer now requires explicit manifest snapshot paths, but lines 61–76/107–108 do not hash the fixture-manifest file against `em.fixture_manifest_sha256`; they compare copied fields only. Add the direct byte hash check and a tampered-snapshot CPU test. The launch-time snapshots already make this easy.
3. **Make the freeze inventory self-consistent.** `tools/build_source_runtime_manifest.py` is included with expected hash `afcc943f…` but currently hashes `a6be5a6b5d0e742acfdb3b771ee67da0ffdaa90b2c335105403fdb5506e266a7`. This auxiliary tool is not used by the native runner. Refresh that inventory entry or explicitly omit it from the narrow freeze; do not claim the complete inventory matched unchanged.

No other experiment or full-model requirement is added. Once these edits are hash-frozen with the affected tests, a narrow native-only approval can be issued immediately.

## Freeze v3 final approval

**APPROVE the native-only Q1.2a launch**, using `FREEZE-Q1_2A.json` frozen at `2026-09-27T21:10:42` UTC, SHA256:

```text
4a506fdddda96d87e8e6338a9e1b7bfc95d5b275c14c648a2ed04a8896ca98ea
```

The parent-synced local files all match the freeze inventory with zero mismatches. The fixture manifest remains `07256c21…` and the execution manifest remains `d4da018a…`; all 20 calibration fixtures and the exact 960-record product were checked in the preceding review and are bound unchanged.

Final changed runtime hashes:

```text
b02b1f9c33ce429849a9ad5632e8854a3cdb24018bd33e257603b9599ef00262 tools/q1_native_baseline_runner.py
35ba59d33f23b3d601b74725da19dba302aa3645911d6220991ea7981717d830 tools/q1_2a_reduce.py
24286867a1e5b7e4c5c78a644fa780448f5b249671ecbc3577d31793c6c93796 tools/run_q1_2a_native_baseline.sh
dd288a4c82fa36fca16450179779a68e0f6de3ed17594d6694118aef04f37bfa tools/tests/test_q1_runner_cpu_loop.py
```

The launcher now captures `PROC_RC`, writes the failure receipt and exits immediately after a failed first process (lines 92–100). It captures and returns `REDUCE_RC` after retaining the final receipt (103–109). The reducer hashes the actual fixture snapshot bytes and compares them with the frozen execution-manifest hash (64–66). The auxiliary source-manifest generator inventory is consistent. These close the last v2 findings.

The refactor extracts `characterize_fixture` without replacing production execution: `main` explicitly passes `run_native` into it. Reset-in-place, cloned CPU output/state retention, separate per-token reference validation, all-repeat checks and finite cross-shape findings remain intact. The five CPU control-flow tests inject a separate native-interface stub only in test code and explicitly make no native-kernel numerical claim.

I independently ran the frozen reducer suite locally: **24 passed in 0.09s**, including the new fixture-snapshot byte-tampering rejection. The source-bound host log reports **48 CPU tests passed**, comprising 12 oracle, 4 fixture, 3 static, 24 reducer and 5 runner control-flow tests. The local static/torch suites could not be collected because local torch is absent; their host log and sources were inspected rather than presented as locally rerun.

Approved executable scope: **20 calibration operator fixtures, 10 strata, primary/per-token/padded16 native variants, 8 repeats in each of two sequential fresh processes, 960 execution records**, on image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`, native module hash `000ab899…`. Use the matching `reviewed_hashes` and `reviewed_scope` in the parent-owned Q1.2a gate. No source changes are covered by this approval; a repaired retry requires retained failure evidence and new hashes if code changes.

This approval permits baseline characterization only. It supplies no tolerance acceptance, Lumo/C0 result, full-model/next-forward/conv qualification, timing result, or later-stage gate. Review actual characterization outputs before freezing any candidate numerical policy. The reviewer has not launched GPU work or modified a gate.
