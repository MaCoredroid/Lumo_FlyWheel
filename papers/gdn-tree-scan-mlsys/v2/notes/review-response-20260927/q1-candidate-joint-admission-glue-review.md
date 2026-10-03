# Joint candidate admission glue: bounded CPU/source review

**PASS for the five-file core glue snapshot below after the live-convolution and request-policy corrections. No remaining blocking incompatibility found in this bounded pass.** This is prospective source readiness, not launch authority, actual candidate collection, native baseline acceptance or numerical qualification. The new outer launcher/gate is a separate review.

| File | SHA-256 |
| --- | --- |
| q1_candidate_job_joint_v1.py | `2657e370d0a8f0f2da6e95bf219c20a987e56fe700ae36ad10b931f32e949a99` |
| q1_candidate_joint_full_audit_v1.py | `86dca38041061e1b806c99c01058af86754f39b06060e7b9b305b49616c505c3` |
| q1_candidate_joint_import_audit_v1.py | `4502e52756b4e231d39736cac3da7292f5116fe57a54702cfe6644e8e9f9bd0b` |
| q1_candidate_joint_seal_auth_v1.py | `d8c928cffa4a12dbafd281efe6eb69affb1c743f5d2c9878989ead1e3fc6aaa0` |
| q1_candidate_driver_joint_v1.py | `391ad5ecc678fa8b057e87f53a6f027a0af06648feb6099759439cdc358ea0bf` |

## Corrections closed

1. The initial importer auditor compared the exported convolution shape `[3,10240]` directly with the physical bank slice `[34,10240]`. This would reject the actual unchanged candidate snapshot (hooks v9 lines 286–293). The parent repaired `same_storage`, lines 24–31: full bank pointer, stride, offset and dtype remain equal; `candidate_state_len=34` and bank dimensions `[34,10240]` are required; the logical live export is exactly `[3,10240]`. Independent extracted-function controls pass the valid join and refuse missing/wrong physical length, raw shape, pointer and stride. SSM remains the full bank-row join.
2. The initial job copied historical `ignore_eos=False` while the new driver sent `True`. The current builder (line 102), full-auditor request validation (line 65), and driver before-control check (line 85) now agree on `ignore_eos=True`, null request seed and the unchanged prospective length budget. Wrong EOS/seed/budget contract is refused before a control write or model request.
3. The parent corrected host/container paths during review. Builder line 103 now provides `/workspace/...` source-manifest/root paths for the candidate worker and separate host paths for offline audit; full auditor lines 178–184 reads those host fields. Same-case native binding paths remain container paths in the job, while the host seal wrapper resolves the native comparison under the exact original joint-corpus directory. This matches the production repository mount; the native comparison's own job/source paths remain original host paths.

## Admission and connection checks

The builder requires the exact joint-native reduction schema, **336 observations**, complete target and categorical eligibility for all **84 cases**, the approved corpus/reduction/source/policy digests, and fixed case ordering. It predetermines aligned native **A/r0 for every same case**, verifies the accepted native terminal/job/record bindings, and binds the common source into each comparison. The prior 672-observation requirement is absent. No real successful 336-observation acceptance was supplied or created in this review.

Current runtime pins cover the joint hooks/patcher, inherited v9 hooks, joint mapper/union/source helpers, after-Z binding/graph observer and retained publication/hydration dependencies. Seal authentication validates the canonical job, caller's expected observation/job identity, exact native host directory, native terminal receipt and full candidate raw audit. The full auditor calls the accepted joint-native auditor before comparison, retains raw disagreement as an explicit result, and joins the import receipt to the actual state/cache descriptors. State/replay/publication/graph internals remain in their separately assigned reviews; this note does not duplicate or supersede them.

The request count is coherent with unchanged forced acceptance: initial forced root emits 1 token, first tree acceptance emits `L+1` (drafts plus pending z), and second root-only event emits the actual target O2 token. Total **L+3 = len(chain_tokens)+1**. Ignoring EOS prevents that actual O2 token from changing the stopping mechanism. The retained production runner at `workload-plan/inspections/cpu-lumotree-source-20260929T012000Z/result/sources/v1/worker/gpu_model_runner.py`, lines 7413–7486, performs the actual proposal, deferred proposal completion and seal anchor before return/bookkeeping; its proposal admission is governed by drafter context capacity, not a new max-output-token bypass. The v9 patch inserts `on_sealed` at that exact anchor, and the joint hook requires the second event, all three phases and both graph observations before sealing. Source review supports the intended placement; actual second-event execution still must be observed in the eventual run.

## CPU evidence and limits

**23 independent CPU controls pass.** Admission controls use the actual frozen plan/source and exact builder admission prefix, with explicitly synthetic future reduction/acceptance documents. They reject old672, incomplete collection, target or MTP ineligibility, a single bad target/MTP case, policy/source drift and case-order drift. No actual candidate job is written.

Actual driver functions, real case-binding logic and the frozen three prefix binaries exercised the complete **84 × R2** request schedule using an injected HTTP function and raw-auth boundary. Exact budgets are checked for every request. Wrong completion counts or invalid raw authentication stop after one request. Numerical disagreement remains visible (`all_agree=false`) while valid instrumentation collection can finish; the control does not claim raw numerical evidence because that boundary is explicitly stubbed. Storage-join controls cover the corrected convolution representation.

The parent separately reports six connected real CPU mapper/union/Base-snapshot tests passing after JSON serialization. I inspected that fixture but did not independently rerun it successfully: the local Python 3.9 Torch environment stops during an unchanged production-topology import using `zip(strict=True)`, before tests execute. That compatibility failure is preserved as `import-controls.log`; it is not evidence of a candidate implementation failure. No implementation was changed to accommodate the local interpreter.

The first independent control attempt reached all cases but its final source-stability check caught the concurrent two-file host/container correction. Original snapshots/logs are retained; final controls bind the corrected bytes above. No GPU, model, container, recovery, preparation or remote write occurred.

Evidence: `p0/monitor/review-response-20260927/candidate-joint-admission-review/`, including source snapshot, runnable CPU controls, final `RESULTS.json`, preserved prior attempt and compatibility log. Final launch admission must still bind these exact source bytes together with the separately reviewed raw/state implementation and outer operational wrapper, and wait for actual joint-native A/B acceptance.
