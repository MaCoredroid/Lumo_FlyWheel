# Owned case-task boundary: bounded independent source review

Verdict: PASS for the reviewed source delta and CPU integration controls. No new blocking defect found. This is not workload admission, numerical qualification, a live launcher check, or an authenticated-ingress claim. Workload scope remains the single selected task under four arms, one attempt each; this review ran no workload attempts.

## Reviewed bytes

| Source | SHA256 |
| --- | --- |
| `tools/attempt-runtime/case_task_boundary_v1.py` | `f94d983031ad7cb63ea25f8081582ff370c8dea22959b7e4a623b705606aaa17` |
| `tools/attempt-runtime/sole_executor_v3_7.py` | `4fe88d9bd4a23cca788487fdec61a9fb17318f7407e691df9b54b99a24a4a101` |
| `tools/attempt-runtime/test_case_task_boundary_v1.py` | `d5fd4688849f2bf23c309279500c4be7b57a2779a753b28bcc0b0feec74b729b` |
| `tools/attempt-runtime/dependencies_v3_3.py` | `573321913926f2b9109ab1a4d76e8bfffecf438d8f09054404aec6101262e259` |
| `tools/attempt-runtime/pin_observation_successors_v1.py` | `34062c4af4f924a5ebc426f1a9d4cf824533577701981faf95da23c6af8d542e` |
| `task-binding-v1/task_binding.py` | `0d0af3e28eab69a14c0fe7b671a355aac4194c3ddc655ff454ba7e9276565847` |

Paths above are relative to `experiments/review-response-20260927/workload-plan/`. Original sources still matched the retained snapshot after controls. The helper's scientific lineage and generated launcher are separately reviewed by qualification_redteam; this review checked the helper's exact-byte invocation and returned source bindings, not a new scientific qualification.

## Findings and closure evidence

1. **Admission precedes the one-use record.** The caller invokes `CT.validate(runtime)` at line 501 before retaining `STARTED_NO_RETRY` at line 511. The boundary checks the frozen boundary/helper hashes, exact selected task/token/scope and helper-verified input/output maps (boundary lines 51–69). It requires both ingress mechanisms disabled and refuses all ten task-binding keys in the caller's boot environment. The helper is loaded by hashing then compiling its exact bytes; changed or symlinked helper files refuse before execution.

2. **Preparation requires ownership and completes before boot.** Within `runtime.own()`, the existing launch gate and observer preparation precede `CT.prepare`; its returned environment is passed through the caller's default boot closure (caller lines 620–629). Preparation first requires mutating ownership, then revalidates the source recipe. The connected control uses the unchanged `run_attempt` function body, real boundary/helper verification and private-file creation, with all external host, Docker, probe, agent and evaluator adapters stubbed. A post-admission input change or owner refusal creates no task-private directory, never reaches boot/agent/measurement, retains a `NOT_REDUCED` result and cannot retry the same ordinal.

3. **Private files and public evidence have distinct content.** An absolute, resolved attempt path without symlink components is required. A fresh 0700 subdirectory and exclusive no-follow 0600 boot-pin file are created; the public 0600 preparation receipt contains its path, task/source/attempt/boot/freeze bindings and disabled-ingress declaration, without either secret value (boundary lines 81–114). A preexisting private destination or public-receipt symlink refuses without overwriting its target. If public-receipt creation fails, already-created private evidence remains; there is no retry or rollback claim. These random boot pins satisfy the required private boot file only and do not establish request authentication or a bearer lifecycle.

4. **Owned values reach the launcher, without changing the other arms.** The caller control verifies exactly the ten owned task keys plus the observer fixture key at the default boot adapter. A separate AST control executes the real `boot_v3_3.launcher_env` function (lines 204–218), showing that every owned task value overrides a conflicting inherited value. AR, CHAIN_MTP and SGLANG_EAGLE retain their path without owner checks or task-private mutation. Repeat complete attempts refuse before new actions.

5. **Source seals close the new boundary dependency.** All three generated successor-manifest hashes and all 19 listed member sizes/hashes match. The runtime-collectors seal includes the task boundary; its literal helper hash and read-only `helper.verify()` bind the recipe transitively. The generator was inspected, not executed. The caller diff against `before-case-task-caller-20260929T003932Z` contains only the import, pre-reservation validation and owned pre-boot preparation additions. Unchanged accepted caller/collector/closure behavior was not re-audited.

## Controls and limits

Six supplied tests pass. Fourteen independent controls pass: seven connected caller cases (Lumo positive, foreign task, owner refusal, post-admission source change, and all three other arms), five targeted module refusals (symlinked ancestor, public receipt symlink, helper symlink, changed helper, missing Lumo spec), successor-manifest closure, and actual boot environment precedence. Positive caller cases also assert same-ordinal refusal after completion. The initial 13-control run and subsequent 14-control run are both retained. No Docker, network, GPU, model, evaluator, agent or live launcher calls occurred.

Evidence is retained under `p0/monitor/review-response-20260927/case-task-boundary-independent-20260929T004344Z/`: source snapshot, caller diff, supplied test output, independent control source/results/logs, successor manifests/member copies and hashes. `MANIFEST.json` binds that review payload. Reproduce with configured Python `-B controls.py`; the script requires the reviewed repository/helper bytes and uses temporary local files only.

The designated final launcher is `launchers/lumotree_workload_owned_v2_2.sh`, identity checked here as `eb8143836beddc5446b132381ad39c5c4625c89b873aa4b4d7b5566f0b5a6d5d`; v2.1 is the historical predecessor. Final parent freeze must select the reviewed v2.2 launcher and task-boundary specification, alongside the separately reviewed lineage and existing runtime prerequisites. This is the remaining admission binding, not a new source defect or authority to open WP. Live mounted files, emitted receipts and actual runtime behavior remain unobserved in this review.
