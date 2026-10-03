# Candidate joint parent planner: bounded independent review

**PASS at source SHA256 `56c9e0394cae43e48477cff33bebc531aacef233d9bdf16ce6ce8698b4497cfe`.** This closes the pure preparation helper's dependency resolution, source-authentication and namespace boundaries. It does not create a real approval, admit a native result, authorize a machine operation, launch candidate A, or qualify scientific data.

Reviewed `experiments/review-response-20260927/tools/q1_candidate_joint_parent_plan_v1.py`. The accepted job/gate/wrapper and scientific predicates were reused at their seams; no unchanged broad suite was rerun. All tests operated on temporary CPU-only copies. No SSH, GPU/runtime query, container, model request, source edit, or real gate write occurred.

## Findings and repairs

Initial SHA `beb7100ec6b0dd0a76f2e28ffd7ce940d18d2ddad2a5e5cc27ca98eb337ada8b` passed 22/24 controls. Two connected output-namespace cases returned a gate under the explicitly injected positive Admission boundary:

- A dangling leaf symlink was invisible to `Path.exists()`. The actual wrapper would reject its `mkdir`, but preparation falsely called the namespace fresh.
- A symlinked run-root ancestor pointing outside the owned repository made both sides of the resolved-path equality escape together. The wrapper could create a child through that ancestor.

Final lines 94–98 require resolved output containment within the resolved repository, exact expected namespace, and `os.path.lexists` absence before any dependency/admission work. Both controls now refuse before Admission import, and the positive remains accepted. The only other source change renames the predecessor check accurately as a source-hash inventory. Original source, initial 22/23 results, extended 22/24 results and final 24/24 results are preserved in the review directory.

## Dependency and authority checks

The parser authenticates the exact wrapper SHA, requires exactly 112 unique dependency assignments, substitutes only known variables, rejects unsupported expressions and paths outside the owned repo, and binds the wrapper/job anchors (lines 45–84). An independent shell assignment-expansion control agrees on all 112 paths. This control evaluated only the declaration prefix in local zsh, rebasing `REPO`; it did not execute the wrapper body or any declared helper function. It is an independent path-resolution check, not a Linux launch test.

All **107 static inputs** authenticate before importing campaign Admission: 62 from the exact joint source freeze, 13 explicit successors, and 32 predecessor entries. The five future run-specific inputs are corpus, native reduction, native-result acceptance, native terminal receipt and prepared job. Their identities/eligibility are delegated to the already reviewed real Admission rebuild; the helper does not substitute an abbreviated scientific check.

The pinned predecessor file `GATE-Q1-CANDIDATE-STAGE1.json` SHA `a29db844567f45053f88af2a3430b9e8e181c1746e6ded54c22b263322ba53bd` is historically closed (`approved=false`). It is used only as a frozen source-hash inventory. Independently, all 32 fallback hashes match the actual accepted 092706 root gate `GATE-Q1-CANDIDATE-20260929T092706Z.json`, SHA `c4a31e0e68c0fbe61ff32b3f685d6583e9921d8569ac573144049409016b7fc5`. No consumed launch authority is reused.

The 13 explicit successor pins have existing source/result evidence:

| Successor keys | Prior evidence under `notes/review-response-20260927/` |
|---|---|
| `swap_observer` | `q1-candidate-v32-unfinished-handoff-review.md:16,33–35`; retained in v2.5 closure |
| `diagnostic_swap_policy` | `fullmodel-memory-readiness-source-audit.md:18`; v2.5 closure |
| `validation_interpreter_record`, `host_validation_binding`, `host_validation_verifier` | `q1-candidate-v25-codex-launch-repair-review.md:15–16,25–29`; provisioning-record pin also matches accepted 092706 root gate |
| `job_builder_v2` | `candidate-cycle0-v2-case-isolation-review.md:8` |
| `model_config` | `native-corpus-a-prelaunch-config-retry-review.md:10` |
| `accepted_base_diag_launcher` | `q1-candidate-joint-outer-glue-review.md:28`; `q1-candidate-cycle0-v9-propagation-review.md:47` |
| `memory_runner` | `memory-global-v28-source-review.md:28`; `memory-global-v28-actual-result-review.md:21` |
| `launch_env`, `host_env`, `launch_env_renderer` | `q1-candidate-v25-codex-launch-repair-review.md:11–13` |
| `stage1_result_acceptance` | `root-stage1-result-review-092706.md`, joined to parent result SHA `cc759978ad89587a0c3232852943e2ae1623b5e6f54179f79b0034b7af853393` |

`AUTHORITY-TRACE.json` stores exact note paths/hashes, all successor hashes, and the 32-entry accepted-root cross-check. The old provisioning source note had open concerns; the v2.5 closure and accepted root evidence, rather than that earlier note alone, close the final host binding.

## Focused controls and scope

All 24 controls pass: exact 112-path expansion and 107-static-input census; a stubbed positive with 84 actual frozen case IDs, 168 requests, process A/R2, one-use gate, 113 reviewed hash keys including image, and false full-Q1/workload flags; scientific Admission refusal propagated; changed frozen module, successor, predecessor-dependent source, freeze, predecessor inventory and wrapper; missing dynamic/static file; extra/missing/retargeted dependency entries; unsafe run/process, naive/non-UTC time, foreign/existing output, escaped dynamic input, dangling output leaf and escaped output ancestor. Authentication failures reach neither Admission import nor its injected callback.

The positive substitutes only future scientific Admission, with clearly synthetic job/reduction/acceptance/receipt files. It therefore proves construction and call ordering, not full 336-observation eligibility. Real execution must still pass the unmodified `q1_candidate_joint_gate_v1.validate`, including the actual native A/B reduction, parent acceptance, selected native A/r0 inputs and exact prepared job. Actual readiness, complete Git backup, fresh recovery authority, live exclusivity and atomic wrapper namespace claim remain separate. The helper contains no CLI or file/network/runtime side effects; returning an `approved=true` dictionary is parent preparation, not a persisted or consumed gate.

Artifacts: `p0/monitor/review-response-20260927/candidate-joint-parent-plan-review/`. Reproduce the CPU controls with `PLAN_REVIEW_SOURCE=source.final.py PLAN_REVIEW_OUTPUT=RESULTS.replay.json CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B controls.py` from that directory (tested local Python 3.9.7; only standard-library modules and local zsh used). The review captures this exact repaired source; no scientific criterion or reviewed runtime source was altered.
