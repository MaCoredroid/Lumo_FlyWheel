# Current qualification boundary: bounded paper-readiness review

2026-09-30. **The completed evidence supports a narrowed implementation/component paper with exploratory historical workload observations. It does not support a qualified current full-model candidate, cache lifecycle, or current same-stack advantage.** The parent's latest abstract, introduction, lifecycle paragraph, runtime paragraph and conclusion now expose this boundary. **PASS for the current claim boundary.** The two wording findings below are closed on the final checked bytes; downstream closure language remains conditional on a recorded parent disposition.

This is a read-only claim review. No paper/gate/runtime edits, tests, inference, external contact or new evidence collection were performed. No ongoing-run partial counts are used. Native reference repeatability failure prevents the declared comparison; it is not an observed candidate failure or evidence of algorithmic incorrectness.

## Wording findings closed on latest bytes

1. **`abstract.tex:2`: declared phase scope and paired error bounds.** The original new draft said “complete draft-token selection” and “numerical non-regression.” The parent corrected these to “the ordered top-three MTP choices at all declared phases” and “recurrent agreement within the declared paired error bounds.” Both corrections are accurate: the68/84 count comes from completed joint-reference evidence, later proposal levels remain outside the frozen phase set, and the error rule explicitly allows1.10 times native error plus declared slack. No partial ongoing-run number appears.

2. **`main.tex:323`: archived engagement versus current qualification.** The paragraph now begins “Archived task-run receipts record engagement” and explicitly separates that evidence from current full-model qualification. It no longer ambiguously attributes all those mechanisms to every native/reference run. The exact stored settings remain valid descriptive evidence without promoting them to lifecycle correctness or a distribution proof.

No further factual correction is needed in those passages. Existing mechanism descriptions can stay; changing them all into unimplemented proposals would also misrepresent the source evidence.

## Conditional final closure if the parent ends downstream stages

The approved `workload-case-study-v1/SCOPE.json` fixes exactly one predetermined sklearn task, four methods in AR/CHAIN_MTP/SGLANG_EAGLE/LUMOTREE order and one attempt per method. Its WP prerequisites include qualified current routes. The current status and parent instruction agree that zero of four workload attempts have started. A source review does not revoke that scope or mark the attempts completed.

If the parent records supported narrowing after the terminal native result is audited, **`main.tex:357–359` must stop promising forthcoming results.** Rename the section to `Unexecuted Workload Case Study` and replace the paragraph with:

> The approved case study specified one SWE-bench Verified task, \texttt{scikit-learn\_\_scikit-learn-9288}, with one attempt each under autoregressive decoding, chain MTP, SGLang EAGLE, and \sys{}, in that order. Execution required qualified current routes. That prerequisite was not met, so none of the four attempts was started; this paper reports no task outcome, agent time, pooled decode rate, or pass for that planned comparison. The fixed plan and the decision not to proceed are retained in the artifact.

Use this only after the actual disposition is recorded. Until then, retain the planned status but explicitly add “No attempts have started (0/4), and the qualification prerequisite remains unmet.” Do not convert an unattempted workload into a failed agent task, a benchmark failure, or a successful experiment. Do not replace the four attempts with a favorable subset.

`results/review-response-20260927/m1-qualification.tex:26` currently says full-model continuation/logits are “separate pending checks.” On final scope closure, change that ending to “remain outside this component result and are not qualified here.” Likewise, `component-qualification.tex:26` can end “These component results do not establish complete convolution/KV continuation or next-forward model agreement.” These are status clarifications only; component counts and comparator findings stay unchanged.

## Claims already appropriately bounded; retain them

- **Introduction `main.tex:54`:** distinguishes implemented mechanisms and exploratory deployments from qualified end-to-end continuation/current same-stack performance. The architectural statements and figures describe the implementation/required state contract; no blanket rewrite of those mechanisms is needed.
- **Lifecycle `main.tex:259`:** native state representation supports an intended interface but explicitly does not qualify cold/hit behavior, row reuse or stale-state refusal. A configuration with prefix caching/graphs enabled (`review-configuration.tex:1`) establishes settings, not lifecycle correctness.
- **Completed native evidence `main.tex:300–311` and included files:** natural-prefill adverse results, target-only common-O0 results, and the completed joint336-observation result remain distinct. The joint result is68/84, with16 long-prefix categorical failures, and explicitly denies candidate admission on the remaining subset. The ongoing retry must not replace this with partial counts or be described as completed before its terminal audit.
- **M1 `m1-qualification.tex:20–26`, `main.tex:333`:** component-only, specified comparators, finite/repeatable tensors and three failing comparator envelopes remain reported. No four-policy timing or full-system ranking is claimed. Keep the failures; do not promote Lumo's component pass to a current hybrid-model pass.
- **Historical rates `case-study.tex:49–80`, `main.tex:342`:** the three revisions, retrospective nonempty-patch selection, maximum-selection caveat, excluded faster empty-patch comparator, separate ten-task deployment and absent matched native arm are disclosed. The range is not a fresh current-build measurement, statistical replication, best-method envelope, or confirmed same-stack advantage. No new experiment is needed merely to retain these already-scoped descriptive observations.
- **Conclusion `main.tex:344` and probability discussion `main.tex:353–355`:** now deny full-model/request-lifecycle qualification, general superiority, task-quality preservation and distribution preservation. The abstract's theoretical requirement to preserve coherent continuation is normative; it should not be changed into a claim that the implementation has proved it.

The final paper may honestly close with the design, bounded component results, complete adverse native reference evidence and clearly retrospective task observations. Further Q1, lifecycle, workload or timing experiments are not required for that narrowed claim set. Any later stronger claim would require its own qualifying evidence; this review neither starts that work nor grants a gate.

## Exact reviewed bytes

Line references above apply to this snapshot (including the parent's new front/back-matter limits).

| File | SHA256 |
|---|---|
| `main.tex` | `348be4640890514fe557dd3aca60b9010676380341dadf7ef4e8ddd08a7ac176` |
| `abstract.tex` | `700b16c6cb1085bbd2ba1203d2b721634151079b6a6b0b306b716ebccfd3c7bc` |
| `results/agent-workload/case-study.tex` | `798470a439c2bebad6956a779803d459a2d8572217d446b1f794b855296e87c3` |
| `results/agent-workload/review-configuration.tex` | `aba40eb0e68052d81a6f69744af16267d3a3cb5f6c0e65c4419e3447457f895f` |
| `results/review-response-20260927/component-qualification.tex` | `dbdb81ffb279923c2cc7ace1cbf5bc6cb11c3921338457e8dbd85c350979c57d` |
| `results/review-response-20260927/m1-qualification.tex` | `a47d54a8e55a2cda867c9494a59c74adf0e44399991356d7c9241571a21570df` |
| `results/review-response-20260927/native-joint-AB-v1.tex` | `a2b7541686d2df0e41392f6b9191c3e145a78d60e3a527bd017a423fcbb1e461` |
| `experiments/review-response-20260927/workload-case-study-v1/SCOPE.json` | `f9a9d7cd4a0be3da43a9f6a92f6125a199af96515707863aafee4acb9e3ac822` |
| `p0/monitor/review-response-20260927/Q1-JOINT-MTP-CATEGORICAL-CONTRACT-v1.json` | `4dad3bf5918c31d734214d8d2579ca1bb5b75500b575dbf2c0718287a287c593` |
