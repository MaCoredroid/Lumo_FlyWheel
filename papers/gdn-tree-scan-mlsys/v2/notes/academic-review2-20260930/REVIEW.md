# Second academic editorial review — 30 September 2026

The pasted review was checked against the 13-page source and PDF. This pass changes presentation and bibliography, not experiment data or numerical criteria. It does not execute inference or add repetitions. The latest completed ten-task NVFP4 workload is a historical deployment; it is not a new September qualification campaign result. Era-specific raw evidence and all prior receipts remain intact.

## Disposition of the requested edits

- The abstract and workload section lead with the ten-task deployment: 25.63 pooled tokens/s, six resolved tasks, four test-failing patches. No older fastest revision is relabeled as the final implementation, and no repetitions are implied.
- Builds A–C disappear from the paper. Their two-task rate range appears in one contextual sentence; the original rows and source mapping remain in the preserved pre-edit source and original raw results. The acceptance/phase table keeps the ten-task measurements (4.05 drafts, 114.89/54.76/20.15 ms).
- Both SGLang rates (26.89 and 30.70 tokens/s) are disclosed together. A single footnote explains retrospective selection, shared task outcomes, and the excluded empty-patch/capped run. The abstract makes no cross-stack speed comparison.
- Interrupted-campaign chronology, raw latency/token sums, attempt-group tables, and the second two-task attempt table are removed from the manuscript. No raw evidence is deleted. One ten-task appendix table remains, with agent times rounded to minutes.
- The serving table retains precision/conversion, proposal geometry, attention, context/scheduling, response caps, sampling, and the agent harness. Network, worker architecture, engine seed, compaction caps, timeout, and proxy trivia remain in artifact configuration records.
- Numerical and performance scope are consolidated in one Limitations section. It retains the adverse native reference, absent full-model candidate/lifecycle qualification, differing recurrent comparators, unmatched workloads, incomplete development outcomes, and missing ablations/memory measurements.
- The recurrent numerical table visibly separates the native GPU reference from sequential software references. It reports unchanged counts outside each reference's paired bounds without declaring a winner. Weaver author-default versus aligned-local preparation and TreeWY normalization/arithmetic remain explicit. Accepted replay and final deferred commitment stay included.
- The reconstruction-numerics label now points to Sequential arithmetic and accepted-path replay, Section IV-B. Introduction navigation is rewritten.
- Repository/blog citations are removed from design prose. A paper-specific artifact is cited once in the reproducibility appendix. The two unused author-blog BibTeX entries are removed; their previous contents remain in the snapshot.
- Twelve publication records are changed from arXiv to verified venue versions. Citation keys remain stable. Publisher title differences for SpecInfer and Sequoia are corrected; capitalization and access-date formatting are repaired. Weaver's version chronology is removed. FastTree and FlashInfer are identified by their complete venue metadata, with full publisher links in VERIFIED-CITATIONS.json.

## Evidence preservation

The pre-edit PDF, source dependency closure, bibliography, configuration and label map are preserved with hashes in `artifacts/before-academic-review2-20260930/`. Original generated numerical-result files and their JSON remain unchanged. The manuscript uses `sections/recurrent-validation.tex` for edited explanatory prose, preserving its numerical table exactly. Other unchanged method tables, figures, and algorithms remain in place.

The rate estimator still pools matching output intervals and request-duration totals and excludes time to first output; it is not wall-clock application throughput. Output-side acceptance is still distinct from the committed path. Logical exports remain distinct from allocated scratch.

## Verification

Final build, render inspection, citation/label resolution, isolated source-bundle build, and scoped recoverable Git backup are recorded in the delivery receipt. No experiment gate is opened and no broader empirical claim is added. The scheduled monitor remains deleted.
