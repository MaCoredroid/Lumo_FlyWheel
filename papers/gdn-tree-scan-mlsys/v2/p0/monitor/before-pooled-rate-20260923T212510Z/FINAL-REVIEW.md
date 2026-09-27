# Agent-workload revision — 23 September 2026

Comparison-history correction: SGLang EAGLE, DSpark/DFlash-family variants, and native MTP controls were tested in the repository. The inventory in `notes/comparator-git-history-2026-09-23.md` distinguishes real SWE tasks, synthetic measurements, capture/replay, and boot checks. Existing tests are not automatically a matched comparison of the current deployment. No superseded performance numbers were restored to the manuscript.

Abstract update: the headline now concerns sustained tool-using agent execution and comparative decoder speed. The ten-task trace audit found no detected degeneration or malformed tool-call arguments, with detector scope and blind spots stated in the results. Outcome counts remain in the evaluation; no unchanged-quality or comparative-speed claim was added. See `notes/agent-behavior-abstract-evidence-2026-09-23.md`.

The current 12-page manuscript reports performance only on named coding-agent tasks. E1/E8 internal-document rates, the reuse percentage gain, and superseded proxy/component headlines are absent from the loaded manuscript and PDF. Immutable original evidence and earlier drafts remain audit history.

The application case is Cqc10: the August 24, 2026 completed single-configuration segment on ten selected SWE-bench Verified Astropy tasks, served by the Qwen3.8-27B NVFP4 Hydra27 route on GB10. It reports all ten evaluator outcomes (six resolved/four test failures), exact task runtimes rounded to minutes, and a request-weighted inverse mean TPOT of 28.20 tokens/s. The 265 observations reconcile with 256 normal requests plus 9 compactions. These measurements are dated descriptive observations; the selected remainder of an interrupted sixteen-task campaign is not a full-benchmark score or a matched native/tree study.

Methods now distinguish the agent route (Hydra27, patched FA2 split-K4, graphs/APC, stochastic sampling) from the separate FP8 Cat10 eager/cache-off numerical qualification. Supporting arithmetic, state-publication and head-equivalence checks remain; neither compact candidate is promoted. They do not prove full-model equivalence or broader task-quality preservation.

Both independent final reviews PASS:

- `p0/monitor/paper-agent-workload-redteam.md`: every task outcome/time, selection lineage, scope, and exclusion of snippet performance; artifact-copy wording corrected and rechecked.
- `p0/monitor/paper-agent-metrics-redteam.md`: model/route/settings, metric support, Hydra geometry, and split-K explanation. The subsequent abstract behavior claim is separately verified by `notes/agent-behavior-abstract-evidence-2026-09-23.md`.
- `p0/monitor/2026-09-23-agent-abstract-build.json`: latest abstract/body update and PDF verification.
- `p0/monitor/2026-09-23-agent-workload-build.json`: prior workload revision: clean 12-page build, 23 citation keys, all-page visual review and final affected-page recheck, exact source/PDF identities.

The compact current evidence has 139 hash-verified raw/projection files, ten original task/evaluator records with metric brackets, explicitly sanitized configuration projections, a reproducible CPU reducer, and source-bound earlier-lineage audit extracts. The parent independently reproduced the reducer output exactly. Raw originals of earlier interrupted segments remain at their recorded source paths; complete external model/runtime/evaluator dependencies are not bundled.

**Remaining scientific gap:** a matched current native-versus-tree full-task campaign is needed for an application-speedup claim. `review-experiments.md` gives the proposed qualification pilot and paired confirmation design. No new inference was launched for this revision. The descriptive paper revision is complete; the stronger comparative experiment is proposed, not completed. No public release, arXiv submission, external message or push was performed.

Current source/PDF/private-package identities are in `artifacts/FINAL-DELIVERY.json`. Earlier delivery receipts are preserved in dated snapshots.
