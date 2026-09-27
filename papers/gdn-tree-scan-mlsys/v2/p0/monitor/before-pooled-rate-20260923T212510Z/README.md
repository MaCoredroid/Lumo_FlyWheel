# GDN Tree-Scan v2: verifier design for coding-agent workloads

Comparison-history correction: SGLang EAGLE, DSpark/DFlash-family variants, and native MTP controls were tested in the repository. The inventory in `notes/comparator-git-history-2026-09-23.md` distinguishes real SWE tasks, synthetic measurements, capture/replay, and boot checks. Existing tests are not automatically a matched comparison of the current deployment. No superseded performance numbers were restored to the manuscript.

Abstract update: the headline now concerns sustained tool-using agent execution and comparative decoder speed. The ten-task trace audit found no detected degeneration or malformed tool-call arguments, with detector scope and blind spots stated in the results. Outcome counts remain in the evaluation; no unchanged-quality or comparative-speed claim was added. See `notes/agent-behavior-abstract-evidence-2026-09-23.md`.

Current revision: 23 September 2026. The manuscript explains tree verification, GPU scan/replay, selected-state publication, and route-specific optimization mechanisms. **Only named coding-agent tasks supply performance numbers.** Local-document E1/E8 rates and reuse percentage gains have been removed from the manuscript; their immutable raw data and dated earlier drafts remain audit-only.

The application case is the latest complete single-configuration segment recovered from the Qwen3.8-27B NVFP4 campaign: Cqc10, recorded 24 August 2026 on GB10. All ten named SWE-bench Verified Astropy tasks have evaluations: six resolved and four failed tests. Agent times total 181.15 minutes, ranging 1.67–41.24 minutes. The inverse of mean request TPOT is 28.20 tokens/s across 265 observations; it is not end-to-end task throughput or a native-versus-tree speedup. The cohort is the remaining ten tasks of a resumed sixteen-task development campaign, not a random or full-benchmark sample. Earlier failures/retries and configuration changes are retained in the audit.

The NVFP4 Hydra27 deployment uses patched FA2 split-K4, graphs, prefix caching, and stochastic sampling. It is distinct from the FP8 Cat10 eager/cache-off numerical qualification. Arithmetic, publication, and same-input head checks remain mechanism evidence, not agent-performance measurements. Neither local compact candidate passes all frozen criteria. Full-model equivalence and broad task-quality preservation are unestablished.

A **matched current native/tree full-task campaign remains necessary** for an application-speedup claim. The concrete proposal is [review-experiments.md](review-experiments.md); it has not been launched. This revision reuses and audits existing task records without model inference. No arXiv submission, external message, or push is implied.

- `main.pdf`, `main.tex`, `abstract.tex`, `ref.bib`, `figures/`: current paper.
- `results/agent-workload/`: task case, original safe records, source-bound configuration projections, reducers and manifests.
- `notes/swe-workload-comparison-review-2026-09-23.md`: evaluator/run-lineage audit, including failed and missing outcomes.
- `notes/nvfp4-workload-evidence-review-2026-09-23.md`: model/route/metric audit.
- `notes/e1-workload-provenance-2026-09-23.json`: proof that excluded E1/E8 prompts were internal documents.
- `notes/claim-evidence-ledger.md`: current allowed claims and limits.
- `FINAL-REVIEW.md`, `p0/monitor/`: independent reviews and build checks.
- `artifacts/FINAL-DELIVERY.json`: current source/PDF/archive identity; dated packages preserve prior revisions.

Build from this directory with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. The current task evidence can be checked without inference using `python3 results/agent-workload/cqc10_reduce.py`. `scripts/audit_evidence.py` is the separate historical-accounting checker and does not validate agent performance.
