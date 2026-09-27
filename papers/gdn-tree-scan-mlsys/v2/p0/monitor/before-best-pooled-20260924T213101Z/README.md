# GDN Tree-Scan v2: verifier design for coding-agent workloads

Comparison-history correction: SGLang EAGLE, DSpark/DFlash-family variants, and native MTP controls were tested in the repository. The inventory in `notes/comparator-git-history-2026-09-23.md` distinguishes real SWE tasks, synthetic measurements, capture/replay, and boot checks. Existing tests are not automatically a matched comparison of the current deployment. No superseded performance numbers were restored to the manuscript.

Abstract update: the headline reports sustained tool-using agent execution and a descriptive comparison using one pooled decode estimator: **25.63 tokens/s for GDN Tree-Scan versus 26.89 for SGLang EAGLE**, on ten and two SWE-bench Verified Astropy tasks, respectively. The ten-task tree trace audit found no detected degeneration or malformed tool-call arguments, with detector scope and blind spots stated in the results. This does not claim an isolated mechanism effect or a task-completion speedup.

Current revision: 23 September 2026. The manuscript explains tree verification, GPU scan/replay, selected-state publication, and route-specific optimization mechanisms. **Only named coding-agent tasks supply performance numbers.** Local-document E1/E8 rates and reuse percentage gains have been removed from the manuscript; their immutable raw data and dated earlier drafts remain audit-only.

The tree application case is the latest complete single-configuration segment recovered from the Qwen3.8-27B NVFP4 campaign: Cqc10, recorded 24 August 2026 on GB10. All ten named tasks have evaluations: six resolved and four failed tests. Agent times total 181.15 minutes, ranging 1.67–41.24 minutes. The cohort is the remaining ten tasks of a resumed sixteen-task development campaign, not a random or full-benchmark sample. The SGLang comparator covers two different completed Astropy tasks, one resolved and one failed. Earlier failures/retries and configuration changes are retained in the audit.

Every current decode-rate claim uses `(total output tokens − completed requests) / (summed E2E request latency − summed TTFT)`. Counts and durations are pooled before division: tree 248,077 output intervals / 9,677.7972 seconds; SGLang 47,809 / 1,777.6857 seconds. This covers 265 and 45 completed engine requests, including internal agent traffic, and excludes between-request tools. The tree rate is 4.69% lower on these recorded subsets. Serving configurations differ; accumulated request time is not task or campaign wall time. The common reducer and its 48 hash-bound inputs are included in the private package. The earlier calculation-only notes and manifests remain dated records; this revision replaces the manuscript headline with their verified pooled decode result.

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

Build from this directory with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Reproduce both current decode rates without inference using `python3 results/agent-workload/shared_rate_reduce.py`; its output must match `results/agent-workload/shared-rate-audit.json`. The earlier `cqc10_reduce.py` retains a superseded request-weighted estimator for audit. `scripts/audit_evidence.py` is the separate historical-accounting checker and does not validate agent performance.
