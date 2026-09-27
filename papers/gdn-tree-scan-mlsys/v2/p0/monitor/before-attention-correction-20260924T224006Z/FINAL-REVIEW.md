# Patch-producing performance subset — 24 September 2026

The earlier SGLang run is removed from the abstract, rate table and numerical performance discussion. The current headline is explicitly conditional: among recorded runs producing nonempty patches on both shared SWE-bench Verified tasks, the best tree run achieves 29.09 pooled tokens/s versus SGLang EAGLE's 26.89 (+8.18%).

The criterion was chosen retrospectively after examining archived outcomes. Each task in an eligible run-pair must have a completed agent attempt (exit0, ended_at, no timeout or task-budget cap) and a positive patch_bytes count. Failed tests remain eligible. The same rule is applied to every recovered pair and excludes the whole pair, not just a failed task or selected requests.

| Run | Patch bytes12907 /13033 | Eligible |
| --- | --- | --- |
| Sr12 | 504 /1017 | Yes |
| Cqc16 | 504 /1092 | Yes |
| Cqc15 | 504 /1450 | Yes |
| SGLang later | 504 /912 | Yes |
| SGLang earlier | 504 /0 | No |

The excluded attempt terminated but produced an empty patch. Its final response was thinking-only at the configured output cap; the test harness was not invoked. The evaluation states the exclusion and preserves the failed outcome in the artifact. This changes the performance population, not the recorded arithmetic. No raw file, unfiltered reducer or previous receipt was edited.

All eligible pairs retain one resolved and one failed-test task. Their rates are 29.09, 28.28, 27.27 and 26.89. The separate ten-task Cqc10 result remains 25.63; the complete four-task Sr12 context remains 26.21. Both context cohorts also meet the nonempty-patch rule. The result is not an all-attempt benchmark speed or quality-equivalence estimate.

The complete decoder methods remain the intended comparison. Tree topology, draft length, candidate budget and optimizations may differ; no common tuning sweep is established. All41 headline tree and45 eligible SGLang requests are at most20000tokens, below either configured cap. Numerical qualification remains separate from agent performance.

Independent review verified all10 original shared-task metadata records and outcome semantics. A107-file isolated replay reproduced the overlay exactly. Disposable controls confirmed symmetric exclusion when a tree patch was emptied or its agent failed, and refusal to estimate when no SGLang pair remains eligible. Failed-test patches remain included.

Current records:

- `results/agent-workload/patch-producing-rate-audit.json`: complete eligibility ledger and conditional rates,94 input bindings and five safe projections.
- `notes/patch-producing-comparison-redteam-2026-09-24.md`: source/metadata/selection/replay review; PASS.
- `p0/monitor/2026-09-24-patch-producing-build.json`: clean13-page build, citation preservation, current source/PDF identities and rendered-page inspection.
- `artifacts/FINAL-DELIVERY.json`: final private package and extracted replay/rebuild.

No inference, publication or push occurred. Previous unfiltered data and competitive-comparison snapshots remain intact. Future confirmation should predeclare eligibility and report all attempted outcomes alongside conditional performance.
