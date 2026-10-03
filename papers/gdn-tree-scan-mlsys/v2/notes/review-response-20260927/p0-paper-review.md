# Independent review of P0 manuscript integration

Reviewed September 27, 2026. Scope: P0 report, reducer, evidence and supplementary ledger; current `main.tex`, `abstract.tex`, and the workload/configuration/outcome/telemetry fragments. This review made no GPU calls, remote calls, runtime changes, or manuscript edits. Only this report was written.

**Disposition: approve the scoped P0 integration. No substantive blocker found.** This approval concerns recovery and presentation of the existing development evidence. It does not qualify the pending Q1 continuation tests or establish a new performance result.

## Independent checks completed

- Ran all **11 CPU negative-control tests** with bytecode writing disabled: all passed. These include the extracted engine statistics class, survival-histogram conservation, physical-padding denominator, counter/reset/presence checks, model/series checks, and pooling rather than averaging.
- Verified all **118 source-input hashes** in `p0-evidence.json`, the reducer's own hash, and the three recorded historical patcher git-object hashes. All matched. The extracted base-image identity is `ffa30d66...`, with configured digest `3dbe092e...`; it matches the recorded Cqc10 container identity. The manuscript/artifact retain the limitation that this does not attest every historically installed scheduler byte after runtime patches.
- Re-read and reduced the raw metric/metadata/evaluation records for all **22 unique complete-bracket attempts**, without invoking the file-writing reducer entry point. Every task's output-token count, completed-request count, E2E sum, TTFT sum, agent time, and pooled rate matches the P0 evidence. Raw acceptance and phase reductions also reproduce the saved per-task records.
- Independently pooled each cohort's token/request/latency totals. Rates reproduce as: Sr12 pair **29.093443**, Cqc16 pair **28.283047**, Cqc15 pair **27.268008**, earlier SGLang **30.700495**, later SGLang **26.893956**, Cqc10 **25.633622**, and the overlapping Sr12 four-task cohort **26.205365** tokens/s. The displayed rounding and paired agent times agree.
- Confirmed the supplement retains **six additional comparison-lineage attempts** and **seven provenance-only earlier records**, without fabricating rates or terminal outcomes. Sr12's pair is a subset of its four-task cohort, not an extra pair of attempts. The paper explicitly warns against summing overlapping rows or treating the interrupted development campaign as a completed sixteen-task benchmark.

## Claims and population boundaries

The pooled estimator uses `(N - R) / (sum E2E - sum TTFT)` over the same completed-request population and idle task boundaries. The text distinguishes accumulated request time from machine wall time and from agent duration. It neither substitutes acceptance counts for API output tokens nor divides global counts by the admitted-step wall subset. All recorded internal compaction requests remain in the applicable task population.

The faster earlier SGLang pair, **30.70 tokens/s**, remains visible in the all-attempt diagnostic table and paired outcome table. Its empty-patch attempt is excluded only from the explicitly retrospective, symmetric patch-producing comparison. Failed-test patches remain eligible. The abstract/conclusion report the three tree revisions as exploratory observations and do not present the maximum-selected 8.18% difference as a confirmatory headline. The source and precision identity gaps are explicit, so a common display name is not represented as checkpoint-byte equivalence.

Acceptance is correctly presented as the scheduler's output-side observed length minus one correction/bonus token. The extracted observation class counts survival positions, permitting the reported length histogram; it does not identify branch/node selections or the commit path. Recomputed support is 0–11, with all 31 raw positions retained. The 31-slot denominator includes four padded draft slots and is kept out of the manuscript as a logical acceptance fraction. The per-event means round correctly to **4.61, 4.49, 4.41, and 4.05**. SGLang's saved gauges are not improperly differenced or averaged into comparable task-total acceptance.

The phase table uses each timer's own duration and span count. Forward, drafter, and sampler/commit-dispatch span counts equal the speculative-event counts in the reported tree cohorts; the admitted-step wall population does not. The source brackets the rejection-sampler dispatch, so the roughly 20 ms column is correctly described as combined acceptance/path/commit work, not isolated recurrent replay. Pending-event acknowledgements are zero at the task boundaries. The paper appropriately leaves instrumentation overhead, isolated stages, per-event distributions, and causal attribution unresolved.

## Approval limits

P0 recovers usable diagnostics and makes selection and identity limitations visible. It does not establish fixed-build repeatability, equivalent checkpoints across stacks, preserved task quality, native-versus-tree task advantage, or a current matched LumoTree/author-kernel comparison. The manuscript preserves these limits and reports no result from the worker currently implementing Q1. PDF layout and later manuscript changes remain outside this review.
