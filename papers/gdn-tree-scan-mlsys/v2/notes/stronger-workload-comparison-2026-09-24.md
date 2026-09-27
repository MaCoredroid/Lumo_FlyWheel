# Stronger B1/B4 rate candidates — 24 September 2026

Read-only evidence recovery and calculation; no new inference and no manuscript/PDF change. The existing 25.63 tree versus 26.89 SGLang comparison remains a correct calculation on different task cohorts, but is not the only available workload comparison.

## Same-task B1 comparison recovered

Use exactly the two SWE-bench Verified Astropy IDs with completed SGLang records: `astropy__astropy-12907` and `astropy__astropy-13033`. This selection follows comparator availability, not a search for the fastest tree tasks. Both tasks have completed metadata and evaluator records in each listed run. The estimator is `(sum output tokens - completed requests)/(sum E2E request latency - sum TTFT)` throughout.

| Run | Date of shared tasks | Requests | Output tokens | Pooled decode tokens/s | Versus SGLang |
| --- | --- | ---: | ---: | ---: | ---: |
| SGLang EAGLE run1/r1 | Archived comparator | 45 | 47854 | 26.8939560242 | Reference |
| Tree Sr12 split-K | Aug 19 | 41 | 25027 | 29.0934427002 | +8.18% |
| Tree Cqc16 | Aug 19 | 35 | 26239 | 28.2830468688 | +5.17% |
| Tree Cqc15 | Aug 23 | 53 | 70424 | 27.2680081503 | +1.39% |

Every row has the same two task outcomes: 12907 resolved, 13033 failed. The six selected tree attempts used tools and produced nonempty patches; the recorded heuristic detects no degeneration or malformed tool-call arguments. All selected tree engine requests finished with stop, with zero length/abort/error completions. The parent independently reconstructed all three pooled rates from the extracted original metric lines and checked matching counter populations, idle boundaries and completed task identities.

The three tree engagement receipts identify B1, 32 physical rows, full-vocabulary Hydra, GQA-pair split-K4 on all 16 full-attention layers, and no fallback. They share the loaded FA2 extension identity, but their Python patch hashes differ. These are separate recorded runs, not a frozen repeated experiment. Requested sampling and network policy match SGLang; tree output caps are 32768 versus SGLang's 24000. No tree cap hit was recorded on these two tasks, but that does not prove the different cap could not affect trajectories. The latest shared-task run is Cqc15, not Sr12. Its route is still implemented but is not byte-identical to Cqc10 or final source.

**Recommended use:** use the latest Cqc15 27.27 versus SGLang 26.89 as a clearly named same-task descriptive comparison; retain the earlier 29.09/28.28 observations in chronology, rather than selecting only the maximum. Keep Cqc10 25.63 across ten tasks as the larger deployment/behavior observation. This supports competitive observed throughput against the tested SGLang deployment, with a slightly higher rate on the shared tasks; it is not evidence of globally best performance or a statistically established causal gain.

The public 28.8 headline is Sr12's older inverse mean request-TPOT estimator. Its whole four-task cohort is **26.2053649279 pooled tokens/s**, while its SGLang-overlap pair is 29.0934427002. These different populations and estimators must not be substituted silently. Cp1, before split-K, gives 24.7401875088 on the same pair. Cqc16's third task 13236 is a genuine degeneration case and Cqc15's third task 13398 is incomplete; neither is removed from a claimed full-campaign denominator here. The comparison above is explicitly the two-task overlap.

## B4 records

The remembered 43.57 aggregate step tokens/s is July27 SUBSPAN1 on four SWE tasks and Qwen3.6 FP8. Its native 43.7 comparison is a two-point fitted projection with zero residual degrees of freedom and a health-flagged source arm, not a measured parity result. Source: `FR13_S1_CAMPAIGN_LOG.md:943-958` and the earlier fit audit at lines761-765.

Later August12 pool16 arms record about49.5 aggregate step tokens/s on average; full-width diagnostic windows reach53.82-59.24. August14 Hydra27 GQA-pair reaches55.78 whole-arm aggregate step tokens/s. These are older FP8 configurations and use aggregate step time, not summed completed-request time. There is no corresponding external baseline in those runs.

The common pooled completed-request estimator was reconstructed from16 complete original arm-level brackets (all selected request/token/latency counts align and pre/post gauges are idle): August12 pool11.48-13.16 tokens/s; August14 GQA candidates12.05,13.56,12.56,13.36 versus their stock12.43,11.37,12.07,11.98. The parent independently recomputed all16 values from the retained counter deltas. Summing request times counts concurrent intervals separately, so these rates must not be conflated with aggregate GPU throughput or multiplied by4 without a supported timing denominator. The B4 records establish internal optimization observations, not an external frontier win.

## Evidence

The sibling `rate-candidates-2026-09-24/` directory contains source-hashed metric-line extracts, selected safe metadata/configuration/behavior projections, independent arithmetic, all candidate populations including failures, and32 original B4 metric-file hashes. It does not include unfiltered environment secrets. Original remote paths and source hashes remain recorded. The SGLang original48-input comparison bundle remains unchanged. This is a new audit note, not a revised paper delivery.
