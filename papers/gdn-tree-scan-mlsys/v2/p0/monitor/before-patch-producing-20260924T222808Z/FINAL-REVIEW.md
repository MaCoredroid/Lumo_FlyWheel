# Competitive comparison correction — 24 September 2026

The 13-page paper compares complete speculative-decoding systems. The blanket abstract caveat about different serving configurations and caps has been removed. Proposal topology, draft depth, candidate count and serving optimizations are part of the competing methods. A matched draft-budget ablation would isolate a component effect; competitive system evaluation instead needs comparable workload/resources and a documented tuning opportunity.

The fairness audit recovered a stronger earlier SGLang result on exactly the shared Astropy tasks 12907 and 13033. All rates use `(sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`:

| Recorded system | Pooled tokens/s | Requests |
| --- | ---: | ---: |
| SGLang EAGLE earlier | 30.70 | 17 |
| SGLang EAGLE later | 26.89 | 45 |
| Tree Sr12 | 29.09 | 41 |
| Tree Cqc16 | 28.28 | 35 |
| Tree Cqc15 | 27.27 | 53 |

The best tree result is 5.23% below the earlier SGLang result and 8.18% above the later result. The abstract and conclusion include both SGLang observations. No best-versus-best speed advantage is claimed. The separate ten-task Cqc10 result remains 25.63; Sr12's full four-task cohort remains 26.21.

SGLang's steps/top-k/draft-tokens setting 3/1/4 matches its pinned Qwen cookbook recipe. Hydra27 uses 27 valid draft nodes, maximum depth 11 and 32 physical verification rows. The recovered records establish deployed settings but no common draft-budget tuning sweep. The cookbook and source identities are in the fairness note.

All 41 Sr12 and all 45 later SGLang responses are at most 20,000 tokens, below both configured caps. The earlier SGLang and Sr12 caps both equal 32,768. Earlier SGLang task 13033 is retained despite its empty patch: metadata records zero patch bytes, and the harness assigned a synthetic failed outcome without invoking tests. Its final response has 32,768 thinking-only tokens; the trace stop_reason is null, so no explicit length-finish flag is inferred. One agent task runs at a time; occasional overlapping SGLang auxiliary requests remain in the summed request-time denominator.

All three selected tree pairs retain one resolved and one failed task. Their six attempts used tools and produced nonempty patches with no detected degeneration under the recorded heuristic. The separate ten-task deployment retains six resolved and four failed attempts with no detected degeneration. These observations do not establish broad quality equivalence. Earlier adverse/incomplete attempts remain recorded.

New evidence includes 12 original safe SGLang metric/metadata/eval/launch/note files, two source-bound safe projections, and an unchanged common-estimator audit. The aggregate reducer binds 94 inputs plus five projections and replays both child receipts exactly. Independent review reproduced the result from 105 isolated files and rejected raw-hash and child-receipt negative controls. No new inference ran.

Current verification:

- `notes/serving-comparison-fairness-2026-09-24.md`: primary-source settings, draft geometry, recovered comparator and behavior evidence.
- `notes/competitive-comparison-redteam-2026-09-24.md`: independent raw arithmetic, safe projections, current manuscript and isolated dependency replay; PASS.
- `notes/headline-response-cap-check-2026-09-24.json`: both original selected response-length populations.
- `p0/monitor/2026-09-24-competitive-comparison-build.json`: final source/PDF hashes, speed inventory, citation preservation and visual QA.
- `artifacts/FINAL-DELIVERY.json`: final private package, extracted replay and rebuild identities.

No publication, submission or push occurred. The earlier best-shared-task archive and receipts remain dated snapshots; this correction supersedes their selected-comparator headline. Numerical qualification and the method/optimization focus are unchanged.
