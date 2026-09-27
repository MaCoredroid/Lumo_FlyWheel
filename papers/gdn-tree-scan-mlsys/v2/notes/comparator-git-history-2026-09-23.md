# Comparator experiments recovered from Git — 23 September 2026

## Correction

The repository contains real SGLang SWE-bench runs, DSpark serving and captured-input experiments, DFlash2 capture and served smoke tests, and complete native-MTP/tree agent campaigns. The previous review was too narrow: its absence of a matched Cqc10 comparator must not be paraphrased as no comparator experiments having been run.

This history audit fetched origin without merging and searched all local/remote-tracking refs, commit messages, changed paths, original result records, and later campaign closure decisions. No inference was launched. Old implementation results are kept here for provenance; no superseded performance numbers were restored to the manuscript.

## Recovered inventory

| Experiment | Evidence | Workload and observed result | Applicability |
|---|---|---|---|
| SGLang EAGLE 3/1/4 | `128d27b86`, `results/fr14_nvfp4_port_20260816/ablation_a_step2.md` and `_reduced.json` | Actual Qwen Code SWE-bench Verified Astropy exact4; 3 completed and scored, fourth interrupted. SGLang log-sample mean generation rate 29.215 tokens/s (1,017 decode samples); report compares it with the then-current tree's 25.26. Both scored resolved/failed/resolved on the three finished tasks. | A real cross-stack agent experiment, explicitly diagnostic. Older conservative-NVFP4 tree versus SGLang aggressive checkpoint/FP8 KV; different trajectories and unequal observation populations. Not a current Hydra27 speed ratio. The 29.22 is an arithmetic mean of logged throughput samples, not Cqc10's inverse mean request TPOT. |
| Later SGLang agent campaign | `c27f30207`, `results/fr14_nvfp4_port_20260816/sglang16_evidence/` | Two completed tasks under 24k cap: 12907 resolved, agent 426.131 s; 13033 failed, agent 1617.769 s. Next attempts interrupted. | Runtime/outcome evidence survives. Campaign names do not establish 16 completed tasks: pass238 explicitly drops r2 and pass245 closes the series at two verdicts. Neither task overlaps Cqc10. Commit-title 480.5/1671.5 s includes evaluation/boundary time and must not be called agent duration. |
| DSpark (DFlash-family drafter) serving | `6530b1f17`, `results/fr14_nvfp4_port_20260816/sglang_calibration/dspark_bench_bs{1,8}.jsonl` | Random1024 input/output: 8 completed requests at concurrency1, 8192 output tokens /411.0920 s =19.9274 tokens/s; 32 at concurrency8, 32768/277.8549 s =117.9321 aggregate tokens/s. | Actual serving benchmark, but synthetic. Neither number is SWE agent performance; concurrency8 aggregate is not per-agent rate. |
| DSpark fusion E1 | `c0d5550f4`, campaign REDTEAM pass77 and captured-input evidence | Completed drafter/fusion experiment; reported rejection of proposed fusion. | Captured-input algorithm study, not an end-to-end SWE comparison. The source closeout records the measured decision. |
| DFlash2 E2 | `f5ae4df63`, `b2ff0aaef` | Capture/recapture completed; later August closeout explicitly leaves replay queued/paused. | A capture is not a measured completed replay or full agent run. |
| DFlash2 served smoke | `92fea914a` on `origin/vllm-upstream/plan-v3`, `results/upstream/54076/RESULT.md` | GB10 TP1, vLLM0.28.0, Qwen3.8-27B-FP8 plus separate DFlash2 drafter,k7; booted and answered a chat request. | Definitively served; this corrects any blanket unsupported/never-served assertion. Scheduler/KV-geometry test with short context, not NVFP4 SWE speed measurement. |
| Tree / native MTP5 / native MTP11 | `51b7dafdf`, `FR13_POSTSNAPFIX3_CLOSEOUT.md`, full `output/fr13_kvremap_tail6/` | Three arms ×16 real SWE tasks, B4; archived outcome counts10/16,10/16,8/16. | Completed comparison under an older design/model. The historical decode proxies were separately invalidated by the P0 population audit and remain excluded; existence of the comparison is not in doubt. |
| Later native NVFP4 probes | See `swe-workload-comparison-review-2026-09-23.md` | Four real task runs on two Astropy cases, two replicates each. | Targeted behavior probes with actual outcomes/runtimes; cache/graph settings differ from tree. Retain both favorable and adverse tree cases, not a selected speedup. |

Additional prior methods: native n-gram/PLD warm-request tests (`390739964`) and Arctic/suffix decoding with a later13-task in-house end-to-end harness study (`f4f33ad41`, `cd6cc54b2`) also exist. These are not SWE-bench Verified results and were not raw-requalified in this audit. Details, all-ref search coverage, and DFlash revision corrections are in `external-baseline-history-audit-2026-09-23.md`.

## What this means for the current paper

1. The comparison discussion should name the executed methods and distinguish their purposes.
2. The old SGLang/tree comparison does not determine the current design's ranking. It uses an earlier tree/checkpoint plus a different rate estimator. Cqc10's28.20 cannot be divided by the old29.22 or synthetic26.1 to manufacture a speed ratio.
3. The later SGLang task records can be audited without new inference. Its ITL counters and Cqc10's mean-request TPOT are different estimators; task sets also differ. See `sglang-swe-recovered-evidence-2026-09-23.md` and `sglang16-metric-audit-2026-09-23.json` for exact fields/hashes. Cqc10 also has an ITL-named histogram, but its49,068 observations track decode-step attempts instead of248,077 generated tokens minus requests; matching metric names do not give matching observation populations.
4. Fresh experiments should fill a specifically identified remaining gap after evidence reuse, rather than repeat an imagined absence of all baseline testing. The prior96-attempt proposal remains conditional and unlaunched.
5. The manuscript now acknowledges these executed comparator experiments without restoring superseded rates. The abstract's blanket future-comparison sentence was removed, along with the quality clause the author asked to remove.

## Primary record hashes (live checkout)

- `results/fr14_nvfp4_port_20260816/ablation_a_step2.md`: `94f5910257d03f6b290a2a9d9ccc2563b2e93c87fbaf7cfdeb4858bdd9254445`
- `results/fr14_nvfp4_port_20260816/ablation_a_step2_reduced.json`: `3850e2e8c6d4a65fafe9d27d989b3c452c565b0c3f5ed5ce9d398c55f12c5e92`
- `results/fr14_nvfp4_port_20260816/sglang_calibration/dspark_bench_bs1.jsonl`: `e9467cf4da82be76c70d64d334e4443d2b88f9e59b4e2c696ed99366a9ee1aab`
- `results/fr14_nvfp4_port_20260816/sglang_calibration/dspark_bench_bs8.jsonl`: `bb4ca75a2a29f4232e3690129619442899528ddad97c3d0df0f19d3f8c262af6`
- `FR13_POSTSNAPFIX3_CLOSEOUT.md`: `3ce92516ef36dacb525ed3694734c9de6bb0929a249b1cf43825b1d5a3c1a5a0`
- `papers/gdn-tree-scan-mlsys/v2/p0/P0-MEASUREMENT-ERRATUM.md`: `83ce2026e5bb7e544b210ddd96d3a039989a1f631f8e2b5c6ffd190b98c50d8a`
