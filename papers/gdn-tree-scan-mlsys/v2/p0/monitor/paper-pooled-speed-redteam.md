# Pooled-speed claim review — 2026-09-23

**Disposition: PASS. No unresolved material issue in the reviewed manuscript and current supporting documents.** This is a source/evidence review, not PDF visual QA. FINAL-REVIEW, the rebuilt PDF, and final packaging remain parent-owned. No inference or experiment was run, and no manuscript, raw result, or attempted experiment source was changed.

## Scope and result

Reviewed the full current `main.tex`, every reachable input (abstract, case study, and three TikZ figures), captions, and current README, claim ledger, experiment review, configuration, and design documents. The only performance comparison presented is the common pooled decode estimator on recorded agent workloads. The old 28.20 tokens/s / 35.46 ms inverse-request-mean headline is absent from reachable manuscript inputs. E1/E8 serving performance plots and multipliers are not loaded; numerical qualification and state-continuation observations retain their separate scope.

| Claim or passage | Independent check | Disposition |
|---|---|---|
| Abstract line 2; case study lines 35–63: tree 25.63 versus SGLang 26.89 tokens/s | Both use `D=(N-R)/(sum(E2E)-sum(TTFT))`; counts and durations are pooled before division. Exact totals below agree with the source-bound independent reduction and the common reducer. | Supported descriptive comparison. |
| Case study lines 43–47: population and denominator meaning | Request counts agree with E2E, TTFT, and generation histogram populations; token counters agree with generation sums. Selected boundaries have no outstanding requests or resets. Tree includes 256 normal plus nine successful internal compaction requests. Both include all completed engine requests; no incompatible ITL histogram is substituted. | Supported. Accumulated request time is correctly distinguished from GPU time, campaign wall time, and between-request agent/tool work. |
| Case study table caption and lines 63–66: 4.69% lower; different subsets/configurations | `100*(1-D_tree/D_sglang)=4.6863077979417795%`. Ten tree tasks and two SGLang tasks are disjoint cohorts. Single deployed-stack observations do not isolate tree verification or establish paired task completion speed. | Supported, with the necessary limits stated. |
| Task outcomes and agent wall times; trace checks | The ten-task table preserves six resolved and four failed outcomes and the independently audited task times. SGLang task 12907 is resolved at 7.10 agent minutes; 13033 fails tests at 26.96 minutes. Case study line 33 retains the trace heuristic and its blind spots. | Supported; no unchanged-quality or universal degeneration-freedom inference. |
| Main setup, evidence table, discussion, conclusion, appendix | The current pooled comparison is consistent across sections. Source-level optimizations are not assigned unmeasured gains; no E1/E8 multiplier or best-performance claim remains. The conclusion preserves cohort/configuration differences. | Supported. |
| Completed versus deferred comparisons | Main line 72 limits the unexecuted author-implementation statement to its three listed methods. Main line 350 explicitly defers a Bole/TreeWY system comparison. Case study lines 69–70 acknowledges prior SGLang, native, DSpark, and DFlash2 experiments with their actual scope. | No false claim that SGLang was never tested or that the descriptive comparison remains unperformed. |

Exact common-counter totals:

| Deployment | Tasks | N output tokens | R completed requests | Sum E2E (s) | Sum TTFT (s) | Pooled decode tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Tree Cqc10 | 10 | 248342 | 265 | 10667.621147632599 | 989.8239369392395 | 25.633622465853126 |
| SGLang EAGLE | 2 | 47854 | 45 | 2013.6344224798959 | 235.9487611737568 | 26.893956024189762 |

The common reducer was re-executed read-only and its parsed output matched `results/agent-workload/shared-rate-audit.json`. That audit binds 48 inputs. The independently written reduction in `notes/common-service-independent-2026-09-23.{py,json}` agrees. The operational estimator is common; identical backend timestamp implementation, matched tasks, matched complete configurations, and causal speed superiority are not asserted. The older request-mean and backend ITL statistics remain archival diagnostics, not alternative headline estimates.

## Closed finding and finite closure

The one supporting-document finding was `notes/design/baselines.csv` describing SGLang only as a planned comparison. The final file now marks its completed two-task pooled comparison and separately says a paired causal comparison has not run. It also separates the NVFP4 agent deployment from FP8 qualification and labels archived E1 performance as audit-only. Rechecked at SHA `783d59e90626e7fe53b1fd71c64cd367095b06200c6ecd2898bd3e50441d9031`: **closed**.

Final main changes only relocate the planned-experiment table and explicitly name Bole/TreeWY in E7c; no numerical change was introduced. Reference/citation checks found no orphan references or missing citation keys. No further experiment is necessary for the bounded descriptive claims now made. A matched task campaign remains necessary only to establish the currently unclaimed native/tree task-speed or quality advantage.

## Reviewed SHA-256 bindings

All paths below are relative to `papers/gdn-tree-scan-mlsys/v2`; final source hashes were rechecked after the last parent edits.

```text
2ee3791b9f60bd839ae73a5d81cd106d6725f7192b04de5d85ffc77e70231342  main.tex
40fbbff240d41c1948a2c669fb1f18d524793dc4f7f97bdd9458ef8086a12798  abstract.tex
a689968b51b2ff2701c5d9867535f2f054329d1b21e4dd63fbe2d987002b4007  results/agent-workload/case-study.tex
c16b154703aec18c66c8a917fd0c18bfe90272b2ed871b2a6a392a4e8469e8e1  README.md
24c51524e97dbfc04fac34c4ac4c492c153f14c02c4d1c7ed7d2cf2a794e8e98  notes/claim-evidence-ledger.md
c8adff151e2656c046d0b974d3d0ffd9dbc595e908e903575493f325583be0d9  review-experiments.md
4e14cf7959479e07c361b8837a6d293510005fe697dd2a07623ff14a35e4f3d3  paper.config.yaml
783d59e90626e7fe53b1fd71c64cd367095b06200c6ecd2898bd3e50441d9031  notes/design/baselines.csv
1a0ed7aad226642d59885a8771f156d21887c2f557eced957986b2bbc9891ff1  notes/design/experiment-matrix.csv
5dfd8e136f33a792b941a902b09307000f836f3cd789b8996afc0e28b4be18a2  notes/design/method-components.csv
de4fcfbe8583b886f9ff9b55c412176b1f9c146a86b42287dc0d276cf9a86ee0  notes/design/optimization-evidence-map.md
8c9dbd43516be8703d4ba841a06d1f2d17caad2c434a61ec29ede40035f0762b  figures/pipeline.tikz
50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898  figures/state-contract.tikz
affc55cd6972d929de38dc8ebdd86c6dcea77fb0beb73c96551010bfaffadcfe  figures/scan-replay.tikz
c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f  results/agent-workload/shared_rate_reduce.py
91e38415989781a3d2460729f86ed90aa246167c3850d480b03354c84342e9b8  results/agent-workload/shared-rate-audit.json
5bc20dcb56d9eaa877b986581c3fb1f081253213cbd2e365a57543ed052fcf7a  notes/common-service-independent-2026-09-23.py
04291fd57a373d592f032dfd3229002135e4983410d3383e295c92c6623897e2  notes/common-service-independent-2026-09-23.json
80b83c8c60e807a4eabb8cacba830e3fd92ba74c2f2168402393cd632c7bdff0  notes/common-service-estimator-independent-review-2026-09-23.md
```
