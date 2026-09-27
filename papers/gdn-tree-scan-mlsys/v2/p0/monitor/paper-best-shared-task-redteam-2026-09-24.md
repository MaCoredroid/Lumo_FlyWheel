# Best recorded shared-task result: manuscript red-team, 2026-09-24

**PASS — no unresolved material finding in the reviewed three source files.** The headline reports the maximum among the three recovered tree configurations on the two completed comparator task IDs. It does not relabel that observation as the latest configuration, a whole-campaign rate, a frozen replicated treatment effect, or an application-speed/quality improvement. No additional experiment is indispensable for this explicitly descriptive scope.

## Source and evidence checks

| Passage | Check and disposition |
|---|---|
| `abstract.tex:2` | Correctly says **best recorded**, **two shared tasks**, **pooled decode**, and **different serving configurations and response caps**. The ten-task behavior observation is explicitly a separate deployment; no behavioral claim is transferred from Cqc10 to Sr12. |
| `case-study.tex:35–47` | The shared estimator remains `(N-R)/(sum E2E - sum TTFT)`, with pooling before division and explicit request-time/agent-wall/GPU-time distinctions. SGLang's 26.89 and Cqc10's 25.63 retain their original populations. |
| `case-study.tex:49–55,62–72` | Selection uses comparator availability, IDs 12907 and 13033; all three recovered split-K tree observations are shown. Dates, distinct patch revisions, common attention extension, and response caps 32768 versus 24000 are disclosed. The table separates Cqc10's disjoint ten-task subset. No averaging of different builds or cohorts creates the headline. |
| `case-study.tex:52` | Independent parsing of the original selected pre/post metric files reproduces all three tree rates, counts, and the 8.18% comparison. The request-success, E2E, TTFT, and generation histogram populations agree; generated-token counter equals generation histogram sum; running/waiting gauges are zero at both boundaries. The cap projection records stop completions and zero length terminations for the six selected tree attempts. |
| `case-study.tex:33,52,72` | Source-bound trace projections support tool use, nonempty patches, zero malformed arguments, and zero recorded degeneration flags for the six selected tree attempts. The heuristic and its blind spots remain explicit. Cqc16's later 13236 degeneration and Cqc15's unfinished/unevaluated 13398 remain adverse records, outside the stated shared pair. The wording makes no general degeneration-freedom or unchanged-quality claim. |
| `case-study.tex:72` | Sr12's entire four-task population is retained: 124 requests, two resolved/two failed, **26.21** pooled tokens/s. This explicitly prevents mistaking 29.09 for its full-run rate. |
| `main.tex:241,256,323,330–332` | Setup, evidence table, discussion, and conclusion consistently retain best versus later observations, configuration limits, and separate numerical qualification. No new native, B4, frontier-wide, task-completion, or causal tree-verification superiority claim appears. Appendix rate definitions remain separate from the shared pooled estimator. |

Independent arithmetic from copied original tree metric brackets (CPU reads only):

| Population | N | R | Sum E2E seconds | Sum TTFT seconds | Exact pooled tokens/s |
|---|---:|---:|---:|---:|---:|
| Sr12, shared pair | 25027 | 41 | 989.2611684799194 | 130.44221544265747 | 29.093442700158857 |
| Cqc16, shared pair | 26239 | 35 | 1048.7886128425598 | 122.29720187187195 | 28.28304686877344 |
| Cqc15, shared pair | 70424 | 53 | 2849.883605480194 | 269.1670525074005 | 27.26800815027045 |
| Sr12, all four tasks | 101923 | 124 | 4300.757935523987 | 416.0953757762909 | 26.20536492791583 |

Against the previously independently verified SGLang rate `26.893956024189762`, Sr12's relative difference is `100*(29.093442700158857/26.893956024189762 - 1) = 8.178367935125518%`. All displayed rounding is correct. The saved `shared-task-rate-audit.json` agrees with these independent computations. Raw tree paths are `results/agent-workload/raw/shared-tasks-20260924/{Sr12,Cqc16,Cqc15}/swe_out/verified/per_task/astropy__astropy-{12907,13033}/vllm_metrics_{pre,post}.txt`; Sr12's additional full-run tasks are 13236 and 13398. The raw manifest and audit bind those individual files.

This review does not independently recertify the entire artifact export, all underlying runtime files, or PDF rendering. Those remain separate artifact/build reviews. It writes only this report and launches no inference.

## Reviewed SHA-256

Paths are relative to `papers/gdn-tree-scan-mlsys/v2`.

```text
862a7dd2c33e431ca875a5aebcd8a25db75a45cdba48433578ac5d0aead46274  abstract.tex
dcd452080c116eafa7836b8c4e580ef53406f7bef499fa404638f88609c3cc69  main.tex
2a841d756d0de05b03458c86d00d29ad2c4b19691d7f6722e19bf14109085c13  results/agent-workload/case-study.tex
dcc6f3fb69bdd01f7ff8bed0c920ea797c18c18418c06044bfa039a88a7b286a  results/agent-workload/shared-task-rate-audit.json
d007110971ed8d3d0b2829a4f1f8e617ff12faaf88ca6231263997cfff4881ca  results/agent-workload/shared_task_reduce.py
2e20393e2fa238d489215a11d3b3315a08e9516609f341713ecd1ae62ed4dbd2  results/agent-workload/raw/shared-tasks-20260924/MANIFEST.json
8efcf7a07982f1eda49c2eade9bd62fd1b754180cf2eb62a3dd8c35004602a2e  results/agent-workload/raw/shared-tasks-20260924/audit-projections/nvfp4-b1-cohort-caps-20260924.json
d25ca9d36dbdcb94e701ed6dda1a879212702a49666072d84598786b31d55a77  results/agent-workload/raw/shared-tasks-20260924/audit-projections/nvfp4-b1-cohort-config-behavior-20260924.json
cd3de566e6f14d6dcd98414f240a17ec04bb497f4b16a70858efcc6b1108fae4  results/agent-workload/raw/shared-tasks-20260924/audit-projections/nvfp4-b1-cohort-engagement-20260924.json
```
