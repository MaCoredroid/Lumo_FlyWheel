# Common service estimators — independent raw reduction, 2026-09-23

**PASS for descriptive server-counter estimators on the two recorded cohorts.** This uses neither backend's incompatible ITL histogram nor Cqc10's inverse request-mean TPOT. No model execution or raw-file changes.

For each completed task bracket, let `N` be the delta in generated tokens, `R` the completed request count, `E` the sum of server-reported end-to-end request latency and `F` the sum of server-reported TTFT. Pool the raw deltas first:

- Output-token rate over summed request latency: `N / E`.
- Pooled post-first-token rate: `(N - R) / (E - F)`.

These use the same counter-defined arithmetic in each backend. The second subtracts one first token and one TTFT per completed request; it is not a recovered mean of per-request rates or a physical kernel timing.

| Cohort | N | R | E (s) | F (s) | N/E (tokens/s) | (N−R)/(E−F) (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|
| SGLang, two completed tasks | 47,854 | 45 | 2,013.6344224798959 | 235.9487611737568 | 23.7649890495 | 26.8939560242 |
| Cqc10, ten completed tasks | 248,342 | 265 | 10,667.621147632599 | 989.8239369392395 | 23.2799793471 | 25.6336224659 |

SGLang per-task pairs are 20.5819709169 / 28.3669731872 for 12907, and 24.5637563708 / 26.6039086119 for 13033. Pooling gives the table; averaging task rates would be another estimator.

## Population and boundary checks

Every task bracket passes:

1. Request-counter, E2E-histogram, TTFT-histogram and generation-length-histogram count deltas agree exactly.
2. The generated-token counter delta equals the generation-length-histogram sum delta. Selected counters do not reset. E2E sum exceeds TTFT sum and every completed request has a TTFT observation.
3. All retained running/waiting gauges at both snapshots are zero. SGLang's retained grammar/prefill-inflight/preallocation/transfer queue gauges are also zero. These are observed boundary states, not continuous telemetry.
4. Cqc10's selected counters are contiguous between all ten task brackets. Its task-boundary receipts report zero pending operations at both ends; saved traffic-audit/flush records additionally reconcile all 265 requests and exclude campaign rejections/aborts. SGLang has two separate boots: use each completed task's own bracket, excluding smoke requests before it and unfinished next-task work after it. Both orchestration runs were later preempted, so whole-run post counters are unsuitable substitutes.

The metric families are `generation_tokens_total`, `e2e_request_latency_seconds_{sum,count}` and `time_to_first_token_seconds_{sum,count}` with backend prefixes. SGLang request count comes from `num_requests_total`, and generation histograms from `generation_tokens_histogram_{sum,count}`. vLLM uses `request_success_total` and `request_generation_tokens_{sum,count}`. Subtract exact labeled series before summing; SGLang's non-streaming deltas are zero in both retained task brackets.

## Internal requests and limits

Cqc10 explicitly binds 256 normal requests and nine successful internal compactions, with no failed internal requests. All are included in the rates. SGLang 12907's final trace usage and visible nonzero assistant usage both match 18 requests and 8,314 tokens. For 13033 the final trace usage matches all 39,540 server tokens, while visible assistant usage accounts for 23 messages and 32,361 tokens versus 27 completed engine requests: four additional requests and 7,179 output tokens are included. Its older metadata does not explicitly classify those extra requests; do not label them definitively as compactions. The support is **all completed engine requests in these task brackets**, not only visible final answers.

These rates use each server's reported request timestamps. They exclude tool/evaluation time and are not task wall time, arrival-throughput measurements, request-weighted TPOT, or an isolated mechanism ablation. The SGLang task IDs do not overlap Cqc10, and the runtime/model-view/attention/scheduling settings differ. Thus the same estimator is now available descriptively, but a ratio would remain an unmatched cohort observation, not a causal tree/SGLang speedup or quality comparison. No additional experiment is needed to report this arithmetic with those labels.

## Reproduction and evidence

`notes/common-service-independent-2026-09-23.py` reads original files only and prints JSON; SHA `5bc20dcb56d9eaa877b986581c3fb1f081253213cbd2e365a57543ed052fcf7a`. The script retains the exact local repository root used in this audit.

`notes/common-service-independent-2026-09-23.json` contains every task's numerator/denominator, count-equality checks, zero boundary gauges, compaction/trace reconciliation, and all read input SHA-256 hashes; SHA `04291fd57a373d592f032dfd3229002135e4983410d3383e295c92c6623897e2`. Earlier incompatible-ITL and interrupted-SGLang findings remain in `notes/sglang-swe-recovered-evidence-2026-09-23.md`; this common-estimator reconstruction supplements rather than alters those records.
