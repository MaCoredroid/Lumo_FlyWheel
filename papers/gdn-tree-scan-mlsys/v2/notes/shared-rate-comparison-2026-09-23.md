# Common rate calculation from saved SWE-bench records

**Yes: a shared estimator can be reconstructed without new inference.** This resolves the averaging mismatch between the paper's request-weighted 28.20 and the older SGLang log-sample mean 29.22. Neither old value is used to compute the comparison below.

| Recorded configuration | Completed Astropy tasks | Completed model requests | Output tokens | Pooled first-output-adjusted rate | Pooled request rate including first-token latency |
|---|---:|---:|---:|---:|---:|
| GDN Tree-Scan, Cqc10 | 10 | 265 | 248,342 | **25.63 tokens/s** | **23.28 tokens/s** |
| SGLang EAGLE, saved run1/r1 | 2 | 45 | 47,854 | **26.89 tokens/s** | **23.76 tokens/s** |

Both cohorts are real Qwen Code SWE-bench Verified Astropy tasks with Qwen3.8-27B NVFP4 serving on GB10. They use different task subsets and deployed serving configurations. The table is a descriptive cross-stack comparison; it is not a paired task-speed result or an isolated effect of tree verification. The tree rate is 4.69% lower on the first-output-adjusted measure and 2.04% lower on the request measure including first-token latency. These percentages describe these saved cohorts only.

## Identical formulas

Let N be the sum of output tokens on completed requests, R the completed-request count, L the sum of their recorded end-to-end request latencies, and F the sum of their time-to-first-output observations.

- **Pooled first-output-adjusted rate:** `(N - R) / (L - F)`.
- **Pooled request rate including first-token latency:** `N / L`.

Pooling happens before division. Requests contribute according to their token and latency totals, rather than each request or each printed log sample receiving equal weight. The first formula is the conventional N−1 output-interval convention; it subtracts one output token per request. Other tokens emitted in the same first batch can have zero observed arrival gaps. This is a rate under the server's first-output/finish timing convention, not a measurement of tokens generated strictly after first emission or kernel-only GPU time.

| Counter | Tree | SGLang |
|---|---:|---:|
| N | 248342 | 47854 |
| R | 265 | 45 |
| L, seconds | 10667.621147632599 | 2013.6344224798959 |
| F, seconds | 989.8239369392395 | 235.9487611737568 |
| N − R | 248077 | 47809 |
| L − F, seconds | 9677.79721069336 | 1777.685661306139 |

The second rate includes the recorded pre-first-output work and waiting; both rates exclude agent tool time. Summed request latency counts concurrent request intervals separately and is not elapsed machine wall time. The separately computed output-tokens/agent-elapsed-time figures are 22.85 and 23.41 respectively, but those are not task completion rates and do not establish a time-to-solution advantage.

## Support checks

For every completed task bracket:

1. The completion count, generation-token histogram count, request-latency count, and TTFT count agree.
2. The completed-request token histogram sum matches the generation-token counter delta exactly.
3. The running and queued request gauges are zero at both boundaries; there is no observed in-flight carryover or negative selected-counter delta.
4. Task metadata and evaluator identities agree; neither selected SGLang attempt nor any Cqc10 attempt timed out.
5. Every selected SGLang bracket has zero non-streaming request, token, and timing contribution. This matters because a non-streaming first response is not an independently observed first generated token.

Use the completed task's pre/post snapshots, not the whole SGLang run's last snapshot: the latter also contains an interrupted subsequent task. SGLang's streaming series may be instantiated lazily; the pre-task non-streaming warmup series is present and is subtracted. Original records remain unchanged.

Support is **all completed engine requests inside the task brackets**, including internal agent traffic. Cqc10 identifies 256 normal requests and nine successful compactions. SGLang task13033 has four additional requests beyond visible assistant-usage messages; the older metadata does not classify these, so they must not be called confirmed compactions. Their tokens and latency are included together.

## Agent behavior and abstract use

The existing Cqc10 trace audit found no degeneration flags or malformed tool-argument flags in the ten tree attempts; every attempt used tools, produced a nonempty patch, and terminated with an evaluator report. Six tasks resolved and four failed tests. This behavior statement refers to the ten-task cohort, not the earlier campaign containing a detected degeneration case.

Suggested results sentence for discussion (not yet substituted into the manuscript):

> On selected SWE-bench Verified Astropy tasks on GB10, GDN Tree-Scan achieves a pooled first-output-adjusted rate of 25.63 tokens/s, compared with 26.89 tokens/s for SGLang EAGLE on its evaluated subset. No degeneration was detected in the ten tree-served agent episodes.

The body should identify both task lists, task counts, serving configurations, and the common formula. The abstract need not recount the metric audit. These observations support competitive descriptive rates; they do not support claiming that the tree beats SGLang.

## Reproduction

From the repository root:

```sh
python3 papers/gdn-tree-scan-mlsys/v2/results/agent-workload/shared_rate_reduce.py
```

The checked output is `results/agent-workload/shared-rate-audit.json`. It records every task, the exact selected metric deltas, all 48 raw input hashes, the reducer hash, both pooled estimates, and descriptive relative differences. It requires no network, GPU, model weights, or external Python packages.

Independent arithmetic and support review: `common-service-estimator-independent-review-2026-09-23.md`, with separate script and output `common-service-independent-2026-09-23.py` / `.json`. The independent implementation reproduces both pooled rates.

The implementation-source semantics review is `common-rate-estimator-semantics-2026-09-23.md`. `results/agent-workload/SHARED_RATE_MANIFEST.json` binds this calculation and both independent reviews. The canonical manuscript/PDF was not changed during this calculation.
