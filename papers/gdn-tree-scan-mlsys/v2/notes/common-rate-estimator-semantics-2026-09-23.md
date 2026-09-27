# Common rate estimator from saved request metrics — 2026-09-23

## Verdict

**Yes: a common completed-token / summed-request-latency estimator is reconstructible for Cqc10 and both completed SGLang tasks.** Use `N / ΣE2E`, where N is the summed completed-request generation-token histogram (cross-checked against the generation counter), not a speculative-token/accepted-draft counter. This is tokens per summed server-observed request-second, not machine throughput, task wall-time speed, or a matched-workload causal comparison. Cqc10 contains ten tasks/265 requests, including nine compaction requests; SGLang here contains two different completed tasks/45 requests. The populations must remain visible.

| Population | N completed generated tokens | R completed requests | ΣE2E (s) | ΣTTFT (s) | N/ΣE2E | (N−R)/(ΣE2E−ΣTTFT) |
|---|---:|---:|---:|---:|---:|---:|
| Cqc10 ten tasks | 248342 | 265 | 10667.621147632599 | 989.8239369392395 | 23.2799793471 | 25.6336224659 |
| SGLang 12907 | 8314 | 18 | 403.9457656198647 | 111.49299156968482 | 20.5819709169 | 28.3669731872 |
| SGLang 13033 | 39540 | 27 | 1609.6886568600312 | 124.45576960407197 | 24.5637563708 | 26.6039086119 |
| SGLang pooled two tasks | 47854 | 45 | 2013.6344224798959 | 235.9487611737568 | 23.7649890495 | 26.8939560242 |

Values were independently parsed from exact pre/post brackets; no generation or benchmark ran. Raw paths and SHA-256s are in `results/agent-workload/metric-semantics-source-review/independent-estimator-check.json`.

## Completion support and boundaries

- **vLLM:** `metrics/stats.py:362–402` counts `len(output.new_token_ids)` and records the first/last engine output. `stats.py:428–475` creates a completed-request record containing generation-token count and E2E latency. `metrics/loggers.py:1178–1208` observes E2E and request-generation-token histograms from that same completed-request record. The continuously updated generation counter can contain unfinished work in general; therefore prefer the completed-request histogram sum. In Cqc10 both are exactly 248342, all histogram/request/TTFT counts equal 265, all finishes are `stop`, and running/waiting gauges are zero at both bracket boundaries.
- **SGLang:** `tokenizer_manager.py:2864–2892` invokes `observe_one_finished_request` only for a finished request. `metrics_collector.py:1740–1797` increments generation tokens, completed-request count, E2E histogram and generation-token histogram from that same call. For both task brackets, histogram sums equal counter deltas, histogram/E2E/TTFT counts equal 18 and 27, and running/queue gauges are zero at both boundaries. Non-streaming counter/latency deltas are exactly zero: the two earlier non-streaming smoke requests subtract away. This audit excludes the subsequently interrupted tasks and does not pool whole-run counters contaminated by their partial work.
- Server E2E measures each request's observed lifetime. Summing it does **not** add idle/tool/agent gaps between requests. The old narrative that the surviving Prometheus brackets necessarily conflate all agent think time is too broad; it applies to dividing by task elapsed time, not to this summed request-latency estimator. Request latency can include queuing/preemption and each engine's frontend work; it is not isolated GPU execution.

## First-output-adjusted estimator

`(N−R)/(ΣE2E−ΣTTFT)` is also reproducible as a **first-output-adjusted pooled rate**, the conventional N−1 token-interval definition applied to the same completed population. It is not the older inverse arithmetic mean of per-request mean TPOT (28.1973882 for Cqc10), which weights requests differently.

SGLang `tokenizer_manager.py:2422–2433` timestamps the **first output batch** and completion; `req_time_stats.py:459–466` defines TTFT = first-output time − creation time and E2E = finish time − creation time. First batch can contain more than one token. Under an observed-arrival convention, tokens in the same batch have zero interarrival gaps, so this does not invalidate subtracting one for N−1 intervals. It does mean the numerator must not be described as exclusively tokens generated after first emission, and the result is not a pure post-prefill kernel rate. Saved aggregate histograms do not recover first-batch cardinality, and none is assumed here. Non-streaming SGLang TTFT can nearly equal E2E; that pitfall does not affect these two all-streaming bracket deltas.

vLLM E2E and TTFT are frontend iteration timestamps (`stats.py:350–370,437`), whereas `request_decode_time` is last minus first **engine-core** output timestamp (`stats.py:448`). Consequently Cqc10 ΣE2E−ΣTTFT = 9677.79721069336 s, while saved Σrequest_decode_time = 9677.884510502685 s. The latter yields 25.6333912366; do not mix it silently with SGLang's frontend-adjusted denominator. The common formula uses E2E−TTFT for both.

## Source provenance and reproducibility limits

Source copies and hashes are in `results/agent-workload/metric-semantics-source-review/MANIFEST.json`. Six small source files were extracted from **stopped**, uniquely named containers; none was started and all created container objects were removed. Original run/source files were unchanged.

The retained vLLM image ID and digest match Cqc10's original `fixed32_container_identity.json`. SGLang was extracted from the retained `lmsysorg/sglang:qwen38-27b` image named by historical launch records, ID `0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`, created August 14. Its base-build OCI commit label is not equated with the separately recorded qwen38 runtime package version. The original SGLang containers were removed without full loaded-module attestation; this audit does not manufacture retrospective byte-level identity proof. The source behavior is directly consistent with the independently matched saved counters and histograms. No new inference is needed to compute or report these descriptive common-estimator values.
