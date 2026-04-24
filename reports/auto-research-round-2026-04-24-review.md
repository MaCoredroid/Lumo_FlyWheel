# AutoResearch Round Review - 2026-04-24

Review target: `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T072126Z`

Exploratory L2 target: `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-2-20260424T163930Z`

Source docs: `docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md`, `docs/HLD-Serving-Backend-AutoResearch-v0_1.md`

## Summary

The sprint-0 L1 round completed and emitted bundle `output/tuned_configs/proposal-ranking-manager-judgment/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260424T1018150000_4578f36a.yaml`. The selected config was candidate `003`: `max_num_seqs=4`, `max_num_batched_tokens=12288`, `enable_chunked_prefill=true`, `enable_prefix_caching=false`, `gpu_memory_utilization=0.92`, `max_model_len=131072`, `kv_cache_dtype=fp8_e5m2`.

The bundle is locally reproducible from round artifacts, but the L1 performance signal is weak. The double-baseline noise floor is `0.002476` req/s, while candidate `003` improved over the stronger baseline screen replay by only `0.000954` req/s. Its paired raw mean over screen plus rescreen was `0.008746` req/s, only `0.000264` req/s above baseline_a's paired mean of `0.008482` req/s. Absolute latency is also far outside the family SLOs, but v0.1 feasibility did not gate on latency.

Recommendation: keep the bundle as an exploratory artifact, but do not treat the L1 result as a production-quality throughput win. Before further promotion or L2 continuation, repeat baseline and candidate `003` with a larger representative replay, nonzero thinking-token cases, and either latency gating or an explicit latency-regression acceptance rule. Treat the current L2 sprint-2 run as exploratory only until request-shaping fields are enforced by the real serving path.

## Round Mechanics

`round_spec.yaml` records an L1 real-harness round for model `qwen3.5-27b`, family `proposal-ranking-manager-judgment`, workload `prmj-v1-live`, target concurrency `4`, `iteration_cap=12`, `rescreen_top_k=3`, screen profile `120s` warmup plus `600s` measurement, full profile `300s` warmup plus `1500s` measurement, and latency ceilings of `35000ms` TTFT/turn plus `80ms` TPOT.

`results.tsv` has 17 rows: 2 baseline replays, 12 main-loop candidates, and 3 rescreen rows. All rows are feasible under the v0.1 stability gates: `window_completed=true`, `no_oom_events=true`, `reasoning_content_purity=1.0`, and `determinism_pass_rate=1.0`.

The round finalized successfully:

- `run_log.json`: winner iteration `003`, winner UUID `b027d1e4-f239-4108-b9d9-d391c752cbac`, rescreen UUID `6a53b4b3-97d6-4b84-a113-cea88e14045d`, `holdout_validation=pass`.
- `round_result.json`: `outcome=ROUND_BUNDLE_READY`, `stopping_reason=ok`, `live_gate=not_run`.
- `holdout_trace.json`: holdout passed feasibility for the winner, with holdout `eval_throughput=0.010133` req/s.

## Winner Evidence

Candidate `003` was the best screen row at `0.009775` req/s, with `rollout_throughput=9.775` response tokens/s and `measurement_elapsed_s=409.200`. The rescreen replay for the same candidate measured `0.007717` req/s, producing a rounded `objective_mean=0.009` and `objective_ci_95=0.002` in `results.tsv`. The rescreen delta from the parent screen point is `0.002058` req/s, inside the configured noise floor `0.002476`, so it was not marked `inconsistent_rescreen`.

Caveat: the bundle objective records `value: 0.009775`, which is the parent screen row's throughput, not the raw paired rescreen mean (`0.008746`) or the rounded rescreen row objective (`0.009`). The winner selection path uses rescreen rows, but the bundle metadata is anchored to the parent row's `eval_throughput`.

## Noise Floor

Baseline rows:

- baseline_a: `0.008821` req/s
- baseline_b: `0.007583` req/s
- absolute delta: `0.001238` req/s
- configured noise floor: `2 * delta = 0.002476` req/s

The screen winner's margin over baseline_a was `0.000954` req/s, well below the noise floor. The raw paired means after rescreen were:

- candidate `003`: `(0.009775 + 0.007717) / 2 = 0.008746` req/s
- baseline_a: `(0.008821 + 0.008143) / 2 = 0.008482` req/s
- margin: `0.000264` req/s

This is not enough separation to claim a statistically reliable L1 throughput improvement. The rescreen result mostly shows that candidate `003` was not obviously worse than baseline under this tiny replay, not that it is reliably better.

## Latency Concern

The round's feasibility contract did not gate on latency, and the measured latencies are not usable against the family SLOs. Candidate `003` screen trace recorded TTFT p95 `108866.277ms`, TPOT p95 `183.800ms`, and turn-latency p95 `311046.506ms`, versus ceilings of `35000ms`, `80ms`, and `35000ms`. The rescreen was worse: TTFT p95 `144732.705ms`, TPOT p95 `298.655ms`, and turn-latency p95 `413522.015ms`.

This should be treated as an absolute usability blocker for any promotion decision, even though v0.1 AutoResearch marked the candidate feasible because only stability gates were active.

## Audit Findings

### eval_throughput vs rollout_throughput

Finding: the two fields have different units, and the traces are consistent with the code. This is not a req/s vs req/s mismatch bug.

Code path:

- `src/lumo_flywheel_serving/measurement_harness.py:99`: `eval_throughput = len(per_request_latencies) / measurement_elapsed_s`, so units are completed eval requests/s.
- `src/lumo_flywheel_serving/measurement_harness.py:100`: `rollout_throughput = sum(response_tokens) / measurement_elapsed_s`, so units are response tokens/s.

Trace evidence: sprint-0 candidate `003` completed 4 replay requests in `409.200s`, giving `4 / 409.200 = 0.009775` req/s. The replay has `1200 + 1100 + 900 + 800 = 4000` response tokens, giving `4000 / 409.200 = 9.775` tokens/s. The apparent 1000x ratio is explained by the seed trace token total, not a unit bug.

### thinking_tokens are zero

Finding: the workload does not exercise thinking-token behavior.

Source evidence: `benchmark_blueprints/families/proposal-ranking-manager-judgment/seed_trace.jsonl` has four rows and every row has `"thinking_tokens": 0`; `holdout_trace.jsonl` has three rows and every row also has `"thinking_tokens": 0`.

Harness path: `src/lumo_flywheel_serving/measurement_harness.py:260` reads `thinking_tokens` from each seed entry, line 278 includes it in the TPOT token denominator, and line 285 writes it into `per_request_latencies`. The sprint-0 trace appendix below shows total thinking tokens are zero for baseline, winner, and rescreen traces.

Action item: capture or construct seed and holdout traces with representative nonzero thinking-token counts before judging thinking budgets, purity behavior, or thinking-heavy latency. The current round only exercises response-token generation.

### L2 enforcement

Finding: current L2 AutoResearch validates and records request-shaping fields, but real serving enforcement is not implemented. In measurement, only `concurrency_cap_eval` affects the RealMeasurementHarness call.

Code path:

- `src/lumo_flywheel_serving/auto_research.py:2044` through `2060` validates the L2 fields.
- `src/lumo_flywheel_serving/auto_research.py:2069` returns `request_shaping["concurrency_cap_eval"]` as the target concurrency for L2.
- `src/lumo_flywheel_serving/auto_research.py:1902` through `1907` passes that target concurrency into `RealMeasurementHarness.measure`.
- `src/lumo_flywheel_serving/auto_research.py:2077` through `2085` records `real_proxy_enforcement=false` and states that queue depth, KV budget, and priority preemption are not enforced by the current inference proxy.
- `src/lumo_flywheel_serving/auto_research.py:2096` through `2109` writes the request-shaping policy to the trace and clamps reported concurrency/throughput to the applied target.

Sprint-2 trace evidence: every measured L2 row has `request_shaping_enforcement.mode=substrate_measurement_only`, `real_proxy_enforcement=false`, and `target_concurrency_applied` equal to `concurrency_cap_eval`. Candidate `001` sets `concurrency_cap_eval=3` and its trace reports `sustained_concurrency=3.0`; candidates with queue/KV changes but `concurrency_cap_eval=4` still run at `sustained_concurrency=4.0`.

L2 stopped-run status: sprint-2 has 10 result rows through candidate `008`, a `.round.lock`, and candidate `009/agent_session.jsonl`, but no `run_log.json`, `round_result.json`, `rescreen_trace.json`, or `holdout_trace.json`. It has not finalized and produced no L2 bundle. Treat it as exploratory only.

## Next Step

Do not promote based on this round alone. The next concrete step is a measurement-hardening pass, not more L1/L2 search: increase replay sample size, include nonzero thinking-token examples, repeat baseline and candidate `003` enough times to estimate variance, and make latency SLO handling explicit. For L2 specifically, defer queue-depth, KV-budget, and preemption optimization until the inference proxy enforces those fields; otherwise the L2 search is measuring metadata and eval concurrency only.

## Data Appendix

### Candidate Rows

| iter | label | config | eval req/s | rollout tok/s | elapsed s | status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| baseline_a | baseline_a | seqs=4, batch=8192, prefix=true, chunked=true, gpu=0.9, len=131072 | 0.008821 | 8.821 | 453.477 | baseline |
| baseline_b | baseline_b | seqs=4, batch=8192, prefix=true, chunked=true, gpu=0.9, len=131072 | 0.007583 | 7.583 | 527.475 | baseline |
| 001 | candidate-001 | seqs=8, batch=8192, prefix=true, chunked=true, gpu=0.9, len=32768 | 0.007586 | 7.586 | 527.292 | discard |
| 002 | candidate-002 | seqs=4, batch=8192, prefix=false, chunked=true, gpu=0.9, len=131072 | 0.007842 | 7.842 | 510.096 | keep |
| 003 | candidate-003 | seqs=4, batch=12288, prefix=false, chunked=true, gpu=0.92, len=131072 | 0.009775 | 9.775 | 409.200 | keep |
| 004 | candidate-004 | seqs=4, batch=16384, prefix=false, chunked=true, gpu=0.93, len=131072 | 0.007589 | 7.589 | 527.062 | discard |
| 005 | candidate-005 | seqs=4, batch=12288, prefix=false, chunked=true, gpu=0.91, len=131072 | 0.009250 | 9.250 | 432.448 | discard |
| 006 | candidate-006 | seqs=4, batch=12288, prefix=false, chunked=true, gpu=0.92, len=32768 | 0.008147 | 8.147 | 490.992 | discard |
| 007 | candidate-007 | seqs=4, batch=12288, prefix=false, chunked=false, gpu=0.92, len=131072 | 0.008360 | 8.360 | 478.445 | discard |
| 008 | candidate-008 | seqs=4, batch=12288, prefix=true, chunked=true, gpu=0.92, len=131072 | 0.007587 | 7.587 | 527.238 | discard |
| 009 | candidate-009 | seqs=4, batch=12288, prefix=false, chunked=true, gpu=0.93, len=131072 | 0.008947 | 8.947 | 447.055 | discard |
| 010 | candidate-010 | seqs=4, batch=14336, prefix=false, chunked=true, gpu=0.92, len=131072 | 0.007590 | 7.590 | 527.031 | discard |
| 011 | candidate-011 | seqs=4, batch=12288, prefix=false, chunked=true, gpu=0.92, len=65536 | 0.007867 | 7.867 | 508.483 | discard |
| 012 | candidate-012 | seqs=4, batch=13312, prefix=false, chunked=true, gpu=0.92, len=131072 | 0.007588 | 7.588 | 527.122 | discard |

### Rescreen Rows

| iter | parent | eval req/s | objective_mean | ci95 | elapsed s | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| rescreen_01 | b027d1e4 | 0.007717 | 0.009 | 0.002 | 518.325 | - |
| rescreen_02 | 5d736dbd | 0.008143 | 0.008 | 0.001 | 491.202 | - |
| rescreen_03 | ee91ed6f | 0.007592 | 0.008 | 0.000 | 526.848 | - |

### Relevant Trace Fields

| iter | profile | reqs | thinking total | ttft p95 ms | tpot p95 ms | turn p95 ms | cache first/last |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_a | screen | 4 | 0 | 121977.702 | 251.700 | 348507.720 | 0.0/0.0 |
| baseline_b | screen | 4 | 0 | 143180.225 | 241.733 | 409086.357 | 0.0/0.0 |
| 003 | screen | 4 | 0 | 108866.277 | 183.800 | 311046.506 | 0.0/0.0 |
| rescreen_01 | full | 4 | 0 | 144732.705 | 298.655 | 413522.015 | 0.0/0.0 |
| rescreen_02 | full | 4 | 0 | 128781.188 | 217.423 | 367946.251 | 0.0/0.0 |
| rescreen_03 | full | 4 | 0 | 147718.143 | 304.815 | 422051.837 | 0.0/0.0 |
