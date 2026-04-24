# AutoResearch Round Candidate Report: qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z

Generated: 2026-04-24T07:08:38Z

## Summary

- Round outcome: `ROUND_INFEASIBLE`
- Stopping reason: `no_feasible_rescreen_winner`
- Bundle path: `-`
- Model: `qwen3.5-27b`
- Family: `proposal-ranking-manager-judgment`
- Weight version: `2e1b21350ce589fcaafbb3c7d7eac526a7aed582`
- Harness: `real`
- Round branch: `autoresearch/qwen3.5-27b/proposal-ranking-manager-judgment/sprint-0/20260424T034004Z`
- Rows measured: `16` total, `2` baselines, `12` main-loop candidates, `2` rescreens
- Feasible rows: `16`
- Rescreened rows: `2`
- Noise floor: `0.0`

No tuned bundle was accepted. The terminal result is `ROUND_INFEASIBLE` because no rescreened row remained eligible for winner selection: both rescreen rows were marked `inconsistent_rescreen`. Serving should therefore retain the default baseline config.

## Data Sources

- `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z/round_spec.yaml`
- `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z/round_result.json`
- `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z/results.tsv`
- `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z/rescreen_trace.json`
- `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z/holdout_trace.json`
- `output/auto_research/.../candidates/<iteration>/candidate.yaml`
- `output/auto_research/.../candidates/<iteration>/measurement_trace.json`
- Round branch commit trailers, read from `git log` on the round branch

## Baseline Fallback Config

```yaml
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.9
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
```

## Candidate Results

| Iteration | Status | Feasible | Eval throughput | Objective mean | TTFT p95 ms | TPOT p95 ms | Turn p95 ms | Rollout throughput | Target concurrency | Config summary | UUID | Commit | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| baseline_a | baseline | true | 0.006667 | - | 50730.7 | 85.649 | 144945 | 6.667 | 4 | max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_model_len=131072, kv_cache_dtype=fp8_e5m2 | 92c6559a-55ac-49e9-925f-3b1f52de3666 | 7b616f9db041 | default-config baseline replay a |
| baseline_b | baseline | true | 0.006667 | - | 41465.6 | 85.153 | 118473 | 6.667 | 4 | max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_model_len=131072, kv_cache_dtype=fp8_e5m2 | b262c644-9274-45e3-991b-4d4ea6bd99d3 | f198f63ac452 | default-config baseline replay b |
| 001 | discard | true | 0.006667 | - | 41481.2 | 85.24 | 118518 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | 618e2036-1219-46df-a805-cc8ea059ba84 | 86bcc2bbcc30 | feasible and stable, but eval_throughput stayed at 0.006667 despite sustained_concurrency 16, so there was no gain over baseline |
| 002 | discard | true | 0.006667 | - | 50789.2 | 85.748 | 145112 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=False, gpu_memory_utilization=0.92, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | 0f06010a-a0b0-4e96-909a-e2b1288f8c71 | 2fcbbd7deb06 | Feasible and stable, but eval_throughput stayed at 0.006667 and disabling prefix caching regressed TTFT_p95 to 50789 ms with prefix-cache hit rate still 0.0. |
| 003 | discard | true | 0.006667 | - | 50781.2 | 85.735 | 145089 | 6.667 | 4 | max_num_seqs=4, max_num_batched_tokens=6144, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.7, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | 60745b03-a4ff-463c-88f1-8de477eb5ce7 | c11ab3f3bef8 | Feasible and stable, but eval_throughput stayed at 0.006667 and TTFT_p95/turn_latency_p95 regressed to 50781 ms/145089 ms versus candidate-001. |
| 004 | discard | true | 0.006667 | - | 50759.4 | 85.698 | 145027 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=4096, enable_chunked_prefill=False, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=16384, kv_cache_dtype=fp8_e5m2 | 692d3b35-84dd-43ef-bdf3-4ec48661ea44 | d77099eadf4f | Feasible and stable, but eval_throughput stayed at 0.006667 and disabling chunked prefill regressed TTFT_p95/turn_latency_p95 to 50759 ms/145027 ms versus candidate-001. |
| 005 | discard | true | 0.006667 | - | 50744.5 | 85.672 | 144984 | 6.667 | 24 | max_num_seqs=24, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | 2494a48f-a344-4abe-bb8a-0beccbe59865 | 9f0ff7296480 | Feasible and stable at sustained_concurrency 24, but eval_throughput stayed at 0.006667 and TTFT_p95/turn_latency_p95 regressed to 50744 ms/144984 ms versus candidate-001. |
| 006 | discard | true | 0.006667 | - | 50746.4 | 85.676 | 144990 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=28672, kv_cache_dtype=fp8_e5m2 | 2a80ea07-547d-42f9-8e58-0ec036c35fb6 | e3092d78d241 | Feasible and stable, but eval_throughput stayed at 0.006667 and reducing max_model_len to 28672 regressed TTFT_p95/turn_latency_p95 to 50746 ms/144990 ms versus candidate-001. |
| 007 | discard | true | 0.006667 | - | 48095.9 | 85.453 | 137417 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=6144, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | 191cd0ef-4bd9-476a-99d4-0ae3be185dc6 | c29fd2392670 | Feasible and stable, but eval_throughput stayed at 0.006667 and TTFT_p95/turn_latency_p95 at 48096 ms/137417 ms still trailed candidate-001. |
| 008 | discard | true | 0.006667 | - | 50745 | 85.673 | 144986 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=4096, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | 712be8c7-c5a0-48a1-97d6-fea55c9e530c | 823d734658e6 | Feasible and stable, but eval_throughput stayed at 0.006667 and max_num_batched_tokens=4096 regressed TTFT_p95/turn_latency_p95 to 50745 ms/144986 ms versus candidate-001 and candidate-007. |
| 009 | discard | true | 0.006667 | - | 50777.2 | 85.539 | 145078 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=12288, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | a3e249c9-6ea9-4445-b81e-1ff12623f338 | 1b0270cfe3e2 | Feasible and stable, but eval_throughput stayed at 0.006667 and TTFT_p95/turn_latency_p95 regressed to 50777 ms/145078 ms in measurement_trace.json. |
| 010 | discard | true | 0.006667 | - | 50758.2 | 85.696 | 145023 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.88, max_model_len=32768, kv_cache_dtype=fp8_e5m2 | f101a83c-e86c-4707-994e-0a6e82dc1cd0 | 715694fd2757 | Feasible and stable, but eval_throughput stayed at 0.006667 and gpu_memory_utilization=0.88 regressed TTFT_p95/turn_latency_p95 to 50758 ms/145023 ms. |
| 011 | discard | true | 0.006667 | - | 50770.6 | 85.717 | 145059 | 6.667 | 32 | max_num_seqs=32, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=16384, kv_cache_dtype=fp8_e5m2 | 1576c730-d159-484f-a8ac-34ffbcbb571e | 6bcb52730b83 | Feasible and stable at sustained_concurrency 32, but eval_throughput stayed at 0.006667 and TTFT_p95/turn_latency_p95 regressed to 50771 ms/145059 ms in measurement_trace.json. |
| 012 | discard | true | 0.006667 | - | 47133.5 | 85.303 | 134667 | 6.667 | 16 | max_num_seqs=16, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.92, max_model_len=16384, kv_cache_dtype=fp8_e5m2 | edbacc25-71f3-4290-acf7-061057488c0b | 598f883e6081 | Feasible and stable, but eval_throughput stayed at 0.006667 and max_model_len=16384 still trailed candidate-001 on TTFT_p95/turn_latency_p95 at 47133 ms/134667 ms. |
| rescreen_01 | rescreened | true | 0.002667 | 0.005 | 50760.8 | 85.7 | 145031 | 2.667 | 4 | max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_model_len=131072, kv_cache_dtype=fp8_e5m2 | 5e5d2e8f-845d-4a40-baea-20590b8dbed5 | e8f78212e8ec | inconsistent_rescreen |
| rescreen_02 | rescreened | true | 0.002667 | 0.005 | 50719.2 | 85.63 | 144912 | 2.667 | 4 | max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_model_len=131072, kv_cache_dtype=fp8_e5m2 | f20d92e3-f9ac-4fee-afc5-b2df74e174d7 | 147fe9dc6b90 | inconsistent_rescreen |

## Rescreen Interpretation

| Rescreen | Parent iteration | Parent UUID | Rescreen eval | Objective mean | Objective CI 95 | Notes | Rescreen commit |
|---|---|---|---:|---:|---:|---|---|
| rescreen_01 | baseline_a | 92c6559a-55ac-49e9-925f-3b1f52de3666 | 0.002667 | 0.005 | 0.004 | inconsistent_rescreen | e8f78212e8ec |
| rescreen_02 | baseline_b | b262c644-9274-45e3-991b-4d4ea6bd99d3 | 0.002667 | 0.005 | 0.004 | inconsistent_rescreen | 147fe9dc6b90 |

Both rescreens were feasible but inconsistent with their parent screen throughput. Because `noise_floor` was `0.0`, the drop from `0.006667` to `0.002667` made both rows ineligible for winner contention.

## Holdout Trace Note

- A holdout trace exists at `output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T034004Z/holdout_trace.json`.
- Holdout candidate UUID: `618e2036-1219-46df-a805-cc8ea059ba84`
- Holdout candidate iteration: `001`
- Holdout pass: `true`
- Holdout eval throughput: `0.002`

This holdout trace is retained as raw evidence, but it is not an accepted winner. It was produced before the resume fix that changed no eligible rescreens from a generic blocked exception into the terminal `ROUND_INFEASIBLE` result.

## Final Decision

The round evaluated stable candidates, but all main-loop rows were committed as `discard`, the only rescreened rows were baseline replays, and both rescreens were inconsistent. The data-backed decision is to emit no tuned bundle and keep the baseline/default vLLM config for `qwen3.5-27b` on `proposal-ranking-manager-judgment`.
