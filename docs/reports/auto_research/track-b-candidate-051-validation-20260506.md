# Track B Candidate 051 Validation Recheck

Generated: 2026-05-06

## Objective

Validate candidate `051` directly, outside the stopped auto-research loop:

- Launch the already-authored `051` tuned-config bundle.
- Re-measure speed on the real Track B first-five workload shape.
- Re-run B-1/B-2/B-3 correctness gates explicitly.
- Record what was verified, how it was verified, and the result.

## Candidate Under Test

- Bundle: `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/051/tuned_config_bundle.yaml`
- Model: `qwen3.5-27b`
- Weight version: `2e1b21350ce589fcaafbb3c7d7eac526a7aed582`
- Serving change: vLLM ngram speculative decoding
- Spec decode config: `method=ngram`, `num_speculative_tokens=4`, `prompt_lookup_min=7`, `prompt_lookup_max=8`
- Launch evidence: `/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.5-27b.log` recorded `--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 8, "prompt_lookup_min": 7}'`.

## Speed Verification

Speed was rechecked as a warm-cache first-five measurement at `warm_concurrency=1`. That means the harness reset the prefix cache, ran one cold completion, then measured the next four warm completions sequentially.

Command shape:

```bash
scripts/measure_track_b_real_workload.py \
  --workload-file benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml \
  --endpoint http://127.0.0.1:9950/v1 \
  --health-url http://127.0.0.1:9950/health \
  --metrics-url http://127.0.0.1:9950/metrics \
  --reset-prefix-cache-url http://127.0.0.1:9950/reset_prefix_cache \
  --model qwen3.5-27b \
  --task-count 1 \
  --completions-per-task 5 \
  --cold-completions 1 \
  --warm-concurrency 1 \
  --baseline-decode-tps 7.5 \
  --target-multiplier 1.2 \
  --reset-prefix-cache
```

The speed metric is vLLM decode-time throughput from Prometheus deltas: `generation_tokens / decode_sum_s`. This is not wall-clock aggregate throughput and not Codex-agent concurrency.

Primary speed recheck artifact: `candidates/051/speed_recheck_result.json`

- Measured at: `2026-05-06T19:04:24Z`
- Decode throughput: `7.67911 tok/s`
- Baseline: `7.5 tok/s`
- Speedup: `1.024x`
- 20% preflight threshold over fixed baseline: `9.0 tok/s`
- Result: speed recheck failed the 20% threshold.

Because the original `17.087062 tok/s` run looked suspiciously favorable, I ran five independent concurrency-1 speed repeats. Summary artifact: `candidates/051/speed_recheck_c1_summary.json`.

| Run | Artifact | Decode tok/s | Generated Tokens | Result |
|---:|---|---:|---:|---|
| 1 | `speed_recheck_c1_run_01.json` | `7.679110` | `1617` | Fail |
| 2 | `speed_recheck_c1_run_02.json` | `7.670172` | `1447` | Fail |
| 3 | `speed_recheck_c1_run_03.json` | `7.658967` | `1545` | Fail |
| 4 | `speed_recheck_c1_run_04.json` | `7.663855` | `1627` | Fail |
| 5 | `speed_recheck_c1_run_05.json` | `7.713841` | `1483` | Fail |

Summary:

- Mean: `7.677189 tok/s`
- Median: `7.670172 tok/s`
- Min/max: `7.658967` / `7.713841 tok/s`
- Standard deviation: `0.021830 tok/s`
- Acceptance threshold passes: `0/5`
- Cache evidence: every run had `816` prefix-cache hits over `2643` cache queries.

The earlier candidate measurement remains recorded in `candidates/051/throughput.json`:

- Measured at: `2026-05-06T17:06:18Z`
- Decode throughput: `17.087062 tok/s`
- Speedup: `2.278x`

The repeated concurrency-1 recheck did not reproduce that earlier speed. Runtime metrics confirmed the same warm-cache pattern (`816` prefix-cache hits over `2643` queried tokens), but accepted speculative-token throughput was too low to clear the speed gate. I also preserved the earlier accidental `warm_concurrency=4` recheck as `candidates/051/speed_recheck_result_concurrency4.json`; it measured `7.561378 tok/s` and also failed the `9.0 tok/s` threshold.

The task inputs were the same at the prompt/window level: warm prompt-token requests `[1245, 1242, 58, 58]`, max-output caps `[512, 512, 4096, 4096]`, no request overrides, and the same workload distribution ID. The original high-throughput run generated far more measured warm tokens (`5335`) than the concurrency-1 repeats (`1447-1627`). Since the harness does not pin deterministic sampling parameters and vLLM applies the model generation config, natural stop behavior can change output length substantially. This is why the original single run is not sufficient evidence of stable speed.

## Correctness Verification

B-gates were run with `concurrent_requests=1`, matching the requested candidate validation shape. Each gate compares serial and concurrent outputs for exact text equality over short deterministic probes.

| Gate | Artifact | Result | Match Rate | Measured At |
|---|---|---:|---:|---|
| B-1 batch equivalence | `candidates/051/b1_result_recheck_concurrency1.json` | Pass | `1.0` | `2026-05-06T18:34:10Z` |
| B-2 workload equivalence | `candidates/051/b2_result_recheck_concurrency1.json` | Pass | `1.0` | `2026-05-06T18:34:29Z` |
| B-3 workload equivalence | `candidates/051/b3_result_recheck_concurrency1.json` | Pass | `1.0` | `2026-05-06T18:34:59Z` |

The previous concurrency-4 stress run is preserved separately:

- `candidates/051/b1_result_concurrency4.json`: pass, `1.0`
- `candidates/051/b2_result_concurrency4.json`: pass, `1.0`
- `candidates/051/b3_result_concurrency4.json`: fail, `0.75`

## Conclusion

Candidate `051` passed B-1/B-2/B-3 at concurrency 1 on the fresh validation run, but the warm-cache concurrency-1 speed rechecks did not reproduce the earlier `17.087062 tok/s` result. Based on the repeated recheck evidence, I would not call `051` speed-accepted: five runs measured `7.658967-7.713841 tok/s`, all below the `9.0 tok/s` 20% threshold over the fixed `7.5 tok/s` baseline.

The speedup mechanism, when present, is vLLM ngram speculative decoding. It is not serving three Codex agents.
