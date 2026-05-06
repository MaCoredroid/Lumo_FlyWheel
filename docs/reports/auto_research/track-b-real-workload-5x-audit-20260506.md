# Track B Real-Workload 5x Audit

Generated: 2026-05-06

## Objective

Build the Track B auto-research loop from `docs/reports/auto_research/l0-warm-decode-quality-bounded-track-20260505.md`, run it against the same real vLLM workload measurement shape used by `docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md`, and optimize toward a 5x warm decode target.

Concrete success criteria:

- Use the real heavy workload descriptor `benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml`.
- Measure 5 completions per task, discarding the first cold completion and counting the next 4 warm completions.
- Keep the final round target at `37.5 tok/s` (`5x` over `7.5 tok/s` baseline).
- Accept a candidate for deeper gates when it improves at least `20%` over the previous best measured real-workload warm decode.
- Do not treat concurrency-only aggregate proxy throughput as the final answer.

## Implemented Artifacts

- Commit: `2a7d7a3 Add real workload Track B gate`
- Follow-up: runtime-config candidate applicator added after the initial audit so workers can propose real vLLM launch-shape changes via `serve_config.yaml:vllm_config`.
- Follow-up: speculative decode candidate support added so workers can propose vLLM `--speculative-config` via `serve_config.yaml:spec_decode`.
- Round directory: `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z`
- Measurement script: `scripts/measure_track_b_real_workload.py`
- Controller: `scripts/run_track_b_loop.py`
- Audit script: `scripts/audit_track_b_round.py`
- Launch integration: `src/lumo_flywheel_serving/track_b.py`, `src/lumo_flywheel_serving/cli.py`
- Tests: `tests/test_track_b.py`, `tests/test_track_b_runtime_loop.py`

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Build Track B loop infra | `src/lumo_flywheel_serving/track_b.py`, `scripts/run_track_b_loop.py` | Done |
| Reference prior CUTLASS rounds | `prior_cutlass_memory.json`, `prior_cutlass_memory.md` in the round directory | Done |
| Use the same real workload family as the CUTLASS round | `round_spec.yaml` uses `benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml` | Done |
| Measure first 5 completions with 4 warm counted | Candidate throughput artifacts use schema `lumo.track_b.real_workload_first_five.v1`, `cold_completions_discarded: 1`, `warm_completions_measured: 4` | Done |
| Keep final 5x target | `round_spec.yaml` has `target_decode_tps: 37.5` | Done |
| Use 20% incremental candidate preflight | `round_spec.yaml` has `candidate_acceptance_incremental_speedup_at_least: 1.2`; initial preflight is `9.0 tok/s` | Done |
| Let auto-research author candidates | Candidates `001`-`012` were generated through `codex exec` worker calls and controller-owned measurement | Done |
| Allow real runtime launch-shape candidates | Controller supports `vllm_config` overrides converted into tuned-config bundles and applied with `--apply-runtime-config` | Done |
| Allow speculative decode candidates | Controller supports `spec_decode` overrides converted into tuned-config bundles and applied as vLLM `--speculative-config` | Done |
| Achieve an accepted candidate | Best candidate `015` measured `7.651060 tok/s`, below the 20% preflight | Not met |
| Achieve final 5x goal | Best candidate `015` measured `7.651060 tok/s`, below `37.5 tok/s` final target | Not met |
| Run full `50*5` benchmark | Not run because no candidate cleared the `20%` preflight | Not met, intentionally gated |

## Candidate Results

| Candidate | Surface | Warm decode tok/s | Speedup vs 7.5 baseline | Result |
|---|---|---:|---:|---|
| `001` | request shaping, concurrency 4 | `7.346421` | `0.980x` | Rejected |
| `003` | request shaping, concurrency 8 | `7.327980` | `0.977x` | Rejected |
| `004` | request shaping, concurrency 4 | `7.363091` | `0.982x` | Rejected |
| `005` | request shaping, concurrency 2 | `7.488368` | `0.998x` | Rejected |
| `006` | request shaping, concurrency 6 | `7.367211` | `0.982x` | Rejected |
| `007` | request shaping, concurrency 4 | `7.322860` | `0.976x` | Rejected |
| `008` | runtime config, invalid `kv_cache_dtype: fp8_e4m3` | n/a | n/a | Rejected before launch |
| `009` | runtime config, blocked by stale active tuned-config state | n/a | n/a | Rejected before launch |
| `010` | runtime config, `max_num_batched_tokens: 2048` | n/a | n/a | vLLM started, then rejected on proxy-start failure |
| `011` | runtime config, `kv_cache_dtype: fp8_e5m2` only | n/a | n/a | Rejected before launch by missing concurrency default |
| `012` | runtime config, `max_num_seqs: 5`, `max_num_batched_tokens: 2048`, `gpu_memory_utilization: 0.88` | `7.640033` | `1.019x` | Rejected |
| `013` | spec decode, `ngram`, 4 speculative tokens, prompt lookup 2-32 | n/a | n/a | vLLM launched, then rejected on HTTP 500 during warm workload |
| `014` | runtime config, `max_num_seqs: 5`, `max_num_batched_tokens: 2048`, `gpu_memory_utilization: 0.88`, explicit prefix caching | `7.584679` | `1.011x` | Rejected |
| `015` | runtime config, `max_num_seqs: 6`, `max_num_batched_tokens: 2048`, `gpu_memory_utilization: 0.88`, explicit prefix caching | `7.651060` | `1.020x` | Rejected |

Candidate `002` proposed a native prefix-cache config, but the live server was already launched with `--enable-prefix-caching`; after the controller was fixed to accept prefix-cache-shaped configs, later candidates still stayed at baseline-level throughput.

Candidate `007` was launched after adding the runtime-config applicator with `--apply-runtime-config` enabled. The worker did not choose the new `vllm_config` surface; it proposed another concurrency-4 request-shaping candidate, so no vLLM restart was needed for that attempt.

Candidates `008`-`012` exercised the runtime-config path. This exposed and fixed loop-infra issues: duplicate request-shaping steering, invalid `kv_cache_dtype` prevalidation, stale tuned-config runtime state, a dedicated Track B proxy port, child `PYTHONPATH` propagation for the proxy process, and default warm concurrency for runtime-only configs. Candidate `012` completed the full runtime-config apply, first-five real-workload measurement, and baseline restore cycle.

Candidate `013` exercised the speculative-decode path. The controller generated a tuned-config bundle with `spec_decode: {method: ngram, num_speculative_tokens: 4, prompt_lookup_min: 2, prompt_lookup_max: 32}` and vLLM launched with `--speculative-config`. vLLM metrics confirmed speculative decoding was active, but the concurrent warm workload hit an HTTP 500 from `/v1/responses`, so the candidate was rejected without a valid throughput measurement. The controller restored the baseline runtime afterward.

Candidate `014` was auto-authored after the speculative-decode failure. It retried the best completed runtime shape from `012` with explicit `enable_prefix_caching: true`; the effective launch was not meaningfully different because prefix caching is already enabled by default in this serving path. It measured `7.584679 tok/s`, below the `9.1680396 tok/s` current preflight threshold. The controller restored the baseline runtime afterward. Follow-up infra now normalizes default-enabled runtime flags in duplicate-surface signatures so future workers do not spend another restart on this effective duplicate.

Candidate `015` increased the same runtime family to `max_num_seqs: 6`. It produced the current best valid first-five measurement at `7.651060 tok/s`, but that is only `1.020x` over the nominal `7.5 tok/s` baseline and below the 20% preflight threshold. It remained blocked at speed and did not advance to B-1/B-2/B-3.

## Runtime Capability Audit

Live container: `lumo-vllm-l0c-fp8-cutlass-run30`

Observed runtime command includes:

- `--enable-prefix-caching`
- `--enable-chunked-prefill`
- `--max-num-batched-tokens 8192`
- `--max-num-seqs 4`
- `--no-async-scheduling`

Capability checks:

- `vllm.config.speculative.SpeculativeConfig`: present
- `vllm.v1.spec_decode`: present
- `vllm.spec_decode`: absent as a legacy module path, but not required for the vLLM 0.19 `--speculative-config` launch path
- `lmcache`: absent
- `xgrammar`: present
- vLLM help exposes `--speculative-config` and `--kv-transfer-config`. The installed runtime accepted `method: ngram`; `method: ngram_gpu` did not construct cleanly during introspection, so the controller currently allows only `ngram`.

## Completion Audit

`scripts/audit_track_b_round.py` reports:

- `complete: false`
- `target_decode_tps: 37.5`
- `candidate_accept_decode_tps_initial: 9.0`
- `best_decode_tps: 7.651060`
- `incremental_candidates: []`
- `promoted_candidates: []`

The loop is therefore correctly blocked at the speed preflight. B-1/B-2/B-3 were not run because no candidate reached the incremental speed acceptance bar. Candidate `013` is excluded from `best_decode_tps` because it failed the real-workload measurement instead of producing a valid warm decode metric. Candidate `015` is the current best valid measurement, but it is still far below both preflight and the final 5x target.

## Blocker

The current live runtime surface has not produced enough real-workload warm decode speedup. Request shaping, native prefix-cache variations, and tested vLLM launch-shape mutations remain near baseline when measured with the CUTLASS-style decode metric.

The next productive Track B branch is not more concurrency search. It should be one of:

1. Continue speculative-decode auto-search only after capturing the vLLM HTTP 500 body/server traceback and constraining the spec-decode surface to stable combinations.
2. Add a real candidate surface for `xgrammar` / guided decoding and run it only on tool-call-heavy workload slices where constrained generation can affect decode.
3. Install LMCache or another KV-transfer path before launching a cache-oriented Track B round, because LMCache is not present in the current container.
4. Continue runtime-config auto-search only if it explores a new launch surface beyond the tested `max_num_batched_tokens`/`max_num_seqs`/`gpu_memory_utilization` variants, because the first completed runtime-config candidate only reached `7.640033 tok/s`.

Until a candidate changes a more productive runtime surface, continuing the same loop is expected to keep generating rejected candidates around `7.3-7.7 tok/s`.
