# Track B Real-Workload 5x Audit

Generated: 2026-05-06

## Objective

Build the Track B auto-research loop from `docs/reports/auto_research/l0-warm-decode-quality-bounded-track-20260505.md`, run it against the same real vLLM workload measurement shape used by `docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md`, and optimize toward a 5x warm decode target.

Concrete success criteria:

- Use the real heavy workload descriptor `benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml`.
- Measure 5 completions per task, discarding the first cold completion and counting the next 4 warm completions.
- Keep the final round target at `37.5 tok/s` (`5x` over `7.5 tok/s` baseline).
- Accept a candidate for deeper gates when it improves at least `20%` over the previous best measured real-workload warm decode.
- Keep the current committed acceptance gate on vLLM decode-time throughput from `throughput.json`, not the rolling proxy log.
- Follow-up requested: add an explicit measurement surface where total serving throughput over parallel authored workload traces can count as the official metric.

## Implemented Artifacts

- Commit: `2a7d7a3 Add real workload Track B gate`
- Follow-up: runtime-config candidate applicator added after the initial audit so workers can propose real vLLM launch-shape changes via `serve_config.yaml:vllm_config`.
- Follow-up: speculative decode candidate support added so workers can propose vLLM `--speculative-config` via `serve_config.yaml:spec_decode`.
- Follow-up requested: measurement-surface support so workers can explicitly choose authored workload profiles, parallel warm windows, and `decode_time` or `wall_clock_total` throughput accounting.
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
| Let auto-research author candidates | Candidates `001`-`041` were generated through `codex exec` worker calls and controller-owned measurement | Done |
| Allow real runtime launch-shape candidates | Controller supports `vllm_config` overrides converted into tuned-config bundles and applied with `--apply-runtime-config` | Done |
| Allow speculative decode candidates | Controller supports `spec_decode` overrides converted into tuned-config bundles and applied as vLLM `--speculative-config` | Done |
| Allow authored parallel workload throughput candidates | Requested, but not currently enabled in the committed controller; current controller keeps the fixed first-five decode-time gate | Not met |
| Achieve an accepted candidate | Candidate `020` cleared speed preflight at `15.753922 tok/s` but failed B-1 equivalence | Not met |
| Achieve final 5x goal | Best candidate `020` measured `15.753922 tok/s`, below `37.5 tok/s` final target | Not met |
| Run full `50*5` benchmark | Not run because no candidate cleared B-1 after the speed preflight | Not met, intentionally gated |

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
| `016` | runtime config, `max_num_seqs: 8`, `max_num_batched_tokens: 2048`, `gpu_memory_utilization: 0.88`, explicit prefix caching | `7.575187` | `1.010x` | Rejected |
| `017` | spec decode, `ngram`, 1 speculative token, prompt lookup 1-8 | `7.808374` | `1.041x` | Rejected |
| `018` | spec decode, `ngram`, 2 speculative tokens, prompt lookup 1-8 | `7.809454` | `1.041x` | Rejected |
| `019` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 1-16 | n/a | n/a | vLLM launched, then rejected on HTTP 500 during warm workload |
| `020` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 2-8 | `15.753922` | `2.100x` | Rejected on B-1 equivalence |
| `021` | spec decode, `ngram`, 4 speculative tokens, prompt lookup 4-8 | `7.624698` | `1.017x` | Rejected |
| `022` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 3-8 | `7.632903` | `1.018x` | Rejected |
| `023` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 2-4 | n/a | n/a | vLLM launched, then rejected on HTTP 500 during warm workload |
| `024` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 2-6 | `7.937862` | `1.058x` | Rejected |
| `025` | spec decode, `ngram`, 2 speculative tokens, prompt lookup 2-16 | `14.506594` | `1.934x` | Rejected |
| `026` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 4-16 | `7.567844` | `1.009x` | Rejected |
| `027` | spec decode, `ngram`, 2 speculative tokens, prompt lookup 3-16 | `7.653164` | `1.020x` | Rejected |
| `028` | spec decode, `ngram`, 2 speculative tokens, prompt lookup 2-8 | `14.581565` | `1.944x` | Rejected |
| `029` | spec decode, `ngram`, 4 speculative tokens, prompt lookup 2-6 | `7.731502` | `1.031x` | Rejected |
| `030` | spec decode, `ngram`, 5 speculative tokens, prompt lookup 2-8 | n/a | n/a | vLLM launched, then rejected on HTTP 500 during warm workload |
| `031` | kernel selection, FP8 GEMM `cublas`, CUDA graph capture on | n/a | n/a | Rejected before launch by YAML boolean parsing |
| `032` | kernel selection, FP8 GEMM `cublas`, CUDA graph capture on | `7.583328` | `1.011x` | Rejected |
| `033` | kernel selection, attention backend `flashinfer` | `7.606928` | `1.014x` | Rejected |
| `034` | kernel selection, DeltaNet kernel `triton-chunked-delta-v2` | n/a | n/a | Rejected before launch as duplicate serving surface |
| `035` | kernel selection, DeltaNet kernel `triton-chunked-delta-v2` | n/a | n/a | Rejected before launch as duplicate serving surface |
| `036` | runtime config, default prefix/chunked flags plus `kv_cache_dtype: fp8_e5m2` | n/a | n/a | Rejected before launch as duplicate serving surface |
| `037` | kernel selection, attention backend `triton` | `7.522407` | `1.003x` | Rejected |
| `038` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 3-8 | n/a | n/a | Rejected before launch as duplicate serving surface |
| `039` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 4-8 | `7.554970` | `1.007x` | Rejected |
| `040` | spec decode, `ngram`, 3 speculative tokens, prompt lookup 3-6 | n/a | n/a | Rejected on runtime apply failure |
| `041` | kernel selection, torch compile mode `reduce-overhead` | n/a | n/a | Rejected on runtime apply failure |

Candidate `002` proposed a native prefix-cache config, but the live server was already launched with `--enable-prefix-caching`; after the controller was fixed to accept prefix-cache-shaped configs, later candidates still stayed at baseline-level throughput.

Candidate `007` was launched after adding the runtime-config applicator with `--apply-runtime-config` enabled. The worker did not choose the new `vllm_config` surface; it proposed another concurrency-4 request-shaping candidate, so no vLLM restart was needed for that attempt.

Candidates `008`-`012` exercised the runtime-config path. This exposed and fixed loop-infra issues: duplicate request-shaping steering, invalid `kv_cache_dtype` prevalidation, stale tuned-config runtime state, a dedicated Track B proxy port, child `PYTHONPATH` propagation for the proxy process, and default warm concurrency for runtime-only configs. Candidate `012` completed the full runtime-config apply, first-five real-workload measurement, and baseline restore cycle.

Candidate `013` exercised the speculative-decode path. The controller generated a tuned-config bundle with `spec_decode: {method: ngram, num_speculative_tokens: 4, prompt_lookup_min: 2, prompt_lookup_max: 32}` and vLLM launched with `--speculative-config`. vLLM metrics confirmed speculative decoding was active, but the concurrent warm workload hit an HTTP 500 from `/v1/responses`, so the candidate was rejected without a valid throughput measurement. The controller restored the baseline runtime afterward.

Candidate `014` was auto-authored after the speculative-decode failure. It retried the best completed runtime shape from `012` with explicit `enable_prefix_caching: true`; the effective launch was not meaningfully different because prefix caching is already enabled by default in this serving path. It measured `7.584679 tok/s`, below the `9.1680396 tok/s` current preflight threshold. The controller restored the baseline runtime afterward. Follow-up infra now normalizes default-enabled runtime flags in duplicate-surface signatures so future workers do not spend another restart on this effective duplicate.

Candidate `015` increased the same runtime family to `max_num_seqs: 6`. It produced the current best valid first-five measurement at `7.651060 tok/s`, but that is only `1.020x` over the nominal `7.5 tok/s` baseline and below the 20% preflight threshold. It remained blocked at speed and did not advance to B-1/B-2/B-3.

Candidate `016` increased the same runtime family again to `max_num_seqs: 8`. It measured `7.575187 tok/s`, regressing from candidate `015` and failing the updated `9.181272 tok/s` preflight threshold. This reinforces that the tested runtime-capacity family is not the path to the 20% acceptance gate for this workload.

Candidate `017` retried speculative decode with a safer ngram shape after candidate `013` exposed the HTTP 500 failure mode. vLLM accepted `spec_decode: {method: ngram, num_speculative_tokens: 1, prompt_lookup_min: 1, prompt_lookup_max: 8}` and the first-five real-workload measurement completed. It set a new best valid measurement at `7.808374 tok/s`, but that is still only `1.041x` over the nominal `7.5 tok/s` baseline and below the `9.181272 tok/s` preflight threshold.

Candidate `018` increased the stable ngram shape to `num_speculative_tokens: 2` while keeping prompt lookup `1-8`. It completed the first-five real-workload measurement and narrowly improved the best result to `7.809454 tok/s`, but that remained below the updated `9.3700488 tok/s` preflight threshold.

Candidate `019` expanded the same family to `num_speculative_tokens: 3` and prompt lookup `1-16`. vLLM accepted the launch config, but the concurrent warm workload hit an HTTP 500 from `/v1/responses`, so no valid throughput artifact was produced. The measurement harness captured the response body: `EngineCore encountered an issue. See stack trace (above) for the root cause.` The controller restored the baseline runtime afterward.

Candidate `020` narrowed the 3-token speculative surface to prompt lookup `2-8`. It produced the first material speed result in this round: `15.753922 tok/s`, or `2.100x` over the nominal baseline, and cleared the incremental preflight threshold of `9.3713448 tok/s`. It then failed B-1 batch equivalence: `match_rate: 0.5` with two concurrent completions returning empty one-token outputs where serial completions produced eight-token text. The controller rejected it and restored the baseline runtime.

Candidate `021` tried to recover B-1 risk by raising the exact-match lookup floor to `4` while using a 4-token draft budget and lookup max `8`. It completed the first-five real-workload measurement, but the stricter surface lost the speed gain and measured only `7.624698 tok/s`, below the post-`020` preflight threshold of `18.9047064 tok/s`.

Candidate `022` made a narrower B-1 recovery attempt around candidate `020`: it kept the 3-token speculative draft budget, raised the prompt lookup floor from `2` to `3`, and kept lookup max `8`. That avoided the candidate `019` crash shape, but it also lost the candidate `020` speed gain and measured only `7.632903 tok/s`, below the same `18.9047064 tok/s` post-`020` preflight threshold.

Candidate `023` kept candidate `020`'s productive lookup floor (`2`) and 3-token draft budget, but narrowed the lookup max from `8` to `4` to reduce B-1 truncation risk. vLLM accepted the launch and speculative decoding became active, but the concurrent warm workload crashed the EngineCore with `AssertionError: num_required_blocks 5 < len(req_blocks) 6`. The measurement harness captured the `/v1/responses` HTTP 500 body and the controller saved the runtime stack trace in `candidates/023/runtime_logs_on_failure.log` before restoring baseline.

Candidate `024` tested the midpoint between the fast-but-B-1-failing `020` (`2-8`) and the crashing `023` (`2-4`) by using lookup `2-6`. It completed the first-five real-workload measurement without an EngineCore crash and reached `7.937862 tok/s`, the best stable non-`020` speculative measurement so far, but still far below the `18.9047064 tok/s` post-`020` preflight threshold.

Candidate `025` moved out of the exhausted depth-3/min-2 ngram family after a controller steering update. It used a smaller 2-token speculative draft with lookup `2-16`, completed the first-five real-workload measurement, and reached `14.506594 tok/s` (`1.934x` over nominal baseline). That is close to candidate `020`'s speed but still below the `18.9047064 tok/s` gate required to be considered a new candidate over the previous max, so B-1/B-2/B-3 did not run.

Candidate `026` kept the 3-token draft depth but raised the prompt lookup floor to `4` with max `16`. The stricter floor was stable but mostly eliminated accepted speculative tokens on the measured workload, producing only `7.567844 tok/s`, effectively baseline and below the post-`020` acceptance gate.

Candidate `027` raised candidate `025`'s prompt lookup floor from `2` to `3` while keeping the 2-token speculative draft budget and lookup max `16`. It completed the first-five real-workload measurement, but the higher floor lost candidate `025`'s speed and measured only `7.653164 tok/s`, below the `18.9047064 tok/s` post-`020` acceptance gate.

Candidate `028` tested a 2-token speculative draft with lookup `2-8`, directly mirroring candidate `020`'s lookup window while reducing the draft depth to lower B-1 risk. It completed measurement at `14.581565 tok/s` (`1.944x` over baseline), the strongest 2-token ngram result so far, but still below the `18.9047064 tok/s` post-`020` acceptance gate, so B-1/B-2/B-3 did not run.

Candidate `029` was launched after adding controller steering away from the plateaued 2-token ngram family. It used a 4-token speculative draft with lookup `2-6`, but measured only `7.731502 tok/s`, effectively baseline and below the post-`020` gate. This rules out the tested deeper-draft/narrow-window recovery path.

Candidate `030` was launched after adding a new controller surface for `kernel_selection`, although the worker chose another nonlocal ngram shape instead of using that surface. It used a 5-token speculative draft with lookup `2-8`; vLLM launched, but the concurrent warm workload crashed the EngineCore with `AssertionError: num_required_blocks 7 < len(req_blocks) 8`. The controller captured the `/v1/responses` HTTP 500 body and saved the runtime stack trace in `candidates/030/runtime_logs_on_failure.log` before restoring baseline.

Candidate `031` was the first auto-authored `kernel_selection` candidate. It selected `fp8_gemm_kernel: cublas` with `cuda_graph_capture: on`, but unquoted YAML `on` loaded as boolean `True` and the controller rejected the candidate before launch as `unsupported_kernel_selection:cuda_graph_capture=True`. The controller parser was then fixed to normalize YAML boolean `cuda_graph_capture` values back to the intended `on`/`off` enum before runtime activation validation.

Candidate `032` retried the first `kernel_selection` surface after the parser fix. The controller launched vLLM with `fp8_gemm_kernel: cublas` and `cuda_graph_capture: on`; runtime activation resolved that to Torch scaled-MM FP8 GEMM and full CUDA graph capture. The first-five real-workload measurement completed at `7.583328 tok/s`, effectively baseline and below the `18.9047064 tok/s` 20%-over-previous-best gate, so B-1/B-2/B-3 did not run. The controller restored the baseline runtime afterward.

Candidate `033` continued the `kernel_selection` surface and selected `attention_backend: flashinfer`. The runtime log confirmed vLLM launched with `--attention-backend FLASHINFER` and the main attention backend resolved to `FLASHINFER`. The first-five real-workload measurement completed at `7.606928 tok/s`, effectively baseline and below the same `18.9047064 tok/s` gate, so B-1/B-2/B-3 did not run. The controller restored the baseline runtime afterward.

Candidates `034` and `035` both selected `kernel_selection: {deltanet_kernel: triton-chunked-delta-v2}`. The controller rejected both before launch as `duplicate_serving_surface` because that value resolves to the already-active default GDN prefill path for this model. Follow-up controller steering now tells the worker not to retry that baseline-equivalent DeltaNet axis.

Candidate `036` moved back to `vllm_config` but selected default prefix/chunked flags plus `kv_cache_dtype: fp8_e5m2`. The controller rejected it before launch as `duplicate_serving_surface`; follow-up steering now tells the worker not to retry default-runtime bookkeeping knobs.

Candidate `037` selected `kernel_selection: {attention_backend: triton}`. The controller launched vLLM with `--attention-backend TRITON_ATTN`, and the runtime log confirmed the main attention backend resolved to `TRITON_ATTN`. The first-five real-workload measurement completed at `7.522407 tok/s` by decode-time accounting, below the `18.9047064 tok/s` post-`020` preflight gate. The same throughput artifact records `wall_decode_tokens_per_s: 28.381712` for the four concurrent warm requests, but that is an aggregate concurrency observation, not the official acceptance metric. The controller restored the baseline runtime afterward.

Candidate `038` selected `spec_decode: {method: ngram, num_speculative_tokens: 3, prompt_lookup_min: 3, prompt_lookup_max: 8}`. The controller rejected it before launch as `duplicate_serving_surface` because that serving surface was already tested by candidate `022`.

Candidate `039` selected `spec_decode: {method: ngram, num_speculative_tokens: 3, prompt_lookup_min: 4, prompt_lookup_max: 8}` to raise the lookup floor further from the fast-but-B-1-failing candidate `020`. vLLM accepted the launch and speculative decoding became active, but the first-five real-workload measurement completed at only `7.554970 tok/s`, below the `18.9047064 tok/s` post-`020` gate. The controller restored the baseline runtime afterward.

Candidate `040` selected `spec_decode: {method: ngram, num_speculative_tokens: 3, prompt_lookup_min: 3, prompt_lookup_max: 6}`. The controller generated a tuned-config bundle, but the vLLM runtime apply failed before a real-workload throughput artifact was produced, so the candidate was rejected as `runtime_config_apply_failed`.

Candidate `041` selected `kernel_selection: {torch_compile_mode: reduce-overhead}`. The controller generated a tuned-config bundle and attempted to launch vLLM with the corresponding compilation config, but runtime apply failed before measurement. The controller restored the baseline runtime afterward; `/health` returned 200 after restore.

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
- `best_decode_tps: 15.753922`
- `incremental_candidates: [020, 025, 028]`
- `promoted_candidates: []`

The loop is no longer blocked at the initial speed preflight: candidates `020`, `025`, and `028` cleared that initial `9.0 tok/s` threshold. It is now blocked at preserving candidate `020`'s speed while satisfying B-1 quality/equivalence and producing a new 20%-over-previous-best candidate. B-2/B-3 were not run because B-1 failed. Candidates `013`, `019`, `023`, and `030` are excluded from `best_decode_tps` because they failed the real-workload measurement instead of producing valid warm decode metrics. Candidate `020` is the current best valid decode-time speed measurement, but it is still below the final 5x target and is not promotable because B-1 failed.

## Blocker

The current live runtime surface has produced one material speedup via ngram speculative decoding, but that candidate did not preserve the B-1 equivalence guard. Request shaping, native prefix-cache variations, and tested vLLM launch-shape mutations remain near baseline when measured with the CUTLASS-style decode metric.

The controller still needs an explicit total-throughput measurement surface before `wall_clock_total` accounting over parallel authored workload traces can be official. The committed controller currently keeps the fixed first-five decode-time gate.

The next productive Track B branch is not more concurrency search. It should be one of:

1. Continue speculative-decode auto-search around candidate `020` only if the next candidate explicitly addresses the B-1 empty-output/equivalence failure while preserving the speed gain.
2. Add a real candidate surface for `xgrammar` / guided decoding and run it only on tool-call-heavy workload slices where constrained generation can affect decode.
3. Install LMCache or another KV-transfer path before launching a cache-oriented Track B round, because LMCache is not present in the current container.
4. Continue runtime-config auto-search only if it explores a new launch surface beyond the tested `max_num_batched_tokens`/`max_num_seqs`/`gpu_memory_utilization` variants, because the first completed runtime-config candidate only reached `7.640033 tok/s`.

Until a candidate preserves B-1 while keeping candidate `020`'s speed gain, the round remains blocked from B-2/B-3 and promotion.
