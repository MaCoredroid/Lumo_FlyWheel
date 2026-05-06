# Track B Auto-Research Candidate Authoring

You are a fresh implementation worker inside a Karpathy-style auto-research loop.
The controller owns measurement, gates, keep/discard, and ledgers. Your job is to author exactly one candidate artifact.

## Hard Rules

- Candidate id: `013`
- Candidate directory: `/home/mark/shared/lumoFlyWheel/output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/013`
- Write only inside that candidate directory.
- Do not edit source files, tests, quality fixtures, prior memory, or round ledgers.
- Do not run expensive live benchmarks; the controller runs gates after you exit.
- Preserve target model weights and sampling behavior.
- Build on prior CUTLASS negative memory; do not propose another tile/schedule/stage mutation unless your config changes the available serving surface.
- Do not repeat an exact serving surface already measured in this round; the controller may reject duplicate surfaces before benchmarking.

## Required Files

Write these files before exiting:

1. `candidate_analysis.md` with these bullets:
   - speed_thesis
   - expected_affected_counter
   - quality_risk
   - why_not_prior_failure

2. `serve_config.yaml` with one of these supported controller surfaces:
   - `request_shaping.target_concurrency: <1-8>` for batching/concurrency experiments
   - `prefix_cache` settings for prefix-cache experiments
   - `vllm_config` runtime overrides for max_num_seqs (1-64), max_num_batched_tokens (1-16384), enable_chunked_prefill (bool), enable_prefix_caching (bool), gpu_memory_utilization (0.0-0.95), max_model_len (1-131072), or kv_cache_dtype (`fp8_e5m2` or `auto` only)
   - `spec_decode` settings for vLLM ngram speculative decoding: method `ngram`, num_speculative_tokens 1-8, prompt_lookup_min 1-16, prompt_lookup_max 1-64

3. Optional `notes.md` with any blocker or measurement caveat.

## Current Objective

- Baseline decode: `7.5` tok/s
- Final target decode: `37.5` tok/s
- Candidate acceptance gate this iteration: `9.168` tok/s (`1.20x` over previous best `7.640` tok/s)
- Speed gate: real vLLM workload window; 5 completions per task, first cold completion discarded, next 4 warm completions counted.
- Best audit so far: `7.640033` tok/s

## Recent Controller Outcomes

Exhausted serving surfaces:

```text
001: rejected 7.346421 tok/s speed_below_target surface={"request_shaping":{"target_concurrency":4}}
002: rejected n/a unsupported_or_missing_serve_config surface={"prefix_cache":{"block_size":16,"enabled":true,"external_tiers":false,"scope":"native_vllm_in_memory"}}
003: rejected 7.327980 tok/s speed_below_target surface={"request_shaping":{"target_concurrency":8}}
004: rejected 7.363091 tok/s speed_below_target surface={"request_shaping":{"target_concurrency":4}}
005: rejected 7.488368 tok/s speed_below_target surface={"request_shaping":{"target_concurrency":2}}
006: rejected 7.367211 tok/s speed_below_target surface={"request_shaping":{"target_concurrency":6}}
007: rejected 7.322860 tok/s speed_below_candidate_acceptance surface={"request_shaping":{"target_concurrency":4}}
008: rejected n/a runtime_config_apply_failed surface={"vllm_config":{"gpu_memory_utilization":0.92,"kv_cache_dtype":"fp8_e4m3","max_num_batched_tokens":8192,"max_num_seqs":8}}
009: rejected n/a runtime_config_apply_failed surface={"vllm_config":{"enable_chunked_prefill":true,"enable_prefix_caching":true,"gpu_memory_utilization":0.9,"kv_cache_dtype":"fp8_e5m2","max_num_batched_tokens":4096,"max_num_seqs":16}}
010: rejected n/a runtime_config_apply_failed surface={"vllm_config":{"max_num_batched_tokens":2048,"max_num_seqs":4}}
011: rejected n/a unsupported_or_missing_serve_config surface={"vllm_config":{"kv_cache_dtype":"fp8_e5m2"}}
012: rejected 7.640033 tok/s speed_below_candidate_acceptance surface={"vllm_config":{"gpu_memory_utilization":0.88,"max_num_batched_tokens":2048,"max_num_seqs":5}}

Controller guidance: request_shaping-only candidates are exhausted; prefer a vllm_config runtime candidate that changes actual vLLM launch capacity.

Controller guidance: vLLM ngram spec_decode is supported and unmeasured in this round; prefer a spec_decode candidate before more launch-shape-only variants.
```

Quality gate history tail:

```tsv
candidate_id	tier	status	score_json	artifact_ref	recorded_at
```

Branch log summary:

```json
[]
```

## Strategy Brief

# Track B Strategy Brief

- source_report: `docs/reports/auto_research/l0-warm-decode-quality-bounded-track-20260505.md`
- extends: `L0c FP8 CUTLASS auto-research`; prior memory at `prior_cutlass_memory.md`
- baseline_decode_tps: 7.5
- target_decode_tps: 37.5
- mode: `round0_prefix_cache`
- workload_trace_sha256: `4bbcfe34a7f703e0d86f9c5ea92abdb157d636501b038e144dbd8343d656a736`

## Bottleneck

- Warm-cache decode is anchored on the FP8 GEMM family: ffn_linear, deltanet_projection_linear, and gatedattn_projection_linear.
- The prior Track A tile/schedule surface is bandwidth bounded and exhausted for the Track B speed target.
- Track B changes serving behavior or runtime bytes-per-token while preserving shipped FP8 target weights.

## Prior CUTLASS Round Memory

- indexed_round_count: 30
- observed_warm_decode: 7.36-7.39 tok/s in May 5 CUTLASS diagnostics
- prior_surface_status: exhausted_for_2x_target
- Do not retry schedule/tile/stage/caller mutations unless a new low-level timing lever proves a material per-kernel win.
- The May 5 speed-gate failures improved only around 0.18-0.24%, so they are explicit negative memory for this objective.

## Required Gates

- B-1 distributional gate before ranking any candidate.
- B-2 behavioral gate for top candidates.
- B-3 full benchmark plus human review before promotion.


## Prior CUTLASS Memory

# Prior CUTLASS Auto-Research Memory

- indexed_round_count: 30
- warm_decode_observed_tps: 7.36-7.39 tok/s in May 5 CUTLASS diagnostics
- track_a_surface_status: exhausted_for_2x_target

## Closeout Reports

- `docs/reports/auto_research/l0c-fp8-cutlass-loop-20260505.md`
- `docs/reports/auto_research/l0c-fp8-cutlass-round-20260505-closeout.md`
- `docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md`

## Negative Memory

- CUTLASS schedule/tile/stage/caller edits left B-weight bytes unchanged.
- Warm speed-gate failures were below 0.25% lift, far below the 2x target.
- MX/NV block-scaled OpClassBlockScaledTensorOp is not a semantics-preserving direct swap for vLLM's FP32-scale path.
- Further CUTLASS-only work needs a new low-level timing lever before full vLLM validation.

## Recent Rounds

- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z` outcome=None terminal=None
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z` outcome=ROUND_BLOCKED terminal=compile_failures_3x
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204103Z` outcome=None terminal=None
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z` outcome=None terminal=None
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z` outcome=ROUND_BLOCKED terminal=compile_failures_3x
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T133749Z` outcome=ROUND_NULL_RESULT terminal=accepted_cap_reached
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T114722Z` outcome=ROUND_BLOCKED terminal=proposer_stuck
- `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T104920Z` outcome=None terminal=None
