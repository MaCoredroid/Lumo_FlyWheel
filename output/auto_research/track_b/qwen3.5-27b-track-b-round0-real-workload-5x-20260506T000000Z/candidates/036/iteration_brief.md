# Track B Auto-Research Candidate Authoring

You are a fresh implementation worker inside a Karpathy-style auto-research loop.
The controller owns measurement, gates, keep/discard, and ledgers. Your job is to author exactly one candidate artifact.

## Hard Rules

- Candidate id: `036`
- Candidate directory: `/home/mark/shared/lumoFlyWheel/output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/candidates/036`
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
   - `kernel_selection` settings for repo-owned vLLM launch choices: attention_backend (`flashinfer`, `triton`, `flash-attn-3`, `flash-attn-4`, or `vllm-default`), fp8_gemm_kernel (`cublas` or `cutlass`), torch_compile_mode (`default`, `reduce-overhead`, `max-autotune`, or `max-autotune-no-cudagraphs`), cuda_graph_capture (`on` or `off`), deltanet_kernel (`triton-chunked-delta-v2`)

3. Optional `notes.md` with any blocker or measurement caveat.

## Current Objective

- Baseline decode: `7.5` tok/s
- Final target decode: `37.5` tok/s
- Candidate acceptance gate this iteration: `18.905` tok/s (`1.20x` over previous best `15.754` tok/s)
- Speed gate: real vLLM workload window; 5 completions per task, first cold completion discarded, next 4 warm completions counted.
- Best audit so far: `15.753922` tok/s

## Recent Controller Outcomes

Exhausted serving surfaces:

```text
024: rejected 7.937862 tok/s speed_below_candidate_acceptance surface={"spec_decode":{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":6,"prompt_lookup_min":2}}
025: rejected 14.506594 tok/s speed_below_candidate_acceptance surface={"spec_decode":{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":16,"prompt_lookup_min":2}}
026: rejected 7.567844 tok/s speed_below_candidate_acceptance surface={"spec_decode":{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":16,"prompt_lookup_min":4}}
027: rejected 7.653164 tok/s speed_below_candidate_acceptance surface={"spec_decode":{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":16,"prompt_lookup_min":3}}
028: rejected 14.581565 tok/s speed_below_candidate_acceptance surface={"spec_decode":{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":8,"prompt_lookup_min":2}}
029: rejected 7.731502 tok/s speed_below_candidate_acceptance surface={"spec_decode":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":6,"prompt_lookup_min":2}}
030: rejected n/a throughput_measure_failed surface={"spec_decode":{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":2}}
031: rejected n/a unsupported_kernel_selection:cuda_graph_capture=True surface={"kernel_selection":{"cuda_graph_capture":true,"fp8_gemm_kernel":"cublas"}}
032: rejected 7.583328 tok/s speed_below_candidate_acceptance surface={"kernel_selection":{"cuda_graph_capture":"on","fp8_gemm_kernel":"cublas"}}
033: rejected 7.606928 tok/s speed_below_candidate_acceptance surface={"kernel_selection":{"attention_backend":"flashinfer"}}
034: rejected n/a duplicate_serving_surface surface={"kernel_selection":{"deltanet_kernel":"triton-chunked-delta-v2"}}
035: rejected n/a duplicate_serving_surface surface={"kernel_selection":{"deltanet_kernel":"triton-chunked-delta-v2"}}

Controller guidance: request_shaping-only candidates are exhausted; prefer a vllm_config runtime candidate that changes actual vLLM launch capacity.

Controller guidance: tested runtime-capacity variants are flat at baseline-level throughput; avoid another candidate that only changes max_num_seqs, max_num_batched_tokens, or gpu_memory_utilization.

Controller guidance: a spec_decode candidate cleared speed preflight but failed B-1 equivalence with empty or truncated concurrent outputs; the next candidate must explicitly reduce that quality risk while preserving the speculative-decode speed gain.

Controller guidance: ngram spec_decode launched but failed real-workload measurement on a broader shape; avoid retrying aggressive ngram settings without a narrower lookup window or captured server-stack evidence.

Controller guidance: a stable ngram spec_decode candidate produced the best valid measurement but still missed preflight; continue only with a distinct ngram spec_decode shape, not another flat launch-capacity-only variant.

Controller guidance: the local ngram family with num_speculative_tokens=3 and prompt_lookup_min=2 is exhausted: max=8 was fast but failed B-1, max=6 was stable but far below preflight, and max=4 crashed the warm measurement. Do not spend the next candidate on max-only interpolation inside this family; move to a different speculative-depth/minimum pair or a different supported serving surface.

Controller guidance: the 2-token ngram family has plateaued below the post-best acceptance gate: lookup 2-16 and 2-8 were fast-but-insufficient, while lookup 3-16 fell back to baseline. Do not spend the next candidate on 2-token ngram lookup-window interpolation; move to a different serving surface such as kernel_selection or an evidence-backed nonlocal spec_decode shape.

Controller guidance: high-depth ngram shapes with prompt_lookup_min=2 are unstable or flat in this workload; the next candidate should not spend another attempt on num_speculative_tokens>=4 with min=2. Prefer the unmeasured kernel_selection surface before more speculative-depth search.

Controller guidance: kernel_selection.deltanet_kernel=triton-chunked-delta-v2 is baseline-equivalent for this model and has already been rejected as a duplicate serving surface. Do not retry that axis; choose a measured-distinct kernel_selection axis or return to a genuinely new speculative-decode shape.
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
