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
